#!/usr/bin/env -S colab run --gpu A100 --timeout 900
"""Profile the full search pipeline on a Colab GPU: speed only, development only, never a score.

    colab run --gpu A100 --timeout 900 colab/pipeline_profile.py [branch-or-commit] [--queries=1000]
        [--runtime=engine[,auto,...]]  (the pipeline is measured once per ranker runtime)

The pipeline is the prototype's (S6): tuned BM25F and dense retrieval (tfidf-svd-128), RRF over a 20,000 window, the
GBDT ranker over the top 1,000, two terms facets and page 1. It reads only what members may read (referee/loader.py):
the catalogue, the fit split (to train the ranker) and the test queries' texts; never a judgment. Quality is the
referee's job; this measures latency at batch 1 with the per-stage times (took_ms), and throughput at several batch
sizes, against S6's bar (at least 694 QPS with page-1 p99 at or under 35 ms on an A100). Lexical-only is measured too.
Writes /content/pipeline-profile.json and prints it.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
REF = args[0] if args else "main"
N_QUERIES = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--queries=")), 1000)
# The ranker's runtime (Index.add_ranker): "engine" (its own device walk) was the only one before 2026-11-13.
RUNTIMES = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--runtime=")), "engine").split(",")
# --torch-profile=1: after the throughput runs, one batch of 64 of each kind under torch.profiler (device time by
# kernel, and the number of kernel launches and host syncs per request), to show where a batch's time goes.
TORCH_PROFILE = any(a == "--torch-profile=1" for a in sys.argv[1:])
BATCHES = (1, 16, 64, 128)
S6_QPS, S6_P99_MS = 694, 35.0

t_start = time.time()


def say(msg):
    print(f"[profile {time.time() - t_start:7.1f} s] {msg}", flush=True)


def heartbeat(stop):
    while not stop.wait(60):
        say("still running")


def pct(xs, p):
    import numpy as np
    return float(np.percentile(xs, p)) if xs else None


def lexical(q):
    return {"multi_match": {"query": q, "fields": LEX_FIELDS}}


def pipeline_request(q):
    return {"query": {"hybrid": {"queries": [lexical(q), {"neural": {"embedding": {"query_text": q, "k": 1000}}}]}},
            "rank": {"rrf": {"rank_constant": 60, "window_size": 20000}},
            "rescore": {"window_size": 1000, "s3_ranker": {"model": "gbdt"}},
            "aggs": FACETS, "size": 10}


def lexical_request(q):
    return {"query": lexical(q), "aggs": FACETS, "size": 10}


def deep_request(q):
    """The matched load test's request (colab/fair_bench.py): tuned BM25F, the top 2,000, no aggregations."""
    return {"query": lexical(q), "size": 2000}


# --deep=1: also measure deep_request in-process, asking for the hits as serving does (hits="columns", ids="bytes"):
# the engine's own ceiling for the founder's workload (1,000 QPS at depth 2,000), without HTTP or pickling.
DEEP = any(a == "--deep=1" for a in sys.argv[1:])


def measure(index, make, texts, label, kw=None):
    """Batch-1 latency with stage times, then throughput per batch size (each batch's wall time is its latency)."""
    import numpy as np
    for t in texts[:20]:                                   # warm-up: kernels, caches, the encoder
        index.search(make(t), **(kw or {}))
    lat, stages = [], {}
    for t in texts:
        t0 = time.perf_counter()
        r = index.search(make(t), **(kw or {}))
        lat.append((time.perf_counter() - t0) * 1e3)
        for s, v in (r.get("took_ms") or {}).items():
            stages.setdefault(s, []).append(float(v))
    out = {"batch_1": {"p50_ms": pct(lat, 50), "p99_ms": pct(lat, 99), "mean_ms": float(np.mean(lat))},
           "stage_ms_p50": {s: pct(v, 50) for s, v in stages.items()},
           "stage_ms_p99": {s: pct(v, 99) for s, v in stages.items()}, "throughput": {}}
    say(f"{label}: batch 1 p50 {out['batch_1']['p50_ms']:.2f} ms, p99 {out['batch_1']['p99_ms']:.2f} ms; "
        f"stages p50 {json.dumps({k: round(v, 2) for k, v in out['stage_ms_p50'].items()})}")
    for bs in BATCHES[1:]:
        reqs = [make(t) for t in texts]
        index.search_batch(reqs[:bs], **(kw or {}))        # warm-up at this shape
        walls = []
        c0 = time.process_time()                           # this process's CPU time: the host core it uses
        t0 = time.perf_counter()
        for i in range(0, len(reqs), bs):
            s = time.perf_counter()
            index.search_batch(reqs[i:i + bs], **(kw or {}))
            walls.append((time.perf_counter() - s) * 1e3)
        total = time.perf_counter() - t0
        cpu = time.process_time() - c0
        qps = len(reqs) / total
        out["throughput"][str(bs)] = {"qps": qps, "batch_p50_ms": pct(walls, 50), "batch_p99_ms": pct(walls, 99),
                                      "cpu_ms_per_request": cpu * 1e3 / len(reqs), "cpu_cores": cpu / total}
        say(f"{label}: batch {bs}: {qps:.0f} QPS, batch latency p50 {pct(walls, 50):.1f} ms, p99 {pct(walls, 99):.1f} ms, "
            f"host CPU {cpu * 1e3 / len(reqs):.3f} ms per request ({cpu / total:.2f} cores)")
    return out


def torch_profile(index, make, texts, label, bs=64, kw=None):
    """One batch of ``bs`` under torch.profiler: the top CUDA kernels by device time, and launches and syncs per
    request."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    reqs = [make(t) for t in texts[:bs]]
    kw = kw or {}
    index.search_batch(reqs, **kw)                         # warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        index.search_batch(reqs, **kw)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) * 1e3
    ev = prof.key_averages()
    dev = lambda e: getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)  # noqa: E731
    kernels = [e for e in ev if dev(e) > 0 and e.device_type.name == "CUDA"]
    launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC"))
    syncs = sum(e.count for e in ev if e.key in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync"))
    top = sorted(kernels, key=dev, reverse=True)[:12]
    out = {"batch": bs, "wall_ms": wall, "device_ms": sum(dev(e) for e in kernels) / 1000.0,
           "launches_per_request": launches / bs, "syncs_and_copies_per_request": syncs / bs,
           "top_kernels": [{"name": e.key[:90], "device_ms": dev(e) / 1000.0, "count": e.count} for e in top]}
    say(f"{label} torch profile: wall {wall:.1f} ms, device {out['device_ms']:.1f} ms, launches/request "
        f"{out['launches_per_request']:.0f}, syncs+copies/request {out['syncs_and_copies_per_request']:.1f}")
    return out


def main():
    global LEX_FIELDS, FACETS
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO], check=True)
    # a branch or a full sha fetches by name; a short sha does not, so then fetch every branch and resolve it
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"],
                       check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", REF], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.environ.setdefault("S3_SEARCH_DATA", os.path.join(REPO, "data", "esci"))
    import importlib.util
    missing = [m for m in ("snowballstemmer",) if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    from engine import api
    from engine.rank.train import train_ranker
    from referee import loader, runtime
    env = runtime.environment()
    say(f"{REF} at {commit}; {env['gpu']}")
    loader.fetch()
    ds = loader.load_dataset("esci-us")
    tmpl = json.loads(open("referee/requests/bm25f-tuned.json").read())
    LEX_FIELDS = tmpl["query"]["multi_match"]["fields"]
    FACETS = {"brand": {"terms": {"field": "product_brand.keyword", "size": 20}},
              "color": {"terms": {"field": "product_color.keyword", "size": 20}}}
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["bm25"] = json.loads(open("referee/requests/bm25f-tuned.bm25.json").read())
    m["_meta"]["engine"]["embeddings"] = {"embedding": {"model": "tfidf-svd-128",
                                                       "inputs": ["product_title", "product_brand", "product_color"]}}
    m["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128,
                                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
    say(f"build: {len(ds.catalogue):,} products on cuda, with the tfidf-svd-128 encoder")
    t0 = time.perf_counter()
    index = api.build_index(ds.catalogue, m, device="cuda")
    build_s = time.perf_counter() - t0
    say(f"build: done in {build_s:.1f} s")
    t0 = time.perf_counter()
    model = train_ranker(index, ds.fit, lambda q: {"query": pipeline_request(q)["query"],
                                                   "rank": pipeline_request(q)["rank"]},
                         window=1000, n_random=20, max_queries=2000)
    index.add_ranker("gbdt", model, runtime=RUNTIMES[0])
    train_s = time.perf_counter() - t0
    say(f"ranker: trained on 2,000 fit queries in {train_s:.1f} s")
    texts = [str(t) for t in ds.test_queries["query"].iloc[:N_QUERIES]]
    result = {"kind": "pipeline-profile", "label": "development only, never a score", "ref": REF, "commit": commit,
              "runtime": env, "n_products": int(len(ds.catalogue)), "n_queries": len(texts),
              "build_s": build_s, "ranker_train_s": train_s, "ranker_runtime": RUNTIMES[0],
              "lexical": measure(index, lexical_request, texts, "lexical"),
              "pipeline": measure(index, pipeline_request, texts, "pipeline")}
    best = max(result["pipeline"]["throughput"].values(), key=lambda v: v["qps"] if v["batch_p99_ms"] <= S6_P99_MS else -1)
    result["s6"] = {"bar": {"qps": S6_QPS, "p99_ms": S6_P99_MS},
                    "best_qps_within_p99": best["qps"] if best["batch_p99_ms"] <= S6_P99_MS else None,
                    "page1_p99_ms_batch_1": result["pipeline"]["batch_1"]["p99_ms"]}
    for rt in RUNTIMES[1:]:                       # the same pipeline with the ranker on another runtime
        index.add_ranker("gbdt", model, runtime=rt)
        say(f"pipeline with the ranker on runtime {rt!r} ({type(index.rankers['gbdt']).__name__})")
        result.setdefault("pipeline_by_runtime", {})[rt] = measure(index, pipeline_request, texts, f"pipeline/{rt}")
    if DEEP:
        result["deep"] = measure(index, deep_request, texts, "deep", {"hits": "columns", "ids": "bytes"})
        b = index._st.b
        if hasattr(b, "blocking_sync"):                    # S6-15: the same with the other kind of host wait
            was = b.blocking_sync
            b.blocking_sync = not was
            result["deep_blocking" if not was else "deep_spinning"] = measure(
                index, deep_request, texts, "deep/blocking" if not was else "deep/spinning",
                {"hits": "columns", "ids": "bytes"})
            b.blocking_sync = was
    if TORCH_PROFILE:
        result["torch_profile"] = {}
        kinds = [("lexical", lexical_request), ("pipeline", pipeline_request)] + ([("deep", deep_request)] if DEEP else [])
        for label, make in kinds:
            try:
                result["torch_profile"][label] = torch_profile(
                    index, make, texts, label, kw={"hits": "columns", "ids": "bytes"} if label == "deep" else None)
            except Exception as e:                   # the profile is extra: never lose the run's numbers to it
                result["torch_profile"][label] = {"error": repr(e)[:500]}
                say(f"{label} torch profile failed: {e!r}")
    stop.set()
    json.dump(result, open("/content/pipeline-profile.json", "w"), indent=1)
    say("DONE")
    print(json.dumps(result, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
