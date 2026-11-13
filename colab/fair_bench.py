#!/usr/bin/env -S colab run --gpu A100 --timeout 1200
"""Matched load test, the engine against OpenSearch on one Colab GPU machine: speed only, development only.

    colab run --gpu A100 --timeout 1200 colab/fair_bench.py [branch-or-commit] [--qps=250,500,1000] [--k=2000]
        [--duration=60] [--engines=engine,opensearch]

What is matched, for each engine in turn (never both at once):
- the corpus: every ESCI-US product (1,215,854), through referee/loader.py;
- the queries: the ESCI-US test queries' texts, replayed in the same order;
- the depth: every request asks for the top --k hits (default 2,000), ids and scores only (no _source);
- the interface: HTTP on localhost. The engine behind `python -m serving` (its batching scheduler, query cache off);
  OpenSearch 2.19.1 single node, as the prototype ran it;
- each engine's own tuned lexical request: the engine's tuned BM25F (referee/requests/bm25f-tuned*.json),
  OpenSearch's tuned cross_fields (opensearch-cross-fields-tuned.json, `english` analyzer, as the prototype);
- the load: open loop at each --qps for --duration seconds. Request i is due at t0 + i/qps whatever happened
  before it, and its latency counts from that due time, so a slow engine queues instead of slowing the clock.
- the CPUs: the load generator (4 processes) is pinned to 4 cores; the engine under test gets the others.
  OpenSearch runs on those CPUs; the engine on the GPU plus those CPUs.

Reports, per engine and rate: achieved QPS, latency p50/p95/p99/max from the due time, errors, timeouts, hits per
response (sampled), and the client's send lag (if it falls behind, the client was the limit and the run says so).
Quality at the same depth is the referee's: `referee full --k=2000` for each engine. Writes /content/fair-bench.json.
"""
from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
OS_VERSION = "2.19.1"
OS_HOME = Path(f"/content/opensearch-{OS_VERSION}")
SERVE_DIR = Path("/content/serve")
CLIENT_PROCS = 4
TIMEOUT_S = 10.0


def opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
REF = ARGS[0] if ARGS else "main"
RATES = [float(x) for x in opt("qps", "250,500,1000").split(",")]
K = int(opt("k", "2000"))
DURATION = float(opt("duration", "60"))
ENGINES = opt("engines", "engine,opensearch").split(",")
t_start = time.time()


def say(msg):
    print(f"[fair {time.time() - t_start:7.1f} s] {msg}", flush=True)


def cores():
    n = os.cpu_count() or 8
    client = list(range(min(CLIENT_PROCS, n // 2)))
    server = list(range(len(client), n))
    return client, server


def http_json(method, url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def wait_until(check, what, seconds=900):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            if check():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{what} not ready after {seconds} s")


# ---- the client: open loop, several processes ---------------------------------------------------------------------

def _client_proc(p, n_procs, url, bodies, rate, duration, t0, out_path, cpu):
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass
    import aiohttp

    async def run():
        n_total = int(rate * duration)
        mine = range(p, n_total, n_procs)
        lat, errors, timeouts, hits, lag = [], 0, 0, [], []
        conn = aiohttp.TCPConnector(limit=0, force_close=False)
        tout = aiohttp.ClientTimeout(total=TIMEOUT_S)
        async with aiohttp.ClientSession(connector=conn, timeout=tout) as s:
            async def one(i, due):
                nonlocal errors, timeouts
                try:
                    async with s.post(url, data=bodies[i % len(bodies)],
                                      headers={"Content-Type": "application/json"}) as r:
                        raw = await r.read()
                        if r.status != 200:
                            errors += 1
                            return
                    lat.append((time.time() - due) * 1e3)
                    if i % 97 == 0:
                        hits.append(len(json.loads(raw)["hits"]["hits"]))
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception:
                    errors += 1

            tasks = []
            for i in mine:
                due = t0 + i / rate
                delay = due - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                lag.append((time.time() - due) * 1e3)
                tasks.append(asyncio.create_task(one(i, due)))
            await asyncio.gather(*tasks)
        json.dump({"lat": lat, "errors": errors, "timeouts": timeouts, "hits": hits, "lag": lag,
                   "sent": len(mine)}, open(out_path, "w"))

    asyncio.run(run())


def load(url, bodies, rate, duration, client_cpus):
    t0 = time.time() + 2.0
    procs, outs = [], []
    for p in range(len(client_cpus)):
        out = f"/content/fair-client-{p}.json"
        outs.append(out)
        pr = mp.Process(target=_client_proc, args=(p, len(client_cpus), url, bodies, rate, duration, t0, out,
                                                   client_cpus[p]))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join(duration + TIMEOUT_S + 120)
    lat, hits, lag, errors, timeouts, sent = [], [], [], 0, 0, 0
    for out in outs:
        d = json.load(open(out))
        lat += d["lat"]; hits += d["hits"]; lag += d["lag"]
        errors += d["errors"]; timeouts += d["timeouts"]; sent += d["sent"]
    wall = max(duration, (max(lat) / 1e3) if lat else 0) + 0.0
    import numpy as np

    def pct(x, q):
        return float(np.percentile(x, q)) if x else None

    res = {"target_qps": rate, "sent": sent, "ok": len(lat), "errors": errors, "timeouts": timeouts,
           "achieved_qps": len(lat) / wall,
           "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "max": max(lat) if lat else None},
           "hits_per_response": {"min": min(hits) if hits else None, "median": pct(hits, 50)},
           "client_send_lag_ms_p99": pct(lag, 99)}
    res["client_limited"] = bool(res["client_send_lag_ms_p99"] and res["client_send_lag_ms_p99"] > 50)
    return res


# ---- the engines -------------------------------------------------------------------------------------------------

def start_opensearch(server_cpus):
    if not (OS_HOME / "bin" / "opensearch").exists():
        tgz = Path(f"/content/opensearch-{OS_VERSION}.tar.gz")
        if not tgz.exists():
            say(f"opensearch: downloading {OS_VERSION}")
            urllib.request.urlretrieve(f"https://artifacts.opensearch.org/releases/bundle/opensearch/{OS_VERSION}/"
                                       f"opensearch-{OS_VERSION}-linux-x64.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall("/content")
    subprocess.run(["useradd", "-m", "osuser"], stderr=subprocess.DEVNULL)
    subprocess.run(["chown", "-R", "osuser:osuser", str(OS_HOME)], check=True)
    knn_lib = OS_HOME / "plugins" / "opensearch-knn" / "lib"
    heap = "8g"
    cmd = ["taskset", "-c", ",".join(map(str, server_cpus)), "runuser", "-u", "osuser", "--", "env",
           f"OPENSEARCH_JAVA_OPTS=-Xms{heap} -Xmx{heap}", f"LD_LIBRARY_PATH={knn_lib}",
           str(OS_HOME / "bin" / "opensearch"), "-Ediscovery.type=single-node", "-Eplugins.security.disabled=true",
           "-Ehttp.port=9200", f"-Enode.processors={len(server_cpus)}"]
    proc = subprocess.Popen(cmd, stdout=open("/content/opensearch.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    wait_until(lambda: http_json("GET", "http://localhost:9200")["version"]["number"], "opensearch", 300)
    say(f"opensearch: up, heap {heap}, {len(server_cpus)} processors")
    return proc


def stop(proc):
    if proc and proc.poll() is None:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def opensearch_target(ds, server_cpus):
    os.environ["S3_SEARCH_OS_ANALYZER"] = "english"         # the prototype indexed OpenSearch with `english`
    from referee import opensearch as osref
    proc = start_opensearch(server_cpus)
    t0 = time.time()
    say("opensearch: indexing 1.2M products (the referee's adapter: same mapping, english analyzer)")
    ix = osref.build_index(ds.catalogue, ds.mapping)
    build_s = time.time() - t0
    http_json("POST", f"http://localhost:9200/{ix.name}/_forcemerge?max_num_segments=1", timeout=3600)
    say(f"opensearch: index {ix.name} ready in {build_s:.0f} s (force-merged to one segment)")
    tmpl = json.loads(Path("referee/requests/opensearch-cross-fields-tuned.json").read_text())

    def body(q):
        b = copy.deepcopy(tmpl)
        b["query"]["multi_match"]["query"] = q
        return {**b, "size": K, "_source": False, "track_total_hits": False}

    return proc, f"http://localhost:9200/{ix.name}/_search", body, build_s


def engine_target(ds, server_cpus):
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["bm25"] = json.loads(Path("referee/requests/bm25f-tuned.bm25.json").read_text())
    (SERVE_DIR / "mapping.json").write_text(json.dumps(m))
    src = SERVE_DIR / "products.parquet"
    ds.catalogue.to_parquet(src)
    t0 = time.time()
    if not (SERVE_DIR / "esci").exists():
        say("engine: building the serving index (python -m serving build)")
        subprocess.run([sys.executable, "-m", "serving", "build", "--name", "esci", "--mapping",
                        str(SERVE_DIR / "mapping.json"), "--source", str(src), "--data-dir", str(SERVE_DIR),
                        "--termination-log", str(SERVE_DIR / "build.json")],
                       check=True)
    build_s = time.time() - t0
    env = {**os.environ, "S3_SERVING_CACHE": "0", "S3_SERVING_MAX_BATCH": opt("max-batch", "64"),
           "S3_SERVING_MAX_WAIT_MS": opt("max-wait-ms", "2"), "S3_SEARCH_DEVICE": "cuda"}
    cmd = ["taskset", "-c", ",".join(map(str, server_cpus)), sys.executable, "-m", "serving", "--data-dir",
           str(SERVE_DIR), "--device", "cuda", "--preload", "esci", "--port", "8080"]
    proc = subprocess.Popen(cmd, env=env, stdout=open("/content/serving.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    wait_until(lambda: urllib.request.urlopen("http://localhost:8080/readyz", timeout=5).status == 200,
               "serving", 1800)
    say(f"engine: serving on cuda (index built in {build_s:.0f} s; query cache off, max_batch "
        f"{env['S3_SERVING_MAX_BATCH']}, max_wait {env['S3_SERVING_MAX_WAIT_MS']} ms)")
    tmpl = json.loads(Path("referee/requests/bm25f-tuned.json").read_text())

    def body(q):
        b = copy.deepcopy(tmpl)
        b["query"]["multi_match"]["query"] = q
        return {**b, "size": K}

    return proc, "http://localhost:8080/indexes/esci/_search", body, build_s


def main():
    subprocess.run(["bash", "-lc", f"[ -d {REPO} ] || git clone --quiet {REPO_URL} {REPO}"], check=True)
    subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF], check=True)
    subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.environ.setdefault("S3_SEARCH_DATA", os.path.join(REPO, "data", "esci"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp", "snowballstemmer"], check=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    from referee import loader, runtime
    env = runtime.environment()
    client_cpus, server_cpus = cores()
    say(f"{REF} at {commit}; {env['gpu']}; {os.cpu_count()} CPUs: client {client_cpus}, server {server_cpus}")
    loader.fetch()
    ds = loader.load_dataset("esci-us")
    texts = [str(t) for t in ds.test_queries["query"]]
    say(f"{len(ds.catalogue):,} products, {len(texts):,} query texts, depth {K}, rates {RATES} QPS x {DURATION:.0f} s")
    out = {"kind": "fair-bench", "label": "development only, never a score", "ref": REF, "commit": commit,
           "runtime": env, "cpus": os.cpu_count(), "client_cpus": client_cpus, "server_cpus": server_cpus,
           "n_products": int(len(ds.catalogue)), "k": K, "duration_s": DURATION, "engines": {}}
    for name in ENGINES:
        target = engine_target if name == "engine" else opensearch_target
        proc, url, make_body, build_s = target(ds, server_cpus)
        try:
            bodies = [json.dumps(make_body(t)).encode() for t in texts]
            for b in bodies[:200]:                               # warm-up, one at a time
                urllib.request.urlopen(urllib.request.Request(url, data=b, method="POST",
                                                              headers={"Content-Type": "application/json"}),
                                       timeout=60).read()
            runs = []
            for rate in RATES:
                say(f"{name}: {rate:.0f} QPS for {DURATION:.0f} s")
                r = load(url, bodies, rate, DURATION, client_cpus)
                say(f"{name}: {rate:.0f} QPS -> achieved {r['achieved_qps']:.0f}, p50 {r['latency_ms']['p50']:.1f} ms, "
                    f"p99 {r['latency_ms']['p99']:.1f} ms, errors {r['errors']}, timeouts {r['timeouts']}, "
                    f"hits {r['hits_per_response']}, client lag p99 {r['client_send_lag_ms_p99']:.1f} ms"
                    + (" (CLIENT-LIMITED)" if r["client_limited"] else ""))
                runs.append(r)
                time.sleep(5)
            out["engines"][name] = {"build_s": build_s, "url": url, "runs": runs}
        finally:
            stop(proc)
            time.sleep(5)
        json.dump(out, open("/content/fair-bench.json", "w"), indent=1)
    say("DONE")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork")
    sys.exit(main())
