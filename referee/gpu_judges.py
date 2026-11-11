"""S9's GPU judges for predictions 1 to 3, run by the founder in Colab (`referee/colab_referee.ipynb`, the S9 cell).
Written 2026-11-10 before any GPU result existed (owner: evaluation-steward, P6). Through `engine.api` only.

    python -m referee.gpu_judges [--out=results-s9.json] [--queries=2000]

- **p1** (cuVS ANN): ESCI-US with the mapping's tfidf-svd-128 embeddings and a deterministic integer `s9_bucket` (a
  hash of the product id, 0-9; `s9_bucket == 0` keeps about 10%). One index on `cuda`,
  searched with each method on the request (`exact`, `cagra`, `ivf_pq`, at their default parameters; INTERFACES §5). For `--queries` fit query texts,
  a `neural` knn request (k 10), unfiltered for CAGRA and IVF-PQ and with the bucket pre-filter for filtered CAGRA;
  recall@10 = mean |approx top 10 ∩ exact top 10| / 10. Bars: CAGRA 0.95, filtered CAGRA 0.90, IVF-PQ 0.90. The
  response must name the method that ran (`s9_checks.method_reported`); if it says exact knn ran, the method is "not
  judged: fell back to exact", never a pass.
- **p2** (cuML encoder): ESCI-sample with the tfidf-svd-128 encoder, built on `cuda`: the time inside the encoder's
  fit, fit_transform and encode (whatever `engine.encode.get_encoder` returns, wrapped), bar 2 s; then the bench's
  nDCG@10 for a hybrid request (BM25F plus `neural`, RRF) with the cuda-fitted index against the cpu-fitted one,
  within 0.005; and the cpu bench's lexical nDCG@10 and Exact recall@100 unchanged (0.4644, 0.7551).
- **p3** (GPU GBDT): the reference GBDT trained on ESCI-sample's fit split (`engine.rank.train.train_ranker`), added
  to a cpu and a cuda index; at least 10,000 rescored candidate scores compared (within 1e-5), and the rescored order
  of every fit query (rescore window 100, size 10) compared exactly.

Every number is labelled with the device and GPU. Each part records an exception as "not judged" with the reason.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from referee import bench, loader
from referee.s9_checks import method_reported

DEV = "cuda"   # the device under test; --device=cpu only smoke-tests the code paths (no verdict)
ENC = {"embedding": {"model": "tfidf-svd-128", "inputs": ["product_title", "product_brand", "product_color"]}}


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _build_mode():
    """Which way the next engine build runs (474f9f5); information only, no verdict depends on it."""
    if os.environ.get("S3_BUILD_WORKERS"):
        return f"S3_BUILD_WORKERS={os.environ['S3_BUILD_WORKERS']} (set by the caller)"
    try:
        from engine.build import cuda_maybe_started
    except ImportError:
        return "unknown"
    return "serial: CUDA had already started" if cuda_maybe_started() else "parallel workers: no CUDA work before it"


def _with_embeddings(mapping, method=None, fit="numpy"):
    """The mapping with the tfidf-svd-128 embeddings. `fit` (INTERFACES, S9-2): "numpy" (the reference, the default)
    or "auto" (the GPU fit when CUDA is present). The cuda builds pass "auto"; p2's cpu comparison passes "numpy".
    Fixed 2026-11-11 before any GPU result existed: the first version set no `fit`, so every build fitted in numpy
    (the architect found it)."""
    m = copy.deepcopy(mapping)
    m["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128,
                                    "method": {"name": method or "exact", "space_type": "cosinesimil"}}
    if method is None:
        m["properties"]["embedding"].pop("method")
    enc = copy.deepcopy(ENC)
    enc["embedding"]["fit"] = fit
    m["_meta"]["engine"]["embeddings"] = enc
    return m


def _bucketed(cat):
    cat = cat.copy()
    cat["s9_bucket"] = [int(hashlib.md5(str(p).encode()).hexdigest()[:8], 16) % 10 for p in cat["product_id"]]
    return cat


def _queries(ds, n, seed=20261121):
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    rng = np.random.default_rng(seed)
    return [str(t) for t in q["query"].iloc[np.sort(rng.choice(len(q), min(n, len(q)), replace=False))]]


def p1_ann(n_queries=2000, dataset="esci-us"):
    import engine.api as api
    ds = loader.load_dataset(dataset)
    cat = _bucketed(ds.catalogue)
    base = copy.deepcopy(ds.mapping)
    base["properties"]["s9_bucket"] = {"type": "integer"}
    texts = _queries(ds, n_queries)
    mode = _build_mode()
    t = time.perf_counter()
    exact = api.build_index(cat, _with_embeddings(base, fit="auto"), device=DEV)  # the GPU fit on cuda
    out = {"n_queries": len(texts), "n_docs": len(cat), "build_exact_s": time.perf_counter() - t, "build_mode": mode,
           "bucket_share": float(np.mean(np.asarray(cat["s9_bucket"]) == 0))}
    flt = {"term": {"s9_bucket": 0}}

    def req(text, filtered, method):
        # INTERFACES section 5 (S9-1, 6d1f7f4): the method is on the request, at its default parameters here
        spec = {"query_text": text, "k": 10, "method": method}
        if filtered:
            spec["filter"] = flt
        return {"query": {"neural": {"embedding": spec}}, "size": 10}

    def ids(filtered, method):
        res, ran = [], set()
        for i in range(0, len(texts), 64):
            for r in exact.search_batch([req(t, filtered, method) for t in texts[i:i + 64]]):
                res.append([h["_id"] for h in r["hits"]["hits"]])
                rep = method_reported(r)
                ran.add(rep[0] if rep else None)
        return res, ran

    truth = {False: ids(False, "exact")[0], True: ids(True, "exact")[0]}
    # The method ran as itself when the response says `ran` is the method or "<method>+postfilter" (INTERFACES
    # section 5 after b0d79f5: IVF-PQ answers a filter by post-filtering with exact re-scoring). "exact" (cuVS did
    # not run) is never counted. Filtered IVF-PQ is information only: prediction 1 judges IVF-PQ unfiltered.
    for method, filtered, bar, key in (("cagra", False, 0.95, "cagra"), ("cagra", True, 0.90, "cagra_filtered"),
                                       ("ivf_pq", False, 0.90, "ivf_pq"), ("ivf_pq", True, None, "ivf_pq_filtered_info")):
        try:
            t = time.perf_counter()
            got, ran = ids(filtered, method)
            out[f"search_{key}_s"] = time.perf_counter() - t  # includes building the method's index on first use
            rec = float(np.mean([len(set(a) & set(b)) / max(1, len(b)) for a, b in zip(got, truth[filtered])]))
            entry = {"recall@10": rec, "bar": bar, "method_reported": sorted(map(str, ran))}
            as_itself = bool(ran) and ran <= {method, f"{method}+postfilter"}
            if bar is None:
                entry["verdict"] = "information only (not part of prediction 1)"
            elif not as_itself:
                entry["verdict"] = f"not judged: the response says {sorted(map(str, ran))} ran, not {method}"
            else:
                entry["verdict"] = "held" if rec >= bar else "wrong"
            out[key] = entry
        except Exception as e:  # noqa: BLE001
            out[key] = {"verdict": f"not judged: {type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
    return out


class _EncoderClock:
    """Time inside the engine's encoders (fit, fit_transform, encode; outermost calls only), at class level: every
    class defined in `engine.encode` (and any `engine.encode_*` or `engine.encode.*` module already imported) that has
    those methods, so it does not matter how the engine gets its encoder. A run that observes no call is not judged
    (the 2026-11-10 CPU smoke test found the first version, which wrapped `get_encoder`, observed none)."""

    METHODS = ("fit", "fit_transform", "encode")

    def __init__(self):
        self.s, self.calls, self.depth, self.saved, self.types = 0.0, [], 0, [], set()
        self.fitted_with = set()   # the encoders' own record of the fit that ran (TfidfSvdEncoder.fitted_with)

    def __enter__(self):
        import inspect
        import engine.encode  # noqa: F401
        mods = [m for n, m in list(sys.modules.items()) if m is not None and
                (n == "engine.encode" or n.startswith("engine.encode") or n.startswith("engine.encode."))]
        clock = self
        for mod in mods:
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if not cls.__module__.startswith("engine"):
                    continue
                for name in self.METHODS:
                    if name in cls.__dict__:
                        orig = cls.__dict__[name]

                        def wrapped(obj, *aa, _o=orig, _n=name, _c=cls, **kk):
                            clock.depth += 1
                            t = time.perf_counter()
                            try:
                                return _o(obj, *aa, **kk)
                            finally:
                                clock.depth -= 1
                                if clock.depth == 0:
                                    clock.s += time.perf_counter() - t
                                    clock.calls.append(_n)
                                    clock.types.add(f"{_c.__module__}.{_c.__name__}")
                                    if _n in ("fit", "fit_transform"):
                                        clock.fitted_with.add(str(getattr(obj, "fitted_with", None)))
                        self.saved.append((cls, name, orig))
                        setattr(cls, name, wrapped)
        return self

    def __exit__(self, *exc):
        for cls, name, orig in self.saved:
            setattr(cls, name, orig)


def p2_encoder():
    import engine.api as api
    ds = loader.load_dataset("esci-sample")
    m_gpu = _with_embeddings(ds.mapping, fit="auto")    # the GPU fit (S9-2) that prediction 2 is about
    m_cpu = _with_embeddings(ds.mapping, fit="numpy")   # the reference fit, for the comparison
    out = {"build_mode": _build_mode()}
    with _EncoderClock() as clock:
        t = time.perf_counter()
        gpu_idx = api.build_index(ds.catalogue, m_gpu, device=DEV)
        out["build_cuda_s"] = time.perf_counter() - t
    out["encoder_fit_and_encode_s"] = clock.s
    out["encoder_types"] = sorted(clock.types)
    out["encoder_calls"] = clock.calls
    out["fitted_with"] = sorted(clock.fitted_with)
    if not clock.calls:
        out["fit_verdict"] = "not judged: no encoder fit or encode call was observed"
    elif DEV == "cuda" and not (clock.fitted_with and clock.fitted_with <= {"gpu", "cuml"}):
        out["fit_verdict"] = f"not judged: the encoder says it was fitted with {sorted(clock.fitted_with)}, not on the GPU"
    else:
        out["fit_verdict"] = "held" if clock.s <= 2.0 else "wrong"
    with _EncoderClock() as cclock:
        cpu_idx = api.build_index(ds.catalogue, m_cpu, device="cpu")
    out["cpu_fitted_with"] = sorted(cclock.fitted_with)
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    judg = loader.fit_judgments(ds)
    hybrid = {"query": {"hybrid": {"queries": [bench.DEFAULT_TEMPLATE["query"],
                                               {"neural": {"embedding": {"query_text": "{{query}}", "k": 100}}}]}},
              "rank": {"rrf": {"rank_constant": 60, "window_size": 100}}}
    rg = bench.run(ds, q, judg, api, index=gpu_idx, template=hybrid, device=DEV, latency_queries=50)
    rc = bench.run(ds, q, judg, api, index=cpu_idx, template=hybrid, device="cpu", latency_queries=50)
    out["hybrid_ndcg@10"] = {"cuda": rg["quality"]["ndcg@10"], "cpu": rc["quality"]["ndcg@10"],
                             "diff": abs(rg["quality"]["ndcg@10"] - rc["quality"]["ndcg@10"])}
    out["hybrid_verdict"] = "held" if out["hybrid_ndcg@10"]["diff"] <= 0.005 else "wrong"
    lex = bench.run(ds, q, judg, api, device="cpu", latency_queries=50)
    out["cpu_lexical"] = {"ndcg@10": lex["quality"]["ndcg@10"], "exact_recall@100": lex["quality"]["exact_recall@100"],
                          "unchanged_4dp": round(lex["quality"]["ndcg@10"], 4) == 0.4644 and
                          round(lex["quality"]["exact_recall@100"], 4) == 0.7551}
    return out


def p3_gbdt(min_rows=10_000):
    import engine.api as api
    from engine.rank.train import train_ranker
    from tools.common import text_request
    ds = loader.load_dataset("esci-sample")
    cpu = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    gpu = api.build_index(ds.catalogue, ds.mapping, device=DEV)
    model = train_ranker(cpu, ds.fit, lambda t: text_request(t), window=200, max_queries=300)
    cpu.add_ranker("judge", model)
    gpu.add_ranker("judge", model)
    texts = [str(t) for t in ds.fit["query"].drop_duplicates()]
    rows, worst, order_diff, n_q = 0, 0.0, 0, 0
    for t in texts:
        big = {**text_request(t, size=100), "rescore": {"window_size": 100, "s3_ranker": {"model": "judge"}}}
        a, b = cpu.search(big), gpu.search(big)
        sa = {h["_id"]: h["_score"] for h in a["hits"]["hits"]}
        for h in b["hits"]["hits"]:
            if h["_id"] in sa:
                worst = max(worst, abs(h["_score"] - sa[h["_id"]]))
                rows += 1
        page = {**big, "size": 10}
        if [h["_id"] for h in cpu.search(page)["hits"]["hits"]] != [h["_id"] for h in gpu.search(page)["hits"]["hits"]]:
            order_diff += 1
        n_q += 1
    return {"candidate_rows_compared": rows, "max_abs_diff": worst, "fit_queries": n_q, "orders_differing": order_diff,
            "verdict": "held" if rows >= min_rows and worst <= 1e-5 and order_diff == 0 else
            ("not judged: fewer than 10,000 rows" if rows < min_rows else "wrong")}


def _mark_smoke(d):
    """Every verdict field, at any depth, of a non-cuda run says it is a smoke test with no verdict."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k.endswith("verdict") and isinstance(v, str):
                d[k] = f"smoke test on {DEV}, no verdict (the judge's code said: {v})"
            else:
                _mark_smoke(v)
    elif isinstance(d, list):
        for v in d:
            _mark_smoke(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee.gpu_judges")
    ap.add_argument("--out", default="results-s9.json")
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--parts", default="p1,p2,p3")
    ap.add_argument("--device", default="cuda", help="cuda (the judged run) or cpu (a smoke test of the code; no verdict)")
    ap.add_argument("--p1-dataset", default="esci-us")
    a = ap.parse_args(argv)
    global DEV
    DEV = a.device
    # 474f9f5: no CUDA work before p1's ESCI-US build (the GPU's name is read after the parts), so that build may
    # fork its workers; later builds in this process run serial, as the engine decides (`build_mode` says which).
    import engine.api as api
    res = {"kind": "gpu-judges-s9", "devices": api.available_devices(),
           "label": "score candidate (GPU), the founder's Colab run is the record"}
    if a.device != "cuda":
        res["label"] = "SMOKE TEST on cpu: checks the judges' code only; no verdict"
    if a.device not in res["devices"]:
        res["not_judged"] = "no cuda device: run this in Colab on a GPU"
    else:
        for part, fn in (("p1", lambda: p1_ann(a.queries, a.p1_dataset)), ("p2", p2_encoder), ("p3", p3_gbdt)):
            if part not in a.parts.split(","):
                continue
            try:
                t = time.perf_counter()
                res[part] = fn()
                res[part]["wall_s"] = time.perf_counter() - t
                if DEV != "cuda":   # a smoke test gives no verdict: say so on every verdict field, not only the label
                    _mark_smoke(res[part])
            except Exception as e:  # noqa: BLE001
                res[part] = {"verdict": f"not judged: {type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
    res["gpu"] = _gpu_name()
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "trace"})
                      for k, v in res.items()}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
