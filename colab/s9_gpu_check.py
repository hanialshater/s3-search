"""S9's GPU measurements for development, in a CUDA runtime (``pip install -r requirements-gpu.txt``):

    python -m colab.s9_gpu_check --out results_s9.json [--only p1,p2,p3] [--p1-dataset esci-us] [--queries 1000]

**Development only; this is not the score of record.** A maker never supplies the numbers its claims are judged on:
S9's predictions 1 to 3 are judged only by the evaluation steward's S9 cell in ``referee/colab_referee.ipynb``. This
script lets the architect see the GPU paths work before that run. ESCI is read only through ``referee.loader`` and
only the **fit** split (queries and their judgments for nDCG and for training the ranker).

- **p1 (S9-1, cuVS):** on ``--p1-dataset`` (ESCI-US by default) with ``tfidf-svd-128`` vectors (fitted on the GPU),
  knn top-10 by ``cagra`` and ``ivf_pq`` at their default search parameters against ``exact``, for the first
  ``--queries`` fit queries (their encoded text as the vector): mean recall@10, unfiltered and with a filter that
  about 10% of the documents pass (the most frequent brands, greedily, up to 10%; the share is reported); also each
  method's ``ran`` and the index build time per method.
- **p2 (S9-2, the GPU fit):** on ESCI-sample, ``tfidf-svd-128`` ``fit_transform`` (fit and encode of every document)
  timed with ``fit="gpu"`` and ``fit="numpy"`` (each after one warm-up), and the bench's nDCG@10 on the fit split
  with a hybrid request (BM25F over the five fields + neural k=100) for an index fitted each way.
- **p3 (S9-3, GPU trees):** a GBDT trained by the S1-5 recipe on ESCI-sample's fit split, its predictions on 10,000
  candidate rows by the engine and by XGBoost on cuda and cuML FIL (max abs difference), and the rescored order of
  the fit queries' hits (window 100) by each runtime against the engine's (queries whose order differs).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

FIELDS5 = ["product_title", "product_brand", "product_color", "product_bullet_point", "product_description"]
INPUTS = ["product_title", "product_brand", "product_color"]


def _rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(__file__)).strip()
    except Exception:
        return None


def _gpu():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        return None


def dense_mapping(base: dict, fit: str) -> dict:
    m = copy.deepcopy(base)
    m["_meta"]["engine"]["embeddings"] = {"embedding": {"model": "tfidf-svd-128", "inputs": INPUTS, "fit": fit}}
    m["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128,
                                    "method": {"name": "exact", "space_type": "cosinesimil"}}
    return m


def ten_percent_filter(ix, n_docs: int):
    """The most frequent brands, added while the share stays at most 10%: a terms filter and its share."""
    aggs = ix.search({"query": {"match_all": {}}, "size": 0,
                      "aggs": {"b": {"terms": {"field": "product_brand.keyword", "size": 5000}}}})
    picked, total = [], 0
    for bk in aggs["aggregations"]["b"]["buckets"]:
        if total + bk["doc_count"] <= 0.10 * n_docs:
            picked.append(bk["key"])
            total += bk["doc_count"]
    return {"terms": {"product_brand.keyword": picked}}, total / n_docs


def p1(dataset: str, n_queries: int, device: str) -> dict:
    from referee import loader
    from engine.api import build_index
    from engine import ann
    ds = loader.load_dataset(dataset)
    t = time.perf_counter()
    ix = build_index(ds.catalogue, dense_mapping(ds.mapping, "gpu" if device == "cuda" else "numpy"), device=device)
    build_s = time.perf_counter() - t
    texts = list(dict.fromkeys(str(q) for q in ds.fit["query"]))[:n_queries]
    enc = ix._st.encoders["embedding"]
    vecs = enc.encode_queries(texts)
    flt, share = ten_percent_filter(ix, len(ds.catalogue))

    def top(v, method, f=None):
        spec = {"vector": [float(x) for x in v], "k": 10, "method": method}
        if f is not None:
            spec["filter"] = f
        r = ix.search({"query": {"knn": {"embedding": spec}}, "size": 10})
        return [h["_id"] for h in r["hits"]["hits"]], r["knn"]["embedding"]["ran"]

    out = {"dataset": dataset, "n_docs": int(len(ds.catalogue)), "n_queries": len(texts), "index_build_s": build_s,
           "filter_share": share, "methods": {}}
    exact = [top(v, "exact")[0] for v in vecs]
    exact_f = [top(v, "exact", flt)[0] for v in vecs]
    for method in ("cagra", "ivf_pq"):
        t = time.perf_counter()
        _, ran = top(vecs[0], method)                       # builds the index on the segment
        first_s = time.perf_counter() - t
        rec, rec_f = [], []
        t = time.perf_counter()
        for v, e, ef in zip(vecs, exact, exact_f):
            got, _ = top(v, method)
            got_f, _ = top(v, method, flt)
            rec.append(len(set(got) & set(e)) / max(1, len(e)))
            rec_f.append(len(set(got_f) & set(ef)) / max(1, len(ef)))
        out["methods"][method] = {"ran": ran, "recall_at_10": float(np.mean(rec)),
                                  "recall_at_10_filtered": float(np.mean(rec_f)),
                                  "first_query_s_including_build": first_s,
                                  "query_ms_mean": (time.perf_counter() - t) * 1e3 / (2 * len(vecs)),
                                  "available": ann.available(ix._st.b)}
    return out


def p2(device: str) -> dict:
    from referee import bench, loader
    from engine.api import build_index
    from engine.encode import TfidfSvdEncoder, doc_texts
    ds = loader.load_dataset("esci-sample")
    texts = doc_texts(ds.catalogue, INPUTS)
    out = {"dataset": "esci-sample", "n_docs": len(texts), "fit_s": {}, "fitted_with": {}, "ndcg_at_10": {}}
    fits = ("numpy", "gpu") if device == "cuda" else ("numpy",)
    for fit in fits:
        TfidfSvdEncoder(128, fit=fit).fit_transform(texts)            # warm-up (CUDA context, kernels)
        e = TfidfSvdEncoder(128, fit=fit)
        t = time.perf_counter()
        e.fit_transform(texts)
        out["fit_s"][fit] = time.perf_counter() - t
        out["fitted_with"][fit] = e.fitted_with
    template = {"query": {"hybrid": {"queries": [{"multi_match": {"query": "{{query}}", "fields": FIELDS5}},
                                                 {"neural": {"embedding": {"query_text": "{{query}}", "k": 100}}}]}}}
    out["request_template"] = template
    judg = loader.fit_judgments(ds)
    queries = ds.fit[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)
    for fit in fits:
        ix = build_index(ds.catalogue, dense_mapping(ds.mapping, fit), device=device)
        r = bench.run(ds, queries, judg, sys.modules["engine.api"], device=device, template=template, index=ix,
                      latency_queries=10)
        out["ndcg_at_10"][fit] = r["quality"].get("ndcg@10", r["quality"].get("ndcg_at_10"))
        out.setdefault("quality", {})[fit] = r["quality"]
    return out


def p3(device: str, n_queries: int) -> dict:
    from referee import loader
    from engine.api import build_index
    from engine.rank.gpu_trees import GpuTrees
    from engine.rank.train import train_ranker
    ds = loader.load_dataset("esci-sample")
    ix = build_index(ds.catalogue, dense_mapping(ds.mapping, "numpy"), device=device)

    def req(text, window=1000):
        return {"query": {"hybrid": {"queries": [{"multi_match": {"query": text, "fields": FIELDS5}},
                                                 {"neural": {"embedding": {"query_text": text, "k": 100}}}]}}}
    model = train_ranker(ix, ds.fit, req, max_queries=400, n_rounds=200, max_depth=6, min_leaf=20)
    all_texts = list(dict.fromkeys(str(q) for q in ds.fit["query"]))
    texts = all_texts[:n_queries]
    rows = []
    for t in all_texts:
        _, X = ix.candidate_features(req(t), 100)
        rows.append(np.asarray(X, np.float32))
        if sum(len(r) for r in rows) >= 10_000:
            break
    X = np.concatenate(rows)[:10_000]
    ref = model.predict(X)
    out = {"n_rows": int(len(X)), "runtimes": {}}
    nf = len(ix.feature_names())
    runtimes = [("xgboost", device), ("fil", device)]
    for rt, dev in runtimes:
        try:
            g = GpuTrees(model, nf, rt, dev)
        except ImportError as e:
            out["runtimes"][rt] = {"error": f"not installed: {e}"}
            continue
        d = np.abs(g.predict(X) - ref)
        ix.add_ranker("ref", model)
        ix.add_ranker("gpu", model, runtime=rt)
        differ = []
        for t in texts:
            body = {**req(t), "size": 100, "rescore": {"window_size": 100, "s3_ranker": {"model": "ref"}}}
            a = [h["_id"] for h in ix.search(body)["hits"]["hits"]]
            body["rescore"]["s3_ranker"]["model"] = "gpu"
            b = [h["_id"] for h in ix.search(body)["hits"]["hits"]]
            if a != b:
                differ.append(t)
        out["runtimes"][rt] = {"device": dev, "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                               "rows_over_1e-5": int((d > 1e-5).sum()), "order_queries": len(texts),
                               "order_differs": len(differ), "order_differs_examples": differ[:10]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m colab.s9_gpu_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results_s9.json")
    ap.add_argument("--only", default="p1,p2,p3")
    ap.add_argument("--p1-dataset", default="esci-us")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    a = ap.parse_args(argv)
    from engine.api import available_devices
    device = a.device or ("cuda" if "cuda" in available_devices() else "cpu")
    res = {"kind": "s9-gpu-check", "commit": _rev(), "device": device, "gpu": _gpu(),
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": {"cpus": os.cpu_count(), "machine": platform.machine(), "python": platform.python_version()},
           "label": "development only (not the score of record; S9 is judged by referee/colab_referee.ipynb)"}
    only = set(a.only.split(","))
    if "p1" in only:
        res["p1"] = p1(a.p1_dataset, a.queries, device)
    if "p2" in only:
        res["p2"] = p2(device)
    if "p3" in only:
        res["p3"] = p3(device, a.queries)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("device", "gpu", "commit")}), "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
