"""S9 prediction 1's CPU half (owner: evaluation-steward, P6): without cuVS, a knn request for an approximate method
returns exactly the exact-knn answer and names the method that ran. CPU, development only.

    python -m referee.judge_s9p1_cpu [--device=cpu|torch-cpu] [--out=...]

1. The contract check `s9_checks.check_s9_1` on toy (given vectors; knn with cagra and ivf_pq, parameters, filters,
   the list form, bad parameters refused).
2. ESCI-sample with the tfidf-svd-128 encoder: 200 sampled fit queries as `neural` requests (k 10 and 100), with
   `cagra` and `ivf_pq`, unfiltered and with a keyword pre-filter, against the same request with `exact`: ids in
   order and scores (1e-6) equal, and `knn.embedding == {requested: method, ran: "exact"}`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from referee import loader, s9_checks
from referee.gpu_judges import _with_embeddings


def main(argv=None) -> int:
    import engine.api as api
    ap = argparse.ArgumentParser(prog="python -m referee.judge_s9p1_cpu")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/judge-s9p1-cpu.json")
    a = ap.parse_args(argv)
    res = {"kind": "judge-s9p1-cpu", "device": a.device, "devices": api.available_devices(),
           "label": "CPU, development only"}
    toy = loader.load_dataset("toy")
    tidx = api.build_index(toy.catalogue, toy.mapping, device=a.device)
    res["toy_contract_check"] = {"problems": s9_checks.check_s9_1(tidx, toy)}
    ds = loader.load_dataset("esci-sample")
    idx = api.build_index(ds.catalogue, _with_embeddings(ds.mapping), device=a.device)
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    rng = np.random.default_rng(20261123)
    texts = [str(t) for t in q["query"].iloc[np.sort(rng.choice(len(q), 200, replace=False))]]
    brands = ds.catalogue["product_brand"].dropna().value_counts()
    flt = {"terms": {"product_brand.keyword": list(brands.index[:20])}}
    n, differ, unnamed = 0, 0, 0
    for t in texts:
        for k in (10, 100):
            for f in (None, flt):
                base = {"query_text": t, "k": k, **({"filter": f} if f else {})}
                e = idx.search({"query": {"neural": {"embedding": {**base, "method": "exact"}}}, "size": k})
                for m in s9_checks.METHODS:
                    r = idx.search({"query": {"neural": {"embedding": {**base, "method": m}}}, "size": k})
                    n += 1
                    same = [h["_id"] for h in r["hits"]["hits"]] == [h["_id"] for h in e["hits"]["hits"]] and all(
                        abs(x["_score"] - y["_score"]) <= 1e-6 * max(1.0, abs(y["_score"]))
                        for x, y in zip(r["hits"]["hits"], e["hits"]["hits"]))
                    differ += not same
                    unnamed += s9_checks.method_reported(r) != ("exact", m)
    res["esci_sample_neural"] = {"requests": n, "differ_from_exact": differ, "method_not_named": unnamed}
    ok = not res["toy_contract_check"]["problems"] and differ == 0 and unnamed == 0
    res["verdict_cpu_half"] = "held" if ok else "wrong"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
