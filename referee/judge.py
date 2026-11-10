"""The steward's judging runs for pre-registered predictions (S2-a, S2-b, S4-b, S5-a, S5-b), on ESCI-sample, CPU.

    taskset -c 0,1 python -m referee.judge s2 [--out=results/judge-s2.json]
    taskset -c 0,1 python -m referee.judge s4b
    taskset -c 0,1 python -m referee.judge s5a
    taskset -c 0,1 python -m referee.judge s5b

Every number here is CPU, development only. The judge drives the makers' code through its public surfaces only
(dataplane.intake/kv/indexer/updater, engine.api, serving's HTTP API, tools) and edits none of it.

Setup choices, stated with the results:
- ESCI-sample has no price or stock, so S2-a adds deterministic synthetic ones (`with_hot_fields`, seed 20261112)
  and marks them hot in a copy of the ESCI mapping. The same catalogue is used for S2-b.
- Freshness is timed from `store.apply` returning; the replica's own background poller (`Replica.start`) makes
  changes visible. Polling inside the hook would hide the latency being judged.

Owner: evaluation-steward (P6).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from referee import bench, freshness, loader

HOT_SEED = 20261112


def with_hot_fields(ds):
    """ESCI-sample plus synthetic `price` (float) and `stock` (integer, often 0), both hot."""
    rng = np.random.default_rng(HOT_SEED)
    cat = ds.catalogue.copy()
    n = len(cat)
    cat["price"] = np.round(rng.uniform(1, 500, n), 2)
    cat["stock"] = (rng.integers(0, 4, n) * rng.integers(0, 40, n)).astype(np.int32)
    m = copy.deepcopy(ds.mapping)
    m["properties"]["price"] = {"type": "float"}
    m["properties"]["stock"] = {"type": "integer"}
    m["_meta"]["engine"]["hot"] = ["price", "stock"]
    return cat, m


def fit_requests(ds, size=100):
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    return [{**bench.fill(bench.DEFAULT_TEMPLATE, t), "size": size} for t in q["query"]]


def compare_all(a, b, reqs, batch=64, exact=True):
    """Responses of two indexes on every request: ids, totals, scores (exactly, or within the contract's tolerance)."""
    from referee import checks
    diffs, max_abs = [], 0.0
    for i in range(0, len(reqs), batch):
        ra, rb = a.search_batch(reqs[i:i + batch]), b.search_batch(reqs[i:i + batch])
        for j, (x, y) in enumerate(zip(ra, rb)):
            xi = [h["_id"] for h in x["hits"]["hits"]]
            yi = [h["_id"] for h in y["hits"]["hits"]]
            xs = [h["_score"] for h in x["hits"]["hits"]]
            ys = [h["_score"] for h in y["hits"]["hits"]]
            if xs and len(xs) == len(ys):
                max_abs = max(max_abs, float(np.max(np.abs(np.asarray(xs, float) - np.asarray(ys, float)))))
            ok = x["hits"]["total"] == y["hits"]["total"] and xi == yi and \
                (xs == ys if exact else all(checks._close(s, t) for s, t in zip(xs, ys)))
            if x.get("aggregations") != y.get("aggregations"):
                ok = False
            if not ok and len(diffs) < 10:
                diffs.append({"request": reqs[i + j], "total": [x["hits"]["total"], y["hits"]["total"]],
                              "first_ids": [xi[:5], yi[:5]]})
            elif not ok:
                diffs.append(None)
    return {"n": len(reqs), "differ": len(diffs), "examples": [d for d in diffs if d][:5], "max_abs_score_diff": max_abs}


def judge_s2(out_path):
    import engine.api as api
    from dataplane import indexer, intake
    from dataplane.kv import Store
    from dataplane.updater import Replica

    res = {"kind": "judge-s2", "label": "CPU, development only", "dataset": "esci-sample"}
    ds = loader.load_dataset("esci-sample")
    cat, mapping = with_hot_fields(ds)
    work = Path(tempfile.mkdtemp(prefix="judge-s2-"))
    src = work / "esci-sample.parquet"
    cat.to_parquet(src, index=False)

    store = Store(work / "store.sqlite", mapping)
    t = time.perf_counter()
    rep = intake.ingest(src, mapping, store, work / "rejected.jsonl")
    res["ingest"] = {"s": time.perf_counter() - t, "report": repr(rep), "head": store.head(), "count": store.count()}

    # S2-b: a full re-index from the store, timed; its segment against engine.api.build_index on every fit query.
    t = time.perf_counter()
    seg = indexer.build(store, mapping, work / "segments")
    reindex_s = time.perf_counter() - t
    snap_ids = list(store.snapshot()["product_id"])
    same_order = snap_ids == list(cat["product_id"])
    from_seg = api.Index.from_segment(seg, device="cpu")
    t = time.perf_counter()
    direct = api.build_index(cat, mapping, device="cpu")
    direct_s = time.perf_counter() - t
    reqs = fit_requests(ds)
    cmp_b = compare_all(from_seg, direct, reqs)
    res["S2-b"] = {"reindex_s": reindex_s, "engine_build_index_s": direct_s, "snapshot_in_catalogue_order": same_order,
                   "segment": seg.name, "fit_queries": cmp_b,
                   "verdict": "held" if reindex_s < 180 and cmp_b["differ"] == 0 else "wrong"}
    print(f"S2-b: reindex {reindex_s:.1f} s; {cmp_b['differ']} of {cmp_b['n']} fit queries differ "
          f"(max |score diff| {cmp_b['max_abs_score_diff']:.3g}) -> {res['S2-b']['verdict']}", flush=True)

    # S2-a: 10,000 price and stock changes written to the store; the replica's poller makes them visible.
    replica = Replica(store, seg, "cpu").start(interval=0.05)
    try:
        def apply(changes):
            store.apply({"op": "patch", "_id": c["_id"], "fields": c["fields"]} for c in changes)
        t = time.perf_counter()
        fr = freshness.check_freshness(replica, cat, mapping, apply, api.build_index, n_changes=10_000, batch=100,
                                       timeout_s=30.0)
        fr["wall_s"] = time.perf_counter() - t
        fr["poller_error"] = repr(replica.last_error) if replica.last_error else None
        # Every fit query, under filters on the hot fields, against a rebuild of the updated catalogue.
        updated = cat.copy()
        row = {d: j for j, d in enumerate(updated["product_id"])}
        for c in freshness.make_changes(cat, mapping, 10_000):
            for f, v in c["fields"].items():
                updated.iat[row[c["_id"]], updated.columns.get_loc(f)] = v
        snap = store.snapshot().set_index("product_id")
        store_agrees = bool(np.allclose(snap.loc[updated["product_id"], "price"].to_numpy(), updated["price"]) and
                            (snap.loc[updated["product_id"], "stock"].to_numpy() == updated["stock"].to_numpy()).all())
        rebuilt = api.build_index(updated, mapping, device="cpu")
        filt = [{"query": {"bool": {"must": [r["query"]], "filter": [{"range": {"price": {"lte": 250}}},
                                                                      {"range": {"stock": {"gt": 0}}}]}},
                 "size": 100, "aggs": {"s": {"terms": {"field": "stock", "size": 5}}}} for r in reqs]
        by_price = [{"query": r["query"], "size": 20, "sort": [{"price": "asc"}]} for r in reqs[:500]]
        cmp_a = compare_all(replica, rebuilt, filt + by_price)
    finally:
        replica.stop()
    p99 = fr["visible_ms"]["p99"]
    res["S2-a"] = {"freshness": fr, "store_equals_expected_catalogue": store_agrees,
                   "fit_queries_with_hot_filters_vs_rebuild": cmp_a,
                   "verdict": "held" if (p99 is not None and p99 <= 1000 and fr["passed"] and cmp_a["differ"] == 0
                                         and store_agrees) else "wrong"}
    print(f"S2-a: {fr['visible']} changes visible, p50 {fr['visible_ms']['p50']:.1f} ms, p99 {p99:.1f} ms, "
          f"max {fr['visible_ms']['max']:.1f} ms; probes equal a rebuild: {fr['equal_to_rebuild']}; "
          f"{cmp_a['differ']} of {cmp_a['n']} filtered fit queries differ -> {res['S2-a']['verdict']}", flush=True)
    res["work_dir"] = str(work)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee.judge")
    ap.add_argument("what", choices=("s2", "s4b", "s5a", "s5b"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or f"results/judge-{a.what}.json"
    {"s2": judge_s2}[a.what](out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
