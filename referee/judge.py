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


# ---- a real serving replica over HTTP -------------------------------------------------------------------------------

class Server:
    """`python -m serving` in a subprocess on 2 cores, and a small JSON client for it."""

    def __init__(self, data_dir, port=None):
        import socket
        import subprocess
        import os
        if port is None:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
        self.port, self.base = port, f"http://127.0.0.1:{port}"
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "S3_SEARCH_DEVICE": "cpu"}
        self.proc = subprocess.Popen(["taskset", "-c", "0,1", sys.executable, "-m", "serving", "--data-dir",
                                      str(data_dir), "--port", str(port), "--device", "cpu"],
                                     cwd=Path(__file__).resolve().parent.parent, env=env,
                                     stdout=subprocess.DEVNULL, stderr=open(Path(data_dir) / "serving.log", "w"))
        for _ in range(200):
            try:
                self.call("GET", "/healthz")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("serving did not start")

    def call(self, method, path, body=None, timeout=600):
        import urllib.request
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")

    def msearch(self, name, reqs, batch=64):
        out = []
        for i in range(0, len(reqs), batch):
            out += self.call("POST", f"/indexes/{name}/_msearch", reqs[i:i + batch])
        return out

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except Exception:
            self.proc.kill()


def _key(resp):
    return (resp["hits"]["total"], [(h["_id"], h["_score"]) for h in resp["hits"]["hits"]])


def judge_s4b(out_path):
    res = {"kind": "judge-s4b", "label": "CPU, development only", "dataset": "esci-sample"}
    ds = loader.load_dataset("esci-sample")
    work = Path(tempfile.mkdtemp(prefix="judge-s4b-"))
    src = work / "esci-sample.parquet"
    ds.catalogue.to_parquet(src, index=False)
    reqs = fit_requests(ds, size=100)
    srv = Server(work / "serving")
    try:
        t = time.perf_counter()
        created = srv.call("PUT", "/indexes/esci", {"mapping": ds.mapping, "source": {"path": str(src)}})
        res["create"] = {"s": time.perf_counter() - t, "response": created}
        v1 = created["segment_id"]
        r1 = [_key(r) for r in srv.msearch("esci", reqs)]
        # A change only a segment carries: new text for 300 products (upserts are pending until a reindex).
        rng = np.random.default_rng(HOT_SEED + 1)
        rows = ds.catalogue.iloc[np.sort(rng.choice(len(ds.catalogue), 300, replace=False))]
        changes = []
        for rec in rows.to_dict("records"):
            doc = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in rec.items()}
            doc["product_title"] = "zzrollback " + str(doc.get("product_title") or "")
            doc["product_description"] = None
            changes.append({"op": "upsert", "_id": doc["product_id"], "doc": doc})
        res["docs"] = {k: v for k, v in srv.call("POST", "/indexes/esci/_docs", changes).items() if k != "rejected"}
        t = time.perf_counter()
        re2 = srv.call("POST", "/indexes/esci/_reindex", {})
        res["reindex"] = {"s": time.perf_counter() - t, "response": re2}
        r2 = [_key(r) for r in srv.msearch("esci", reqs)]
        n_changed = sum(a != b for a, b in zip(r1, r2))
        res["versions_before"] = srv.call("GET", "/indexes/esci/versions")
        t = time.perf_counter()
        rb = srv.call("POST", "/indexes/esci/_rollback", {"segment_id": v1})
        rollback_s = time.perf_counter() - t
        r3 = [_key(r) for r in srv.msearch("esci", reqs)]
        differ = [i for i, (a, b) in enumerate(zip(r1, r3)) if a != b]
        res["versions_after"] = srv.call("GET", "/indexes/esci/versions")
        res["S4-b"] = {"rollback_s": rollback_s, "response": rb, "v1": v1, "v2": re2.get("segment_id"),
                       "fit_queries": len(reqs), "changed_by_v2": n_changed, "differ_after_rollback_vs_v1": len(differ),
                       "examples": [reqs[i] for i in differ[:3]],
                       "verdict": "held" if rollback_s < 10 and not differ and n_changed > 0 and rb["segment_id"] == v1
                       else "wrong"}
    finally:
        srv.close()
    s = res["S4-b"]
    print(f"S4-b: rollback {s['rollback_s']:.2f} s over HTTP; v2 changed {s['changed_by_v2']} of {s['fit_queries']} "
          f"fit queries; after rollback {s['differ_after_rollback_vs_v1']} differ from v1 -> {s['verdict']}", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


def _report_on(results_path, run_path, work, tag):
    import contextlib
    import io
    from tools import report
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = report.main([str(results_path), "--out", str(work / f"report-{tag}.html"), "--recompute-run",
                            str(run_path), "--data-dir", str(loader.SAMPLE_DIR)])
    return {"exit_code": code, "output": buf.getvalue().strip().splitlines()[-12:]}


def judge_s5a(out_path):
    import engine.api as api
    from tools import explain
    from tools.common import engine_search, text_request

    res = {"kind": "judge-s5a", "label": "CPU, development only", "dataset": "esci-sample"}
    ds = loader.load_dataset("esci-sample")
    work = Path(tempfile.mkdtemp(prefix="judge-s5a-"))
    index = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    queries = ds.fit[["query_id", "query"]].drop_duplicates("query_id").reset_index(drop=True)
    judg = loader.fit_judgments(ds)

    # Part 1: the referee's bench run (fit split) and the report recomputing it from the same run.
    res["report"] = {}
    for k in (100, 1000):
        runs = {}
        r = bench.run(ds, queries, judg, api, index=index, k=k, latency_queries=200, runs_out=runs)
        rp, run_p = work / f"results-k{k}.json", work / f"run-k{k}.jsonl"
        rp.write_text(json.dumps(r, indent=1) + "\n")
        run_p.write_text("".join(json.dumps({"query_id": q, "ids": ids}) + "\n" for q, ids in runs.items()))
        rep = _report_on(rp, run_p, work, f"k{k}")
        rep["referee_quality"] = r["quality"]
        res["report"][f"k={k}"] = rep
        print(f"S5-a report (bench --k={k}): exit {rep['exit_code']}; " + " | ".join(rep["output"][-3:]), flush=True)

    # Part 2a: the explain tool's own sample check (lexical, no ranker), 100 pairs from 200 fit queries.
    rng = np.random.default_rng(HOT_SEED + 2)
    sample_q = [str(t) for t in queries["query"].iloc[np.sort(rng.choice(len(queries), 200, replace=False))]]
    t = time.perf_counter()
    lex = explain.sample_check(engine_search(index), sample_q, n=100, seed=1)
    res["explain_lexical"] = {**lex, "s": time.perf_counter() - t}
    print(f"S5-a explain (lexical, tool's sample_check): {lex['n_ok']}/{lex['n']} ok, {lex['n_complete']} complete",
          flush=True)
    # The judge's own arithmetic on the same kind of trees: root = sum of terms = hit score; a term's pseudo_tf (in its
    # description, 4 significant digits) = the sum of its field contributions.
    import re
    rng2 = np.random.default_rng(HOT_SEED + 4)
    pairs, n_root_ok, n_ptf_ok = [], 0, 0
    for q in sample_q[:100]:
        r = index.search({**text_request(q, size=100), "explain": True})
        for h, e in zip(r["hits"]["hits"], r["explain"]):
            pairs.append((h, e))
    for i in rng2.permutation(len(pairs))[:100]:
        h, e = pairs[i]
        terms = e.get("details") or []
        if abs(sum(t["value"] for t in terms) - h["_score"]) <= 1e-4 * max(1.0, abs(h["_score"])) and \
                abs(e["value"] - h["_score"]) <= 1e-4 * max(1.0, abs(h["_score"])):
            n_root_ok += 1
        ok = True
        for t in terms:
            m = re.search(r"pseudo_tf ([0-9.e+-]+)$", t["description"])
            fs = sum(f["value"] for f in t.get("details") or [])
            ok &= m is not None and abs(float(m.group(1)) - fs) <= 1e-3 * max(1.0, fs)
        n_ptf_ok += ok
    res["explain_lexical_judge_arithmetic"] = {"n": 100, "root_is_sum_of_terms_and_score": n_root_ok,
                                               "pseudo_tf_is_sum_of_fields": n_ptf_ok}
    print(f"S5-a explain (lexical, judge's arithmetic): root = sum of terms = score in {n_root_ok}/100; "
          f"pseudo_tf = sum of fields in {n_ptf_ok}/100", flush=True)

    # Part 2b: the ranker. Train on fit queries, register it, explain 100 rescored (query, product) pairs.
    from engine.rank.train import train_ranker
    t = time.perf_counter()
    model = train_ranker(index, ds.fit, lambda q: text_request(q), window=200, max_queries=300)
    index.add_ranker("judge", model)
    train_s = time.perf_counter() - t
    resc = lambda q, size=100: {**text_request(q, size=size),  # noqa: E731
                                "rescore": {"window_size": 100, "s3_ranker": {"model": "judge"}}}
    pairs = []
    for q in sample_q[:50]:
        pairs += [(q, h["_id"]) for h in index.search(resc(q))["hits"]["hits"]]
    order = np.random.default_rng(HOT_SEED + 3).permutation(len(pairs))[:100]
    rows, feature_sums = [], []
    for i in order:
        q, pid = pairs[i]
        e = explain.explain_product(engine_search(index), resc(q), pid, windows=(100,))
        if not e["found"]:
            rows.append({"found": False})
            continue
        root = e["tree"]
        kids = root.get("details") or []
        kid_sum = float(sum(k["value"] for k in kids))
        feature_sums.append(abs(kid_sum - float(e["score"])) <= 1e-4 * max(1.0, abs(float(e["score"]))))
        rows.append({"found": True, "ok": e["ok"], "complete": e["complete"], "root_matches_score": e["root_matches_score"],
                     "root": root["description"], "n_children": len(kids), "children_sum": kid_sum, "score": e["score"],
                     "n_leaves": len(e["leaves"]),
                     "feature_leaves": sum("feature" in l["description"].lower() for l in e["leaves"])})
    found = [r for r in rows if r["found"]]
    res["explain_ranker"] = {
        "train_s": train_s, "n": len(rows), "found": len(found),
        "tool_ok": sum(r["ok"] for r in found), "tool_complete": sum(r["complete"] for r in found),
        "children_sum_to_ranker_score": int(sum(feature_sums)),
        "with_feature_leaves": sum(r["feature_leaves"] > 0 for r in found), "examples": found[:3]}
    er = res["explain_ranker"]
    print(f"S5-a explain (ranker): {er['found']}/{er['n']} found; tool ok {er['tool_ok']}, complete "
          f"{er['tool_complete']}; children sum to the ranker score in {er['children_sum_to_ranker_score']}; "
          f"with any per-feature leaf: {er['with_feature_leaves']}", flush=True)
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
    {"s2": judge_s2, "s4b": judge_s4b, "s5a": judge_s5a}[a.what](out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
