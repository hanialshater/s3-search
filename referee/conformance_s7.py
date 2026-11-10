"""S7-9: conformance families for S7's features, over ESCI-sample, recorded against OpenSearch 2.19 (owner:
evaluation-steward, P6). S7's predictions 1 and 2 are judged on these families.

    python -m referee.conformance_s7 --record            # a live OpenSearch (S3_SEARCH_OPENSEARCH_URL), once
    python -m referee.conformance_s7 [--device=cpu]      # the engine against the recording

**The data.** ESCI-sample as the loader serves it, plus the judges' deterministic synthetic `price` (float) and
`stock` (integer) (`referee.judge.with_hot_fields`, seed 20261112): ESCI has no numeric fields, and the aggregations
need them. Both sides index exactly this catalogue, in this order, with this mapping; OpenSearch with
`referee/opensearch.py`'s `s3_english` analyzer.

**Roles and rules** (fixed before the recording and before any engine run):

| Role | Families | Prediction | Rule |
|---|---|---|---|
| `exact` | exists, ids, prefix, post_filter, collapse, agg_min_max, agg_sum, agg_value_count, agg_cardinality, agg_range, agg_histogram | S7-1, at least 98% identical | totals, hit ids in order (every request sorts by exact keys ending in `product_id`), bucket keys, bounds and counts, min, max, value_count and cardinality exactly; sum within 1e-12 relative (the order of a float64 sum; stated as a tolerance, not a semantic difference) |
| `approx` | agg_avg_stats | S7-1, within 1e-6 relative | avg and every stats value within 1e-6 relative; count, min and max exactly |
| `phrase` | phrase_slop0, phrase_slop2 | S7-2, at least 95% same hit set | the same total and the same set of ids (size 1,000) |
| `highlight` | highlight | S7-2, every fragment holds a query term | judged on the engine alone: every fragment holds at least one analysed query term (OpenSearch's answer is recorded for information) |
| `info` | fuzzy, match_fuzziness, match_bool_prefix, function_score | none | the same total and hit set, shown, not counted |

`cardinality` requests keep match sets well under OpenSearch's default precision threshold (3,000), where its
HyperLogLog++ counts exactly. Synonyms (S7-6) are not in these families: OpenSearch would need a synonym analyzer at
search time, and S7's predictions 1 and 2 do not name them.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from referee import checks, conformance, loader
from referee.analysis import EN_STOP, WORD, analyze

SUITE = "s7-v1"
RECORDING = Path(__file__).resolve().parent / "conformance" / f"opensearch-2.19-esci-sample-{SUITE}.jsonl.gz"
TIE = {"product_id": "asc"}
PLAN = [("exists", "exact", 12), ("ids", "exact", 12), ("prefix", "exact", 20), ("post_filter", "exact", 20),
        ("collapse", "exact", 20), ("agg_min_max", "exact", 12), ("agg_sum", "exact", 8), ("agg_value_count", "exact", 8),
        ("agg_cardinality", "exact", 10), ("agg_range", "exact", 12), ("agg_histogram", "exact", 12),
        ("agg_avg_stats", "approx", 16),
        ("phrase_slop0", "phrase", 20), ("phrase_slop2", "phrase", 20),
        ("highlight", "highlight", 20),
        ("fuzzy", "info", 8), ("match_fuzziness", "info", 8), ("match_bool_prefix", "info", 8), ("function_score", "info", 8)]


def dataset():
    from referee.judge import with_hot_fields
    ds = loader.load_dataset("esci-sample")
    cat, mapping = with_hot_fields(ds)
    return ds, cat.reset_index(drop=True), mapping


def build_suite(ds=None, cat=None, seed=20261112) -> list:
    if cat is None:
        ds, cat, _ = dataset()
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    titles = [t for t in cat["product_title"].dropna().astype(str) if len(t) > 20]
    ids = list(cat["product_id"])
    brands = cat["product_brand"].dropna().astype(str).value_counts()
    common_brands = list(brands[brands >= 20].index)
    queries = [str(q) for q in ds.fit["query"].drop_duplicates()]
    text_fields = checks.FIELDS5

    def price_band(lo_w=5, hi_w=60):
        lo = round(float(rng.uniform(1, 400)), 2)
        return {"range": {"price": {"gte": lo, "lt": round(lo + float(rng.uniform(lo_w, hi_w)), 2)}}}

    def two_words(gap_ok):
        for _ in range(200):
            words = WORD.findall(pick(titles).lower())
            pairs = [(a, b) for a, b in zip(words, words[1:]) if a not in EN_STOP and b not in EN_STOP
                     and len(a) > 2 and len(b) > 2]
            if pairs:
                a, b = pick(pairs)
                return f"{b} {a}" if gap_ok and rng.random() < 0.5 else f"{a} {b}"
        return "red shoes"

    def a_prefix():
        for _ in range(200):
            w = [t for t in analyze(pick(titles)) if len(t) >= 5 and t.isalpha()]
            if w:
                return pick(w)[: int(rng.integers(3, 5))]
        return "sho"

    def a_typo():
        w = [t for t in analyze(pick(titles)) if len(t) >= 5 and t.isalpha()]
        w = pick(w) if w else "shoes"
        i = int(rng.integers(1, len(w) - 1))
        return w[:i] + w[i + 1:]

    srt = lambda f="price": [{f: pick(["asc", "desc"])}, TIE]  # noqa: E731
    make = {
        "exists": lambda: {"query": {"bool": {"filter": [{"exists": {"field": pick(
            ["product_color", "product_bullet_point", "product_description", "product_brand.keyword",
             "product_color.keyword", "price"])}}, price_band()]}}, "sort": srt(), "size": 100},
        "ids": lambda: {"query": {"ids": {"values": [pick(ids) for _ in range(int(rng.integers(1, 30)))] +
                                                     ["NO-SUCH-PRODUCT"]}}, "sort": [TIE], "size": 100},
        "prefix": lambda: {"query": {"prefix": {pick(["product_title", "product_description"]): a_prefix()}},
                           "sort": srt(), "size": 100},
        "post_filter": lambda: {"query": {"bool": {"filter": [price_band()]}},
                                "post_filter": {"terms": {"product_brand.keyword": [pick(common_brands)
                                                                                   for _ in range(2)]}},
                                "aggs": {"b": {"terms": {"field": "product_brand.keyword", "size": 10}}},
                                "sort": srt(), "size": 50},
        "collapse": lambda: {"query": {"bool": {"filter": [price_band()]}},
                             "collapse": {"field": pick(["product_brand.keyword", "stock"])},
                             "sort": [{"price": "asc"}, TIE], "from": int(rng.integers(0, 10)),
                             "size": int(rng.integers(5, 30))},
        "agg_min_max": lambda: {"query": {"bool": {"filter": [price_band()]}}, "size": 0, "aggs": {
            "mn": {"min": {"field": pick(["price", "stock"])}}, "mx": {"max": {"field": pick(["price", "stock"])}}}},
        "agg_sum": lambda: {"query": {"bool": {"filter": [price_band()]}}, "size": 0,
                            "aggs": {"s": {"sum": {"field": pick(["price", "stock"])}}}},
        "agg_value_count": lambda: {"query": {"match": {"product_title": pick(queries)}}, "size": 0, "aggs": {
            "v": {"value_count": {"field": pick(["price", "product_brand.keyword", "product_color.keyword"])}}}},
        "agg_cardinality": lambda: {"query": {"bool": {"filter": [price_band(2, 8)]}}, "size": 0, "aggs": {
            "c": {"cardinality": {"field": pick(["product_brand.keyword", "product_color.keyword", "stock"])}}}},
        "agg_range": lambda: {"query": {"match": {"product_title": pick(queries)}}, "size": 0, "aggs": {
            "r": {"range": {"field": "price", "ranges": [{"to": 50}, {"from": 50, "to": 150.5}, {"from": 150.5}]}}}},
        "agg_histogram": lambda: {"query": {"match": {"product_title": pick(queries)}}, "size": 0, "aggs": {
            "h": {"histogram": {"field": pick(["price", "stock"]), "interval": pick([10, 25, 50])}}}},
        "agg_avg_stats": lambda: {"query": {"match": {"product_title": pick(queries)}}, "size": 0, "aggs": {
            "a": {"avg": {"field": pick(["price", "stock"])}}, "s": {"stats": {"field": pick(["price", "stock"])}},
            "t": {"terms": {"field": "product_brand.keyword", "size": 3}, "aggs": {"p": {"avg": {"field": "price"}}}}}},
        "phrase_slop0": lambda: {"query": {"match_phrase": {"product_title": two_words(False)}}, "size": 1000},
        "phrase_slop2": lambda: {"query": {"match_phrase": {"product_title": {"query": two_words(True), "slop": 2}}},
                                 "size": 1000},
        "highlight": lambda: {"query": {"multi_match": {"query": pick(queries), "fields": text_fields}}, "size": 20,
                              "highlight": {"fields": {"product_title": {}, "product_description": {},
                                                       "product_bullet_point": {}}}},
        "fuzzy": lambda: {"query": {"fuzzy": {"product_title": a_typo()}}, "size": 1000},
        "match_fuzziness": lambda: {"query": {"match": {"product_title": {"query": a_typo(), "fuzziness": "AUTO"}}},
                                    "size": 1000},
        "match_bool_prefix": lambda: {"query": {"match_bool_prefix": {"product_title": two_words(False)[:-1]}},
                                      "size": 1000},
        "function_score": lambda: {"query": {"function_score": {
            "query": {"match": {"product_title": pick(queries)}},
            "field_value_factor": {"field": "stock", "modifier": "log1p", "missing": 1},
            "boost_mode": pick(["multiply", "sum"])}}, "size": 1000},
    }
    suite, i = [], 0
    for fam, role, n in PLAN:
        for _ in range(n):
            suite.append({"id": i, "family": fam, "role": role, "request": make[fam]()})
            i += 1
    return suite


def _close(a, b, tol):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol * max(1e-300, abs(float(b)))


def _agg_diffs(g, e, approx=False):
    """Aggregations: keys, bounds, counts exactly; min/max/value_count/cardinality exactly; sum 1e-12; avg/stats
    1e-6 (approx families) or exactly-as-sum otherwise."""
    out = []
    for name, ea in (e or {}).items():
        ga = (g or {}).get(name)
        if ga is None:
            out.append(f"aggregation {name} missing")
            continue
        if "buckets" in ea:
            gb, eb = ga.get("buckets", []), ea["buckets"]
            if len(gb) != len(eb):
                out.append(f"{name}: {len(gb)} buckets != {len(eb)}")
                continue
            for x, y in zip(gb, eb):
                for k in ("key", "from", "to", "doc_count"):
                    if k in y and (k not in x or (x[k] != y[k] and not (isinstance(y[k], float) and _close(x[k], y[k], 0)))):
                        out.append(f"{name}: bucket {y.get('key')} {k} {x.get(k)} != {y[k]}")
                subs = {k: v for k, v in y.items() if isinstance(v, dict)}
                out += _agg_diffs({k: x.get(k) for k in subs}, subs, approx=True)
            if "sum_other_doc_count" in ea and ga.get("sum_other_doc_count") != ea["sum_other_doc_count"]:
                out.append(f"{name}: sum_other_doc_count {ga.get('sum_other_doc_count')} != {ea['sum_other_doc_count']}")
        elif "count" in ea:  # stats
            for k in ("count", "min", "max"):
                if not _close(ga.get(k), ea.get(k), 0):
                    out.append(f"{name}.{k}: {ga.get(k)} != {ea.get(k)}")
            for k in ("avg", "sum"):
                if not _close(ga.get(k), ea.get(k), 1e-6):
                    out.append(f"{name}.{k}: {ga.get(k)} != {ea.get(k)} (1e-6)")
        elif "value" in ea:
            tol = 1e-6 if approx else (1e-12 if name in ("s",) else 0)
            if not _close(ga.get("value"), ea.get("value"), tol):
                out.append(f"{name}: {ga.get('value')} != {ea.get('value')}")
    return out


def _fragment_problems(resp, req):
    """Every fragment holds at least one analysed query term (the query's text, analysed)."""
    q = req["query"]
    text = q.get("multi_match", {}).get("query") or json.dumps(q)
    terms = set(analyze(text))
    bad = []
    for h in resp["hits"]["hits"]:
        for f, frags in (h.get("highlight") or {}).items():
            for fr in frags:
                spans = conformance_tag(fr)
                if not any(set(analyze(sp)) & terms for sp in spans):
                    bad.append(f"{h['_id']} {f}: a fragment without a query term: {fr[:60]!r}")
    return bad


def conformance_tag(fr, pre="<em>", post="</em>"):
    import re
    return re.findall(re.escape(pre) + r"(.*?)" + re.escape(post), fr)


def judge(index, recorded, suite) -> dict:
    rows = []
    for s in suite:
        g, e = conformance.outcome(index, s), conformance.outcome(recorded, s)
        row = {"id": s["id"], "family": s["family"], "role": s["role"]}
        if "error" in g:
            row["diffs"] = [f"engine refused: {g['error']}: {g.get('reason', '')[:120]}"]
        elif s["role"] == "highlight":
            row["diffs"] = _fragment_problems(g["ok"], s["request"])
            row["fragments"] = sum(len(fr) for h in g["ok"]["hits"]["hits"] for fr in (h.get("highlight") or {}).values())
        elif "error" in e:
            row["diffs"] = [f"OpenSearch refused: {e.get('reason', '')[:160]}"]
        else:
            ge, ee = g["ok"], e["ok"]
            d = []
            if ge["hits"]["total"] != ee["hits"]["total"]:
                d.append(f"total {ge['hits']['total']} != {ee['hits']['total']}")
            gi, ei = [h["_id"] for h in ge["hits"]["hits"]], [h["_id"] for h in ee["hits"]["hits"]]
            if s["role"] in ("exact", "approx"):
                if gi != ei:
                    k = next((i for i, (a, b) in enumerate(zip(gi, ei)) if a != b), min(len(gi), len(ei)))
                    d.append(f"ids differ from rank {k} ({len(gi)} vs {len(ei)})")
                d += _agg_diffs(ge.get("aggregations"), ee.get("aggregations"), approx=s["role"] == "approx")
            elif set(gi) != set(ei):
                d.append(f"{len(set(gi) ^ set(ei))} ids differ (as sets)")
            row["diffs"] = d
        row["passed"] = not row["diffs"]
        rows.append(row)

    def rate(rs):
        n = len(rs)
        p = sum(r["passed"] for r in rs)
        return {"passed": p, "n": n, "rate": p / n if n else None}
    fams = {f: {"role": role, **rate([r for r in rows if r["family"] == f])} for f, role, _ in PLAN}
    return {"kind": "conformance-s7", "suite": SUITE, "suite_sha256": conformance.suite_sha256(suite),
            "recording": getattr(recorded, "meta", None),
            "prediction_1_exact": rate([r for r in rows if r["role"] == "exact"]),
            "prediction_1_avg_stats": rate([r for r in rows if r["role"] == "approx"]),
            "prediction_2_phrase": rate([r for r in rows if r["role"] == "phrase"]),
            "prediction_2_highlight": {**rate([r for r in rows if r["role"] == "highlight"]),
                                       "fragments_checked": sum(r.get("fragments", 0) for r in rows)},
            "info": rate([r for r in rows if r["role"] == "info"]),
            "by_family": fams, "failures": [r for r in rows if not r["passed"]][:60]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee.conformance_s7")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--engine", default="engine", choices=("engine", "reference"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/conformance-s7.json")
    a = ap.parse_args(argv)
    ds, cat, mapping = dataset()
    suite = build_suite(ds, cat)
    if a.record:
        from referee import opensearch
        p = conformance.record(opensearch.build_index(cat, mapping), suite, path=RECORDING,
                               meta={"server": "OpenSearch 2.19.1 (Lucene 9.12.1), single node, one shard",
                                     "dataset": "esci-sample + synthetic price and stock (seed 20261112)",
                                     "analyzer": os.environ.get("S3_SEARCH_OS_ANALYZER") or "s3_english"})
        print(f"recorded {len(suite)} answers -> {p}")
        return 0
    from referee.bench import get_api
    index = get_api(a.engine).build_index(cat, mapping, device=a.device)
    rep = judge(index, conformance.Recorded(suite, path=RECORDING), suite)
    rep.update({"engine": a.engine, "device": a.device, "label": "CPU, development only"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k in ("prediction_1_exact", "prediction_1_avg_stats", "prediction_2_phrase", "prediction_2_highlight", "info"):
        print(f"{k:24s} {rep[k]['passed']}/{rep[k]['n']}")
    for f, v in rep["by_family"].items():
        print(f"  {f:18s} {v['role']:9s} {v['passed']}/{v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
