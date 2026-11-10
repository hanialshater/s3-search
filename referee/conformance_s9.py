"""S9-6: geo and nested conformance families over ESCI-sample, recorded against OpenSearch 2.19 (owner:
evaluation-steward, P6). S9's predictions 4 (geo) and 5 (nested) are judged on them.

    python -m referee.conformance_s9 --record            # a live OpenSearch (S3_SEARCH_OPENSEARCH_URL), once
    python -m referee.conformance_s9 [--device=cpu]      # the engine against the recording

**The data** (deterministic, `SEED`): ESCI-sample as the loader serves it, plus the judges' synthetic `price` and
`stock` (`referee.judge.with_hot_fields`), plus
- `location` (`geo_point`, `{"lat", "lon"}`): 70% of products around 20 city centres (Gaussian, about 30 km), 30%
  uniform over the inhabited latitudes. **Every coordinate is an exact multiple of Lucene's quantum** (latitude
  180/2^32 degrees, longitude 360/2^32), so OpenSearch stores it without loss and a distance can agree to 1e-6;
- `offers` (`nested`: `seller` keyword, `price` float with 2 decimals, `stock` integer): 0 to 5 per product, sellers
  from 40 names.
Both sides index exactly this catalogue, in this order, with this mapping.

**Roles and rules** (fixed before the recording and before any engine run):

| Role | Families | Prediction | Rule |
|---|---|---|---|
| `geo` | geo_distance (km, mi, m, nmi, ft), geo_bounding_box, geo_sort | S9-4: at least 98% identical | totals and hit ids in order exactly (each request sorts by exact keys ending in `product_id`); `_geo_distance` sort values within 1e-6 relative |
| `nested` | nested_query (score_mode avg, max, min, sum, none; the conditions must hold in one offer), nested_in_bool, nested_aggs | S9-5: at least 98% identical | totals, hit ids in order, scores exactly (the inner queries score constants, so every score_mode gives exact values); nested aggregations: doc_count, bucket keys and counts, min/max/value_count exactly, sum and avg within 1e-6 relative |

OpenSearch's `_geo_distance` defaults (`arc`, metres unless `unit`) and `nested` defaults are what the requests
leave unset; the engine must give OpenSearch's answer.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from referee import conformance, loader

SUITE = "s9-v1"
SEED = 20261120
RECORDING = Path(__file__).resolve().parent / "conformance" / f"opensearch-2.19-esci-sample-{SUITE}.jsonl.gz"
TIE = {"product_id": "asc"}
LAT_Q, LON_Q = 180.0 / 2 ** 32, 360.0 / 2 ** 32
SELLERS = [f"seller-{i:02d}" for i in range(40)]
CITIES = [(40.71, -74.01), (34.05, -118.24), (41.88, -87.63), (29.76, -95.37), (47.61, -122.33), (25.76, -80.19),
          (51.51, -0.13), (48.86, 2.35), (52.52, 13.40), (40.42, -3.70), (41.90, 12.50), (35.68, 139.69),
          (37.57, 126.98), (1.35, 103.82), (-33.87, 151.21), (-23.55, -46.63), (19.43, -99.13), (28.61, 77.21),
          (30.04, 31.24), (-26.20, 28.05)]
PLAN = [("geo_distance", "geo", 40), ("geo_bounding_box", "geo", 30), ("geo_sort", "geo", 30),
        ("nested_query", "nested", 40), ("nested_in_bool", "nested", 20), ("nested_aggs", "nested", 30)]


def _quantise(x, q):
    return float(np.floor(x / q) * q)


def dataset():
    from referee.judge import with_hot_fields
    ds = loader.load_dataset("esci-sample")
    cat, mapping = with_hot_fields(ds)
    cat = cat.reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    n = len(cat)
    locs, offers = [], []
    for i in range(n):
        if rng.random() < 0.7:
            lat0, lon0 = CITIES[int(rng.integers(len(CITIES)))]
            lat, lon = lat0 + rng.normal(0, 0.27), lon0 + rng.normal(0, 0.35)
        else:
            lat, lon = rng.uniform(-50, 65), rng.uniform(-179.9, 179.9)
        lat, lon = float(np.clip(lat, -89.9, 89.9)), float(np.clip(lon, -179.99, 179.99))
        locs.append({"lat": _quantise(lat, LAT_Q), "lon": _quantise(lon, LON_Q)})
        k = int(rng.integers(0, 6))
        offers.append([{"seller": SELLERS[int(rng.integers(len(SELLERS)))],
                        "price": round(float(rng.uniform(1, 500)), 2), "stock": int(rng.integers(0, 50))}
                       for _ in range(k)])
    cat["location"] = locs
    cat["offers"] = offers
    m = copy.deepcopy(mapping)
    m["properties"]["location"] = {"type": "geo_point"}
    m["properties"]["offers"] = {"type": "nested", "properties": {
        "seller": {"type": "keyword"}, "price": {"type": "float"}, "stock": {"type": "integer"}}}
    return ds, cat, m


def build_suite(ds=None, cat=None, seed=SEED + 1) -> list:
    if cat is None:
        ds, cat, _ = dataset()
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]  # noqa: E731
    queries = [str(q) for q in ds.fit["query"].drop_duplicates()]

    def centre():
        lat0, lon0 = pick(CITIES)
        return {"lat": round(lat0 + float(rng.normal(0, 0.1)), 6), "lon": round(lon0 + float(rng.normal(0, 0.1)), 6)}

    units = {"km": (5, 60), "mi": (3, 40), "m": (2000, 50000), "nmi": (3, 30), "ft": (10000, 150000)}

    def dist():
        u = pick(list(units))
        lo, hi = units[u]
        return f"{round(float(rng.uniform(lo, hi)), 1)}{u}"

    def price_band():
        lo = round(float(rng.uniform(1, 400)), 2)
        return {"range": {"price": {"gte": lo, "lt": round(lo + float(rng.uniform(20, 150)), 2)}}}

    def offer_cond():
        lo = round(float(rng.uniform(1, 400)), 2)
        conds = [{"range": {"offers.price": {"gte": lo, "lt": round(lo + float(rng.uniform(20, 200)), 2)}}}]
        if rng.random() < 0.6:
            conds.append({"terms": {"offers.seller": [pick(SELLERS) for _ in range(int(rng.integers(1, 6)))]}})
        if rng.random() < 0.3:
            conds.append({"range": {"offers.stock": {"gt": int(rng.integers(0, 40))}}})
        return {"bool": {"filter": conds}} if rng.random() < 0.5 else {"bool": {"must": conds}}

    make = {
        "geo_distance": lambda: {"query": {"bool": {"filter": [{"geo_distance": {"distance": dist(), "location": centre()}}]}},
                                 "sort": [TIE], "size": 200},
        "geo_bounding_box": lambda: (lambda c, h, w: {"query": {"bool": {"filter": [{"geo_bounding_box": {"location": {
            "top_left": {"lat": round(c["lat"] + h, 6), "lon": round(c["lon"] - w, 6)},
            "bottom_right": {"lat": round(c["lat"] - h, 6), "lon": round(c["lon"] + w, 6)}}}}]}},
            "sort": [TIE], "size": 200})(centre(), float(rng.uniform(0.05, 0.6)), float(rng.uniform(0.05, 0.8))),
        "geo_sort": lambda: (lambda c: {"query": {"bool": {"filter": [{"geo_distance": {"distance": dist(), "location": c}},
                                                                      price_band()]}},
                                        "sort": [{"_geo_distance": {"location": c, "order": pick(["asc", "desc"]),
                                                                    **({"unit": pick(["km", "mi", "m"])} if rng.random() < 0.7 else {})}},
                                                 TIE], "size": 50})(centre()),
        "nested_query": lambda: {"query": {"nested": {"path": "offers", "query": offer_cond(),
                                                      "score_mode": pick(["avg", "max", "min", "sum", "none"])}},
                                 "sort": ["_score", TIE], "size": 100},
        "nested_in_bool": lambda: {"query": {"bool": {
            "filter": [price_band()],
            "must": [{"nested": {"path": "offers", "query": offer_cond(), "score_mode": pick(["sum", "max"])}}],
            "must_not": [{"nested": {"path": "offers", "query": {"term": {"offers.seller": pick(SELLERS)}}}}]}},
            "sort": ["_score", TIE], "size": 100},
        "nested_aggs": lambda: {"query": pick([{"match": {"product_title": pick(queries)}}, {"bool": {"filter": [price_band()]}}]),
                                "size": 0, "aggs": {"o": {"nested": {"path": "offers"}, "aggs": {
                                    "sellers": {"terms": {"field": "offers.seller", "size": 10}},
                                    "p": {pick(["avg", "sum", "min", "max"]): {"field": "offers.price"}},
                                    "s": {pick(["max", "value_count", "sum"]): {"field": "offers.stock"}}}}}},
    }
    suite, i = [], 0
    for fam, role, n in PLAN:
        for _ in range(n):
            suite.append({"id": i, "family": fam, "role": role, "request": make[fam]()})
            i += 1
    return suite


def _rel(a, b, tol):
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol * max(1e-300, abs(float(b)))


def _agg_diffs(g, e, path=""):
    out = []
    for k, ev in (e or {}).items():
        gv = (g or {}).get(k) if isinstance(g, dict) else None
        p = f"{path}.{k}" if path else k
        if isinstance(ev, dict):
            if gv is None:
                out.append(f"{p}: missing")
                continue
            if "buckets" in ev:
                gb, eb = gv.get("buckets", []), ev["buckets"]
                if [(x.get("key"), x.get("doc_count")) for x in gb] != [(y.get("key"), y.get("doc_count")) for y in eb]:
                    out.append(f"{p}: buckets differ")
            elif "value" in ev:
                tol = 1e-6 if any(s in p for s in (".p", ".s")) and not isinstance(ev["value"], int) else 0
                if not _rel(gv.get("value"), ev.get("value"), max(tol, 0 if "count" in p else tol)):
                    out.append(f"{p}: {gv.get('value')} != {ev.get('value')}")
            else:
                if ev.get("doc_count") != gv.get("doc_count"):
                    out.append(f"{p}.doc_count: {gv.get('doc_count')} != {ev.get('doc_count')}")
                out += _agg_diffs(gv, {kk: vv for kk, vv in ev.items() if isinstance(vv, dict)}, p)
    return out


def judge(index, recorded, suite) -> dict:
    rows = []
    for s in suite:
        g, e = conformance.outcome(index, s), conformance.outcome(recorded, s)
        row = {"id": s["id"], "family": s["family"], "role": s["role"]}
        if "error" in g:
            row["diffs"] = [f"engine refused: {g['error']}: {g.get('reason', '')[:120]}"]
        elif "error" in e:
            row["diffs"] = [f"OpenSearch refused: {e.get('reason', '')[:160]}"]
        else:
            ge, ee = g["ok"], e["ok"]
            d = []
            if ge["hits"]["total"] != ee["hits"]["total"]:
                d.append(f"total {ge['hits']['total']} != {ee['hits']['total']}")
            gh, eh = ge["hits"]["hits"], ee["hits"]["hits"]
            gi, ei = [h["_id"] for h in gh], [h["_id"] for h in eh]
            if gi != ei:
                k = next((i for i, (a, b) in enumerate(zip(gi, ei)) if a != b), min(len(gi), len(ei)))
                d.append(f"ids differ from rank {k} ({len(gi)} vs {len(ei)})")
            elif s["family"] == "geo_sort":
                for x, y in zip(gh, eh):
                    if not _rel(x["sort"][0], y["sort"][0], 1e-6):
                        d.append(f"{x['_id']}: distance {x['sort'][0]} != {y['sort'][0]}")
                        break
            elif s["role"] == "nested" and s["family"] != "nested_aggs":
                for x, y in zip(gh, eh):
                    if not _rel(x["_score"], y["_score"], 1e-6):
                        d.append(f"{x['_id']}: score {x['_score']} != {y['_score']}")
                        break
            d += _agg_diffs(ge.get("aggregations"), ee.get("aggregations"))
            row["diffs"] = d
        row["passed"] = not row["diffs"]
        rows.append(row)

    def rate(rs):
        p = sum(r["passed"] for r in rs)
        return {"passed": p, "n": len(rs), "rate": p / len(rs) if rs else None}
    return {"kind": "conformance-s9", "suite": SUITE, "suite_sha256": conformance.suite_sha256(suite),
            "recording": getattr(recorded, "meta", None),
            "prediction_4_geo": rate([r for r in rows if r["role"] == "geo"]),
            "prediction_5_nested": rate([r for r in rows if r["role"] == "nested"]),
            "by_family": {f: {"role": role, **rate([r for r in rows if r["family"] == f])} for f, role, _ in PLAN},
            "failures": [r for r in rows if not r["passed"]][:60]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee.conformance_s9")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--engine", default="engine", choices=("engine", "reference"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/conformance-s9.json")
    a = ap.parse_args(argv)
    ds, cat, mapping = dataset()
    suite = build_suite(ds, cat)
    if a.record:
        from referee import opensearch
        p = conformance.record(opensearch.build_index(cat, mapping), suite, path=RECORDING,
                               meta={"server": "OpenSearch 2.19.1 (Lucene 9.12.1), single node, one shard",
                                     "dataset": f"esci-sample + synthetic price, stock, location, offers (seed {SEED})",
                                     "analyzer": os.environ.get("S3_SEARCH_OS_ANALYZER") or "s3_english"})
        print(f"recorded {len(suite)} answers -> {p}")
        return 0
    from referee.bench import get_api
    index = get_api(a.engine).build_index(cat, mapping, device=a.device)
    rep = judge(index, conformance.Recorded(suite, path=RECORDING), suite)
    rep.update({"engine": a.engine, "device": a.device, "label": "CPU, development only"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k in ("prediction_4_geo", "prediction_5_nested"):
        print(f"{k:22s} {rep[k]['passed']}/{rep[k]['n']}")
    for f, v in rep["by_family"].items():
        print(f"  {f:18s} {v['role']:7s} {v['passed']}/{v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
