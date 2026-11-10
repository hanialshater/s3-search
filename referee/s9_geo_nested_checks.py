"""S9-4 (geo) and S9-5 (nested) checked by brute force on toy, with OpenSearch's semantics (owner:
evaluation-steward, P6). Registered in `s9_checks.CHECKS`; run when the items are marked landed in `scope.py`.

The toy copy gains `location` (geo_point; every coordinate an exact multiple of Lucene's quantum; 10% missing) and
`offers` (nested: seller keyword, price float with 2 decimals, stock integer; 0 to 4 per product), seed 20261122.
- geo_distance: haversine on OpenSearch's mean earth radius (6,371,008.7714 m), in m, km, mi, yd, ft, in, nmi, cm,
  mm, as a filter (set and total exact);
- geo_bounding_box: lat within [bottom, top], lon within [left, right], or outside (right, left) when the box crosses
  the dateline (set and total exact);
- _geo_distance sort: asc and desc, in the requested unit (metres by default), a missing location as +infinity,
  ties by product_id; ids in order exact, values within 1e-6 relative;
- nested: the conditions must hold in one offer; constant-scoring inner clauses (range and terms score 1.0 each in
  must, 0 in filter); score_mode avg, sum, max, min over the matching offers, none = 0; ids, totals and scores exact;
  in bool with must_not; a leaf on the nested path outside a nested query matches nothing;
- nested aggregation: doc_count = offers of the matching products; terms on offers.seller by count then key;
  min, max, value_count exact, sum and avg within 1e-6.
"""
from __future__ import annotations

import copy
import math

import numpy as np

R = 6371008.7714
UNITS = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "yd": 0.9144, "ft": 0.3048, "in": 0.0254, "nmi": 1852.0,
         "cm": 0.01, "mm": 0.001}
LAT_Q, LON_Q = 180.0 / 2 ** 32, 360.0 / 2 ** 32
SELLERS = [f"s{i}" for i in range(8)]
ALL = 10_000


def toy_geo_nested(ds):
    rng = np.random.default_rng(20261122)
    cat = ds.catalogue.copy().reset_index(drop=True)
    locs, offers = [], []
    for i in range(len(cat)):
        if rng.random() < 0.1:
            locs.append(None)
        else:
            lat = float(np.floor(rng.uniform(40, 41) / LAT_Q) * LAT_Q)
            lon = float(np.floor(rng.uniform(-74.5, -73.5) / LON_Q) * LON_Q)
            locs.append({"lat": lat, "lon": lon})
        offers.append([{"seller": SELLERS[int(rng.integers(len(SELLERS)))], "price": round(float(rng.uniform(1, 300)), 2),
                        "stock": int(rng.integers(0, 30))} for _ in range(int(rng.integers(0, 5)))])
    cat["location"], cat["offers"] = locs, offers
    m = copy.deepcopy(ds.mapping)
    m["properties"]["location"] = {"type": "geo_point"}
    m["properties"]["offers"] = {"type": "nested", "properties": {"seller": {"type": "keyword"},
                                                                 "price": {"type": "float"}, "stock": {"type": "integer"}}}
    return cat, m


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(h)))


def _ids(index, req):
    r = index.search({**req, "size": req.get("size", ALL)})
    return r


def check_geo(index, ds, build=None):
    if build is None:
        import engine.api
        build = engine.api.build_index
    cat, m = toy_geo_nested(ds)
    idx = build(cat, m)
    ids = list(cat["product_id"])
    locs = list(cat["location"])
    problems = []
    centre = {"lat": 40.5, "lon": -74.0}
    for unit, dist in (("km", 25), ("m", 18000), ("mi", 12), ("yd", 30000), ("ft", 70000), ("in", 900000),
                       ("nmi", 9), ("cm", 2000000), ("mm", 15000000)):
        radius = dist * UNITS[unit]
        exp = {p for p, loc in zip(ids, locs) if loc is not None and haversine(centre, loc) <= radius}
        r = _ids(idx, {"query": {"bool": {"filter": [{"geo_distance": {"distance": f"{dist}{unit}", "location": centre}}]}}})
        got = {h["_id"] for h in r["hits"]["hits"]}
        if got != exp or r["hits"]["total"] != len(exp):
            problems.append(f"geo_distance {dist}{unit}: {len(got)} hits, expected {len(exp)}")
    for tl, br in (((40.8, -74.3), (40.2, -73.7)), ((40.6, -74.0), (40.55, -73.95)),
                   ((40.9, 179.5), (40.1, -179.5))):
        top, left = tl
        bottom, right = br
        def inside(loc):
            if loc is None or not (bottom <= loc["lat"] <= top):
                return False
            return left <= loc["lon"] <= right if left <= right else (loc["lon"] >= left or loc["lon"] <= right)
        exp = {p for p, loc in zip(ids, locs) if inside(loc)}
        r = _ids(idx, {"query": {"bool": {"filter": [{"geo_bounding_box": {"location": {
            "top_left": {"lat": top, "lon": left}, "bottom_right": {"lat": bottom, "lon": right}}}}]}}})
        got = {h["_id"] for h in r["hits"]["hits"]}
        if got != exp or r["hits"]["total"] != len(exp):
            problems.append(f"geo_bounding_box {tl}-{br}: {len(got)} hits, expected {len(exp)}")
    for order in ("asc", "desc"):
        for unit in (None, "km", "mi"):
            spec = {"location": centre, "order": order, **({"unit": unit} if unit else {})}
            r = idx.search({"query": {"match_all": {}}, "sort": [{"_geo_distance": spec}, {"product_id": "asc"}],
                            "size": ALL})
            f = UNITS[unit or "m"]
            d = {p: (haversine(centre, loc) / f if loc is not None else math.inf) for p, loc in zip(ids, locs)}
            exp = sorted(ids, key=lambda p: ((d[p] if order == "asc" else -d[p]), p))
            got = [h["_id"] for h in r["hits"]["hits"]]
            if got != exp:
                k = next((i for i, (a, b) in enumerate(zip(got, exp)) if a != b), None)
                problems.append(f"_geo_distance {order} {unit}: order differs at rank {k}")
                continue
            for h in r["hits"]["hits"]:
                v, e = h["sort"][0], d[h["_id"]]
                if math.isinf(e):
                    if not (v is None or (isinstance(v, (int, float)) and math.isinf(v))):
                        problems.append(f"_geo_distance {order}: a missing location sorts as {v}, not +inf")
                        break
                elif abs(v - e) > 1e-6 * max(1e-300, abs(e)):
                    problems.append(f"_geo_distance {order} {unit}: {h['_id']} {v} != {e}")
                    break
    return problems


def check_nested(index, ds, build=None):
    if build is None:
        import engine.api
        build = engine.api.build_index
    cat, m = toy_geo_nested(ds)
    idx = build(cat, m)
    ids = list(cat["product_id"])
    offs = list(cat["offers"])
    f32 = lambda x: float(np.float32(x))  # noqa: E731
    problems = []

    def child_score(o, clauses, scoring):
        s = 0.0
        for c in clauses:
            (k, v), = c.items()
            (fld, cond), = v.items()
            key = fld.split(".")[1]
            if k == "range":
                val = f32(o[key]) if key == "price" else o[key]
                ok = all({"gte": val >= cond.get("gte", -math.inf) if "gte" in cond else True,
                          "lt": val < cond["lt"] if "lt" in cond else True,
                          "gt": val > cond["gt"] if "gt" in cond else True}.values())
            else:
                ok = o[key] in cond
            if not ok:
                return None
            s += 1.0 if scoring else 0.0
        return s

    cases = []
    for lo, hi, sellers, mode, kind in ((50, 150, ["s1", "s2"], "avg", "must"), (0, 300, None, "sum", "must"),
                                        (100, 200, ["s3"], "max", "must"), (10, 90, ["s0", "s4", "s5"], "min", "must"),
                                        (0, 120, ["s6"], "none", "must"), (40, 260, ["s2", "s7"], "sum", "filter")):
        conds = [{"range": {"offers.price": {"gte": lo, "lt": hi}}}]
        if sellers:
            conds.append({"terms": {"offers.seller": sellers}})
        cases.append((conds, mode, kind))
    for conds, mode, kind in cases:
        req = {"query": {"nested": {"path": "offers", "query": {"bool": {kind: conds}}, "score_mode": mode}},
               "sort": ["_score", {"product_id": "asc"}], "size": ALL}
        exp = {}
        for p, os_ in zip(ids, offs):
            sc = [s for s in (child_score(o, conds, kind == "must") for o in os_) if s is not None]
            if sc:
                exp[p] = {"avg": sum(sc) / len(sc), "sum": sum(sc), "max": max(sc), "min": min(sc), "none": 0.0}[mode]
        order = sorted(exp, key=lambda p: (-exp[p], p))
        r = idx.search(req)
        got = [(h["_id"], h["_score"]) for h in r["hits"]["hits"]]
        if [g for g, _ in got] != order or r["hits"]["total"] != len(exp) or \
                any(abs(s - exp[g]) > 1e-6 for g, s in got):
            problems.append(f"nested {mode} {kind} {conds}: {len(got)} hits (total {r['hits']['total']}), expected {len(exp)}")
    # in bool, with must_not; same-offer semantics are in every case above
    conds = [{"range": {"offers.price": {"gte": 20, "lt": 200}}}]
    req = {"query": {"bool": {"must": [{"nested": {"path": "offers", "query": {"bool": {"must": conds}}, "score_mode": "sum"}}],
                              "must_not": [{"nested": {"path": "offers", "query": {"term": {"offers.seller": "s0"}}}}]}},
           "sort": ["_score", {"product_id": "asc"}], "size": ALL}
    exp = {}
    for p, os_ in zip(ids, offs):
        if any(o["seller"] == "s0" for o in os_):
            continue
        sc = [s for s in (child_score(o, conds, True) for o in os_) if s is not None]
        if sc:
            exp[p] = sum(sc)
    r = idx.search(req)
    if [h["_id"] for h in r["hits"]["hits"]] != sorted(exp, key=lambda p: (-exp[p], p)):
        problems.append("nested in bool with must_not: wrong hits")
    r = idx.search({"query": {"term": {"offers.seller": "s1"}}, "size": ALL})
    if r["hits"]["total"] != 0:
        problems.append(f"a leaf on a nested path outside a nested query matched {r['hits']['total']} documents")
    # nested aggregation, under two queries
    for q, rows in (({"match_all": {}}, range(len(ids))),
                    ({"bool": {"filter": [{"range": {"price": {"lt": 100}}}]}},
                     [i for i, v in enumerate(cat["price"]) if v < 100])):
        children = [o for i in rows for o in offs[i]]
        aggs = {"o": {"nested": {"path": "offers"}, "aggs": {
            "sellers": {"terms": {"field": "offers.seller", "size": 5}},
            "mn": {"min": {"field": "offers.price"}}, "mx": {"max": {"field": "offers.price"}},
            "sm": {"sum": {"field": "offers.price"}}, "av": {"avg": {"field": "offers.stock"}},
            "vc": {"value_count": {"field": "offers.stock"}}}}}
        r = idx.search({"query": q, "size": 0, "aggs": aggs})["aggregations"]["o"]
        from collections import Counter
        cnt = sorted(Counter(o["seller"] for o in children).items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        prices = [f32(o["price"]) for o in children]
        stocks = [o["stock"] for o in children]
        exp = {"doc_count": len(children), "mn": min(prices), "mx": max(prices), "sm": math.fsum(prices),
               "av": sum(stocks) / len(stocks), "vc": len(stocks)}
        if r.get("doc_count") != exp["doc_count"]:
            problems.append(f"nested agg under {q}: doc_count {r.get('doc_count')} != {exp['doc_count']}")
        if [(b["key"], b["doc_count"]) for b in r["sellers"]["buckets"]] != cnt:
            problems.append(f"nested agg under {q}: seller buckets differ")
        for k, tol in (("mn", 0), ("mx", 0), ("vc", 0), ("sm", 1e-6), ("av", 1e-6)):
            g = r[k]["value"]
            if abs(g - exp[k]) > tol * max(1e-300, abs(exp[k])):
                problems.append(f"nested agg under {q}: {k} {g} != {exp[k]}")
    return problems
