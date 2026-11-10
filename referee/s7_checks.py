"""Correctness of S7's constructs as they land, by brute force over `toy` (owner: evaluation-steward, P6).

`referee/scope.py` says which S7 items have landed; `check_landed_correct` runs this file's check for every landed
item, and fails for a landed item that has no check yet (the steward writes it when marking the item landed). The
semantics are OpenSearch's (Lucene's), since S7-9 compares with OpenSearch:

- S7-1 `match_phrase`: the analysed query tokens at consecutive positions in one field; stop words keep their
  position, so a removed stop word leaves a gap. `slop` n: two terms match when |(p2 - p1) - 1| <= n (so slop 2 also
  matches the two words swapped). Checked: the match set and total.
- S7-2 `fuzzy` (the value is not analysed), `match` with `fuzziness` (each analysed token), `prefix` (not analysed),
  `match_bool_prefix` (all tokens but the last as terms, the last as a prefix; any may match). Fuzziness AUTO: 0
  edits for 1-2 characters, 1 for 3-5, 2 for more; Damerau distance (an adjacent transposition is one edit);
  prefix_length 0. Checked: the match set and total.

Scores are not checked here; S7-9's OpenSearch families compare them.
"""
from __future__ import annotations

from referee.analysis import analyze, analyze_positions

ALL = 10_000
FIELDS = ["product_title", "product_bullet_point", "product_description"]


def _osa(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance (Damerau with adjacent transpositions), capped at cap + 1."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def auto_edits(term: str) -> int:
    return 0 if len(term) <= 2 else (1 if len(term) <= 5 else 2)


def _ids(index, req):
    r = index.search({**req, "size": ALL})
    return r["hits"]["total"], {h["_id"] for h in r["hits"]["hits"]}


def _expect(ds, field, pred):
    return {pid for pid, text in zip(ds.catalogue["product_id"], ds.catalogue[field]) if pred(text)}


def _phrase_pred(query, slop):
    q = [t for t, _ in analyze_positions(query)]
    qpos = [p for _, p in analyze_positions(query)]

    def pred(text):
        toks = analyze_positions(text)
        if not q:
            return False
        if len(q) == 1:
            return any(t == q[0] for t, _ in toks)
        if len(q) == 2 and slop:
            want = qpos[1] - qpos[0]
            p1 = [p for t, p in toks if t == q[0]]
            p2 = [p for t, p in toks if t == q[1]]
            return any(abs((b - a) - want) <= slop for a in p1 for b in p2 if a != b)
        positions = {}
        for t, p in toks:
            positions.setdefault(t, set()).add(p)
        offs = [p - qpos[0] for p in qpos]
        return any(all(start + o in positions.get(t, ()) for t, o in zip(q, offs)) for start in positions.get(q[0], ()))
    return pred


def check_s7_1(index, ds):
    problems = []
    cases = [("product_title", "red lamp", 0), ("product_title", "portable gray", 0), ("product_title", "lamp red", 0),
             ("product_title", "lamp red", 2), ("product_title", "kestrel portable", 1),
             ("product_bullet_point", "drinks hot", 0), ("product_bullet_point", "keeps drinks hot", 0),
             ("product_description", "made steel", 0), ("product_description", "made steel", 1),
             ("product_description", "made of steel", 0), ("product_bullet_point", "camping hiking", 0),
             ("product_bullet_point", "camping hiking", 1), ("product_title", "zzzz lamp", 0)]
    for field, text, slop in cases:
        spec = {"query": text, "slop": slop} if slop else text
        total, got = _ids(index, {"query": {"match_phrase": {field: spec}}})
        exp = _expect(ds, field, _phrase_pred(text, slop))
        if got != exp or total != len(exp):
            problems.append(f"match_phrase {field}:{text!r} slop {slop}: {len(got)} hits (total {total}), "
                            f"expected {len(exp)}; missing {sorted(exp - got)[:3]}, extra {sorted(got - exp)[:3]}")
    return problems


def _terms_of(text):
    return set(analyze(text))


def check_s7_2(index, ds):
    problems = []

    def run(label, req, field, pred):
        total, got = _ids(index, req)
        exp = _expect(ds, field, pred)
        if got != exp or total != len(exp):
            problems.append(f"{label}: {len(got)} hits (total {total}), expected {len(exp)}; "
                            f"missing {sorted(exp - got)[:3]}, extra {sorted(got - exp)[:3]}")

    for value in ("lamq", "bots", "kettel", "wooll", "acme", "zzzzzz", "gry"):
        e = auto_edits(value)
        run(f"fuzzy {value!r}", {"query": {"fuzzy": {"product_title": value}}}, "product_title",
            lambda t, v=value, e=e: any(_osa(v, w, e) <= e for w in _terms_of(t)))
    for text in ("red lamq", "blu ketle", "portabel speakr", "the"):
        toks = list(dict.fromkeys(analyze(text)))
        run(f"match fuzziness {text!r}", {"query": {"match": {"product_title": {"query": text, "fuzziness": "AUTO"}}}},
            "product_title",
            lambda t, q=toks: any(_osa(a, w, auto_edits(a)) <= auto_edits(a) for a in q for w in _terms_of(t)))
    for value in ("lam", "ke", "port", "zz"):
        run(f"prefix {value!r}", {"query": {"prefix": {"product_title": value}}}, "product_title",
            lambda t, v=value: any(w.startswith(v) for w in _terms_of(t)))
    for text in ("red la", "portable ke", "acme", "lamps"):
        toks = analyze(text)
        run(f"match_bool_prefix {text!r}", {"query": {"match_bool_prefix": {"product_title": text}}}, "product_title",
            lambda t, q=toks: bool(q) and (bool(set(q[:-1]) & _terms_of(t)) or
                                           any(w.startswith(q[-1]) for w in _terms_of(t))))
    return problems


def check_s7_3(index, ds):
    """exists, ids, post_filter (aggregations before it, hits and total after it), collapse (keyword or numeric; the
    missing values as one group; from/size over groups; total = matching documents)."""
    import pandas as pd
    from referee import reference
    c = ds.catalogue.reset_index(drop=True)
    ids = list(c["product_id"])
    problems = []
    for field, col in (("product_color", "product_color"), ("product_bullet_point", "product_bullet_point"),
                       ("product_brand.keyword", "product_brand"), ("price", "price"), ("stock", "stock")):
        exp = {p for p, v in zip(ids, c[col]) if pd.notna(v)}
        total, got = _ids(index, {"query": {"exists": {"field": field}}})
        if got != exp or total != len(exp):
            problems.append(f"exists {field}: {len(got)} hits, expected {len(exp)}")
    want = [ids[3], ids[77], "NO-SUCH-ID", ids[150]]
    total, got = _ids(index, {"query": {"ids": {"values": want}}})
    if got != set(want) - {"NO-SUCH-ID"} or total != 3:
        problems.append(f"ids: {sorted(got)} (total {total})")
    total, got = _ids(index, {"query": {"bool": {"must": [{"ids": {"values": want}}],
                                                 "filter": [{"range": {"price": {"lt": 1e9}}}]}}})
    if got != set(want) - {"NO-SUCH-ID"}:
        problems.append("ids inside bool: wrong set")

    ref = reference.build_index(ds.catalogue, ds.mapping)
    agg = {"c": {"terms": {"field": "category", "size": 20}}}
    for q, pf in (({"match": {"product_title": "red lamp"}}, {"term": {"category": "lamp"}}),
                  ({"match_all": {}}, {"range": {"price": {"lt": 100}}}),
                  ({"bool": {"filter": [{"range": {"stock": {"gt": 0}}}]}}, {"terms": {"category": ["mug", "tent"]}})):
        r = index.search({"query": q, "post_filter": pf, "aggs": agg, "size": ALL})
        before = ref.search({"query": q, "aggs": agg, "size": 0})
        after = ref.search({"query": {"bool": {"must": [q], "filter": [pf]}}, "size": ALL})
        if r["aggregations"]["c"]["buckets"] != before["aggregations"]["c"]["buckets"]:
            problems.append(f"post_filter {pf}: aggregations are not over the query's match set")
        if r["hits"]["total"] != after["hits"]["total"] or \
                {h["_id"] for h in r["hits"]["hits"]} != {h["_id"] for h in after["hits"]["hits"]}:
            problems.append(f"post_filter {pf}: hits or total are not after the filter "
                            f"({r['hits']['total']} vs {after['hits']['total']})")

    def collapsed(order_req, field, col):
        full = [h["_id"] for h in ref.search({**order_req, "size": ALL})["hits"]["hits"]]
        val = dict(zip(ids, c[col]))
        seen, tops = set(), []
        for p in full:
            v = val[p]
            key = None if pd.isna(v) else v
            if key not in seen:
                seen.add(key)
                tops.append(p)
        return len(full), tops

    cases = [({"query": {"match_all": {}}, "sort": [{"price": "asc"}, {"product_id": "asc"}]}, "category", "category"),
             ({"query": {"match_all": {}}, "sort": [{"price": "desc"}, {"product_id": "asc"}]},
              "product_brand.keyword", "product_brand"),
             ({"query": {"bool": {"filter": [{"range": {"price": {"lt": 200}}}]}},
               "sort": [{"price": "asc"}, {"product_id": "asc"}]}, "product_color.keyword", "product_color"),
             ({"query": {"match_all": {}}, "sort": [{"price": "asc"}, {"product_id": "asc"}]}, "stock", "stock"),
             ({"query": {"multi_match": {"query": "waterproof red lamp boots", "fields": checks_fields()}}},
              "category", "category")]
    for req, field, col in cases:
        n_match, tops = collapsed(req, field, col)
        for frm, size in ((0, ALL), (0, 3), (2, 4), (len(tops) - 1, 5), (len(tops) + 2, 3)):
            r = index.search({**req, "collapse": {"field": field}, "from": frm, "size": size})
            got = [h["_id"] for h in r["hits"]["hits"]]
            if got != tops[frm:frm + size] or r["hits"]["total"] != n_match:
                problems.append(f"collapse {field} from {frm} size {size}: {got[:4]} (total {r['hits']['total']}) "
                                f"!= {tops[frm:frm + size][:4]} (total {n_match})")
                break
    return problems


def checks_fields():
    from referee import checks
    return checks.FIELDS5


def check_s7_4(index, ds):
    """Metric and bucket aggregations against brute force, OpenSearch's definitions: float fields aggregate their
    float32 values; min, max, sum, value_count, cardinality and every bucket key, bound and count exact; avg and
    stats to 1e-6 relative; an empty set gives null (sum 0); range: from inclusive, to exclusive, sorted by from,
    keys "a-b" with "*" for an open end; histogram: key floor(v / interval) * interval, empty buckets between the
    lowest and highest key; one level of sub-aggregations under terms, per bucket. Range buckets come sorted by
    (from, to) as OpenSearch's range aggregator sorts them, not in request order."""
    import math

    import numpy as np
    import pandas as pd
    from referee import reference
    c = ds.catalogue.reset_index(drop=True)
    ids = list(c["product_id"])
    ref = reference.build_index(ds.catalogue, ds.mapping)
    f32 = {"price": [None if pd.isna(v) else float(np.float32(v)) for v in c["price"]],
           "stock": [None if pd.isna(v) else int(v) for v in c["stock"]]}
    problems = []

    def rows_of(q):
        hit = {h["_id"] for h in ref.search({"query": q, "size": ALL})["hits"]["hits"]}
        return [i for i, p in enumerate(ids) if p in hit]

    def rel(a, b, tol):
        if a is None or b is None:
            return a is b
        return abs(a - b) <= tol * max(1e-300, abs(b))

    def metrics(vals):
        v = [x for x in vals if x is not None]
        s = float(math.fsum(v)) if v else 0.0
        return {"min": min(v) if v else None, "max": max(v) if v else None, "sum": s,
                "avg": (s / len(v)) if v else None, "count": len(v)}

    queries = [{"match": {"product_title": "lamp"}}, {"match_all": {}}, {"term": {"category": "no-such"}},
               {"bool": {"filter": [{"range": {"price": {"lt": 120}}}]}}]
    for q in queries:
        rows = rows_of(q)
        for field in ("price", "stock"):
            m = metrics([f32[field][i] for i in rows])
            aggs = {k: {k: {"field": field}} for k in ("min", "max", "sum", "avg", "value_count", "stats")}
            r = index.search({"query": q, "size": 0, "aggs": aggs})["aggregations"]
            for k, exp, tol in (("min", m["min"], 0), ("max", m["max"], 0), ("value_count", m["count"], 0),
                                ("avg", m["avg"], 1e-6)):
                if not rel(r[k]["value"], exp, tol):
                    problems.append(f"{k} {field} under {q}: {r[k]['value']} != {exp}")
            if not rel(r["sum"]["value"], m["sum"], 1e-12):
                problems.append(f"sum {field} under {q}: {r['sum']['value']} != {m['sum']}")
            st = r["stats"]
            for k in ("count", "min", "max", "avg", "sum"):
                if not rel(st.get(k), m[k], 1e-6 if k in ("avg", "sum") else 0):
                    problems.append(f"stats.{k} {field} under {q}: {st.get(k)} != {m[k]}")
        for field, col in (("category", "category"), ("product_brand.keyword", "product_brand")):
            vals = [c[col].iat[i] for i in rows if pd.notna(c[col].iat[i])]
            r = index.search({"query": q, "size": 0, "aggs": {"vc": {"value_count": {"field": field}},
                                                                 "cd": {"cardinality": {"field": field}}}})["aggregations"]
            if r["vc"]["value"] != len(vals) or r["cd"]["value"] != len(set(vals)):
                problems.append(f"value_count/cardinality {field} under {q}: {r['vc']['value']}, {r['cd']['value']} "
                                f"!= {len(vals)}, {len(set(vals))}")
        ranges = [{"to": 50}, {"from": 50, "to": 100}, {"from": 100}, {"from": 75.5, "to": 80}]
        r = index.search({"query": q, "size": 0, "aggs": {"r": {"range": {"field": "price", "ranges": ranges}}}})
        got = r["aggregations"]["r"]["buckets"]
        exp = []
        for rg in ranges:
            lo, hi = rg.get("from"), rg.get("to")
            n = sum(1 for i in rows if f32["price"][i] is not None and (lo is None or f32["price"][i] >= lo)
                    and (hi is None or f32["price"][i] < hi))
            key = f"{'*' if lo is None else float(lo)}-{'*' if hi is None else float(hi)}"
            b = {"key": key, "doc_count": n}
            if lo is not None:
                b["from"] = float(lo)
            if hi is not None:
                b["to"] = float(hi)
            exp.append(b)
        exp.sort(key=lambda b: (b.get("from", -math.inf), b.get("to", math.inf)))
        if [{k: v for k, v in g.items() if k in ("key", "doc_count", "from", "to")} for g in got] != exp:
            problems.append(f"range under {q}: {got} != {exp}")
        for field, interval in (("price", 50), ("price", 7.5), ("stock", 10)):
            vals = [f32[field][i] for i in rows if f32[field][i] is not None]
            keys = [math.floor(v / interval) * interval for v in vals]
            exp = []
            if keys:
                k = min(keys)
                while k <= max(keys) + 1e-9:
                    exp.append({"key": float(k), "doc_count": sum(1 for x in keys if abs(x - k) < 1e-9)})
                    k = k + interval
            r = index.search({"query": q, "size": 0, "aggs": {"h": {"histogram": {"field": field,
                                                                                 "interval": interval}}}})
            got = [{"key": float(b["key"]), "doc_count": b["doc_count"]} for b in r["aggregations"]["h"]["buckets"]]
            if len(got) != len(exp) or any(abs(g["key"] - e["key"]) > 1e-9 or g["doc_count"] != e["doc_count"]
                                           for g, e in zip(got, exp)):
                problems.append(f"histogram {field}/{interval} under {q}: {got[:4]} != {exp[:4]}")
        r = index.search({"query": q, "size": 0, "aggs": {"t": {"terms": {"field": "category", "size": 3},
                                                               "aggs": {"a": {"avg": {"field": "stock"}},
                                                                        "mx": {"max": {"field": "price"}}}}}})
        for b in r["aggregations"]["t"]["buckets"]:
            br = [i for i in rows if c["category"].iat[i] == b["key"]]
            ms, mp = metrics([f32["stock"][i] for i in br]), metrics([f32["price"][i] for i in br])
            if b["doc_count"] != len(br) or not rel(b["a"]["value"], ms["avg"], 1e-6) or b["mx"]["value"] != mp["max"]:
                problems.append(f"terms sub-aggregations, bucket {b['key']} under {q}: {b}")
    return problems


CHECKS = {"S7-1": check_s7_1, "S7-2": check_s7_2, "S7-3": check_s7_3, "S7-4": check_s7_4}


def check_landed_correct(index, ds):
    from referee import scope
    problems = []
    for item in sorted({e[1] for e in scope.landed()}):
        if item not in CHECKS:
            problems.append(f"{item} is marked landed but referee/s7_checks.py has no correctness check for it")
            continue
        problems += CHECKS[item](index, ds)
    assert not problems, "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
