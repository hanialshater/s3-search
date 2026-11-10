"""S7-4: metric and bucket aggregations against brute force (pandas, float64) on toy."""
import math

import numpy as np
import pandas as pd
import pytest

from engine.analysis import analyze

QUERIES = [{"match_all": {}}, {"match": {"product_title": "black shoes wool"}},
           {"bool": {"filter": [{"term": {"category": "socks"}}]}}, {"match": {"product_title": "zzzz"}}]


def matched(index, catalogue, q):
    r = index.search({"query": q, "size": 500})
    ids = {h["_id"] for h in r["hits"]["hits"]}
    return catalogue[catalogue["product_id"].isin(ids)]


def f32(series):
    """Values as the index holds them (float fields are float32), read as float64."""
    return series.dropna().astype(np.float32).astype(np.float64)


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("q", QUERIES)
@pytest.mark.parametrize("field", ["price", "stock"])
def test_metrics(index, catalogue, q, field):
    m = matched(index, catalogue, q)
    v = f32(m[field]) if field == "price" else m[field].dropna().astype(np.float64)
    aggs = {k: {k: {"field": field}} for k in ("min", "max", "avg", "sum", "stats", "value_count", "cardinality")}
    r = index.search({"query": q, "size": 0, "aggs": aggs})["aggregations"]
    n = len(v)
    assert r["value_count"]["value"] == n
    assert r["cardinality"]["value"] == v.nunique()
    assert close(r["sum"]["value"], float(v.sum()))
    assert close(r["min"]["value"], float(v.min()) if n else None)
    assert close(r["max"]["value"], float(v.max()) if n else None)
    assert close(r["avg"]["value"], float(v.sum()) / n if n else None)
    st = r["stats"]
    assert st["count"] == n and close(st["sum"], float(v.sum())) and close(st["min"], float(v.min()) if n else None)


def test_keyword_count_and_cardinality(index, catalogue):
    m = matched(index, catalogue, QUERIES[1])
    r = index.search({"query": QUERIES[1], "size": 0, "aggs": {
        "c": {"cardinality": {"field": "product_brand.keyword"}},
        "n": {"value_count": {"field": "product_brand.keyword"}}}})["aggregations"]
    assert r["c"]["value"] == m["product_brand"].dropna().nunique()
    assert r["n"]["value"] == int(m["product_brand"].notna().sum())


@pytest.mark.parametrize("q", QUERIES)
def test_range(index, catalogue, q):
    ranges = [{"to": 20}, {"from": 20, "to": 50.5, "key": "mid"}, {"from": 50.5}, {"from": 10, "to": 30}]
    r = index.search({"query": q, "size": 0, "aggs": {"r": {"range": {"field": "price", "ranges": ranges}}}})
    v = f32(matched(index, catalogue, q)["price"])
    want = []
    for rg in sorted(ranges, key=lambda x: (x.get("from", -math.inf), x.get("to", math.inf))):
        lo, hi = rg.get("from"), rg.get("to")
        sel = v[(v >= (lo if lo is not None else -math.inf)) & (v < (hi if hi is not None else math.inf))]
        key = rg.get("key") or f"{'*' if lo is None else repr(float(lo))}-{'*' if hi is None else repr(float(hi))}"
        bk = {"key": key, **({"from": float(lo)} if lo is not None else {}), **({"to": float(hi)} if hi is not None else {}),
              "doc_count": len(sel)}
        want.append(bk)
    assert r["aggregations"]["r"]["buckets"] == want


@pytest.mark.parametrize("q", QUERIES)
@pytest.mark.parametrize("interval,offset,mdc", [(10, 0, 0), (7.5, 2, 0), (25, 0, 1), (1, 0, 0)])
def test_histogram(index, catalogue, q, interval, offset, mdc):
    field = "price" if interval != 1 else "stock"
    r = index.search({"query": q, "size": 0, "aggs": {"h": {"histogram": {
        "field": field, "interval": interval, "offset": offset, "min_doc_count": mdc}}}})
    m = matched(index, catalogue, q)
    v = f32(m[field]) if field == "price" else m[field].dropna().astype(np.float64)
    if not len(v):
        assert r["aggregations"]["h"]["buckets"] == []
        return
    k = np.floor((v - offset) / interval).astype(int)
    counts = k.value_counts()
    want = [{"key": i * interval + offset, "doc_count": int(counts.get(i, 0))} for i in range(k.min(), k.max() + 1)
            if counts.get(i, 0) >= mdc and (counts.get(i, 0) > 0 or mdc == 0)]
    assert r["aggregations"]["h"]["buckets"] == want


def test_sub_aggregations_under_terms(index, catalogue):
    q = QUERIES[0]
    r = index.search({"query": q, "size": 0, "aggs": {"c": {"terms": {"field": "category"}, "aggs": {
        "p": {"avg": {"field": "price"}}, "s": {"max": {"field": "stock"}},
        "h": {"histogram": {"field": "price", "interval": 50}},
        "b": {"terms": {"field": "product_brand.keyword", "size": 2}}}}}})
    for bk in r["aggregations"]["c"]["buckets"]:
        g = catalogue[catalogue["category"] == bk["key"]]
        assert bk["doc_count"] == len(g)
        v = f32(g["price"])
        assert close(bk["p"]["value"], float(v.sum()) / len(v))
        assert bk["s"]["value"] == float(g["stock"].max())
        top = sorted(g["product_brand"].dropna().value_counts().items(), key=lambda kv: (-kv[1], kv[0]))[:2]
        assert [(x["key"], x["doc_count"]) for x in bk["b"]["buckets"]] == top
        assert sum(x["doc_count"] for x in bk["h"]["buckets"]) == len(v)
