"""Filters, facets and pagination against brute force over the catalogue (pandas)."""
import numpy as np
import pandas as pd
import pytest

from engine.analysis import analyze
from engine.api import Index

ALL = {"match_all": {}}


def ids_of(r):
    return [h["_id"] for h in r["hits"]["hits"]]


def test_term_terms_range(index, catalogue):
    c = catalogue
    cases = [
        ({"term": {"category": "shoes"}}, c["category"] == "shoes"),
        ({"terms": {"product_brand.keyword": ["Nike", "Levi's", "Nope"]}}, c["product_brand"].isin(["Nike", "Levi's"])),
        ({"range": {"price": {"gte": 20, "lt": 60}}}, (c["price"] >= 20) & (c["price"] < 60)),
        ({"range": {"price": {"gt": 20}}}, c["price"] > 20),
        ({"range": {"stock": {"lte": 1}}}, c["stock"] <= 1),
        ({"term": {"stock": 3}}, c["stock"] == 3),
        ({"range": {"product_color.keyword": {"gte": "Blue", "lte": "Red"}}},
         c["product_color"].notna() & (c["product_color"] >= "Blue") & (c["product_color"] <= "Red")),
    ]
    for q, want in cases:
        r = index.search({"query": {"bool": {"filter": [q]}}, "size": 200, "sort": ["_doc"]})
        assert ids_of(r) == c.loc[want.fillna(False), "product_id"].tolist(), q
        assert all(h["_score"] == 0.0 for h in r["hits"]["hits"])


def test_bool_logic(index, catalogue):
    c = catalogue
    has = lambda t: c["product_title"].map(lambda s: t in analyze(s))
    req = {"query": {"bool": {"must": [{"match": {"product_title": "shoes"}}],
                              "should": [{"term": {"category": "shoes"}}],
                              "must_not": [{"term": {"product_brand.keyword": "Nike"}}],
                              "filter": [{"range": {"price": {"gte": 5}}}]}}, "size": 200, "sort": ["_doc"]}
    want = has("shoe") & (c["product_brand"] != "Nike") & (c["price"] >= 5).fillna(False)
    assert ids_of(index.search(req)) == c.loc[want, "product_id"].tolist()
    req = {"query": {"bool": {"should": [{"match": {"product_title": "wool"}}, {"match": {"product_title": "jeans"}}],
                              "minimum_should_match": 2}}, "size": 200, "sort": ["_doc"]}
    assert ids_of(index.search(req)) == c.loc[has("wool") & has("jean"), "product_id"].tolist()


def test_facets_full_match_set(index, catalogue):
    c = catalogue
    req = {"query": {"match": {"product_title": "black"}}, "size": 1,
           "aggs": {"b": {"terms": {"field": "product_brand.keyword", "size": 2}},
                    "c": {"terms": {"field": "category"}}}}
    r = index.search(req)
    m = c["product_title"].map(lambda s: "black" in analyze(s))
    for name, col, size in (("b", "product_brand", 2), ("c", "category", 10)):
        counts = c.loc[m, col].dropna().value_counts()
        brute = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        got = [(x["key"], x["doc_count"]) for x in r["aggregations"][name]["buckets"]]
        assert got == brute[:size]
        assert r["aggregations"][name]["sum_other_doc_count"] == sum(v for _, v in brute[size:])


def all_pages(index, req, size):
    out, start = [], 0
    while True:
        r = index.search({**req, "from": start, "size": size})
        if not r["hits"]["hits"]:
            return out
        out += r["hits"]["hits"]
        start += size


def after_pages(index, req, size):
    out, after = [], None
    while True:
        r = index.search({**req, "size": size, **({"search_after": after} if after else {})})
        if not r["hits"]["hits"]:
            return out
        out += r["hits"]["hits"]
        after = r["hits"]["hits"][-1]["sort"]


@pytest.mark.parametrize("req", [
    {"query": {"match": {"product_title": "shoes socks jeans"}}},
    {"query": ALL, "sort": [{"price": {"order": "desc"}}, "_score"]},
    {"query": {"match": {"product_title": "black"}}, "sort": [{"product_brand.keyword": "asc"}, {"stock": "desc"}]},
    {"query": {"term": {"category": "shoes"}}},                         # all scores tie: doc id order
])
def test_pagination_consistent(index, req):
    whole = index.search({**req, "size": 200})["hits"]["hits"]
    assert len(whole) == index.search({**req, "size": 0})["hits"]["total"]
    for size in (1, 7):
        assert [h["_id"] for h in all_pages(index, req, size)] == [h["_id"] for h in whole]
        assert [h["_id"] for h in after_pages(index, req, size)] == [h["_id"] for h in whole]


def test_sort_semantics(index, catalogue):
    r = index.search({"query": ALL, "sort": [{"price": "asc"}], "size": 200})
    prices = [h["sort"][0] for h in r["hits"]["hits"]]
    n_missing = int(catalogue["price"].isna().sum())
    assert prices[-n_missing:] == [None] * n_missing                 # missing last
    assert prices[:-n_missing] == sorted(prices[:-n_missing])
    r = index.search({"query": {"term": {"category": "socks"}}, "size": 200})
    docs = [h["sort"][-1] for h in r["hits"]["hits"]]
    assert docs == sorted(docs)                                       # ties by doc id ascending
    r = index.search({"query": ALL, "sort": [{"product_brand.keyword": {"order": "desc", "missing": "_first"}}],
                      "size": 200})
    keys = [h["sort"][0] for h in r["hits"]["hits"]]
    nb = int(catalogue["product_brand"].isna().sum())
    assert keys[:nb] == [None] * nb and keys[nb:] == sorted(keys[nb:], reverse=True)


def test_patch_and_delete(segment):
    import copy
    ix = Index(copy.deepcopy(segment), "cpu")
    req = {"query": {"bool": {"filter": [{"range": {"price": {"gte": 1000}}}]}}}
    assert ix.search(req)["hits"]["total"] == 0
    ix.patch("price", ["P0003", "P0007"], [1500.0, None])
    r = ix.search(req)
    assert ids_of(r) == ["P0003"]
    assert ix.search({"query": ALL, "sort": [{"price": "desc"}], "size": 1})["hits"]["hits"][0]["_id"] == "P0003"
    ix.delete(["P0003"])
    assert ix.search(req)["hits"]["total"] == 0
    n = ix.search({"query": ALL, "size": 0})["hits"]["total"]
    assert n == len(segment.ids) - 1
    from engine.errors import NotHot
    with pytest.raises(NotHot):
        ix.patch("category", ["P0001"], ["x"])
