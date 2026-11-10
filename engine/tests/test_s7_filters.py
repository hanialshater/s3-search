"""S7-3: exists, ids, post_filter and collapse, against brute force on toy."""
import numpy as np
import pytest

from engine.analysis import analyze


def ids_of(r):
    return [h["_id"] for h in r["hits"]["hits"]]


def all_hits(index, req):
    return index.search({**req, "size": 500})


@pytest.mark.parametrize("field,want", [
    ("product_bullet_point", lambda c: c["product_bullet_point"].map(lambda s: len(analyze(s)) > 0)),
    ("product_brand.keyword", lambda c: c["product_brand"].notna()),
    ("price", lambda c: c["price"].notna()),
    ("product_title", lambda c: c["product_title"].map(lambda s: len(analyze(s)) > 0)),
])
def test_exists(index, catalogue, field, want):
    r = all_hits(index, {"query": {"exists": {"field": field}}, "sort": ["_doc"]})
    assert ids_of(r) == catalogue.loc[want(catalogue), "product_id"].tolist()
    assert all(h["_score"] == 1.0 for h in r["hits"]["hits"])
    neg = all_hits(index, {"query": {"bool": {"must_not": [{"exists": {"field": field}}]}}})
    assert neg["hits"]["total"] + r["hits"]["total"] == len(catalogue)


def test_exists_on_vectors(index, catalogue):
    assert all_hits(index, {"query": {"exists": {"field": "embedding"}}})["hits"]["total"] > 0


def test_ids(index):
    r = all_hits(index, {"query": {"ids": {"values": ["P0007", "P0003", "nope", "P0003"]}}, "sort": ["_doc"]})
    assert ids_of(r) == ["P0003", "P0007"]
    r = all_hits(index, {"query": {"bool": {"must": [{"match": {"product_title": "shoes"}}],
                                            "filter": [{"ids": {"values": ["P0001", "P0002", "P0004"]}}]}}})
    assert set(ids_of(r)) <= {"P0001", "P0002", "P0004"}


def test_post_filter_facets_before_hits_after(index, catalogue):
    q = {"match": {"product_description": "black wool shoes"}}
    pf = {"term": {"product_brand.keyword": "Nike"}}
    aggs = {"b": {"terms": {"field": "product_brand.keyword"}}}
    plain = all_hits(index, {"query": q, "aggs": aggs})
    filtered = all_hits(index, {"query": q, "aggs": aggs, "post_filter": pf})
    both = all_hits(index, {"query": {"bool": {"must": [q], "filter": [pf]}}})
    assert filtered["aggregations"] == plain["aggregations"]                 # facets: before the post_filter
    assert ids_of(filtered) == ids_of(both) and filtered["hits"]["total"] == both["hits"]["total"]
    assert [h["_score"] for h in filtered["hits"]["hits"]] == [h["_score"] for h in both["hits"]["hits"]]


def brute_collapse(full_hits, catalogue, field, col):
    """The first hit of each value (missing values form one group) in the full ordering."""
    val = dict(zip(catalogue["product_id"], catalogue[col]))
    seen, out = set(), []
    for h in full_hits:
        v = val[h["_id"]]
        v = None if v is None or (isinstance(v, float) and v != v) else v
        if v not in seen:
            seen.add(v)
            out.append((h["_id"], v))
    return out


@pytest.mark.parametrize("req", [
    {"query": {"match": {"product_title": "shoes black wool"}}},
    {"query": {"match_all": {}}, "sort": [{"price": "asc"}]},
    {"query": {"match_all": {}}, "sort": [{"stock": "desc"}, {"price": "desc"}]},
])
@pytest.mark.parametrize("field,col", [("product_brand.keyword", "product_brand"), ("stock", "stock"),
                                       ("category", "category")])
def test_collapse(index, catalogue, req, field, col):
    full = all_hits(index, req)["hits"]["hits"]
    want = brute_collapse(full, catalogue, field, col)
    r = all_hits(index, {**req, "collapse": {"field": field}})
    assert [(h["_id"], h["fields"][field][0]) for h in r["hits"]["hits"]] == want
    assert r["hits"]["total"] == len(full)                                 # total counts docs, not groups
    for frm, size in ((0, 2), (1, 2), (2, 10)):
        p = index.search({**req, "collapse": {"field": field}, "from": frm, "size": size})
        assert ids_of(p) == [i for i, _ in want[frm:frm + size]]
