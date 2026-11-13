"""S9-5: nested documents (the nested type, the nested query with score_mode, the nested aggregation) against brute
force on toy, on every device, across segments, deltas, merges and a save."""
import copy
import math
import tempfile

import numpy as np
import pandas as pd
import pytest

from engine.api import Index, InvalidMapping, InvalidRequest, UnsupportedQuery, build_index
from engine.build import build_segment, merge_segments
from engine.tests.conftest import toy_catalogue, toy_mapping

SELLERS = ["acme", "bolt", "core", "dash"]
COND = ["new", "used like new", "used", "refurbished new"]


def with_offers(cat, seed=0):
    rng = np.random.default_rng(seed)
    cat = cat.copy()
    offers = []
    for i in range(len(cat)):
        k = int(rng.integers(0, 4))
        offers.append(None if k == 0 else [{"seller": SELLERS[int(rng.integers(4))],
                                            "price": round(float(rng.uniform(1, 90)), 2),
                                            "condition": COND[int(rng.integers(4))]} for _ in range(k)])
    cat["offers"] = offers
    return cat


def nmapping():
    m = toy_mapping()
    m["_meta"]["engine"].pop("embeddings")
    m["properties"].pop("embedding")
    m["properties"]["offers"] = {"type": "nested", "properties": {
        "seller": {"type": "keyword"}, "price": {"type": "float"}, "condition": {"type": "text"}}}
    return m


@pytest.fixture(scope="module")
def ncat():
    return with_offers(toy_catalogue())


@pytest.fixture(scope="module")
def nindex(ncat, device):
    return build_index(ncat, nmapping(), device)


def children(cat):
    rows = []
    for pid, offs in zip(cat["product_id"], cat["offers"]):
        for i, o in enumerate(offs or []):
            rows.append({"pid": pid, "i": i, **o})
    return pd.DataFrame(rows)


def child_scores(cat, q):
    """BM25 of each child for a match on offers.condition, from an index of the children alone (the children's
    statistics are their own, as in OpenSearch's hidden child documents)."""
    ch = children(cat)
    df = pd.DataFrame({"_child_id": [f"{p}\x1f{i}" for p, i in zip(ch["pid"], ch["i"])],
                       "offers.condition": ch["condition"], "offers.price": ch["price"], "offers.seller": ch["seller"]})
    m = {"_meta": {"engine": {"id_field": "_child_id"}},
         "properties": {"offers.condition": {"type": "text"}, "offers.price": {"type": "float"},
                        "offers.seller": {"type": "keyword"}}}
    r = build_index(df, m).search({"query": q, "size": 10_000})
    return {h["_id"]: h["_score"] for h in r["hits"]["hits"]}


@pytest.mark.parametrize("mode", ["avg", "sum", "max", "min", "none"])
@pytest.mark.parametrize("text", ["new", "used", "like refurbished"])
def test_nested_query_score_modes(nindex, ncat, mode, text):
    inner = {"match": {"offers.condition": text}}
    cs = child_scores(ncat, inner)
    want = {}
    for cid, s in cs.items():
        want.setdefault(cid.split("\x1f")[0], []).append(s)
    agg = {"avg": lambda v: sum(v) / len(v), "sum": sum, "max": max, "min": min, "none": lambda v: 0.0}[mode]
    r = nindex.search({"query": {"nested": {"path": "offers", "query": inner, "score_mode": mode}}, "size": 500})
    got = {h["_id"]: h["_score"] for h in r["hits"]["hits"]}
    assert set(got) == set(want) and r["hits"]["total"] == len(want)
    for pid, v in want.items():
        assert math.isclose(got[pid], agg(v), rel_tol=1e-5, abs_tol=1e-6)


def test_nested_filters_are_per_child(nindex, ncat):
    """Both conditions on the same child (the point of nested): seller bolt and price under 30."""
    q = {"nested": {"path": "offers", "query": {"bool": {"filter": [
        {"term": {"offers.seller": "bolt"}}, {"range": {"offers.price": {"lt": 30}}}]}}}}
    want = {pid for pid, offs in zip(ncat["product_id"], ncat["offers"])
            if any(o["seller"] == "bolt" and np.float32(o["price"]) < np.float32(30) for o in (offs or []))}
    r = nindex.search({"query": {"bool": {"filter": [q]}}, "size": 500})
    assert {h["_id"] for h in r["hits"]["hits"]} == want
    outside = nindex.search({"query": {"term": {"offers.seller": "bolt"}}})       # no nested: no docs
    assert outside["hits"]["total"] == 0


def test_nested_aggregation(nindex, ncat):
    q = {"match": {"product_title": "shoes black"}}
    r = nindex.search({"query": q, "size": 500, "aggs": {"o": {"nested": {"path": "offers"}, "aggs": {
        "s": {"terms": {"field": "offers.seller"}, "aggs": {"p": {"avg": {"field": "offers.price"}}}},
        "mx": {"max": {"field": "offers.price"}}}}}})
    hit = {h["_id"] for h in r["hits"]["hits"]}
    ch = children(ncat)
    ch = ch[ch["pid"].isin(hit)]
    o = r["aggregations"]["o"]
    assert o["doc_count"] == len(ch)
    assert math.isclose(o["mx"]["value"], float(ch["price"].astype(np.float32).max()), rel_tol=1e-9)
    want = sorted(ch["seller"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
    assert [(x["key"], x["doc_count"]) for x in o["s"]["buckets"]] == want
    for x in o["s"]["buckets"]:
        v = ch[ch["seller"] == x["key"]]["price"].astype(np.float32).astype(np.float64)
        assert math.isclose(x["p"]["value"], float(v.sum()) / len(v), rel_tol=1e-9)


def test_refusals(nindex, ncat):
    for bad, err in [({"query": {"nested": {"path": "product_title", "query": {"match_all": {}}}}}, InvalidRequest),
                     ({"query": {"nested": {"path": "offers", "query": {"match_all": {}}, "score_mode": "median"}}},
                      InvalidRequest),
                     ({"aggs": {"x": {"terms": {"field": "offers.seller"}}}}, UnsupportedQuery),
                     ({"aggs": {"x": {"terms": {"field": "category"}, "aggs": {"y": {"nested": {"path": "offers"}}}}}},
                      UnsupportedQuery)]:
        with pytest.raises(err):
            nindex.search(bad)
    m = nmapping()
    m["properties"]["offers"]["properties"]["deep"] = {"type": "nested", "properties": {"x": {"type": "keyword"}}}
    with pytest.raises(InvalidMapping):
        build_index(ncat, m)


REQS = [
    {"query": {"nested": {"path": "offers", "query": {"match": {"offers.condition": "new"}}, "score_mode": "max"}},
     "size": 200},
    {"query": {"nested": {"path": "offers", "query": {"range": {"offers.price": {"gte": 20, "lte": 60}}}}}, "size": 200},
    {"query": {"match_all": {}}, "size": 0, "aggs": {"o": {"nested": {"path": "offers"}, "aggs": {
        "s": {"terms": {"field": "offers.seller"}}, "p": {"sum": {"field": "offers.price"}}}}}},
]


def same(a, b):
    assert a["hits"]["total"] == b["hits"]["total"]
    assert [h["_id"] for h in a["hits"]["hits"]] == [h["_id"] for h in b["hits"]["hits"]]
    assert np.allclose([h["_score"] for h in a["hits"]["hits"]], [h["_score"] for h in b["hits"]["hits"]], atol=1e-4)
    assert a.get("aggregations") == b.get("aggregations") or \
        json_close(a.get("aggregations"), b.get("aggregations"))


def json_close(x, y):
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(json_close(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(json_close(a, b) for a, b in zip(x, y))
    if isinstance(x, float):
        return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9)
    return x == y


def test_nested_across_segments_merges_and_saves(ncat, device, tmp_path):
    cat = ncat.assign(_first_offset=range(len(ncat)))
    m = nmapping()
    base = cat.iloc[:90]
    rng = np.random.default_rng(3)
    edits = cat.iloc[[2, 5, 9, 40]].copy()
    edits["offers"] = [[{"seller": "zed", "price": 5.0, "condition": "new new"}] * 5, None,
                       [{"seller": "acme", "price": 77.0, "condition": "used"}], []]
    new = cat.iloc[90:]
    delta = pd.concat([edits, new]).sort_values("_first_offset")
    dels = ["P0011", "P0012"]
    ix = Index(build_segment(base, m), device)
    ix.add_segment(build_segment(delta, m, version=2), version=2, delete_ids=dels + list(delta["product_id"]))
    final = cat.set_index("product_id")
    final.loc[edits["product_id"], "offers"] = pd.Series(list(edits["offers"]), index=edits["product_id"])
    final = final.drop(index=dels).reset_index()
    rebuilt = Index(build_segment(final, m), device)
    for r in REQS:
        same(ix.search(r), rebuilt.search(r))
    segs, v = ix.snapshot_segments()
    ix.replace_segments([s["segment_id"] for s in ix.segments], merge_segments(segs, version=v))
    assert len(ix.segments) == 1
    for r in REQS:
        same(ix.search(r), rebuilt.search(r))
    back = Index.from_segment(ix.save(tmp_path / "s"), device)
    for r in REQS:
        same(back.search(r), rebuilt.search(r))
    ix.delete(["P0002"])
    final2 = final[final["product_id"] != "P0002"]
    for r in REQS:
        same(ix.search(r), Index(build_segment(final2, m), device).search(r))


def test_commit_reload_after_nested_edits_and_deletes(tmp_path, device):
    """§4: a commit's live bitmaps cover parents; children's live bits are re-derived from their parents on load,
    so a commit taken after an edit (the old parent copy tombstoned with its children) reloads equal to the index."""
    m = {"_meta": {"engine": {"id_field": "product_id"}},
         "properties": {"product_id": {"type": "keyword"}, "title": {"type": "text"},
                        "offers": {"type": "nested", "properties": {"seller": {"type": "keyword"},
                                                                    "price": {"type": "float"}}}}}
    base = pd.DataFrame({"product_id": ["a", "b", "c"], "title": ["red lamp", "blue lamp", "green lamp"],
                         "offers": [[{"seller": "s1", "price": 3.0}, {"seller": "s2", "price": 5.0}],
                                    [{"seller": "s3", "price": 7.0}], [{"seller": "s1", "price": 9.0}]],
                         "_first_offset": [1, 2, 3]})
    ix = Index(build_segment(base, m, version=2), device)
    delta = build_segment(pd.DataFrame({"product_id": ["a", "d"], "title": ["red lamp", "white lamp"],
                                        "offers": [[{"seller": "s9", "price": 4.0}], [{"seller": "s2", "price": 1.0}]],
                                        "_first_offset": [1, 4]}), m, version=3)
    ix.add_segment(delta, version=3)                    # edits a (its old children die), adds d
    ix.delete(["c"])
    ix.set_version(4)
    reqs = [{"size": 0, "aggs": {"o": {"nested": {"path": "offers"},
                                       "aggs": {"s": {"terms": {"field": "offers.seller"}},
                                                "p": {"sum": {"field": "offers.price"}}}}}},
            {"query": {"nested": {"path": "offers", "query": {"term": {"offers.seller": "s1"}}}}},
            {"query": {"nested": {"path": "offers", "score_mode": "max",
                                  "query": {"range": {"offers.price": {"gte": 0}}}}}, "size": 10}]
    want = [ix.search(r) for r in reqs]
    assert want[0]["aggregations"]["o"]["doc_count"] == 3 and want[1]["hits"]["total"] == 0
    back = Index.from_commit(ix.save_commit(tmp_path), device)
    for r, w in zip(reqs, want):
        g = back.search(r)
        assert g["hits"] == w["hits"] and g.get("aggregations") == w.get("aggregations")
