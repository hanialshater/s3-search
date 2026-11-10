"""The segment format, the DSL's named errors, cross-device agreement and the import rule."""
import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from engine.api import Index, InvalidMapping, InvalidRequest, UnsupportedQuery, available_devices, build_index
from engine.segment import CorruptSegment, read_manifest, read_segment, write_segment

REQS = [
    {"query": {"multi_match": {"query": "black running shoes", "fields": ["product_title^3", "product_brand^2"]}},
     "aggs": {"b": {"terms": {"field": "product_brand.keyword"}}}},
    {"query": {"bool": {"must": [{"match": {"product_description": "wool socks"}}],
                        "filter": [{"range": {"price": {"lte": 50}}}]}}, "sort": [{"price": "desc"}], "size": 20},
    {"query": {"neural": {"embedding": {"query_text": "blue cotton shirt", "k": 15}}}, "size": 15},
    {"query": {"hybrid": {"queries": [{"match": {"product_title": "black jeans"}},
                                      {"neural": {"embedding": {"query_text": "black jeans", "k": 30}}}]}},
     "rank": {"rrf": {"rank_constant": 60, "window_size": 50}}, "size": 25},
]


def strip(r):
    r = dict(r)
    r.pop("took_ms")
    return r


def test_round_trip(segment, tmp_path):
    p = write_segment(copy.deepcopy(segment), tmp_path / "seg")
    m = read_manifest(p)
    assert m["format"] == "s3seg" and m["n_docs"] == len(segment.ids) and m["device_nbytes"] > 0
    assert m["files"]["text/tf.npy"]["dtype"] == "uint8"
    a = Index(segment, "cpu")
    b = Index.from_segment(p, "cpu")
    for r in REQS:
        assert strip(a.search(r)) == strip(b.search(r))


def test_corruption_detected(segment, tmp_path):
    p = write_segment(copy.deepcopy(segment), tmp_path / "seg")
    tf = np.load(p / "text/tf.npy")
    tf[0, 0] ^= 1
    np.save(p / "text/tf.npy", tf)
    with pytest.raises(CorruptSegment):
        read_segment(p)
    read_segment(p, verify=False)


def test_save_keeps_patches_and_deletes(segment, tmp_path):
    ix = Index(copy.deepcopy(segment), "cpu")
    ix.patch("stock", ["P0001"], [99])
    ix.delete(["P0002"])
    ix.set_version(7)
    back = Index.from_segment(ix.save(tmp_path / "s2"))
    assert back.version == 7
    r = back.search({"query": {"term": {"stock": 99}}})
    assert [h["_id"] for h in r["hits"]["hits"]] == ["P0001"]
    assert back.search({"query": {"match_all": {}}, "size": 0})["hits"]["total"] == len(segment.ids) - 1


@pytest.mark.parametrize("req,path", [
    ({"query": {"regexp": {"product_title": "sho.*"}}}, "query.regexp"),
    ({"query": {"bool": {"must": [{"wildcard": {"product_title": "sh*"}}]}}}, "query.bool.must[0].wildcard"),
    ({"query": {"match_all": {}}, "suggest": {}}, "suggest"),
    ({"aggs": {"x": {"geo_bounds": {"field": "price"}}}}, "aggs.x.geo_bounds"),
    ({"query": {"multi_match": {"query": "x", "type": "phrase"}}}, "query.multi_match.type"),
    ({"query": {"match": {"product_title": {"query": "x", "lenient": True}}}}, "query.match.product_title.lenient"),
])
def test_unsupported_named(index, req, path):
    with pytest.raises(UnsupportedQuery) as e:
        index.search(req)
    assert e.value.path == path


@pytest.mark.parametrize("req", [
    {"from": 9995, "size": 10},
    {"size": -1},
    {"query": {"match": {"nope": "x"}}},
    {"query": {"term": {"category": "shoes"}}, "search_after": [1]},
    {"query": {"knn": {"embedding": {"vector": [0.1, 0.2], "k": 3}}}},
])
def test_invalid_requests(index, req):
    with pytest.raises(InvalidRequest):
        index.search(req)


def test_invalid_mapping(catalogue):
    with pytest.raises(InvalidMapping):
        build_index(catalogue, {"properties": {"x": {"type": "geo_point"}}})
    with pytest.raises(InvalidMapping):
        build_index(catalogue, {"_meta": {"engine": {"hot": ["category"]}},
                                "properties": {"category": {"type": "keyword"}}})


def test_response_shape(index):
    r = index.search({**REQS[0], "explain": True})
    assert set(r) == {"hits", "aggregations", "took_ms", "explain"}
    assert isinstance(r["hits"]["total"], int) and "total" in r["took_ms"]
    h = r["hits"]["hits"][0]
    assert set(h) == {"_id", "_score", "sort"} and isinstance(h["_score"], float)
    assert r["explain"][0]["_id"] == h["_id"] and abs(r["explain"][0]["value"] - h["_score"]) < 1e-5
    json.dumps(r)


def test_batch_equals_single(index):
    batch = index.search_batch(REQS)
    assert [strip(x) for x in batch] == [strip(index.search(r)) for r in REQS]


def test_deterministic_and_devices_agree(segment):
    ref = [strip(Index(segment, "cpu").search(r)) for r in REQS]
    for dev in available_devices():
        ix = Index(segment, dev)
        for _ in range(2):
            for r, want in zip(REQS, ref):
                got = strip(ix.search(r))
                assert [h["_id"] for h in got["hits"]["hits"]] == [h["_id"] for h in want["hits"]["hits"]], dev
                assert np.allclose([h["_score"] for h in got["hits"]["hits"]],
                                   [h["_score"] for h in want["hits"]["hits"]], rtol=1e-5)
                assert got["hits"]["total"] == want["hits"]["total"]
                assert got.get("aggregations") == want.get("aggregations")


def test_engine_imports_nothing_outside():
    root = Path(__file__).resolve().parents[1]
    bad = re.compile(r"^\s*(from|import)\s+(dataplane|serving|tools|referee)\b", re.M)
    hits = [p.name for p in root.rglob("*.py") if "tests" not in p.parts and bad.search(p.read_text())]
    assert hits == []
    numpy_or_torch = re.compile(r"^\s*(import|from)\s+(numpy|torch|cupy)\b", re.M)
    assert [p.name for p in (root / "ops").glob("*.py") if numpy_or_torch.search(p.read_text())] == []
