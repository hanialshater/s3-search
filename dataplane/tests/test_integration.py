"""The data plane against the real engine (engine.build, engine.segment, engine.api). Skipped until ``engine.api``
imports; then these are the checks that S2's two predictions rest on, at toy size."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from dataplane import freshness
from dataplane.indexer import build
from dataplane.updater import Replica
from dataplane.tests.conftest import load

pytestmark = pytest.mark.skipif(importlib.util.find_spec("engine.api") is None,
                                reason="waits on engine.api (S1-6), engine.build and engine.segment (S1-1)")

BIG = 1 << 40


def queries():
    v = np.zeros(16, dtype=np.float32)
    v[3] = 1.0
    return [
        {"query": {"match": {"product_title": "red shoe"}}, "size": 20},
        {"query": {"multi_match": {"query": "blue jacket acme", "fields": ["product_title^2", "product_brand"]}},
         "size": 20, "aggs": {"cats": {"terms": {"field": "category"}}}},
        {"query": {"bool": {"must": [{"match": {"product_title": "lamp"}}],
                            "filter": [{"range": {"price": {"gte": 50}}}]}}, "sort": [{"price": {"order": "asc"}}]},
        {"query": {"bool": {"filter": [{"range": {"stock": {"gt": 10}}}]}}, "sort": [{"stock": {"order": "desc"}}],
         "size": 50},
        {"query": {"knn": {"embedding": {"vector": v.tolist(), "k": 10}}}},
    ]


@pytest.fixture
def real(tmp_path, store, catalogue):
    load(store, catalogue)
    seg = build(store, None, tmp_path / "segs")
    return store, seg


def test_a_segment_from_the_store_equals_build_index(real):
    from engine.api import Index, build_index
    store, seg = real
    from_store = Index.from_segment(seg)
    direct = build_index(store.snapshot(), store.mapping.raw)
    assert from_store.version == store.head() and from_store.n_docs == len(store.snapshot())
    for i, q in enumerate(queries()):
        assert freshness.compare_responses(from_store.search(q), direct.search(q)) is None, i


def test_patches_and_deletes_equal_a_rebuild(real, tmp_path):
    store, seg = real
    r = Replica(store, seg, budget=BIG)
    rep = freshness.run(store, r, n_changes=500, batch=50, interval=0.002, queries=queries()[2:4])
    assert rep.equal_to_rebuild, rep.mismatches
    store.apply([{"op": "delete", "_id": "P00004"}])
    r.poll()
    hits = r.search({"query": {"match_all": {}}, "size": 1000})["hits"]
    assert "P00004" not in {h["_id"] for h in hits["hits"]} and hits["total"] == len(store.snapshot())


def test_swap_onto_a_rebuild(real, tmp_path):
    store, seg = real
    r = Replica(store, seg, budget=BIG)
    store.apply([{"op": "upsert", "_id": "NEW", "doc": {"product_id": "NEW", "product_title": "red shoe deluxe",
                                                         "price": 5.0, "stock": 2}}])
    r.poll()
    r.swap(build(store, None, tmp_path / "segs"))
    ids = {h["_id"] for h in r.search({"query": {"match": {"product_title": "deluxe"}}})["hits"]["hits"]}
    assert ids == {"NEW"}
