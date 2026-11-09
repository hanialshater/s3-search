"""The data plane's eight questions, as the engine answers them (docs/INTERFACES.md, Changes, 2026-11-09)."""
import copy
import threading

import numpy as np
import pandas as pd
import pytest

from engine.api import Index
from engine.build import build_segment
from engine.segment import read_manifest, write_segment


def test_snapshot_frame_contract(catalogue, mapping, tmp_path):
    snap = catalogue.copy()
    snap["stock"] = snap["stock"].astype("Int32")
    snap.loc[3, "stock"] = pd.NA
    snap["_first_offset"] = np.arange(len(snap), dtype=np.int64) * 10 + 5
    snap["_last_offset"] = 0
    snap["unmapped"] = "x"
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"].pop("embeddings")
    snap["embedding"] = [None if i == 2 else np.eye(16, dtype=np.float32)[i % 16] for i in range(len(snap))]
    seg = build_segment(snap, m, vector_models={"embedding": "hash-16"})
    assert seg.first_offset[:2].tolist() == [5, 15]
    assert seg.manifest["vector_fields"]["embedding"]["model"] == "hash-16"
    assert not np.asarray(seg.vectors["embedding"])[2].any()
    seg.manifest["version"] = 42                                   # the indexer's way to set the version
    d = tmp_path / "empty"
    d.mkdir()
    p = write_segment(seg, d)
    man = read_manifest(p)
    assert p == d and man["version"] == 42 and man["segment_id"].startswith("seg-42-")
    with pytest.raises(FileExistsError):
        write_segment(seg, d)
    ix = Index.from_segment(p)
    assert ix.version == 42
    r = ix.search({"query": {"bool": {"filter": [{"range": {"stock": {"gte": 0}}}]}}, "size": 0})
    assert r["hits"]["total"] == len(snap) - 1                     # the NA stock is missing


def test_nullable_boolean_and_int64(catalogue):
    snap = catalogue[["product_id", "product_title"]].copy()
    snap["flag"] = pd.array([True, None, False] * 40, dtype="boolean")
    snap["n"] = pd.array([1, None, 3] * 40, dtype="Int64")
    m = {"_meta": {"engine": {"id_field": "product_id"}},
         "properties": {"product_title": {"type": "text"}, "flag": {"type": "boolean"}, "n": {"type": "long"}}}
    ix = Index(build_segment(snap, m))
    assert ix.search({"query": {"term": {"flag": True}}, "size": 0})["hits"]["total"] == 40
    assert ix.search({"query": {"range": {"n": {"gte": 0}}}, "size": 0})["hits"]["total"] == 80


def test_patch_delete_unknown_and_deleted(segment):
    ix = Index(copy.deepcopy(segment), "cpu")
    with pytest.raises(KeyError):
        ix.patch("stock", ["NOPE"], [1])
    with pytest.raises(KeyError):
        ix.delete(["NOPE"])
    ix.delete(["P0005"])
    ix.delete(["P0005"])                                           # already deleted: no-op
    ix.patch("stock", ["P0005", "P0006"], [77, 77])                # the deleted one is skipped
    r = ix.search({"query": {"term": {"stock": 77}}})
    assert [h["_id"] for h in r["hits"]["hits"]] == ["P0006"]


def test_patch_is_atomic_for_searches(segment):
    ix = Index(copy.deepcopy(segment), "cpu")
    ids = [f"P{i:04d}" for i in range(0, 120, 2)]
    stop, seen = threading.Event(), set()

    def writer():
        v = 0
        while not stop.is_set():
            v = (v + 1) % 3
            ix.patch("stock", ids, [1000 + v] * len(ids))

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(100):
            r = ix.search({"query": {"range": {"stock": {"gte": 1000}}}, "size": 0,
                           "aggs": {"s": {"terms": {"field": "stock"}}}})
            seen.add(tuple((b["key"], b["doc_count"]) for b in r["aggregations"]["s"]["buckets"]))
    finally:
        stop.set()
        t.join()
    assert all(len(s) <= 1 and (not s or s[0][1] == len(ids)) for s in seen), seen


def test_numeric_facets_have_typed_keys(index, catalogue):
    r = index.search({"size": 0, "aggs": {"s": {"terms": {"field": "stock", "size": 10}},
                                          "p": {"terms": {"field": "price", "size": 2}}}})
    want = sorted(catalogue["stock"].value_counts().items(), key=lambda kv: (-kv[1], kv[0]))
    assert [(b["key"], b["doc_count"]) for b in r["aggregations"]["s"]["buckets"]] == [(int(k), v) for k, v in want]
    assert all(type(b["key"]) is int for b in r["aggregations"]["s"]["buckets"])
    assert all(type(b["key"]) is float for b in r["aggregations"]["p"]["buckets"])
