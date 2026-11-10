from __future__ import annotations

import json

import pytest

from dataplane.indexer import MergePolicy, build, gc, latest_segment, list_segments, read_manifest
from dataplane.indexer.__main__ import main as indexer_main
from dataplane.tests.conftest import load, segment_ids


def test_build_writes_a_segment_at_head(tmp_path, store, catalogue):
    load(store, catalogue)
    seg = build(store, None, tmp_path / "segs")
    man = read_manifest(seg)
    assert man["version"] == store.head() == len(catalogue)
    assert seg.name == man["segment_id"] and seg.name.startswith(f"seg-{store.head()}-")
    assert not [p for p in (tmp_path / "segs").iterdir() if p.name.startswith(".")]   # no temp left behind
    from engine.api import Index
    idx = Index.from_segment(seg)
    assert segment_ids(seg) == list(catalogue.product_id) and idx.version == store.head()


def test_rebuild_drops_deletes_and_keeps_order(tmp_path, store, catalogue):
    load(store, catalogue.head(10))
    store.apply([{"op": "delete", "_id": "P00003"},
                 {"op": "upsert", "_id": "P00001", "doc": {"product_id": "P00001", "price": 2.0}},
                 {"op": "upsert", "_id": "NEW", "doc": {"product_id": "NEW", "price": 1.0}}])
    seg = build(store, None, tmp_path / "segs")
    ids = segment_ids(seg)
    assert ids == [f"P{i:05d}" for i in range(10) if i != 3] + ["NEW"]   # P00001 keeps its place


def test_device_budget_refuses_before_anyone_loads(tmp_path, store, catalogue):
    from engine.backend import OverBudget
    load(store, catalogue)
    with pytest.raises(OverBudget):
        build(store, None, tmp_path / "segs", device_budget=10)
    assert list_segments(tmp_path / "segs") == []


def test_gc_and_latest(tmp_path, store, catalogue):
    load(store, catalogue.head(3))
    segs = []
    for k in range(4):
        store.apply([{"op": "patch", "_id": "P00000", "fields": {"stock": k}}])
        segs.append(build(store, None, tmp_path / "segs"))
    assert latest_segment(tmp_path / "segs") == segs[-1]
    removed = gc(tmp_path / "segs", keep=2, protect=(segs[0],))
    assert removed == [segs[1]] and [p for p, _ in list_segments(tmp_path / "segs")] == [segs[0]] + segs[2:]


def test_merge_policy():
    p = MergePolicy(max_deleted_ratio=0.1, max_pending=10, max_age_s=60)
    assert not p.due({"n_docs": 100, "deleted": 5, "pending_upserts": 0, "loaded_at": 1e18})
    assert p.due({"n_docs": 100, "deleted": 11, "pending_upserts": 0})
    assert p.due({"n_docs": 100, "deleted": 0, "pending_upserts": 10})
    assert p.due({"n_docs": 100, "deleted": 1, "pending_upserts": 0, "loaded_at": 0})


def test_cli_ingests_and_builds(tmp_path, catalogue, mapping, capsys):
    (tmp_path / "m.json").write_text(json.dumps(mapping))
    catalogue.to_parquet(tmp_path / "b.parquet")
    assert indexer_main(["--store", str(tmp_path / "s.sqlite"), "--mapping", str(tmp_path / "m.json"),
                         "--out", str(tmp_path / "segs"), "--ingest", str(tmp_path / "b.parquet")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == len(catalogue) and out["intake"]["accepted"] == len(catalogue)


def test_segment_carries_the_store_encoder_for_neural(tmp_path, store, catalogue):
    from engine.api import Index, build_index
    from dataplane.freshness import compare_responses
    load(store, catalogue)
    assert store.meta("encoder.embedding") == "tfidf-svd-16" and store.blob("encoder.embedding")
    seg = build(store, None, tmp_path / "segs")
    vf = read_manifest(seg)["vector_fields"]["embedding"]
    assert vf["model"] == "tfidf-svd-16" and vf["encoder"] is True
    req = {"query": {"neural": {"embedding": {"query_text": "red shoe acme", "k": 20}}}, "size": 20}
    got = Index.from_segment(seg).search(req)
    assert got["hits"]["hits"] and compare_responses(got, build_index(catalogue, store.mapping.raw).search(req)) is None
