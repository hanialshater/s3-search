from __future__ import annotations

from dataplane import freshness
from dataplane.indexer import build
from dataplane.updater import Replica
from dataplane.tests.conftest import load


def test_freshness_measures_and_equals_a_rebuild(tmp_path, store, catalogue):
    load(store, catalogue)
    r = Replica(store, build(store, None, tmp_path / "segs"), budget=1 << 40)
    rep = freshness.run(store, r, n_changes=2000, batch=50, interval=0.002, seed=1)
    assert rep.n_changes == 2000 and rep.batches == 40 and rep.version == store.head()
    assert len(rep.latencies_ms) == 2000 and 0 <= rep.p50_ms <= rep.p99_ms <= rep.max_ms < 1000
    assert rep.equal_to_rebuild is True and rep.queries == 6 and rep.mismatches == []
    assert r._thread is None and r._listeners == []


def test_freshness_catches_a_replica_that_misses_changes(tmp_path, store, catalogue):
    load(store, catalogue)
    r = Replica(store, build(store, None, tmp_path / "segs"), budget=1 << 40)
    real = store.delta

    def lossy(after):                                              # a replica that drops some changes
        head, recs, docs = real(after)
        return head, [x for x in recs if x["offset"] % 7], docs

    store.delta = lossy
    rep = freshness.run(store, r, n_changes=300, batch=30, interval=0.002)
    assert rep.equal_to_rebuild is False and rep.mismatches


def test_compare_responses():
    a = {"hits": {"total": 2, "hits": [{"_id": "x", "_score": 1.0, "sort": [3.0, 0]},
                                       {"_id": "y", "_score": 1.0, "sort": [4.0, 1]}]}}
    b = {"hits": {"total": 2, "hits": [{"_id": "x", "_score": 1.0000001, "sort": [3.0, 9]},
                                       {"_id": "y", "_score": 1.0, "sort": [4.0, 10]}]}}
    assert freshness.compare_responses(a, b) is None          # doc ids may differ; scores within rtol
    b["hits"]["hits"][1]["sort"][0] = 4.5
    assert "sort" in freshness.compare_responses(a, b)
    b["hits"]["total"] = 3
    assert "total" in freshness.compare_responses(a, b)
