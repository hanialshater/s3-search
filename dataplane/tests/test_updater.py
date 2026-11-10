from __future__ import annotations

import time

import pytest

from dataplane.indexer import build
from dataplane.updater import OverBudget, Replica
from dataplane.tests.conftest import load, value

BIG = 1 << 40


def price(replica, _id):
    return value(replica.index, _id, "price")


@pytest.fixture
def setup(tmp_path, store, catalogue):
    load(store, catalogue)
    seg = build(store, None, tmp_path / "segs")
    return store, seg, tmp_path / "segs"


def test_poll_patches_in_place_and_tombstones(setup, monkeypatch):
    store, seg, _ = setup
    r = Replica(store, seg, budget=BIG)
    calls = []
    real_patch = r.index.patch
    monkeypatch.setattr(r.index, "patch", lambda f, ids, vals: (calls.append(f), real_patch(f, ids, vals)))
    v0 = r.version
    store.apply([{"op": "patch", "_id": "P00001", "fields": {"price": 3.5}},
                 {"op": "patch", "_id": "P00001", "fields": {"price": 4.5, "stock": 0}},   # the last value wins
                 {"op": "patch", "_id": "P00002", "fields": {"price": 9.0}},
                 {"op": "delete", "_id": "P00002"},                                         # never patched
                 {"op": "upsert", "_id": "NEW", "doc": {"product_id": "NEW", "price": 1.0}},
                 {"op": "patch", "_id": "NEW", "fields": {"price": 2.0}}])                  # waits for a build
    rep = r.poll()
    assert rep.version == v0 + 6 == r.version and rep.changes == 6
    assert (rep.patched, rep.deleted, rep.pending_upserts) == (2, 1, 2)
    assert price(r, "P00001") == 4.5 and sorted(calls) == ["price", "stock"]   # one Index.patch per field
    hits = r.search({"query": {"match_all": {}}, "size": 1000})["hits"]
    assert hits["total"] == len(r.live_ids()) and "P00002" not in {h["_id"] for h in hits["hits"]}
    assert r.poll().changes == 0


def test_a_new_replica_catches_up(setup):
    store, seg, _ = setup
    store.apply([{"op": "patch", "_id": "P00005", "fields": {"price": 0.25}}])
    r = Replica(store, seg, budget=BIG)
    assert r.version == store.head() and price(r, "P00005") == 0.25


def test_swap_replays_and_keeps_old_searches_alive(setup):
    store, seg, out = setup
    r = Replica(store, seg, budget=BIG)
    store.apply([{"op": "upsert", "_id": "NEW", "doc": {"product_id": "NEW", "price": 1.0, "stock": 1}}])
    r.poll()
    seg2 = build(store, None, out)
    store.apply([{"op": "patch", "_id": "NEW", "fields": {"price": 6.0}}])   # after the new segment's version
    old = r.index
    rep = r.swap(seg2)
    assert rep.new_version == store.head() and rep.replayed == 1 and r.segment == seg2
    assert price(r, "NEW") == 6.0 and r.stats()["pending_upserts"] == 0
    assert value(old, "NEW", "price") is None and old.search({"size": 1})["hits"]["total"] == 200   # the old one still answers
    # rollback: swap back to the first segment, replaying the log after it
    r.swap(seg)
    assert r.segment == seg and r.version == store.head() and r.stats()["pending_upserts"] == 2


def test_budget_is_checked_for_load_and_double_buffer(setup):
    store, seg, _ = setup
    need = Replica(store, seg, budget=BIG)
    one = need.stats()["device_nbytes"] + 32 * 200 * 8
    with pytest.raises(OverBudget):
        Replica(store, seg, budget=one - 1)
    r = Replica(store, seg, budget=one)
    with pytest.raises(OverBudget, match="double buffer"):
        r.swap(seg)
    assert r.segment == seg                                     # nothing changed


def test_budget_env(setup, monkeypatch):
    store, seg, _ = setup
    monkeypatch.setenv("S3_SEARCH_DEVICE_BUDGET", "100")
    with pytest.raises(OverBudget):
        Replica(store, seg)


def test_background_poller(setup):
    store, seg, _ = setup
    with Replica(store, seg, budget=BIG).start(interval=0.005) as r:
        _, last = store.apply([{"op": "patch", "_id": "P00007", "fields": {"stock": 42}}])
        deadline = time.time() + 2
        while r.version < last and time.time() < deadline:
            time.sleep(0.002)
        assert r.version == last and r.last_error is None
