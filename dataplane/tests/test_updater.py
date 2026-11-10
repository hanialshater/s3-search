"""The updater on the real engine (S8): deltas, tombstones, merges, commits and swaps, each checked against a rebuild
from the store (dataplane.indexer.build) with the same requests."""
from __future__ import annotations

import time

import pytest

from dataplane.freshness import compare_responses
from dataplane.indexer import TieredMergePolicy, build
from dataplane.intake import apply_changes
from dataplane.updater import OverBudget, Replica
from dataplane.tests.conftest import load, value

BIG = 1 << 40
REQS = [
    {"query": {"match": {"product_title": "lamp"}}, "size": 50},
    {"query": {"multi_match": {"query": "red shoe acme", "fields": ["product_title^2", "product_brand"]}},
     "size": 50, "aggs": {"c": {"terms": {"field": "category"}}}},
    {"query": {"bool": {"filter": [{"range": {"price": {"gte": 100}}}]}}, "sort": [{"price": {"order": "asc"}}],
     "size": 100},
    {"query": {"neural": {"embedding": {"query_text": "blue jacket", "k": 30}}}, "size": 30},
]


def price(replica, _id):
    return value(replica.index, _id, "price")


def assert_equals_rebuild(replica, store, tmp_path, name="rebuild"):
    from engine.api import Index
    fresh = Index.from_segment(build(store, None, tmp_path / name))
    assert replica.index.n_docs == fresh.n_docs == store.count()
    for i, q in enumerate(REQS):
        assert compare_responses(replica.search(q), fresh.search(q)) is None, (i, q)
        assert [h["sort"][-1] for h in replica.search(q)["hits"]["hits"]] == \
               [h["sort"][-1] for h in fresh.search(q)["hits"]["hits"]]          # the tie key is first_offset too


def write(store, changes):
    rep = apply_changes(changes, store.mapping, store)
    assert not rep.rejected, rep.rejected
    return rep.last_offset


def edits(catalogue, k: int, n_new: int = 8) -> list[dict]:
    """New products, one full-text edit, one delete and one patch: the S8 stream's mix."""
    lo = 200 + k * n_new
    out = [{"op": "upsert", "_id": f"N{k:03d}-{i}", "doc": {**r, "product_id": f"N{k:03d}-{i}"}}
           for i, r in enumerate(catalogue.iloc[lo % 200:lo % 200 + n_new].to_dict("records"))]
    out += [{"op": "upsert", "_id": f"P{k:05d}", "doc": {**catalogue.iloc[k].to_dict(), "product_title": f"edited lamp {k}"}},
            {"op": "delete", "_id": f"P{100 + k:05d}"},
            {"op": "patch", "_id": f"P{150 + k:05d}", "fields": {"price": 101.5 + k}}]
    return out


@pytest.fixture
def setup(tmp_path, store, catalogue):
    load(store, catalogue)
    seg = build(store, None, tmp_path / "segs")
    return store, seg, tmp_path / "segs"


def test_refresh_makes_every_change_visible_and_equals_a_rebuild(setup, tmp_path, catalogue, monkeypatch):
    store, seg, _ = setup
    r = Replica(store, seg, budget=BIG)
    calls = []
    real_patch = r.index.patch
    monkeypatch.setattr(r.index, "patch", lambda f, ids, vals: (calls.append(f), real_patch(f, ids, vals)))
    write(store, [{"op": "patch", "_id": "P00001", "fields": {"price": 3.5}},
                  {"op": "patch", "_id": "P00001", "fields": {"price": 4.5, "stock": 0}},   # the last value wins
                  {"op": "patch", "_id": "P00002", "fields": {"price": 9.0}},
                  {"op": "delete", "_id": "P00002"},
                  {"op": "upsert", "_id": "NEW", "doc": {"product_title": "acme red lamp", "price": 1.0}},
                  {"op": "patch", "_id": "NEW", "fields": {"price": 2.0}},
                  {"op": "upsert", "_id": "P00003", "doc": {**catalogue.iloc[3].to_dict(), "product_title": "acme lamp"}}])
    rep = r.refresh()
    assert rep.version == store.head() == r.version and (rep.added, rep.deleted, rep.pending_upserts) == (2, 1, 0)
    assert price(r, "P00001") == 4.5 and sorted(calls) == ["price", "stock"]   # one Index.patch per field
    assert price(r, "NEW") == 2.0 and price(r, "P00002") is None
    assert r.stats()["segments"] == 2 and "NEW" in r.live_ids() and "P00002" not in r.live_ids()
    hits = r.search({"query": {"match": {"product_title": "lamp"}}, "size": 100})["hits"]["hits"]
    assert {"NEW", "P00003"} <= {h["_id"] for h in hits}
    assert r.refresh().changes == 0
    assert_equals_rebuild(r, store, tmp_path)


def test_a_new_replica_catches_up(setup, catalogue):
    store, seg, _ = setup
    write(store, [{"op": "patch", "_id": "P00005", "fields": {"price": 0.25}}] + edits(catalogue, 0))
    r = Replica(store, seg, budget=BIG)
    assert r.version == store.head() and price(r, "P00005") == 0.25 and r.stats()["segments"] == 2


def test_merges_equal_a_rebuild_with_changes_during_the_merge(setup, tmp_path, catalogue, monkeypatch):
    store, seg, _ = setup
    r = Replica(store, seg, budget=BIG, merge_policy=TieredMergePolicy(floor=10, factor=10, merge_factor=4))
    for k in range(8):
        write(store, edits(catalogue, k))
        r.refresh()
    assert r.stats()["segments"] == 9
    import engine.build as eb
    real = eb.merge_segments

    def racing(segs, *a, **kw):               # a refresh lands while the merge runs (between snapshot and replace)
        out = real(segs, *a, **kw)
        write(store, edits(catalogue, 20) + [{"op": "patch", "_id": f"N001-0", "fields": {"price": 777.0}},
                                              {"op": "delete", "_id": "N002-1"},
                                              {"op": "upsert", "_id": "N003-2", "doc": {"product_title": "acme lamp x",
                                                                                         "price": 5.0}}])
        r.refresh()
        return out

    monkeypatch.setattr(eb, "merge_segments", racing)
    rep = r.merge()
    assert rep is not None and len(rep.inputs) == 4 and rep.replayed > 0
    assert price(r, "N001-0") == 777.0 and price(r, "N002-1") is None and price(r, "N003-2") == 5.0
    monkeypatch.setattr(eb, "merge_segments", real)
    while r.merge() is not None:
        pass
    assert r.stats()["segments"] <= 5
    assert_equals_rebuild(r, store, tmp_path)


def test_commit_reload_replays_patches_the_commit_lacks(setup, tmp_path, catalogue):
    store, seg, out = setup
    r = Replica(store, seg, budget=BIG, commit_dir=out)
    write(store, [{"op": "patch", "_id": "P00010", "fields": {"price": 111.0}}] + edits(catalogue, 1))
    r.refresh()
    write(store, [{"op": "patch", "_id": "N001-0", "fields": {"price": 222.0}},     # a patch to a delta's doc
                  {"op": "patch", "_id": "P00010", "fields": {"price": 112.0}}])
    r.refresh()
    c = r.commit()
    assert c.name == f"commit-{store.head()}.json"
    write(store, edits(catalogue, 2))                                               # after the commit
    r2 = Replica(store, out, budget=BIG)
    assert r2.segment == c and r2.version == store.head()
    assert price(r2, "P00010") == 112.0 and price(r2, "N001-0") == 222.0
    r.refresh()
    assert_equals_rebuild(r2, store, tmp_path)
    for q in REQS:
        assert compare_responses(r.search(q), r2.search(q)) is None


def test_swap_onto_a_rebuild_and_back_to_a_commit(setup, tmp_path, catalogue):
    store, seg, out = setup
    r = Replica(store, seg, budget=BIG, commit_dir=out)
    write(store, edits(catalogue, 3))
    r.refresh()
    commit = r.commit()
    old = r.index
    write(store, edits(catalogue, 4))
    rep = r.swap(build(store, None, out))
    assert rep.new_version == store.head() and r.stats()["segments"] == 1
    assert old.search({"query": {"match_all": {}}, "size": 1})["hits"]["total"] > 0    # the old one still answers
    write(store, [{"op": "patch", "_id": "P00020", "fields": {"price": 3.0}}])
    r.swap(commit)                                                                   # rollback, then the log after it
    assert r.segment == commit and r.version == store.head() and price(r, "P00020") == 3.0
    assert_equals_rebuild(r, store, tmp_path)


def test_budget_is_checked_for_load_and_double_buffer(setup):
    store, seg, _ = setup
    one = Replica(store, seg, budget=BIG).stats()["device_nbytes"] + 32 * 200 * 8
    with pytest.raises(OverBudget):
        Replica(store, seg, budget=one - 1)
    r = Replica(store, seg, budget=one)
    with pytest.raises(OverBudget, match="double buffer"):
        r.swap(seg)
    assert r.segment == seg


def test_budget_env(setup, monkeypatch):
    store, seg, _ = setup
    monkeypatch.setenv("S3_SEARCH_DEVICE_BUDGET", "100")
    with pytest.raises(OverBudget):
        Replica(store, seg)


def test_background_refresh_merge_and_commit(setup, tmp_path, catalogue):
    store, seg, out = setup
    pol = TieredMergePolicy(floor=10, factor=10, merge_factor=4)
    with Replica(store, seg, budget=BIG, merge_policy=pol, commit_dir=out, commit_every=0.05).start(0.01) as r:
        for k in range(10):
            last = write(store, edits(catalogue, k))
            deadline = time.time() + 2
            while r.version < last and time.time() < deadline:
                time.sleep(0.002)
            assert r.version == last
        deadline = time.time() + 3
        while (r.stats()["segments"] > 4 or not r.merges) and time.time() < deadline:
            time.sleep(0.01)
    assert r.last_error is None and r.merges and r.stats()["segments"] <= 4
    assert_equals_rebuild(r, store, tmp_path)
