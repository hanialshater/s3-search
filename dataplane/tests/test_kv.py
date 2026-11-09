from __future__ import annotations

import threading

import numpy as np
import pytest

from dataplane.kv import BadChange, NoSuchKey, NotHot, Store
from dataplane.tests.conftest import load


def up(i, **kw):
    d = {"product_id": i, "product_title": f"item {i}", "price": 1.0, "stock": 1}
    d.update(kw)
    return {"op": "upsert", "_id": i, "doc": d}


def test_offsets_are_gapless_and_atomic(store):
    assert store.head() == 0 and store.apply([]) == (0, 0)
    assert store.apply([up("a"), up("b")]) == (1, 2)
    with pytest.raises(NotHot):
        store.apply([up("c"), {"op": "patch", "_id": "a", "fields": {"product_title": "x"}}])
    with pytest.raises(NoSuchKey):
        store.apply([{"op": "patch", "_id": "zz", "fields": {"price": 2.0}}])
    with pytest.raises(BadChange):
        store.apply([{"op": "patch", "_id": "a", "fields": {"price": "cheap"}}])
    with pytest.raises(BadChange):
        store.apply([{"op": "frob", "_id": "a"}])
    assert store.head() == 2 and store.get(["c"]) == [None]          # the refused batches left nothing
    assert store.apply([{"op": "delete", "_id": "nobody"}]) == (3, 3)  # logged, a no-op
    assert [c["offset"] for c in store.changes(0)] == [1, 2, 3]


def test_records_follow_the_interface(store):
    store.apply([up("a", embedding=[1.0] + [0.0] * 15)])
    store.apply([{"op": "patch", "_id": "a", "fields": {"price": 8.49, "stock": 0}}, {"op": "delete", "_id": "a"}])
    u, p, d = store.changes(0)
    assert set(u) == {"offset", "op", "_id", "doc", "ts"} and u["doc"]["embedding"][0] == 1.0
    assert p == {"offset": 2, "op": "patch", "_id": "a", "fields": {"price": 8.49, "stock": 0}, "ts": p["ts"]}
    assert set(d) == {"offset", "op", "_id", "ts"} and d["ts"].endswith("Z")
    assert "doc" not in store.changes(0, docs=False)[0]
    assert [c["offset"] for c in store.changes(1, limit=1)] == [2]


def test_first_offset_rules(store):
    store.apply([up("a"), up("b")])
    store.apply([up("a", price=2.0), {"op": "patch", "_id": "b", "fields": {"stock": 5}}])
    snap = store.snapshot()
    assert list(snap.product_id) == ["a", "b"]
    assert list(snap._first_offset) == [1, 2] and list(snap._last_offset) == [3, 4]
    store.apply([{"op": "delete", "_id": "a"}, up("a")])           # delete then re-insert resets first_offset
    snap = store.snapshot()
    assert list(snap.product_id) == ["b", "a"] and list(snap._first_offset) == [2, 6]


def test_snapshot_at_an_offset_replays_the_log(store, catalogue):
    load(store, catalogue.head(10))
    store.apply([{"op": "patch", "_id": "P00001", "fields": {"price": 0.5}}, {"op": "delete", "_id": "P00002"}])
    mid = store.head()
    store.apply([up("P00002"), {"op": "patch", "_id": "P00001", "fields": {"price": 7.0}}, up("Z")])
    then = store.snapshot(at=mid)
    assert then.attrs["version"] == mid and len(then) == 9
    assert then.set_index("product_id").price["P00001"] == 0.5
    now = store.snapshot()
    assert list(now.product_id)[-2:] == ["P00002", "Z"] and now.set_index("product_id").price["P00001"] == 7.0
    # the replayed snapshot at head equals the table's
    full = store._frame(store._replay(store._conn(), store.head()))
    assert full.drop(columns=["embedding"]).equals(now.drop(columns=["embedding"]))
    with pytest.raises(ValueError):
        store.snapshot(at=store.head() + 1)


def test_snapshot_dtypes(store):
    store.apply([up("a", embedding=[0.0, 1.0] + [0.0] * 14), up("b", stock=None, price=None)])
    snap = store.snapshot()
    assert str(snap.stock.dtype) == "Int32" and snap.price.dtype == np.float64 and np.isnan(snap.price[1])
    assert snap.embedding[0].dtype == np.float32 and snap.embedding[1] is None
    store.apply([{"op": "delete", "_id": "b"}])
    assert store.snapshot().stock.dtype == np.int32


def test_store_reopens_and_refuses_another_mapping(tmp_path, mapping):
    with Store(tmp_path / "s.sqlite", mapping) as s:
        s.apply([up("a")])
    with Store(tmp_path / "s.sqlite", mapping) as s:
        assert s.head() == 1 and s.get(["a"])[0]["price"] == 1.0
    mapping["_meta"]["engine"]["hot"] = ["price"]
    with pytest.raises(ValueError, match="different mapping"):
        Store(tmp_path / "s.sqlite", mapping)


def test_concurrent_writers_keep_offsets_gapless(store):
    def write(t):
        for k in range(20):
            store.apply([up(f"{t}-{k}"), up(f"{t}-{k}b")])

    threads = [threading.Thread(target=write, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    offs = [c["offset"] for c in store.changes(0, docs=False)]
    assert offs == list(range(1, 161)) and store.count() == 160
