"""Serving's Ports.refresh hook (dataplane.updater.serving_refresh) through a real ManagedIndex and the default ports:
a refresh makes stored upserts visible as a delta segment, and the result equals a rebuild."""
from __future__ import annotations

from pathlib import Path

from dataplane.freshness import compare_responses
from dataplane.tests.conftest import MAPPING, make_catalogue


def test_refresh_hook_adds_a_delta(tmp_path):
    import copy
    from engine.api import Index
    from dataplane.indexer import build
    from serving.config import ServingConfig
    from serving.manager import ManagedIndex
    from serving.metrics import standard
    from serving.ports import default_ports
    m = copy.deepcopy(MAPPING)
    cat = make_catalogue(120)
    cat.to_parquet(tmp_path / "c.parquet", index=False)
    cfg = ServingConfig(data_dir=str(tmp_path / "srv"), refresh_interval="-1")
    mi = ManagedIndex("shop", tmp_path / "srv" / "shop", m, cfg, default_ports(), standard())
    try:
        mi.create({"path": str(tmp_path / "c.parquet")})
        before = mi.index
        out = mi.add_docs([{"op": "upsert", "_id": "NEW", "doc": {"product_id": "NEW", "product_title": "acme red lamp deluxe", "price": 4.0}},
                           {"op": "delete", "_id": "P00007"},
                           {"op": "patch", "_id": "P00009", "fields": {"price": 1.25}}])
        assert not out["rejected"] and out["applied"]["pending_upserts"] == 1
        res = mi.refresh()
        assert res["refreshed"] and res["added"] == 1 and res["segments"] == 2 and mi.pending_upserts == 0
        assert mi.index is before                                          # in place, no swap
        req = {"query": {"match": {"product_title": "deluxe"}}}
        assert [h["_id"] for h in mi.search(req)[0]["hits"]["hits"]] == ["NEW"]
        store = mi._get_store()
        fresh = Index.from_segment(build(store, None, tmp_path / "rebuild"))
        for q in ({"query": {"match": {"product_title": "lamp"}}, "size": 50},
                  {"query": {"match_all": {}}, "sort": [{"price": {"order": "asc"}}], "size": 200}):
            assert compare_responses(mi.index.search(q), fresh.search(q)) is None
        assert mi.lag() == 0
    finally:
        mi.close() if hasattr(mi, "close") else None
