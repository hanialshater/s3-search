"""Freshness on toy (B1-4): 10,000 hot-field changes, time to visible, results equal to a rebuild.

Two paths: `Index.patch` directly (S1, INTERFACES section 4), and the data plane's store and updater (S2), which is
skipped until `dataplane.kv` and `dataplane.updater` exist. Owner: evaluation-steward (P6).
"""
import importlib.util

import pytest

api = pytest.importorskip("engine.api")

from referee import freshness, loader  # noqa: E402


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("device", api.available_devices())
def test_patch_freshness(device):
    ds = loader.load_dataset("toy")
    idx = api.build_index(ds.catalogue, ds.mapping, device=device)
    if not hasattr(idx, "patch"):
        pytest.skip("Index.patch not implemented yet")
    rep = freshness.check_freshness(idx, ds.catalogue, ds.mapping, freshness.patch_hook(idx), api.build_index,
                                    n_changes=10_000, device=device)
    assert rep["passed"], rep


@pytest.mark.skipif(not (_has("dataplane.kv") and _has("dataplane.updater")), reason="S2: the data plane is not here yet")
def test_dataplane_freshness(tmp_path):
    from dataplane.kv import Store
    from dataplane.updater import Replica
    from dataplane.indexer import build
    ds = loader.load_dataset("toy")
    store = Store(tmp_path / "store.sqlite", ds.mapping)
    store.apply({"op": "upsert", "_id": r["product_id"], "doc": r} for r in ds.catalogue.to_dict("records"))
    seg = build(store, ds.mapping, tmp_path / "segments")
    rep_ = Replica(store, seg, "cpu")

    def apply(changes):
        store.apply({"op": "patch", "_id": c["_id"], "fields": c["fields"]} for c in changes)
        rep_.poll()
    rep = freshness.check_freshness(rep_.index, ds.catalogue, ds.mapping, apply, api.build_index, n_changes=10_000)
    assert rep["passed"], rep
