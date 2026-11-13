"""Index.patch with the same product twice in one call: the last value wins on every device.

Regression for the cuda freshness failure: a scatter with duplicate row indices is not ordered on the GPU, so
4 of 44 changed prices stayed stale on an L4 while the CPU was exact.
"""
import numpy as np
import pytest

api = pytest.importorskip("engine.api")
from referee import loader  # noqa: E402


@pytest.mark.parametrize("device", api.available_devices())
def test_duplicate_patch_last_write_wins(device):
    ds = loader.load_dataset("toy")
    eng = ds.mapping["_meta"]["engine"]
    id_field, field = eng["id_field"], eng["hot"][0]
    pid = str(ds.catalogue[id_field].iloc[0])
    idx = api.build_index(ds.catalogue, ds.mapping, device=device)
    idx.patch(field, [pid] * 3, [1.0, 2.0, 3.0])
    r = idx.search({"query": {"bool": {"filter": [{"terms": {id_field: [pid]}}]}}, "size": 1,
                    "sort": [{field: "asc"}]})
    assert r["hits"]["hits"][0]["sort"][0] == pytest.approx(3.0, rel=1e-6)


class _NoDuplicateScatter(np.ndarray):
    """An array that refuses an integer-array assignment with repeated indices: the write cuda does not order."""

    def __setitem__(self, key, value):
        if isinstance(key, np.ndarray) and key.dtype.kind in "iu":
            assert len(np.unique(key)) == len(key), f"scatter with duplicate indices {sorted(key.tolist())}"
        super().__setitem__(key, value)


def test_patch_never_scatters_duplicate_indices(monkeypatch):
    """Deterministic on any machine (the cuda test above depends on the GPU's write order): patch's scatters into
    the hot columns, the index's and each part's, see each row once."""
    from engine.backend.numpy_backend import NumpyBackend
    monkeypatch.setattr(NumpyBackend, "copy", lambda self, a: np.array(a).view(_NoDuplicateScatter))
    ds = loader.load_dataset("toy")
    eng = ds.mapping["_meta"]["engine"]
    ids = ds.catalogue[eng["id_field"]].astype(str).tolist()
    idx = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    for field in eng["hot"]:
        idx.patch(field, [ids[0], ids[5], ids[0], ids[7], ids[5], ids[0]], [1, 2, 3, 4, 5, 6])
    r = idx.search({"query": {"ids": {"values": [ids[0], ids[5], ids[7]]}}, "size": 3,
                    "sort": [{eng["hot"][0]: "asc"}]})
    assert [h["sort"][0] for h in r["hits"]["hits"]] == pytest.approx([4.0, 5.0, 6.0])
