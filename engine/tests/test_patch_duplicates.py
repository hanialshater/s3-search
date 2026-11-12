"""Index.patch with the same product twice in one call: the last value wins on every device.

Regression for the cuda freshness failure: a scatter with duplicate row indices is not ordered on the GPU, so
4 of 44 changed prices stayed stale on an L4 while the CPU was exact.
"""
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
