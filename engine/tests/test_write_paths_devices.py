"""Every write path gives the cpu path's results on every other device (torch-cpu, cuda).

The same sequence of writes goes to a cpu index and to one on each other device: hot-field patches with duplicate ids
in a batch, deletes (with repeats and already-deleted ids), and an added segment that edits existing products and adds
new ones. Then score-sorted requests must rank alike, and field-sorted and aggregation requests must match exactly.

Why: the GPU write path once kept a stale value when a product changed twice in one batch (a scatter with duplicate
indices is not ordered on cuda); the cpu-only CI could not see it. Run on a GPU with `colab/gpu_ci.py`.
"""
import numpy as np
import pandas as pd
import pytest

api = pytest.importorskip("engine.api")
from referee import checks, freshness, loader  # noqa: E402

OTHER_DEVICES = [d for d in api.available_devices() if d != "cpu"]


def _writes(ds, rng):
    """A fixed sequence of (method, args) writes on toy."""
    eng = ds.mapping["_meta"]["engine"]
    id_field = eng["id_field"]
    ids = ds.catalogue[id_field].astype(str).to_numpy()
    out = []
    changes = freshness.make_changes(ds.catalogue, ds.mapping, 600, seed=int(rng.integers(1 << 30)))
    for i in range(0, len(changes), 100):
        chunk = changes[i:i + 100]
        chunk = chunk + chunk[:10]                               # the same products again, in the same batch
        by_field: dict = {}
        for c in chunk:
            for f, v in c["fields"].items():
                by_field.setdefault(f, ([], []))
                by_field[f][0].append(c["_id"])
                by_field[f][1].append(v)
        out += [("patch", (f, fids, vals)) for f, (fids, vals) in by_field.items()]
    dead = list(rng.choice(ids, 12, replace=False))
    out.append(("delete", (dead + dead[:3],)))                   # repeats in one call
    out.append(("delete", (dead[:5],)))                          # already deleted
    # A delta segment: 20 existing products with new hot values, and 5 new products.
    edit = ds.catalogue.sample(20, random_state=int(rng.integers(1 << 30))).copy()
    for f in eng["hot"]:
        t = ds.mapping["properties"][f]["type"]
        edit[f] = rng.integers(0, 100, len(edit)) if t in ("integer", "long") else rng.uniform(1, 500, len(edit))
    new = ds.catalogue.head(5).copy()
    new[id_field] = [f"new-{k}" for k in range(len(new))]
    delta = pd.concat([edit, new], ignore_index=True)
    out.append(("add_segment", (api.build_segment(delta, ds.mapping), 2)))
    return out


def _apply(index, writes):
    for method, args in writes:
        getattr(index, method)(*args)


def _field_sorted_same(got, exp, what):
    assert got["hits"]["total"] == exp["hits"]["total"], f"{what}: total"
    g = [(h["_id"], h.get("sort")) for h in got["hits"]["hits"]]
    e = [(h["_id"], h.get("sort")) for h in exp["hits"]["hits"]]
    assert [i for i, _ in g] == [i for i, _ in e], f"{what}: ids"
    for (i, a), (_, b) in zip(g, e):
        assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-6, equal_nan=True), f"{what}: {i} sort"
    assert got.get("aggregations") == exp.get("aggregations"), f"{what}: aggregations"


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.mark.skipif(not OTHER_DEVICES, reason="only the cpu device here")
@pytest.mark.parametrize("device", OTHER_DEVICES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_write_paths_agree_with_cpu(device, seed, ds):
    writes = _writes(ds, np.random.default_rng(seed))
    base = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    other = api.build_index(ds.catalogue, ds.mapping, device=device)
    _apply(base, writes)
    _apply(other, writes)
    for r in checks.lexical_requests(ds)[:40]:
        checks.assert_same_ranking(other.search(r), base.search(r), f"{device} vs cpu, seed {seed}: {r['query']}",
                                   tol=checks.DEVICE_RTOL)
    for r in freshness.probe_requests(ds.mapping):
        _field_sorted_same(other.search(r), base.search(r), f"{device} vs cpu, seed {seed}: {r}")
