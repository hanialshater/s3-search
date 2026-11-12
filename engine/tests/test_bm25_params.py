"""BM25F parameters from the mapping (`_meta.engine.bm25`): k1, and b as one number or a per-field map.

The prototype's tuned BM25F (`baseline/esci_gpu_vs_opensearch.ipynb`, 0.304 nDCG@10 on ESCI-US) has its own b per
field and k1 = 0.036; the engine read a single b, so that result could not be expressed. Scores are checked against
the referee's independent reference scorer on every device.
"""
import copy

import pytest

api = pytest.importorskip("engine.api")
from engine.errors import InvalidMapping  # noqa: E402
from referee import checks, loader, reference  # noqa: E402

TUNED = {"k1": 0.036, "b": {"product_title": 0.742, "product_brand": 0.675, "product_color": 0.475,
                            "product_bullet_point": 0.619, "product_description": 0.551}}


def _with_bm25(mapping, bm25):
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["bm25"] = bm25
    return m


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.mark.parametrize("device", api.available_devices())
@pytest.mark.parametrize("which", ["per-field b", "per-field b, b_default", "one b"])
def test_scores_match_the_reference(device, which, ds):
    bm25 = TUNED
    if which == "per-field b, b_default":
        bm25 = {"k1": 0.5, "b": {"product_title": 0.2}, "b_default": 0.9}
    elif which == "one b":
        bm25 = {"k1": 0.8, "b": 0.4}
    m = _with_bm25(ds.mapping, bm25)
    idx = api.build_index(ds.catalogue, m, device=device)
    ref = reference.build_index(ds.catalogue, m)
    for r in checks.lexical_requests(ds)[:40]:
        checks.assert_same_ranking(idx.search(r), ref.search(r), f"{which} on {device}: {r['query']}")


def test_per_field_b_changes_scores(ds):
    """Not a no-op: the default b and a per-field b give different scores somewhere."""
    a = api.build_index(ds.catalogue, ds.mapping)
    b = api.build_index(ds.catalogue, _with_bm25(ds.mapping, TUNED))
    differ = 0
    for r in checks.lexical_requests(ds)[:40]:
        sa = {h["_id"]: h["_score"] for h in a.search(r)["hits"]["hits"]}
        sb = {h["_id"]: h["_score"] for h in b.search(r)["hits"]["hits"]}
        differ += any(abs(sa[d] - sb.get(d, 0.0)) > 1e-6 for d in sa)
    assert differ


@pytest.mark.parametrize("bm25,path", [
    ({"k1": -1}, "_meta.engine.bm25.k1"),
    ({"k1": "x"}, "_meta.engine.bm25.k1"),
    ({"b": 1.5}, "_meta.engine.bm25.b"),
    ({"b": {"no_such_field": 0.5}}, "_meta.engine.bm25.b.no_such_field"),
    ({"b": {"product_title": -0.1}}, "_meta.engine.bm25.b.product_title"),
    ({"b": 0.5, "b_default": 0.5}, "_meta.engine.bm25.b_default"),
    ("not an object", "_meta.engine.bm25"),
])
def test_bad_parameters_are_refused(bm25, path, ds):
    m = _with_bm25(ds.mapping, bm25)
    with pytest.raises(InvalidMapping) as e:
        api.build_index(ds.catalogue, m)
    assert path in str(e.value)
