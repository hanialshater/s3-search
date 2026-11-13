"""S6-7: ``hits="columns"`` expands to exactly the dict form (INTERFACES §5, "Columnar hits").

Over every construct the referee marks landed, the lexical and sort requests of the contract checks, deep pages,
``search_after``, missing sort values and a rescore (the tier column), on every device.
"""
import numpy as np
import pytest

api = pytest.importorskip("engine.api")
from referee import checks, freshness, loader, scope  # noqa: E402


def _requests(ds):
    out = [req for _, _, _, _, req in scope.landed()]
    out += checks.lexical_requests(ds)[:30] + checks._sort_requests() + freshness.probe_requests(ds.mapping)
    out += [{"query": {"match": {"product_title": "lamp"}}, "size": 2000},
            {"query": {"match_all": {}}, "sort": [{"category": "desc"}, {"price": "asc"}], "size": 50, "from": 7},
            {"query": {"match_all": {}}, "sort": [{"stock": {"order": "asc", "missing": "_first"}}], "size": 30},
            {"query": {"match": {"product_title": "lamp"}}, "size": 0}]
    return out


def _without_took(r):
    return {k: v for k, v in r.items() if k != "took_ms"}


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.mark.parametrize("device", api.available_devices())
def test_columns_expand_to_the_dict_form(device, ds):
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    reqs = _requests(ds)
    first = ix.search(reqs[0])
    if first["hits"]["hits"]:                                 # search_after from a real sort value
        reqs.append({**reqs[0], "search_after": first["hits"]["hits"][0]["sort"]})
    n_columns = 0
    for r, d, c in zip(reqs, ix.search_batch(reqs), ix.search_batch(reqs, hits="columns")):
        if "columns" in c["hits"]:
            n_columns += 1
            assert "hits" not in c["hits"]
            col = c["hits"]["columns"]
            assert isinstance(col["_score"], np.ndarray) and col["_score"].dtype == np.float32
            assert len(col["_id"]) == len(col["_score"]) == len(d["hits"]["hits"])
        else:                                                  # collapse, highlight, explain: the dict form
            assert "hits" in c["hits"]
        assert _without_took(api.expand_hits(c)) == _without_took(d), r
    assert n_columns >= len(reqs) // 2


def test_a_rescored_page_has_the_tier_column(segment, catalogue, device):
    from engine.tests.test_gpu_paths import ranked_index
    ix = ranked_index(segment, catalogue, device)
    req = {"query": {"hybrid": {"queries": [{"match": {"product_title": "black shoes"}},
                                            {"neural": {"embedding": {"query_text": "black shoes", "k": 20}}}]}},
           "rescore": {"window_size": 25, "s3_ranker": {"model": "m"}}, "size": 30}
    c = ix.search(req, hits="columns")
    tier = c["hits"]["columns"]["sort"][0]["values"]
    assert tier.dtype == np.int64 and set(tier.tolist()) <= {0, 1}
    assert _without_took(api.expand_hits(c)) == _without_took(ix.search(req))


def test_dict_form_passes_through_and_bad_values_are_refused(ds):
    ix = api.build_index(ds.catalogue, ds.mapping)
    r = ix.search({"query": {"match_all": {}}})
    assert api.expand_hits(r) is r
    with pytest.raises(ValueError):
        ix.search({"query": {"match_all": {}}}, hits="arrays")


@pytest.mark.parametrize("device", api.available_devices())
def test_ids_as_bytes_expand_to_the_same_dicts(device, ds):
    """S6-12: ids="bytes" gives _id as one S<w> array; expanded, the response equals the dict form."""
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    reqs = _requests(ds)
    for d, c in zip(ix.search_batch(reqs), ix.search_batch(reqs, hits="columns", ids="bytes")):
        if "columns" in c["hits"]:
            col = c["hits"]["columns"]["_id"]
            assert isinstance(col, np.ndarray) and col.dtype.kind == "S"
        assert _without_took(api.expand_hits(c)) == _without_took(d)
    with pytest.raises(ValueError):
        ix.search(reqs[0], ids="bytes")                        # only with hits="columns"


def test_ids_bytes_follow_writes_and_unicode(ds):
    """The bytes copy is extended by add_segment, carried over by clones, and holds UTF-8 of any id."""
    from engine.build import build_segment
    cat = ds.catalogue.copy()
    id_field = ds.mapping["_meta"]["engine"]["id_field"]
    cat[id_field] = [f"é-{i}" if i % 7 == 0 else str(v) for i, v in enumerate(cat[id_field])]
    ix = api.Index(build_segment(cat.iloc[:150], ds.mapping, version=1))
    q = {"query": {"match_all": {}}, "size": 300}
    before = ix.search(q, hits="columns", ids="bytes")
    assert _without_took(api.expand_hits(before)) == _without_took(ix.search(q))
    delta = cat.iloc[150:].copy()
    delta[id_field] = [f"longer-unicode-id-ü-{i}" for i in range(len(delta))]    # wider than any id before
    ix.add_segment(build_segment(delta, ds.mapping, version=2), version=2)
    ix.patch(ds.mapping["_meta"]["engine"]["hot"][0], [cat[id_field].iloc[0]], [1.0])   # a clone of the state
    after = ix.search(q, hits="columns", ids="bytes")
    assert _without_took(api.expand_hits(after)) == _without_took(ix.search(q))
    assert ix._st.ids_bytes().tolist() == [str(i).encode("utf-8") for i in ix._st.ids]
