"""The GPU-friendliness rules (docs/INTERFACES.md §0): what comes to the host per search, rescoring on the device,
numeric codes kept on the device, and the float32 accumulation option."""
import copy

import numpy as np
import pandas as pd
import pytest

from engine.api import Index
from engine.rank.gbdt import GBDT

LEX = {"multi_match": {"query": "black running shoes", "fields": ["product_title^3", "product_brand"]}}


def counting(ix):
    b = ix._st.b
    calls = []
    real = b.to_numpy

    def wrapped(a):
        calls.append(1)
        return real(a)
    b.to_numpy = wrapped
    return calls, lambda: b.__dict__.pop("to_numpy", None)


def ranked_index(segment, catalogue, device):
    ix = Index(copy.deepcopy(segment), device)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(ix.feature_names())))
    X[rng.random(X.shape) < 0.1] = np.nan
    ix.add_ranker("m", GBDT(n_rounds=15, max_depth=3, min_leaf=5).fit(X, np.nan_to_num(X[:, 0])))
    return ix


def test_transfers_per_search_are_bounded(segment, catalogue, device):
    """At most: the page (one transfer), each terms aggregation's buckets (one), and nothing for rescoring; explain
    is allowed more."""
    ix = ranked_index(segment, catalogue, device)
    hyb = {"hybrid": {"queries": [{"match": {"product_title": "black shoes"}},
                                  {"neural": {"embedding": {"query_text": "black shoes", "k": 20}}}]}}
    cases = [
        ({"query": LEX}, 1),
        ({"query": LEX, "sort": [{"price": "desc"}, {"stock": "asc"}]}, 1),
        ({"query": {"match_all": {}}, "sort": [{"product_brand.keyword": "asc"}], "search_after": ["Acme", 3]}, 1),
        ({"query": LEX, "aggs": {"b": {"terms": {"field": "product_brand.keyword"}},
                                 "s": {"terms": {"field": "stock"}}}}, 3),
        ({"query": {"neural": {"embedding": {"query_text": "blue shirt", "k": 10}}}}, 1),
        ({"query": hyb, "rescore": {"window_size": 20, "s3_ranker": {"model": "m"}}}, 1),
    ]
    for req, bound in cases:
        ix.search(req)                                    # warm the per-segment caches (numeric codes)
        calls, undo = counting(ix)
        try:
            ix.search(req)
        finally:
            undo()
        assert len(calls) <= bound, (req, len(calls))


def test_gbdt_on_the_device_equals_numpy(device):
    from engine.backend import get_backend
    b = get_backend(device)
    rng = np.random.default_rng(4)
    X = rng.normal(size=(700, 6)).astype(np.float32)
    X[rng.random(X.shape) < 0.15] = np.nan
    y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 3])
    m = GBDT(n_rounds=40, max_depth=4, min_leaf=5).fit(X[:500], y[:500])
    got = b.to_numpy(m.predict_backend(b, b.asarray(X, "float32")))
    want = m.predict(X)
    assert not np.isnan(want).any() and np.array_equal(got, want)


def test_rescore_on_device_matches_the_host_path(segment, catalogue, device):
    ix = ranked_index(segment, catalogue, device)
    req = {"query": {"hybrid": {"queries": [{"match": {"product_title": "black shoes"}},
                                            {"neural": {"embedding": {"query_text": "black shoes", "k": 20}}}]}},
           "rescore": {"window_size": 25, "s3_ranker": {"model": "m"}}, "size": 30}
    on_device = ix.search(req)["hits"]["hits"]

    class HostOnly:                                       # the same model without predict_backend
        def __init__(self, m):
            self.predict = m.predict
    ix.add_ranker("m", HostOnly(ix.rankers["m"]))
    assert ix.search(req)["hits"]["hits"] == on_device


def test_numeric_codes_cached_and_refreshed_on_patch(segment):
    ix = Index(copy.deepcopy(segment), "cpu")
    agg = {"size": 0, "aggs": {"s": {"terms": {"field": "stock", "size": 20}}}}
    ix.search(agg)
    first = ix._st._value_codes["stock"]
    ix.search(agg)
    assert ix._st._value_codes["stock"] is first          # computed once per segment
    ix.patch("stock", ["P0001", "P0002"], [123, 123])
    assert "stock" not in ix._st._value_codes
    keys = {x["key"]: x["doc_count"] for x in ix.search(agg)["aggregations"]["s"]["buckets"]}
    assert keys[123] == 2


@pytest.mark.parametrize("accum", ["float32"])
def test_float32_accumulation(segment, device, accum):
    ref = Index(segment, device)
    ix = Index(segment, device, accumulate=accum)
    lex = [{"query": {"match": {"product_title": q}}, "size": 20} for q in ("black shoes", "wool socks", "nike men")]
    lex += [{"query": LEX, "size": 20}]
    # BM25F: deterministic and batch-invariant in float32 too
    one = [ix.search(r)["hits"] for r in lex]
    assert [x["hits"] for x in ix.search_batch(lex)] == one
    for r in lex:
        a, b = ref.search(r)["hits"]["hits"], ix.search(r)["hits"]["hits"]
        assert np.allclose([h["_score"] for h in a], [h["_score"] for h in b], rtol=1e-5)
    # dense: close to the float64 reference (batch invariance is not promised for the float32 GEMM)
    kn = [{"query": {"neural": {"embedding": {"query_text": q, "k": 10}}}} for q in ("blue shirt", "black jeans")]
    for r, got in zip(kn, ix.search_batch(kn)):
        want = ref.search(r)["hits"]["hits"]
        assert np.allclose([h["_score"] for h in got["hits"]["hits"]], [h["_score"] for h in want], rtol=1e-5)
