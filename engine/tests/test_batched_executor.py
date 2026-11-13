"""S6-6: one count, one top-k and the terms facets per batch, with ``search_batch == [search]`` exactly.

A mixed batch (plain score-sorted pages of several sizes and offsets, field sorts, search_after, post_filter,
size 0, terms facets of two sizes and with sub-aggregations, collapse, highlight, explain, a rescore) on every
device, in both hit forms, with the chunking forced small; and the host transfers of a batch no longer grow with
its size.
"""
import pytest

api = pytest.importorskip("engine.api")
from engine.plan.execute import Executor  # noqa: E402
from engine.tests.test_gpu_paths import counting, ranked_index  # noqa: E402
from referee import checks, loader  # noqa: E402

FACETS = {"brand": {"terms": {"field": "product_brand.keyword", "size": 5}},
          "cat": {"terms": {"field": "category", "size": 3}}}


def _batch(ds):
    lex = checks.lexical_requests(ds)[:12]
    out = []
    for k, r in enumerate(lex):
        out.append({**r, "size": [10, 3, 50, 0][k % 4], "from": [0, 2, 0, 0][k % 4]})
        out.append({**r, "aggs": FACETS if k % 2 else {"brand": FACETS["brand"]}})
    q = lex[0]["query"]
    out += [
        {"query": q, "sort": [{"price": "desc"}], "size": 7},
        {"query": {"match_all": {}}, "sort": [{"category": "asc"}, {"price": "asc"}], "search_after": ["", 0, 0]},
        {"query": q, "post_filter": {"term": {"category": "kitchen"}}, "aggs": {"cat": FACETS["cat"]}},
        {"query": q, "aggs": {"cat": {"terms": {"field": "category"}, "aggs": {"p": {"max": {"field": "price"}}}}}},
        {"query": q, "collapse": {"field": "category"}},
        {"query": q, "highlight": {"fields": {"product_title": {}}}},
        {"query": q, "explain": True, "size": 3},
        {"query": q, "size": 0, "aggs": FACETS},
    ]
    return out


def _same(a, b):
    strip = lambda r: {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}  # noqa: E731
    return strip(a) == strip(b)


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.mark.parametrize("device", api.available_devices())
@pytest.mark.parametrize("hits", ["dicts", "columns"])
def test_search_batch_equals_search_one_at_a_time(device, hits, ds, monkeypatch):
    monkeypatch.setattr(Executor, "BATCH_ROWS", 5)          # several chunks, and chunks that mix shapes
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    reqs = _batch(ds)
    got = ix.search_batch(reqs, hits=hits)
    monkeypatch.setattr(Executor, "BATCHED", False)         # the per-plan path, as before S6-6
    ref = ix.search_batch(reqs, hits=hits)
    for r, g, e in zip(reqs, got, ref):
        assert _same(g, e), r
    monkeypatch.setattr(Executor, "BATCHED", True)
    for r, g in zip(reqs, got):
        assert _same(g, ix.search(r, hits=hits)), r


@pytest.mark.parametrize("runtime", ["engine", "auto"])
def test_rescored_plans_in_a_batch(segment, catalogue, device, runtime, monkeypatch):
    """S6-8: rescored plans of several windows, queries and shapes, batched, equal the per-plan path and each alone."""
    monkeypatch.setattr(Executor, "BATCH_ROWS", 4)
    ix = ranked_index(segment, catalogue, device)
    if runtime == "auto":
        ix.add_ranker("m", ix.rankers["m"] if not hasattr(ix.rankers["m"], "model") else ix.rankers["m"].model,
                      runtime="auto")
    texts = ["black shoes", "red shirt", "running", "zzzz", "blue"]
    reqs = []
    for k, t in enumerate(texts):
        lex = {"match": {"product_title": t}}
        hyb = {"hybrid": {"queries": [lex, {"neural": {"embedding": {"query_text": t, "k": 20}}}]}}
        reqs.append({"query": hyb if k % 2 else lex, "size": [10, 3, 30][k % 3], "from": [0, 4, 0, 2, 0][k],
                     "rescore": {"window_size": [20, 5, 100, 1, 7][k], "s3_ranker": {"model": "m"}}})
        reqs.append({"query": lex})
    reqs.append({"query": {"match": {"product_title": "shoes"}}, "size": 40, "from": 3,     # deep into tier 0
                 "post_filter": {"range": {"price": {"lte": 80}}},
                 "rescore": {"window_size": 4, "s3_ranker": {"model": "m"}}})
    reqs.append({"query": {"match": {"product_title": "shoes"}}, "explain": True, "size": 3,
                 "rescore": {"window_size": 10, "s3_ranker": {"model": "m"}}})
    got = ix.search_batch(reqs)
    monkeypatch.setattr(Executor, "BATCHED", False)
    ref = ix.search_batch(reqs)
    monkeypatch.setattr(Executor, "BATCHED", True)
    for r, g, e in zip(reqs, got, ref):
        assert _same(g, e), r
        assert _same(g, ix.search(r)), r


def test_a_missing_ranker_is_still_named(segment, catalogue):
    ix = ranked_index(segment, catalogue, "cpu")
    with pytest.raises(api.InvalidRequest, match="no ranker named 'x'"):
        ix.search_batch([{"query": {"match_all": {}}, "rescore": {"window_size": 5, "s3_ranker": {"model": "x"}}}])


def test_host_transfers_do_not_grow_with_the_batch(ds):
    ix = api.build_index(ds.catalogue, ds.mapping)
    lex = checks.lexical_requests(ds)
    for n in (4, 32):
        reqs = [{**lex[k % len(lex)], "aggs": FACETS} for k in range(n)]
        ix.search_batch(reqs)                                  # warm caches
        calls, undo = counting(ix)
        try:
            ix.search_batch(reqs)
        finally:
            undo()
        assert len(calls) == 3, (n, len(calls))                # the pages and totals, and one per facet field


def test_features_equal_the_host_count_route(segment, catalogue, device):
    """The match-count feature from the device table has the bits of the host route (math.log1p of the count)."""
    import numpy as np
    from engine import ops
    from engine.plan.execute import Timer
    from engine.plan.plan import plan as make_plan
    from engine.rank.features import _components, candidate_features_batch
    ix = ranked_index(segment, catalogue, device)
    st = ix._st
    b = st.b
    ex = Executor(st, ix.rankers)
    plans = [make_plan({"query": {"match": {"product_title": t}}}, st) for t in ("black shoes", "shoes", "zzzz")]
    ex._prefetch(plans, Timer(b))
    ctxs = []
    for p in plans:
        ctx = {"explain": None}
        ex.eval(p.root, ctx)
        ctxs.append(ctx)
    cand = b.asarray(np.tile(np.arange(20, dtype=np.int32), (3, 1)), "int32")
    X, _ = candidate_features_batch(ex, ctxs, cand)
    for q, ctx in enumerate(ctxs):
        comps, node, nm = _components(ex, ctx)
        n = int(b.to_numpy(b.count_nonzero(nm))) if nm is not None else 0
        mf = st.match_fields()
        want = ops.gather_features(b, b.astype(cand[q:q + 1], "int64"), *[c.reshape(1, -1) for c in comps],
                                   [[st.keywords[f].matched_codes(set(node.terms))] for f in mf],
                                   [st.keywords[f].codes for f in mf], st.static_features(), [len(node.terms)], [n])
        assert (n > 0) == (q < 2)
        np.testing.assert_array_equal(b.to_numpy(X[q:q + 1]), b.to_numpy(want))   # NaN == NaN here


def test_hybrid_and_knn_plans_in_a_batch(segment, catalogue, device, monkeypatch):
    """S6-9: knn top-k masks and hybrid RRF windows and fusion, batched, equal the per-plan path and each alone."""
    monkeypatch.setattr(Executor, "BATCH_ROWS", 3)
    ix = ranked_index(segment, catalogue, device)
    reqs = []
    for k, t in enumerate(["black shoes", "red shirt", "running", "zzzz", "blue jeans", "shoes"]):
        lex = {"match": {"product_title": t}}
        knn = {"neural": {"embedding": {"query_text": t, "k": [20, 5, 50][k % 3]}}}
        rank = {"rrf": {"rank_constant": [60, 10][k % 2], "window_size": [100, 7, 20000][k % 3]}}
        reqs.append({"query": {"hybrid": {"queries": [lex, knn]}}, "rank": rank, "size": 15})
        reqs.append({"query": knn, "size": 10})
        reqs.append({"query": {"hybrid": {"queries": [knn, lex, {"match": {"product_brand": t}}]}}, "size": 5})
    reqs.append({"query": {"neural": {"embedding": {"query_text": "shoes", "k": 10,
                                                    "filter": {"range": {"price": {"lte": 50}}}}}}})
    reqs.append({"query": {"hybrid": {"queries": [{"match": {"product_title": "shoes"}},
                                                  {"neural": {"embedding": {"query_text": "shoes", "k": 10}}}]}},
                 "explain": True, "size": 3})
    reqs.append({"query": {"hybrid": {"queries": [{"match": {"product_title": "shoes"}},
                                                  {"neural": {"embedding": {"query_text": "shoes", "k": 10}}}]}},
                 "rescore": {"window_size": 10, "s3_ranker": {"model": "m"}}})
    got = ix.search_batch(reqs)
    monkeypatch.setattr(Executor, "BATCHED", False)
    ref = ix.search_batch(reqs)
    monkeypatch.setattr(Executor, "BATCHED", True)
    for r, g, e in zip(reqs, got, ref):
        assert _same(g, e), r
        assert _same(g, ix.search(r)), r
