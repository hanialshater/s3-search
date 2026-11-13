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


def test_a_rescored_request_in_a_batch(segment, catalogue, device):
    ix = ranked_index(segment, catalogue, device)
    lex = {"match": {"product_title": "black shoes"}}
    reqs = [{"query": lex}, {"query": lex, "rescore": {"window_size": 20, "s3_ranker": {"model": "m"}}},
            {"query": lex, "size": 3}]
    for r, got in zip(reqs, ix.search_batch(reqs)):
        assert _same(got, ix.search(r)), r


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
