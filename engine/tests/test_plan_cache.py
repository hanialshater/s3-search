"""S6-19: the per-state plan cache, and phase A's stacked rows reused by the page step; results unchanged."""
import pytest

api = pytest.importorskip("engine.api")
from engine.build import build_segment  # noqa: E402
from referee import checks, loader  # noqa: E402


def _strip(r):
    return {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


def test_cached_plans_give_the_same_responses_and_a_write_drops_them(ds, monkeypatch):
    ix = api.Index(build_segment(ds.catalogue.iloc[:150], ds.mapping, version=1))
    reqs = checks.lexical_requests(ds)[:20] + [{"query": {"match": {"product_title": "lamp"}}, "size": 5}] * 3
    first = ix.search_batch(reqs)
    cache = ix._st.__dict__["_plan_cache"][1]
    assert 0 < len(cache) <= len(reqs)                     # equal requests share one entry
    assert [_strip(r) for r in ix.search_batch(reqs)] == [_strip(r) for r in first]
    monkeypatch.setattr(api, "PLAN_CACHE_SIZE", 0)         # off: plan every time
    assert [_strip(r) for r in ix.search_batch(reqs)] == [_strip(r) for r in first]
    monkeypatch.setattr(api, "PLAN_CACHE_SIZE", 4096)
    ix.add_segment(build_segment(ds.catalogue.iloc[150:], ds.mapping, version=2), version=2)
    assert "_plan_cache" not in ix._st.__dict__ or ix._st.__dict__["_plan_cache"][0] is not None
    after = ix.search_batch(reqs)
    assert ix._st.__dict__["_plan_cache"][0] is ix._st     # a new state, a new cache
    alone = api.Index(build_segment(ds.catalogue, ds.mapping, version=2))
    monkeypatch.setattr(api, "PLAN_CACHE_SIZE", 0)
    assert [_strip(r)["hits"]["total"] for r in after] == [_strip(r)["hits"]["total"] for r in alone.search_batch(reqs)]


def test_a_refused_request_is_not_cached(ds):
    ix = api.build_index(ds.catalogue, ds.mapping)
    bad = {"query": {"no_such_query": {}}}
    for _ in range(2):
        with pytest.raises(api.UnsupportedQuery):
            ix.search(bad)
    assert all("no_such_query" not in k for k in ix._st.__dict__.get("_plan_cache", (None, {}))[1])


def test_phase_a_rows_are_reused_by_the_page_step(ds, monkeypatch):
    """A batch of bare lexical queries stacks its rows once, in phase A, not again for the pages."""
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")    # today's path (S6-22's fused path stacks no rows at all)
    ix = api.build_index(ds.catalogue, ds.mapping)
    b = ix._st.b
    reqs = [{"query": {"match": {"product_title": t}}, "size": 10}
            for t in ["lamp", "mug", "boot", "gray lamp", "portable", "compact mug"]]
    ix.search_batch(reqs)
    calls = []
    real = b.stack
    b.stack = lambda arrays, axis=0: (calls.append(len(list(arrays)) if not isinstance(arrays, list) else len(arrays)),
                                      real(arrays, axis))[1]
    try:
        out = ix.search_batch(reqs)
    finally:
        del b.stack
    big = [n for n in calls if n == len(reqs)]
    assert len(big) == 2, calls                            # score and match counts, once each (phase A)
    assert [_strip(r) for r in out] == [_strip(ix.search(r)) for r in reqs]
