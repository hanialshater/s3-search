"""S6-22: the fused lexical path gives today's responses exactly.

``S3_TRITON_FUSED_LEXICAL=reference`` routes the eligible plans through the fused path's interface with a reference
built from today's operations, on any device. ``=1`` uses the Triton kernel on cuda where the gate allows. Off by
default."""
import numpy as np
import pytest

from engine import api
from engine.backend import available_devices, get_backend
from engine.backend import triton_fused as TF
from referee import checks, loader

AGGS = {"b": {"terms": {"field": "product_brand.keyword", "size": 5}}, "c": {"terms": {"field": "category"}}}


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


def _strip(r):
    return {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}


def _requests(ds):
    lex = checks.lexical_requests(ds)[:24]
    texts = [str(t) for t in ds.test_queries["query"]][:8]
    out = [{**r, "size": 300} for r in lex]                                     # deep, plain
    out += [{**r, "size": 10, "aggs": AGGS} for r in lex[:12]]                  # facets
    out += [{**r, "from": 5, "size": 7} for r in lex[:6]]                       # a later page
    out += [{"query": {"match": {"product_title": {"query": t, "minimum_should_match": "2", "boost": 1.5}}},
             "size": 50} for t in texts]                                        # msm and boost
    out += [{"query": {"multi_match": {"query": t, "fields": ["product_title^2", "product_brand"]}}, "size": 40,
             "aggs": {"b": AGGS["b"]}} for t in texts]
    out += [{"query": {"match": {"product_title": "zzzzunknown"}}, "size": 5}]  # no known term
    out += [{"query": {"bool": {"must": [lex[0]["query"]]}}, "size": 5}]        # not eligible: today's path
    return out


def test_on_by_default_where_the_gate_allows(monkeypatch):
    monkeypatch.delenv("S3_TRITON_FUSED_LEXICAL", raising=False)
    assert TF.mode(get_backend("cpu")) is None                                  # the kernel: cuda only
    if TF.triton is not None and "cuda" in available_devices():
        monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
        assert TF.mode(get_backend("cuda")) == "kernel"                         # S6-26: on by default
        monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")
        assert TF.mode(get_backend("cuda")) is None
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "1")
    assert TF.mode(get_backend("cpu")) is None
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "reference")
    assert TF.mode(get_backend("cpu")) == "reference"


@pytest.mark.parametrize("device", available_devices())
def test_fused_reference_gives_todays_responses(ds, device, monkeypatch):
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    reqs = _requests(ds)
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")                          # the unfused path
    want = [_strip(r) for r in ix.search_batch(reqs)]
    want_cols = [_strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "reference")
    calls = []
    real = TF.run
    monkeypatch.setattr(TF, "run", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    assert [_strip(r) for r in ix.search_batch(reqs)] == want
    assert [_strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")] == want_cols
    assert calls, "the fused path never ran"
    alone = [_strip(ix.search(r)) for r in reqs[:5]]                            # a batch of one: today's path
    assert alone == want[:5]


@pytest.mark.skipif(TF.triton is None or "cuda" not in available_devices(), reason="needs Triton and cuda")
def test_kernel_equals_the_reference_bit_for_bit(ds, monkeypatch):
    import torch
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    st = ix._st
    part, b = st.parts[0], st.b
    F = len(st.text_fields)
    rng = np.random.default_rng(0)
    common = np.argsort(-np.diff(part.host_term_ptr))[:12]
    terms = [sorted(rng.choice(common, int(rng.integers(1, 6)), replace=False).tolist()) for _ in range(40)] + [[]]
    w = np.linspace(0.7, 2.3, F).astype(np.float32)
    bb = np.linspace(0.3, 0.9, F).astype(np.float32)
    msm = [int(rng.integers(1, 3)) for _ in terms]
    boost = [1.0 if i % 3 else 1.5 for i in range(len(terms))]
    live = st.live.clone()
    live[::7] = False
    codes = [st.keywords[f].codes for f in sorted(st.keywords)[:2]]
    sizes = [len(st.keywords[f].values) for f in sorted(st.keywords)[:2]]
    want = TF.reference(b, part, terms, w, bb, 1.2, part.idf, msm, boost, live, codes, sizes)
    got = TF._run_kernel(b, part, terms, w, bb, 1.2, part.idf, msm, boost, live, codes, sizes)
    assert torch.equal(got[0].view(torch.int32), want[0].view(torch.int32))   # every score's bits, -inf included
    assert torch.equal(got[1], want[1])
    for g, wv in zip(got[2], want[2]):
        assert torch.equal(g, wv)
    again = TF._run_kernel(b, part, terms, w, bb, 1.2, part.idf, msm, boost, live, codes, sizes)
    assert torch.equal(again[0].view(torch.int32), got[0].view(torch.int32))  # run to run


@pytest.mark.skipif(TF.triton is None or "cuda" not in available_devices(), reason="needs Triton and cuda")
def test_kernel_responses_equal_todays(ds, monkeypatch):
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    reqs = _requests(ds)
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")                          # the unfused path
    want = [_strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "1")
    calls = []
    real = TF._run_kernel
    monkeypatch.setattr(TF, "_run_kernel", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    assert [_strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")] == want
    assert calls, "the fused kernel never ran"
