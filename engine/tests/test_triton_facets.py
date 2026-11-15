"""S6-18: the Triton facet histogram equals the torch path's counts exactly, and the facets' responses are unchanged.

Runs where Triton and cuda are present (GPU CI); skipped elsewhere.
"""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend
from engine.backend import triton_facets as TF
from referee import checks, loader

pytestmark = pytest.mark.skipif(TF.triton is None or "cuda" not in available_devices(), reason="needs Triton and cuda")


@pytest.fixture(autouse=True)
def _any_gpu_class(monkeypatch):
    """Run the kernel on whatever GPU this is (the gate's list is what these tests decide), and count its calls, so a
    test cannot pass by falling back to the torch path."""
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")    # S6-26: this kernel's own path, not the fused pass's
    calls = []
    real = getattr(TF, "bincount_rows")
    monkeypatch.setattr(TF, "bincount_rows", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


def test_counts_equal_the_torch_path(monkeypatch, _any_gpu_class):
    import torch
    b = get_backend("cuda")
    rng = np.random.default_rng(0)
    for Q, N, n in ((1, 1, 1), (3, 1000, 7), (64, 50_000, 5_000), (5, 4097, 2)):
        codes = b.asarray(rng.integers(0, n, N), "int64")
        mask = b.asarray(rng.random((Q, N)) < 0.3, "bool")
        monkeypatch.setenv("S3_TRITON_FACETS", "0")
        want = b.bincount_rows(codes, mask, n)
        monkeypatch.setenv("S3_TRITON_FACETS", "1")
        got = b.bincount_rows(codes, mask, n)
        again = b.bincount_rows(codes, mask, n)
        assert got.dtype == want.dtype == torch.int64
        assert torch.equal(got, want) and torch.equal(again, want)
    assert _any_gpu_class, "the Triton kernel never ran"


def test_facet_responses_unchanged(monkeypatch, _any_gpu_class):
    from engine import api
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    aggs = {"b": {"terms": {"field": "product_brand.keyword", "size": 5}}, "c": {"terms": {"field": "category"}}}
    reqs = [{**r, "aggs": aggs, "size": 3} for r in checks.lexical_requests(ds)[:40]]
    monkeypatch.setenv("S3_TRITON_FACETS", "0")
    want = ix.search_batch(reqs)
    monkeypatch.setenv("S3_TRITON_FACETS", "1")
    got = ix.search_batch(reqs)
    strip = lambda r: {k: v for k, v in r.items() if k != "took_ms"}  # noqa: E731
    assert [strip(r) for r in got] == [strip(r) for r in want]
    assert _any_gpu_class, "the Triton kernel never ran"


def test_off_by_default():
    assert not TF.wanted(get_backend("cuda"))
    assert not TF.wanted(get_backend("cpu"))
