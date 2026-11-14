"""S6-18: the Triton top-k (float32 threshold, compaction, key sort) equals the int64-key top-k exactly.

Runs where Triton and cuda are present (GPU CI); skipped elsewhere.
"""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend
from engine.backend import triton_topk as TT
from referee import checks, loader

pytestmark = pytest.mark.skipif(TT.triton is None or "cuda" not in available_devices(), reason="needs Triton and cuda")


@pytest.fixture(autouse=True)
def _any_gpu_class(monkeypatch):
    """Run the kernel on whatever GPU this is (the gate's list is what these tests decide), and count its calls, so a
    test cannot pass by falling back to the torch path."""
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")    # S6-26: this kernel's own path, not the fused pass's
    calls = []
    real = getattr(TT, "topk")
    monkeypatch.setattr(TT, "topk", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


def _cases(rng):
    for Q, N in ((1, 300), (4, 5000), (16, 70_000), (3, 4097)):
        for kind in ("ties", "negatives", "zeros", "dense"):
            if kind == "ties":
                x = rng.integers(0, 4, (Q, N)).astype(np.float32) / 2
            elif kind == "negatives":
                x = rng.normal(size=(Q, N)).astype(np.float32)
                x[:, ::5] = -np.inf
            elif kind == "zeros":
                x = np.zeros((Q, N), np.float32)
                x[:, ::3] = -0.0
                x[:, 1::7] = 1.5
            else:
                x = rng.random((Q, N)).astype(np.float32)
            valid = rng.random((Q, N)) < 0.6
            for k in (1, 10, TT.MIN_K, 1000, N // 2 + 1, N):        # S6-27: small k on wide rows too
                if k <= N:
                    yield x, valid, k


def test_triton_topk_equals_the_key_topk(monkeypatch, _any_gpu_class):
    import torch
    b = get_backend("cuda")
    rng = np.random.default_rng(0)
    for x, valid, k in _cases(rng):
        for v in (None, valid):
            vv = None if v is None else b.asarray(v, "bool")
            monkeypatch.setenv("S3_TRITON_TOPK", "0")
            wi, wv = b.topk(b.asarray(x, "float32"), k, vv)
            monkeypatch.setenv("S3_TRITON_TOPK", "1")
            gi, gv = b.topk(b.asarray(x, "float32"), k, vv)
            assert torch.equal(gi, wi), (x.shape, k)
            assert torch.equal(gv.view(torch.int32), wv.view(torch.int32)), (x.shape, k)
    assert _any_gpu_class, "the Triton kernel never ran"


def test_deep_responses_unchanged(monkeypatch, _any_gpu_class):
    from engine import api
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    monkeypatch.setattr(TT, "MIN_K", 1)                     # the toy has 200 docs, fewer than MIN_K: let it run
    reqs = [{**r, "size": 300} for r in checks.lexical_requests(ds)[:40]]
    monkeypatch.setenv("S3_TRITON_TOPK", "0")
    want = [api.expand_hits(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    monkeypatch.setenv("S3_TRITON_TOPK", "1")
    got = [api.expand_hits(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    strip = lambda r: {k: v for k, v in r.items() if k != "took_ms"}  # noqa: E731
    assert [strip(r) for r in got] == [strip(r) for r in want]
    assert _any_gpu_class, "the Triton kernel never ran"


def test_on_by_default_and_off_with_0(monkeypatch):
    monkeypatch.delenv("S3_TRITON_TOPK", raising=False)
    assert TT.wanted(get_backend("cuda"), 2000)                # S6-18c: on by default where the gate allows
    assert TT.wanted(get_backend("cuda"), 10, TT.WIDE_N)          # S6-27: a small k on a wide row
    assert not TT.wanted(get_backend("cuda"), 10, TT.WIDE_N - 1)
    monkeypatch.setenv("S3_TRITON_TOPK", "0")
    assert not TT.wanted(get_backend("cuda"), 2000)
