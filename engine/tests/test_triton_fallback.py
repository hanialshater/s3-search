"""S6-18: where a Triton kernel may not run, the torch path runs instead and gives the same responses.

The gate (``engine/backend/triton_gate.py``) lets a kernel run only on cuda, with Triton importable at the tested version,
on a GPU class that passed GPU CI. These tests run everywhere: the gate's logic with a stand-in backend (no GPU needed),
and the responses with every kernel flag on, on each device here, against the flags off.
"""
import types

import pytest

from engine.backend import available_devices, get_backend
from engine.backend import triton_bm25 as TB
from engine.backend import triton_facets as TF
from engine.backend import triton_gate as G
from engine.backend import triton_topk as TT
from referee import checks, loader

FLAGS = ("S3_TRITON_BM25", "S3_TRITON_TOPK", "S3_TRITON_FACETS")


def _cuda_like(index=0):
    return types.SimpleNamespace(name="cuda", device=types.SimpleNamespace(index=index))


@pytest.fixture
def fake_gpu(monkeypatch):
    """A stand-in cuda backend on a GPU of capability ``cap`` with Triton ``version``; no GPU or Triton needed."""
    def make(cap=(8, 0), version=G.TESTED_TRITON + ".0"):
        monkeypatch.setattr(G, "triton", types.SimpleNamespace(__version__=version))
        monkeypatch.setattr(G, "_capability", lambda index: cap)
        return _cuda_like()
    return make


def test_the_gate_allows_only_the_passed_classes_at_the_tested_version(fake_gpu, monkeypatch):
    monkeypatch.delenv("S3_TRITON_CAPABILITIES", raising=False)
    for cap in G.PASSED_CAPABILITIES:
        assert G.supported(fake_gpu(cap=cap))
    assert not G.supported(fake_gpu(cap=(7, 5)))                          # a class CI did not pass: torch path
    assert not G.supported(fake_gpu(cap=(9, 0)))
    assert not G.supported(fake_gpu(version="3.9.0"))                     # an untested Triton
    assert not G.supported(fake_gpu(version="3.80.1"))
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "9.0, 12.0")
    assert G.supported(fake_gpu(cap=(9, 0))) and G.supported(fake_gpu(cap=(12, 0)))
    assert not G.supported(fake_gpu(cap=(8, 0)))
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    assert G.supported(fake_gpu(cap=(7, 5)))
    assert not G.supported(fake_gpu(cap=(7, 5), version="3.9.0"))         # "any" class, never any version


def test_no_triton_or_not_cuda_means_no_kernel(fake_gpu, monkeypatch):
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    b = fake_gpu()
    assert TB.wanted(b) and TT.wanted(b, TT.MIN_K) and TF.wanted(b)
    assert not TT.wanted(b, TT.MIN_K - 1)                                 # small windows stay on the one-call path
    monkeypatch.setattr(G, "triton", None)                                # Triton not installed
    assert not (TB.wanted(b) or TT.wanted(b, 2000) or TF.wanted(b))
    for dev in ("cpu", "torch-cpu"):
        if dev in available_devices():
            assert not (TB.wanted(get_backend(dev)) or TT.wanted(get_backend(dev), 2000) or TF.wanted(get_backend(dev)))


def _requests(ds):
    aggs = {"b": {"terms": {"field": "product_brand.keyword", "size": 5}}, "c": {"terms": {"field": "category"}}}
    lex = checks.lexical_requests(ds)[:30]
    return ([{**r, "size": 300} for r in lex] + [{**r, "size": 10, "aggs": aggs} for r in lex]
            + [{**r, "from": 250, "size": 20} for r in lex[:10]])


@pytest.mark.parametrize("device", available_devices())
def test_flags_on_without_a_usable_kernel_give_the_same_responses(device, monkeypatch):
    """Every flag on, but no kernel may run (no Triton on CPU; on cuda the gate is told Triton is missing, then that the
    GPU class did not pass): the responses equal the flags-off responses."""
    from engine import api
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    reqs = _requests(ds)
    monkeypatch.setattr(TT, "MIN_K", 1)                   # the toy has 200 docs: the top-k would be wanted here
    strip = lambda r: {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}  # noqa: E731
    for f in FLAGS:
        monkeypatch.setenv(f, "0")
    want = [strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    for f in FLAGS:
        monkeypatch.setenv(f, "1")
    monkeypatch.delenv("S3_TRITON_CAPABILITIES", raising=False)
    calls = []
    for mod, fn in ((TB, "segment_sum_and_count"), (TT, "topk"), (TF, "bincount_rows")):
        real = getattr(mod, fn)
        monkeypatch.setattr(mod, fn, lambda *a, _r=real, **k: (calls.append(1), _r(*a, **k))[1])
    real_triton = G.triton
    ways = [lambda: None]
    if device == "cuda":
        ways = [lambda: monkeypatch.setattr(G, "triton", None),                         # Triton missing
                lambda: (monkeypatch.setattr(G, "triton", real_triton),                 # a class CI did not pass
                         monkeypatch.setattr(G, "_capability", lambda index: (1, 0)))]
    for way in ways:
        way()
        got = [strip(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
        assert got == want
    assert not calls, "a Triton kernel ran where the gate should have refused it"


def _pin(name):
    import pathlib
    import re
    text = (pathlib.Path(__file__).resolve().parents[2] / "requirements-gpu.txt").read_text()
    m = re.search(rf"^{name}>=(\d+\.\d+),<(\d+\.\d+)$", text, re.M)
    assert m, f"requirements-gpu.txt pins {name} to one minor version"
    return m.group(1)


def test_triton_is_pinned_with_torch_at_the_tested_version():
    assert _pin("triton") == G.TESTED_TRITON
    _pin("torch")


@pytest.mark.skipif(G.triton is None, reason="needs Triton")
def test_installed_triton_and_torch_are_the_pinned_ones():
    """GPU CI: the Triton and torch installed are the pinned pair (a newer Triton would silently fall back)."""
    import torch
    assert G._version_ok(), f"Triton {G.triton.__version__} is not the tested {G.TESTED_TRITON}"
    assert torch.__version__.split("+")[0].startswith(_pin("torch") + "."), torch.__version__


def test_effective_says_which_kernels_run(fake_gpu, monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.delenv("S3_TRITON_CAPABILITIES", raising=False)
    e = G.effective(fake_gpu(cap=(8, 0)))
    assert (e["bm25"], e["topk"], e["facets"], e["capability"]) == (True, True, False, [8, 0])   # S6-18c defaults
    monkeypatch.setenv("S3_TRITON_BM25", "0")
    assert not G.effective(fake_gpu(cap=(8, 0)))["bm25"]
    assert G.effective(fake_gpu(cap=(9, 0)))["topk"] is False                                  # an unpassed class
    assert G.effective(get_backend("cpu"))["bm25"] is False
