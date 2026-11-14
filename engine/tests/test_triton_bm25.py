"""S6-18: the Triton segmented sum gives bm25f's scores and match counts bit for bit, batched, alone and run to run.

Runs where Triton and cuda are both present (GPU CI); skipped elsewhere.
"""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend
from engine.backend import triton_bm25 as TB
from engine.ops import lexical as L
from referee import loader

pytestmark = pytest.mark.skipif(TB.triton is None or "cuda" not in available_devices(), reason="needs Triton and cuda")


@pytest.fixture(autouse=True)
def _any_gpu_class(monkeypatch):
    """Run the kernel on whatever GPU this is (the gate's list is what these tests decide), and count its calls, so a
    test cannot pass by falling back to the torch path."""
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    calls = []
    real = getattr(TB, "segment_sum_and_count")
    monkeypatch.setattr(TB, "segment_sum_and_count", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    return calls


@pytest.mark.parametrize("limit", [10**9, 150, 25])          # one chunk, and chunks that cut the batch
def test_triton_sum_equals_the_torch_path(limit, monkeypatch, _any_gpu_class):
    import torch
    from engine import api
    from engine.analysis import analyze
    monkeypatch.setattr(L, "MAX_POSTINGS_PER_CHUNK", limit)
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    st = ix._st
    part, b = st.parts[0], st.b
    texts = list(dict.fromkeys(list(ds.fit["query"]) + list(ds.test_queries["query"])))
    terms = [sorted({part.vocab[t] for t in analyze(x) if t in part.vocab}) for x in texts][:64] + [[]]
    rng = np.random.default_rng(0)
    common = np.argsort(-np.diff(part.host_term_ptr))[:12]
    terms += [sorted(rng.choice(common, 5, replace=False).tolist()) for _ in range(16)]
    F = len(st.text_fields)
    w = np.linspace(0.7, 2.3, F).astype(np.float32)
    bb = np.full(F, 0.75, np.float32)

    def run(on, ts):
        monkeypatch.setenv("S3_TRITON_BM25", "1" if on else "0")
        s, n = L.bm25f(b, part, ts, w, bb, 1.2, part.idf)
        return s.view(torch.int32).cpu().numpy(), n.cpu().numpy()

    s0, n0 = run(False, terms)
    s1, n1 = run(True, terms)
    assert np.array_equal(s0, s1) and np.array_equal(n0, n1)
    s2, _ = run(True, terms)
    assert np.array_equal(s1, s2)                              # run to run
    for q in (0, 5, len(terms) - 1):
        sa, na = run(True, [terms[q]])
        assert np.array_equal(sa[0], s1[q]) and np.array_equal(na[0], n1[q])
    assert _any_gpu_class, "the Triton kernel never ran"


def test_off_by_default_and_never_on_cpu():
    assert not TB.wanted(get_backend("cuda"))                  # S3_TRITON_BM25 unset
    assert not TB.wanted(get_backend("cpu"))
