"""Item 2 (opt-in, ``S3_FUSED_HIST24=1``): the fused kernel writes the top-k's first radix pass, so the top-k does not
read the score rows for it.

The radix select (``triton_topk.kth_value``) makes four 8-bit passes over each row's order-preserving keys. Its first
pass, the histogram of the top digit, reads all of the rows (Q x N float32 per batch). The fused kernel holds every
value it stores, so it can add that histogram as it writes them. Only the non-zero bins are added: matched BM25F scores
have a handful of top digits, and every unmatched -inf goes to bin 0. The select then starts from that histogram.

What must hold, and is checked here:
- the algorithm (numpy, any machine): a radix select started from a given first-pass histogram gives the k-th value
  that sorting gives, on ties, -inf, -0.0 and +0.0. The kernel's key treats -0.0 as +0.0, as ``triton_topk.topk``
  does with ``s + 0.0``;
- the kernel's histogram equals ``first_digit_hist`` of the same rows, bin for bin, and the rows are the same bits
  with it on or off (cuda);
- ``kth_value`` with that histogram equals ``kth_value`` without it, bit for bit (cuda);
- every response is the same with it on or off, and the histogram path really ran (cuda).
"""
import numpy as np
import pytest

from engine import api
from engine.backend import available_devices
from engine.backend import triton_fused as TF
from engine.backend import triton_topk as TT
from referee import loader

CUDA = TF.triton is not None and "cuda" in available_devices()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs Triton and cuda")


# ---- the algorithm, in numpy (runs everywhere) ---------------------------------------------------------------------

def _np_key(x):
    """The kernel's key: the order-preserving 32-bit pattern of x, with -0.0 as +0.0."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.int64)
    b = np.where(b == 0x80000000, 0, b)
    return np.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)


def _np_hist24(x):
    return np.stack([np.bincount((_np_key(r) >> 24).astype(np.int64), minlength=256) for r in x])


def _np_kth(x, k, h24):
    """``kth_value``'s four passes on one row, the first from ``h24``."""
    key = _np_key(x)
    prefix, krem = 0, k
    for shift in (24, 16, 8, 0):
        if shift == 24:
            h = h24
        else:
            m = (key >> (shift + 8)) == (prefix >> (shift + 8))
            h = np.bincount(((key[m] >> shift) & 255).astype(np.int64), minlength=256)
        cum = np.cumsum(h[::-1])
        first = int(np.argmax(cum >= krem))
        krem -= int(h[::-1][:first].sum())
        prefix |= (255 - first) << shift
    b = prefix - 0x80000000 if prefix >= 0x80000000 else 0xFFFFFFFF - prefix
    return np.array([b], np.uint32).view(np.float32)[0]


def _rows(rng):
    for N in (1, 7, 300, 5000):
        yield rng.integers(0, 4, N).astype(np.float32) / 2                 # ties
        x = rng.normal(size=N).astype(np.float32)
        x[::5] = -np.inf
        yield x                                                            # negatives and -inf
        z = np.zeros(N, np.float32)
        z[::3] = -0.0
        z[1::7] = 1.5
        yield z                                                            # -0.0 and +0.0
        m = np.full(N, -np.inf, np.float32)
        m[::11] = (rng.random(len(m[::11])) * 40 + 1).astype(np.float32)
        yield m                                                            # sparse matches among -inf, BM25-like


def test_a_select_started_from_the_first_pass_histogram_gives_the_sorted_kth_value():
    rng = np.random.default_rng(3)
    for x in _rows(rng):
        h = _np_hist24(x[None])[0]
        s = np.sort((x + np.float32(0.0)))[::-1]                          # -0.0 as +0.0, as topk does
        for k in sorted({1, min(2, len(x)), len(x) // 2 + 1, len(x)}):
            got = _np_kth(x, k, h)
            assert got.view(np.uint32) == np.float32(s[k - 1]).view(np.uint32), (x[:8], k)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(TF.HIST24_ENV, raising=False)
    assert TF.hist24_on() is False
    monkeypatch.setenv(TF.HIST24_ENV, "1")
    assert TF.hist24_on() is True


# ---- the kernel (cuda) ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@needs_cuda
def test_kth_value_from_a_given_first_pass_equals_the_four_pass_select(monkeypatch):
    import torch
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    rng = np.random.default_rng(1)
    for Q, N in ((1, 300), (4, 5000), (16, 70_000), (3, 4097)):
        for x in (rng.integers(0, 4, (Q, N)).astype(np.float32) / 2, rng.normal(size=(Q, N)).astype(np.float32)):
            x[:, ::5] = -np.inf
            x[:, 1::9] = -0.0
            s = (torch.as_tensor(x, device="cuda") + 0.0).contiguous()
            for k in (1, 10, TT.MIN_K, N // 2 + 1, N):
                if k > N:
                    continue
                want = TT.kth_value(s, k)
                got = TT.kth_value(s, k, TT.first_digit_hist(s))
                assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (Q, N, k)


@needs_cuda
@pytest.mark.parametrize("impacts", ["0", "1"])
def test_the_kernels_histogram_is_the_first_pass_of_its_rows(ds, impacts, monkeypatch):
    import torch
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    monkeypatch.setenv("S3_TRITON_FUSED_IMPACTS", impacts)
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
    args = (b, part, terms, w, bb, 1.2, part.idf, msm, boost, live, [], [])
    monkeypatch.setenv(TF.HIST24_ENV, "0")
    off = TF._run_kernel(*args)
    assert TF.hist24_of(off[0]) is None
    monkeypatch.setenv(TF.HIST24_ENV, "1")
    on = TF._run_kernel(*args)
    assert torch.equal(on[0].view(torch.int32), off[0].view(torch.int32))   # the rows: the same bits
    assert torch.equal(on[1], off[1])
    h = TF.hist24_of(on[0])
    assert h is not None and TF.hist24_of(on[0]) is None                     # taken once
    want = TT.first_digit_hist((on[0] + 0.0).contiguous())
    assert torch.equal(h, want)
    assert int(h.view(-1, 256).sum(dim=1).min()) == part.N == int(h.view(-1, 256).sum(dim=1).max())


@needs_cuda
@pytest.mark.parametrize("impacts", ["0", "1"])
def test_responses_are_the_same_with_it_on_or_off(ds, impacts, monkeypatch):
    from engine.tests.test_fused_lexical import _requests, _strip
    monkeypatch.setenv("S3_TRITON_CAPABILITIES", "any")
    monkeypatch.setenv("S3_TRITON_FUSED_IMPACTS", impacts)
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "1")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    reqs = _requests(ds)
    reqs = reqs + [dict(r, size=TT.MIN_K + 50) for r in reqs[:10]]            # windows on the Triton top-k
    monkeypatch.setenv(TF.HIST24_ENV, "0")
    want = [_strip(r) for r in ix.search_batch(reqs)]
    used = []
    real = TT.kth_value
    monkeypatch.setattr(TT, "kth_value", lambda s, k, hist24=None: (used.append(hist24 is not None), real(s, k, hist24))[1])
    monkeypatch.setenv(TF.HIST24_ENV, "1")
    got = [_strip(r) for r in ix.search_batch(reqs)]
    assert got == want
    assert any(used), "the histogram path did not run"
