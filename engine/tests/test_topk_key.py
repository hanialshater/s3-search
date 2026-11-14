"""S6-14: the torch backend's top-k over a unique int64 key equals the tie-pass top-k it replaces, and numpy's."""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend

TORCH = [d for d in available_devices() if d != "cpu"]


@pytest.fixture(autouse=True)
def _torch_path(monkeypatch):
    monkeypatch.setenv("S3_TRITON_TOPK", "0")       # S6-18c: this tests the int64-key top-k, not the Triton one


def _tie_pass_topk(torch, s, k, valid):
    """The previous torch top-k (kth value, cumsum over ties, nonzero, stable sort), the reference."""
    s = s.to(torch.float32)
    if valid is not None:
        s = torch.where(valid, s, torch.tensor(float("-inf"), device=s.device))
    Q, N = s.shape
    k = min(int(k), N)
    kth = torch.topk(s, k, dim=1).values[:, -1:]
    gt = s > kth
    need = k - gt.sum(dim=1, keepdim=True)
    eq = s == kth
    sel = gt | (eq & (torch.cumsum(eq.to(torch.int64), dim=1) <= need))
    cols = torch.nonzero(sel, as_tuple=True)[1].view(Q, k)
    vals = s.gather(1, cols)
    order = torch.sort(vals, dim=1, descending=True, stable=True).indices
    cols, vals = cols.gather(1, order), vals.gather(1, order)
    return cols.masked_fill(torch.isneginf(vals), -1).to(torch.int32), vals


def _cases(rng):
    for Q, N in ((1, 1), (3, 7), (8, 1000), (5, 4097)):
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
            for k in (1, 5, N // 2 + 1, N, N + 3):
                yield x, valid, k


@pytest.mark.parametrize("device", TORCH)
def test_key_topk_equals_the_tie_pass(device):
    import torch
    b = get_backend(device)
    nb = get_backend("cpu")
    rng = np.random.default_rng(0)
    for x, valid, k in _cases(rng):
        for v in (None, valid):
            got_i, got_v = b.topk(b.asarray(x, "float32"), k, None if v is None else b.asarray(v, "bool"))
            want_i, want_v = _tie_pass_topk(torch, b.asarray(x, "float32"), k,
                                            None if v is None else b.asarray(v, "bool"))
            assert np.array_equal(b.to_numpy(got_i), b.to_numpy(want_i)), (x.shape, k)
            np.testing.assert_array_equal(b.to_numpy(got_v), b.to_numpy(want_v))
            ni, nv = nb.topk(x, k, v)                          # the numpy reference agrees too
            assert np.array_equal(b.to_numpy(got_i), ni), (x.shape, k)
