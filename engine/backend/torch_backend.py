"""The torch backend: ``"torch-cpu"`` and ``"cuda"`` (docs/INTERFACES.md §0). Imported only when one of those devices
is asked for; torch is in ``requirements-gpu.txt``, not ``requirements.txt``."""
from __future__ import annotations

import os

import numpy as np
import torch

DTYPES = {"bool": torch.bool, "uint8": torch.uint8, "int32": torch.int32, "int64": torch.int64,
          "float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


class TorchBackend:
    def __init__(self, name: str, accumulate: str = "float64"):
        if accumulate not in ("float64", "float32"):
            raise ValueError("accumulate: float64 | float32")
        self.name, self.accumulate = name, accumulate
        self.device = torch.device("cuda" if name == "cuda" else "cpu")

    # ---- creation and transfer
    def asarray(self, x, dtype=None):
        dt = None if dtype is None else DTYPES[dtype]
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=dt or x.dtype)
        a = np.asarray(x)
        if a.dtype == np.uint32:                     # bitmaps: torch has little uint32 support
            a = a.astype(np.int64)
        if a.dtype == object:
            raise TypeError("object arrays stay on the host")
        return torch.as_tensor(np.ascontiguousarray(a), device=self.device, dtype=dt)

    def to_numpy(self, a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def zeros(self, shape, dtype="float32"):
        return torch.zeros(shape, dtype=DTYPES[dtype], device=self.device)

    def full(self, shape, value, dtype="float32"):
        return torch.full(shape if isinstance(shape, tuple) else (shape,) if isinstance(shape, int) else tuple(shape),
                          value, dtype=DTYPES[dtype], device=self.device)

    def arange(self, n, dtype="int64"):
        return torch.arange(n, dtype=DTYPES[dtype], device=self.device)

    def concatenate(self, arrays, axis=0):
        return torch.cat(list(arrays), dim=axis)

    def stack(self, arrays, axis=0):
        return torch.stack(list(arrays), dim=axis)

    def astype(self, a, dtype):
        return a.to(DTYPES[dtype])

    def where(self, cond, a, b):
        ref = a if isinstance(a, torch.Tensor) else b if isinstance(b, torch.Tensor) else None
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=ref.dtype if ref is not None else torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, device=self.device, dtype=a.dtype)
        return torch.where(cond, a, b)

    def copy(self, a):
        return a.clone()

    # ---- elementwise and reductions
    def log1p(self, a):
        return torch.log1p(a)

    def isnan(self, a):
        return torch.isnan(a)

    def sum(self, a, axis=None):
        return a.sum() if axis is None else a.sum(dim=axis)

    def nonzero(self, mask):
        return tuple(torch.nonzero(mask, as_tuple=True))

    def take_along(self, a, idx, axis=-1):
        return torch.take_along_dim(a, idx, dim=axis)

    def unique(self, a):
        return torch.unique(a, sorted=True)

    def searchsorted(self, sorted_a, v, right=False):
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(np.asarray(v), device=self.device, dtype=sorted_a.dtype)
        return torch.searchsorted(sorted_a, v, right=right)

    def argsort(self, a, descending=False):
        if a.dtype == torch.bool:
            a = a.to(torch.int8)
        return torch.sort(a, dim=-1, descending=descending, stable=True).indices

    # ---- sparse CSR
    def csr_expand(self, ptr, rows, owners):
        rows = self.asarray(rows, "int64")
        owners = self.asarray(owners, "int64")
        starts = ptr[rows]
        n = ptr[rows + 1] - starts
        total = int(n.sum())
        offsets = torch.cumsum(n, 0) - n
        entry = torch.arange(total, device=self.device, dtype=torch.int64) + torch.repeat_interleave(starts - offsets, n)
        return torch.repeat_interleave(owners, n), entry

    def segment_sum(self, keys, values, n):
        # float64 accumulation through index_put_(accumulate=True), whose CUDA kernel is sort-based and
        # deterministic; on CPU it is sequential.
        # float32 (the option for T4/L4-class GPUs, where FP64 is slow): the same deterministic kernel in float32;
        # a key's addends keep their order whatever else is in the batch, so BM25F stays batch-invariant.
        dt = torch.float32 if self.accumulate == "float32" else torch.float64
        out = torch.zeros(n, dtype=dt, device=self.device)
        out.index_put_((keys,), values.to(dt), accumulate=True)
        return out.to(torch.float32)

    def segment_count(self, keys, n):
        return torch.bincount(keys, minlength=n).to(torch.int32)

    # ---- dense
    def matmul(self, a, b, chunk=65_536):
        # float64 accumulation, float32 result (see the numpy backend); a float32 GEMM is an S6 fast path.
        if self.accumulate == "float32":     # a float32 GEMM: cuBLAS may pick a kernel by batch shape
            return torch.matmul(a.to(torch.float32), b.to(torch.float32))
        a64 = a.to(torch.float64)
        out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.float32, device=self.device)
        for s in range(0, b.shape[1], chunk):
            out[:, s:s + chunk] = (a64 @ b[:, s:s + chunk].to(torch.float64)).to(torch.float32)
        return out

    def topk(self, scores, k, valid=None):
        s = scores.to(torch.float32)
        if valid is not None:
            s = torch.where(valid, s, torch.tensor(float("-inf"), device=self.device))
        Q, N = s.shape
        k = min(int(k), N)
        if k == 0 or Q == 0:
            return (torch.full((Q, k), -1, dtype=torch.int32, device=self.device),
                    torch.full((Q, k), float("-inf"), device=self.device))
        kth = torch.topk(s, k, dim=1).values[:, -1:]
        gt = s > kth
        need = k - gt.sum(dim=1, keepdim=True)
        eq = s == kth
        sel = gt | (eq & (torch.cumsum(eq.to(torch.int64), dim=1) <= need))
        cols = torch.nonzero(sel, as_tuple=True)[1].view(Q, k)        # ascending index within each row
        vals = s.gather(1, cols)
        order = torch.sort(vals, dim=1, descending=True, stable=True).indices
        cols, vals = cols.gather(1, order), vals.gather(1, order)
        cols = cols.masked_fill(torch.isneginf(vals), -1)
        return cols.to(torch.int32), vals

    # ---- bitmaps (packed form is for storage; on device a bitmap is a bool mask)
    def pack_bits(self, mask):
        from .numpy_backend import NumpyBackend
        return NumpyBackend().pack_bits(self.to_numpy(mask))

    def unpack_bits(self, words, n):
        from .numpy_backend import NumpyBackend
        return self.asarray(NumpyBackend().unpack_bits(self.to_numpy(words).astype(np.uint32), n))

    def isin(self, a, values):
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(np.asarray(values), device=self.device, dtype=a.dtype)
        return torch.isin(a, values)

    def count_nonzero(self, mask, axis=None):
        return torch.count_nonzero(mask) if axis is None else torch.count_nonzero(mask, dim=axis)

    # ---- facets
    def bincount(self, codes, n, weights=None):
        out = torch.bincount(codes.to(torch.int64), weights=weights, minlength=n)
        return out.to(torch.float32) if weights is not None else out.to(torch.int64)

    def bincount_rows(self, codes, mask, n):
        Q = mask.shape[0]
        r, c = torch.nonzero(mask, as_tuple=True)
        return torch.bincount(r * n + codes[c].to(torch.int64), minlength=Q * n).view(Q, n)

    # ---- memory and sync
    def nbytes(self, a):
        return int(a.element_size() * a.numel())

    def memory_budget(self):
        env = os.environ.get("S3_SEARCH_DEVICE_BUDGET")
        if env:
            return int(env)
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            return int(0.8 * free)
        from .numpy_backend import _available_ram
        return int(0.5 * _available_ram())

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
