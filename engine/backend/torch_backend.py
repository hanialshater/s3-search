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
        # S6-15: optionally wait for the device without spinning. A host read (a copy to the host, a scalar) waits
        # for the stream; CUDA's default wait spins a host core while the device runs. With blocking_sync the wait is
        # on a blocking event and the thread sleeps. Off by default: the A100 profile at 3bfeea4 measured no host CPU
        # saved at depth 2,000 (1.224 against 1.248 ms per request, the process CPU-bound by its own work) and
        # 0.34 ms more batch-1 latency. S3_CUDA_BLOCKING_SYNC=1 turns it on. Results are the same either way.
        import os
        self.blocking_sync = self.device.type == "cuda" and os.environ.get("S3_CUDA_BLOCKING_SYNC", "0") == "1"

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

    def _wait(self):
        """Block (not spin) until the current stream's work is done, when ``blocking_sync`` is on."""
        if self.blocking_sync:
            ev = torch.cuda.Event(blocking=True)
            ev.record()
            ev.synchronize()

    def to_numpy(self, a):
        if isinstance(a, torch.Tensor):
            if a.device.type == "cuda":
                self._wait()
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
        # S6-17: next to a tensor, a Python scalar becomes a 0-dim tensor on the host, which torch passes to the kernel
        # as a value. A device scalar was a host-to-device copy per call, and that copy waited for the stream (the GPU
        # profile at 20bda25: about 2 per request, 40% of a depth-2,000 batch's host time on an L4). Same dtypes as
        # before (a scalar takes the other operand's dtype), so the same results.
        ref = a if isinstance(a, torch.Tensor) else b if isinstance(b, torch.Tensor) else None
        if ref is None:                                   # two scalars: tensors on the device, as before
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
            b = torch.tensor(b, device=self.device, dtype=a.dtype)
            return torch.where(cond, a, b)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=ref.dtype)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=a.dtype)
        return torch.where(cond, a, b)

    def copy(self, a):
        return a.clone()

    # ---- elementwise and reductions
    def log1p(self, a):
        return torch.log1p(a)

    def amin(self, a):
        return a.min()

    def sin(self, a):
        return torch.sin(a)

    def cos(self, a):
        return torch.cos(a)

    def arcsin(self, a):
        return torch.asin(a)

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
    def csr_expand(self, ptr, rows, owners, total=None):
        """``total``: the expanded length when the caller knows it on the host (S6-17), so no device read is needed
        (the count, and repeat_interleave's own, each waited for the stream)."""
        rows = self.asarray(rows, "int64")
        owners = self.asarray(owners, "int64")
        starts = ptr[rows]
        n = ptr[rows + 1] - starts
        if total is None:
            total_t = n.sum()
            self._wait()
            total = int(total_t)
        offsets = torch.cumsum(n, 0) - n
        entry = torch.arange(total, device=self.device, dtype=torch.int64) + \
            torch.repeat_interleave(starts - offsets, n, output_size=total)
        return torch.repeat_interleave(owners, n, output_size=total), entry

    def segment_sum(self, keys, values, n, accumulate=None):
        # float64 accumulation through index_put_(accumulate=True), whose CUDA kernel is sort-based and
        # deterministic; on CPU it is sequential.
        # float32 (the option for T4/L4-class GPUs, where FP64 is slow): the same deterministic kernel in float32;
        # a key's addends keep their order whatever else is in the batch, so BM25F stays batch-invariant.
        dt = torch.float32 if (accumulate or self.accumulate) == "float32" else torch.float64
        out = torch.zeros(n, dtype=dt, device=self.device)
        out.index_put_((keys,), values.to(dt), accumulate=True)
        return out.to(torch.float32)

    def segment_max(self, keys, values, n, init=float("-inf")):
        out = torch.full((n,), init, dtype=torch.float64, device=self.device)
        return out.scatter_reduce(0, keys, values.to(torch.float64), reduce="amax", include_self=True)

    def segment_min(self, keys, values, n, init=float("inf")):
        out = torch.full((n,), init, dtype=torch.float64, device=self.device)
        return out.scatter_reduce(0, keys, values.to(torch.float64), reduce="amin", include_self=True)

    def segment_count(self, keys, n):
        return torch.bincount(keys, minlength=n).to(torch.int32)

    def segment_sum_rows(self, docs, values, hits, row_lens, row_owners, n_queries, N, accumulate=None):
        """S6-18: ``(segment_sum(q * N + docs, values), segment_count_where(...))`` for a chunk whose entries are its term
        rows one after another (each sorted by doc), summed by the Triton kernel in the rows' order: the same float64
        additions in the same order as the sort-based kernel, so the same bits. ``None`` when it does not apply
        (no Triton, not cuda, ``S3_TRITON_BM25`` not "1"), and the caller takes the torch path. ``values`` may be float16,
        float32 or float64 and is read as it is; ``accumulate`` (S6-21): the sum's dtype, float64 or float32, else the
        backend's. Each dtype pair gives the torch path's bits: the same adds in the same order."""
        from . import triton_bm25 as TB
        if not TB.wanted(self):
            return None
        acc_dt = torch.float32 if (accumulate or self.accumulate) == "float32" else torch.float64
        row_start, at = [], 0
        for n in row_lens:
            row_start.append(at)
            at += n
        q_index, q_lo, q_hi = [], [], []
        for r, q in enumerate(row_owners):                  # a query's rows are consecutive
            if not q_index or q_index[-1] != q:
                q_index.append(q)
                q_lo.append(r)
                q_hi.append(r)
            q_hi[-1] = r + 1
        acc, cnt = TB.segment_sum_and_count(self, docs, values, hits, row_start, list(row_lens),
                                            q_lo, q_hi, q_index, n_queries, N, acc_dtype=acc_dt)
        return acc.to(torch.float32), cnt

    def segment_count_where(self, keys, mask, n):
        """``segment_count(keys[mask], n)`` without the boolean index (a host sync, S6-17): an integer scatter-add of
        the mask (atomic integer adds: exact in any order). Not index_put(accumulate=True), which is sort-based on
        cuda (22 ms of a depth-2,000 batch's device time on an A100 at 9729340), and not a weighted bincount, which
        reads the keys' maximum from the device to size its result (12 ms of host CPU per batch at 4d5ce23)."""
        return torch.zeros(n, dtype=torch.int32, device=self.device).scatter_add_(0, keys, mask.to(torch.int32))

    def accumulator(self, n):
        return torch.zeros(n, dtype=torch.float32 if self.accumulate == "float32" else torch.float64,
                           device=self.device)

    def add_unique(self, out, keys, values):
        out[keys] = out[keys] + values.to(out.dtype)

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
            s = torch.where(valid, s, torch.tensor(float("-inf"), dtype=torch.float32))  # a host scalar: no copy
        Q, N = s.shape
        k = min(int(k), N)
        if k == 0 or Q == 0:
            return (torch.full((Q, k), -1, dtype=torch.int32, device=self.device),
                    torch.full((Q, k), float("-inf"), device=self.device))
        from . import triton_topk as TT
        if TT.wanted(self, k):                            # S6-18: a float32 threshold and a Triton compaction
            return TT.topk(s, k)
        # S6-14: one top-k over a unique int64 key, (the score's order-preserving bits, then N-1-index), so the
        # order is score descending, then index ascending, with no tie pass (the kth value, a cumsum over the row,
        # a nonzero and a sort before). -0.0 is made +0.0 first: they compare equal as floats.
        s = s + 0.0
        bits = s.view(torch.int32).to(torch.int64)
        ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        shift = max(1, (N - 1).bit_length())
        key = (ordered << shift) | (N - 1 - torch.arange(N, device=self.device, dtype=torch.int64))
        top = torch.topk(key, k, dim=1).values
        cols = (N - 1) - (top & ((1 << shift) - 1))
        vals = s.gather(1, cols)
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
        from . import triton_facets as TF
        if TF.wanted(self):                               # S6-18: one kernel, no nonzero, no device reads
            return TF.bincount_rows(mask, codes, n)
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
            self._wait()
            torch.cuda.synchronize()
