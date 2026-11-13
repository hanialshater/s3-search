"""The reference backend: numpy on CPU (docs/INTERFACES.md §0). Always available; every other device is checked
against it."""
from __future__ import annotations

import os

import numpy as np

DTYPES = {"bool": np.bool_, "uint8": np.uint8, "int32": np.int32, "int64": np.int64, "uint32": np.uint32,
          "float16": np.float16, "float32": np.float32, "float64": np.float64}


def _dt(dtype):
    return None if dtype is None else DTYPES[dtype]


def _available_ram() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 4 << 30


class NumpyBackend:
    name = "cpu"

    def __init__(self, accumulate: str = "float64"):
        if accumulate not in ("float64", "float32"):
            raise ValueError("accumulate: float64 | float32")
        self.accumulate = accumulate

    # ---- creation and transfer
    def asarray(self, x, dtype=None):
        if hasattr(x, "detach"):            # a torch tensor
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=_dt(dtype))

    def to_numpy(self, a):
        return np.asarray(a)

    def zeros(self, shape, dtype="float32"):
        return np.zeros(shape, dtype=DTYPES[dtype])

    def full(self, shape, value, dtype="float32"):
        return np.full(shape, value, dtype=DTYPES[dtype])

    def arange(self, n, dtype="int64"):
        return np.arange(n, dtype=DTYPES[dtype])

    def concatenate(self, arrays, axis=0):
        return np.concatenate(list(arrays), axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(list(arrays), axis=axis)

    def astype(self, a, dtype):
        return a.astype(DTYPES[dtype], copy=False)

    def where(self, cond, a, b):
        return np.where(cond, a, b)

    def copy(self, a):
        return a.copy()

    # ---- elementwise and reductions the operators need
    def log1p(self, a):
        return np.log1p(a)

    def amin(self, a):
        """The minimum of all elements (a 0-d array)."""
        return np.asarray(a.min())

    def sin(self, a):
        return np.sin(a)

    def cos(self, a):
        return np.cos(a)

    def arcsin(self, a):
        return np.arcsin(a)

    def isnan(self, a):
        return np.isnan(a)

    def sum(self, a, axis=None):
        return a.sum(axis=axis)

    def nonzero(self, mask):
        """Indices of true entries, as a tuple of int64 arrays (row-major order)."""
        return tuple(i.astype(np.int64) for i in np.nonzero(mask))

    def take_along(self, a, idx, axis=-1):
        return np.take_along_axis(a, idx, axis=axis)

    def unique(self, a):
        """Sorted distinct values."""
        return np.unique(a)

    def searchsorted(self, sorted_a, v, right=False):
        return np.searchsorted(sorted_a, v, side="right" if right else "left")

    def argsort(self, a, descending=False):
        """Stable argsort along the last axis."""
        if descending:
            if a.dtype == np.bool_:
                a = a.astype(np.int8)
            return np.argsort(-a, axis=-1, kind="stable")
        return np.argsort(a, axis=-1, kind="stable")

    # ---- sparse CSR
    def csr_expand(self, ptr, rows, owners, total=None):
        rows = np.asarray(rows, dtype=np.int64)
        starts = ptr[rows]
        n = ptr[rows + 1] - starts
        total = int(n.sum())
        offsets = np.cumsum(n) - n
        entry = np.arange(total, dtype=np.int64) + np.repeat(starts - offsets, n)
        return np.repeat(np.asarray(owners, dtype=np.int64), n), entry

    def segment_sum(self, keys, values, n):
        # float64: bincount accumulates in float64, in entry order. float32: np.add.at in entry order. Both are
        # deterministic, and a key's addends keep their order whatever else is in the batch.
        if self.accumulate == "float32":
            out = np.zeros(n, np.float32)
            np.add.at(out, keys, np.asarray(values, np.float32))
            return out
        return np.bincount(keys, weights=values, minlength=n).astype(np.float32)

    def accumulator(self, n):
        return np.zeros(n, np.float32 if self.accumulate == "float32" else np.float64)

    def add_unique(self, out, keys, values):
        out[keys] += np.asarray(values, out.dtype)

    def segment_max(self, keys, values, n, init=float("-inf")):
        out = np.full(n, init, np.float64)
        np.maximum.at(out, keys, np.asarray(values, np.float64))
        return out

    def segment_min(self, keys, values, n, init=float("inf")):
        out = np.full(n, init, np.float64)
        np.minimum.at(out, keys, np.asarray(values, np.float64))
        return out

    def segment_count(self, keys, n):
        return np.bincount(keys, minlength=n).astype(np.int32)

    def segment_count_where(self, keys, mask, n):
        return np.bincount(keys[mask], minlength=n).astype(np.int32)

    # ---- dense
    def matmul(self, a, b, chunk=65_536):
        # float32 inputs, float64 accumulation, float32 result: the same values whatever the batch shape, so
        # search_batch equals search one by one. Chunked over b's columns to bound the float64 copy.
        if self.accumulate == "float32":     # a float32 GEMM: results may depend on the batch shape
            return np.matmul(a.astype(np.float32, copy=False), b.astype(np.float32, copy=False))
        a64 = a.astype(np.float64)
        out = np.empty((a.shape[0], b.shape[1]), np.float32)
        for s in range(0, b.shape[1], chunk):
            out[:, s:s + chunk] = a64 @ b[:, s:s + chunk].astype(np.float64)
        return out

    def topk(self, scores, k, valid=None):
        s = np.asarray(scores, dtype=np.float32)
        if valid is not None:
            s = np.where(valid, s, np.float32(-np.inf))
        Q, N = s.shape
        k = min(int(k), N)
        idx = np.full((Q, k), -1, dtype=np.int32)
        val = np.full((Q, k), -np.inf, dtype=np.float32)
        if k == 0:
            return idx, val
        for q in range(Q):
            row = s[q]
            kth = np.partition(row, N - k)[N - k]
            gt = np.flatnonzero(row > kth)
            eq = np.flatnonzero(row == kth)[: k - len(gt)]
            sel = np.concatenate([gt, eq])
            sel = sel[np.lexsort((sel, -row[sel]))]
            idx[q], val[q] = sel, row[sel]
        idx[np.isneginf(val)] = -1
        return idx, val

    # ---- bitmaps
    def pack_bits(self, mask):
        mask = np.asarray(mask, dtype=bool)
        n = mask.shape[-1]
        pad = (-n) % 32
        if pad:
            mask = np.concatenate([mask, np.zeros(mask.shape[:-1] + (pad,), bool)], axis=-1)
        return np.packbits(mask, axis=-1, bitorder="little").view("<u4").astype(np.uint32)

    def unpack_bits(self, words, n):
        words = np.ascontiguousarray(np.asarray(words, dtype="<u4"))
        return np.unpackbits(words.view(np.uint8), axis=-1, bitorder="little")[..., :n].astype(bool)

    def isin(self, a, values):
        return np.isin(a, np.asarray(values))

    def count_nonzero(self, mask, axis=None):
        # A numpy integer, never a Python int: numpy < 2 returns int for axis=None, which has no .astype and broke
        # range aggregations (aggs.range_agg) on numpy 1.26, which requirements.txt allows.
        n = np.count_nonzero(mask, axis=axis)
        return np.int64(n) if axis is None else n

    # ---- facets
    def bincount(self, codes, n, weights=None):
        out = np.bincount(codes, weights=weights, minlength=n)
        return out.astype(np.float32) if weights is not None else out.astype(np.int64)

    def bincount_rows(self, codes, mask, n):
        Q = mask.shape[0]
        r, c = np.nonzero(mask)
        return np.bincount(r * n + codes[c], minlength=Q * n).reshape(Q, n).astype(np.int64)

    # ---- memory and sync
    def nbytes(self, a):
        return int(a.nbytes)

    def memory_budget(self):
        env = os.environ.get("S3_SEARCH_DEVICE_BUDGET")
        return int(env) if env else int(0.5 * _available_ram())

    def synchronize(self):
        return None
