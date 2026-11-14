"""S6-18: BM25F's per-(query, doc) sum as a Triton kernel: a deterministic segmented sum in the postings' own order.

The torch path sums a chunk's addends with ``segment_sum`` (``index_put_(accumulate=True)`` in float64), whose CUDA kernel
sorts every (query, doc) key and then adds each key's addends one after another in entry order. The postings are
already in that order: a chunk's entries are its queries' terms one after another, and each term's postings are sorted
by doc. So a key's addends are the query's terms that hold the doc, in the query's term order.

The kernel gives each program one query and one tile of docs. It walks the query's terms in order; for each, it finds
the term's postings inside the tile (two binary searches in the term's sorted doc ids) and adds their values into the
tile's accumulators, a float64 sum and an int32 count, with a barrier between terms. Inside one term no doc repeats, so
no two lanes write one key at once; across terms the barrier keeps the order. Each key's sum is therefore
0 + a1 + a2 + ..., added in the same order as the sort-based kernel adds them, in float64: the same bits. No atomics.

Used only on cuda, for a single segment whose postings are sorted by doc within each term, when ``S3_TRITON_BM25`` is
"1" and ``triton_gate.supported`` (Triton imports, at the tested version, on a GPU class that passed). The torch path stays the default until the kernel is measured faster on the A100.
"""
from __future__ import annotations

import os

from . import triton_gate

try:                                                        # optional: Colab's torch ships it; CPU CI does not need it
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

TILE = 2048
BLOCK = 256


def wanted(b) -> bool:
    return os.environ.get("S3_TRITON_BM25", "0") == "1" and triton_gate.supported(b)


if triton is not None:
    @triton.jit
    def _segsum_kernel(docs_ptr, vals_ptr, hits_ptr, row_start_ptr, row_len_ptr, q_row_lo_ptr, q_row_hi_ptr,
                       q_index_ptr, acc_ptr, cnt_ptr, N, TILE: tl.constexpr, BLOCK: tl.constexpr):
        g = tl.program_id(0)                                # the query's position in the chunk
        tile = tl.program_id(1)
        d0 = tile * TILE
        d1 = d0 + TILE
        q = tl.load(q_index_ptr + g).to(tl.int64)           # the query's row in the [Q, N] output
        r_lo = tl.load(q_row_lo_ptr + g)
        r_hi = tl.load(q_row_hi_ptr + g)
        base = q * N
        lanes = tl.arange(0, BLOCK)
        for r in range(r_lo, r_hi):
            start = tl.load(row_start_ptr + r)
            n = tl.load(row_len_ptr + r)
            # the first posting with doc >= d0, and the first with doc >= d1, in [start, start + n)
            lo = start
            hi = start + n
            while lo < hi:
                mid = (lo + hi) // 2
                v = tl.load(docs_ptr + mid)
                go = v < d0
                lo = tl.where(go, mid + 1, lo)
                hi = tl.where(go, hi, mid)
            a = lo
            hi = start + n
            while lo < hi:
                mid = (lo + hi) // 2
                v = tl.load(docs_ptr + mid)
                go = v < d1
                lo = tl.where(go, mid + 1, lo)
                hi = tl.where(go, hi, mid)
            e_end = lo
            for e0 in range(a, e_end, BLOCK):
                e = e0 + lanes
                ok = e < e_end
                d = tl.load(docs_ptr + e, mask=ok, other=0)
                x = tl.load(vals_ptr + e, mask=ok, other=0.0)
                h = tl.load(hits_ptr + e, mask=ok, other=0)
                off = base + d
                cur = tl.load(acc_ptr + off, mask=ok, other=0.0)
                tl.store(acc_ptr + off, cur + x, mask=ok)
                c = tl.load(cnt_ptr + off, mask=ok, other=0)
                tl.store(cnt_ptr + off, c + h, mask=ok)
            tl.debug_barrier()                              # this term's adds land before the next term's


def segment_sum_and_count(b, docs, vals64, hits, row_start, row_len, q_row_lo, q_row_hi, q_index, n_queries: int,
                          N: int):
    """``(sum float64[Q*N], count int32[Q*N])`` of a chunk's entries, the sum bit for bit ``segment_sum``'s float64
    accumulation of ``vals64`` at keys ``q * N + doc``, and the count ``segment_count`` of the entries with ``hits``.
    ``row_start``, ``row_len``: each term row's entries in the chunk (host lists); ``q_row_lo``/``q_row_hi``: each query's
    rows; ``q_index``: each query's row in the output."""
    import torch
    dev = docs.device
    acc = torch.zeros(n_queries * N, dtype=torch.float64, device=dev)
    cnt = torch.zeros(n_queries * N, dtype=torch.int32, device=dev)
    G = len(q_index)
    if G == 0 or N == 0:
        return acc, cnt
    t = lambda xs: torch.tensor(xs, dtype=torch.int64, device=dev)  # noqa: E731
    grid = (G, triton.cdiv(N, TILE))
    _segsum_kernel[grid](docs.contiguous(), vals64.contiguous(), hits.to(torch.int32).contiguous(), t(row_start),
                         t(row_len), t(q_row_lo), t(q_row_hi), t(q_index), acc, cnt, N, TILE=TILE, BLOCK=BLOCK)
    return acc, cnt
