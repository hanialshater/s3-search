"""S6-18: top-k over [Q, N] float32 scores without the int64 key: a float32 threshold, then a Triton compaction.

``TorchBackend.topk`` (S6-14) takes one ``torch.topk`` over an int64 key per element (the score's order-preserving bits,
then N - 1 - index), so the order is score descending, then index ascending. On large windows that 64-bit radix select
is the largest top-k cost. Here:

1. ``torch.topk`` of the float32 scores gives each row's k-th largest value ``v`` (its value only; which of several
   equal elements it picked does not matter).
2. A Triton pass counts, per row and block, the elements ``> v`` and ``== v``; a cumulative sum over the blocks gives
   each block's starting positions.
3. A second pass writes the selected indices: every element ``> v``, then the first ``k - count(> v)`` elements
   ``== v`` in index order. That is exactly the set the int64-key top-k selects.
4. The k selected (unordered) are ordered by the same key, score descending then index ascending, with ``torch.sort``
   of a [Q, k] int64 key: small next to the row.

So the indices and values equal the int64-key top-k's exactly, and run to run: no float atomics, no order-dependent
sums. Used on cuda where ``triton_gate.supported`` (Triton imports, at the tested version, on a GPU class that
passed GPU CI), for k at least ``MIN_K`` (small windows stay on the one-call path). On by default since S6-18c (the
founder's decision, 2026-11-14); ``S3_TRITON_TOPK=0`` turns it off.
"""
from __future__ import annotations

import os

from . import triton_gate

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

BLOCK = 2048
MIN_K = 256
WIDE_N = 65_536     # S6-27: on rows at least this wide the kernel serves any k (the facets' top buckets, k ~10)


def wanted(b, k: int, n: int = 0) -> bool:
    """On by default where the gate allows, for k at least ``MIN_K``, or any k on rows of at least ``WIDE_N`` (S6-27:
    the int64-key top-k's radix select over a wide row, such as a facet's counts, costs more than the threshold and
    two passes even for a small k)."""
    return ((k >= MIN_K or n >= WIDE_N) and k >= 1 and os.environ.get("S3_TRITON_TOPK", "1") != "0"
            and triton_gate.supported(b))


if triton is not None:
    @triton.jit
    def _count_kernel(s_ptr, v_ptr, gt_ptr, eq_ptr, N, n_blocks, BLOCK: tl.constexpr):
        q = tl.program_id(0)
        blk = tl.program_id(1)
        d = blk * BLOCK + tl.arange(0, BLOCK)
        inb = d < N
        x = tl.load(s_ptr + q.to(tl.int64) * N + d, mask=inb, other=float("-inf"))
        v = tl.load(v_ptr + q)
        gt = inb & (x > v)
        eq = inb & (x == v)
        tl.store(gt_ptr + q * n_blocks + blk, tl.sum(gt.to(tl.int32), axis=0))
        tl.store(eq_ptr + q * n_blocks + blk, tl.sum(eq.to(tl.int32), axis=0))

    @triton.jit
    def _write_kernel(s_ptr, v_ptr, gt_off_ptr, eq_off_ptr, n_gt_ptr, out_ptr, N, n_blocks, k, BLOCK: tl.constexpr):
        q = tl.program_id(0)
        blk = tl.program_id(1)
        d = blk * BLOCK + tl.arange(0, BLOCK)
        inb = d < N
        x = tl.load(s_ptr + q.to(tl.int64) * N + d, mask=inb, other=float("-inf"))
        v = tl.load(v_ptr + q)
        gt = inb & (x > v)
        eq = inb & (x == v)
        g0 = tl.load(gt_off_ptr + q * n_blocks + blk)       # elements > v before this block
        e0 = tl.load(eq_off_ptr + q * n_blocks + blk)       # elements == v before this block
        n_gt = tl.load(n_gt_ptr + q)                        # elements > v in the row
        rg = g0 + tl.cumsum(gt.to(tl.int32), axis=0) - 1    # each > v element's place among them
        re = e0 + tl.cumsum(eq.to(tl.int32), axis=0) - 1    # each == v element's place among them, in index order
        take_eq = eq & (re < k - n_gt)
        pos = tl.where(gt, rg, n_gt + re)
        tl.store(out_ptr + q.to(tl.int64) * k + pos, d.to(tl.int64), mask=gt | take_eq)


def topk(scores, k: int):
    """``(cols int32[Q, k], vals float32[Q, k])`` equal to ``TorchBackend.topk``'s key top-k of the float32 ``scores``
    (invalid entries already ``-inf``): score descending, then index ascending; ``-1`` where the value is ``-inf``."""
    import torch
    s = (scores.to(torch.float32) + 0.0).contiguous()      # -0.0 -> +0.0, as the key path
    Q, N = s.shape
    v = torch.topk(s, k, dim=1, sorted=False).values.min(dim=1).values.contiguous()   # each row's k-th value
    nb = triton.cdiv(N, BLOCK)
    gt = torch.empty((Q, nb), dtype=torch.int32, device=s.device)
    eq = torch.empty((Q, nb), dtype=torch.int32, device=s.device)
    _count_kernel[(Q, nb)](s, v, gt, eq, N, nb, BLOCK=BLOCK)
    gt_off = (torch.cumsum(gt, dim=1, dtype=torch.int32) - gt).contiguous()
    eq_off = (torch.cumsum(eq, dim=1, dtype=torch.int32) - eq).contiguous()
    n_gt = gt.sum(dim=1, dtype=torch.int32).contiguous()
    sel = torch.empty((Q, k), dtype=torch.int64, device=s.device)
    _write_kernel[(Q, nb)](s, v, gt_off, eq_off, n_gt, sel, N, nb, k, BLOCK=BLOCK)
    vals = s.gather(1, sel)
    bits = vals.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    shift = max(1, (N - 1).bit_length())
    key = (ordered << shift) | (N - 1 - sel)
    order = torch.sort(key, dim=1, descending=True).indices
    cols, vals = sel.gather(1, order), vals.gather(1, order)
    cols = cols.masked_fill(torch.isneginf(vals), -1)
    return cols.to(torch.int32), vals
