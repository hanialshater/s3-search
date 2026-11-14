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


if triton is not None:
    @triton.jit
    def _hist_kernel(s_ptr, prefix_ptr, hist_ptr, N, SHIFT: tl.constexpr, BLOCK: tl.constexpr):
        """One radix pass: per row, a 256-bin histogram of the 8-bit digit at ``SHIFT`` of the keys whose higher digits
        equal the row's prefix (every key on the first pass). A key is the score's order-preserving 32-bit pattern,
        held in int64: larger scores, larger keys."""
        q = tl.program_id(0)
        blk = tl.program_id(1)
        d = blk * BLOCK + tl.arange(0, BLOCK)
        inb = d < N
        x = tl.load(s_ptr + q.to(tl.int64) * N + d, mask=inb, other=0.0)
        b = x.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        key = tl.where(b >= 0x80000000, 0xFFFFFFFF - b, b + 0x80000000)
        if SHIFT == 24:
            m = inb
        else:
            prefix = tl.load(prefix_ptr + q)
            m = inb & ((key >> (SHIFT + 8)) == (prefix >> (SHIFT + 8)))
        digit = ((key >> SHIFT) & 255).to(tl.int32)
        h = tl.histogram(tl.where(m, digit, 0), 256)
        miss = tl.sum((m == 0).to(tl.int32), axis=0)
        bins = tl.arange(0, 256)
        h = h - tl.where(bins == 0, miss, 0)
        tl.atomic_add(hist_ptr + q * 256 + bins, h)

    @triton.jit
    def _select_kernel(hist_ptr, prefix_ptr, krem_ptr, SHIFT: tl.constexpr):
        """Per row: the digit the k-th largest key has at ``SHIFT``; the row's prefix gains it, and the remaining k
        drops by the count in the bins above it. The histogram is cleared for the next pass."""
        q = tl.program_id(0)
        i = tl.arange(0, 256)
        h_rev = tl.load(hist_ptr + q * 256 + (255 - i))          # bins from the top digit down
        cum = tl.cumsum(h_rev, axis=0)
        k = tl.load(krem_ptr + q)
        first = tl.min(tl.where(cum >= k, i, 256), axis=0)
        above = tl.sum(tl.where(i < first, h_rev, 0), axis=0)
        digit = (255 - first).to(tl.int64)
        tl.store(krem_ptr + q, k - above)
        tl.store(prefix_ptr + q, tl.load(prefix_ptr + q) | (digit << SHIFT))
        tl.store(hist_ptr + q * 256 + i, tl.zeros([256], tl.int32))


def kth_value(s, k: int):
    """``float32[Q]``: each row's k-th largest value of the float32 ``s`` [Q, N], multiplicity counted. It is the value
    ``torch.topk(s, k).values.min(dim=1)`` gives (S6-29). Four radix passes of 8 bits over the scores'
    order-preserving bits, each a block histogram (one atomic add per bin and block) then a per-row digit select.
    Exact: integer counts on the bits, no float comparisons."""
    import torch
    Q, N = s.shape
    dev = s.device
    hist = torch.zeros(Q * 256, dtype=torch.int32, device=dev)
    prefix = torch.zeros(Q, dtype=torch.int64, device=dev)
    krem = torch.full((Q,), int(k), dtype=torch.int32, device=dev)
    nb = triton.cdiv(N, BLOCK)
    for shift in (24, 16, 8, 0):
        _hist_kernel[(Q, nb)](s, prefix, hist, N, SHIFT=shift, BLOCK=BLOCK)
        _select_kernel[(Q,)](hist, prefix, krem, SHIFT=shift)
    bits = torch.where(prefix >= 0x80000000, prefix - 0x80000000, 0xFFFFFFFF - prefix)
    bits = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)          # as a signed 32-bit pattern
    return bits.to(torch.int32).view(torch.float32).contiguous()


def topk(scores, k: int):
    """``(cols int32[Q, k], vals float32[Q, k])`` equal to ``TorchBackend.topk``'s key top-k of the float32 ``scores``
    (invalid entries already ``-inf``): score descending, then index ascending; ``-1`` where the value is ``-inf``."""
    import torch
    s = (scores.to(torch.float32) + 0.0).contiguous()      # -0.0 -> +0.0, as the key path
    Q, N = s.shape
    if os.environ.get("S3_TRITON_TOPK_THRESHOLD", "radix") == "torch":                # S6-29's comparison switch
        v = torch.topk(s, k, dim=1, sorted=False).values.min(dim=1).values.contiguous()
    else:
        v = kth_value(s, k)                                 # each row's k-th value (S6-29: a radix select)
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
