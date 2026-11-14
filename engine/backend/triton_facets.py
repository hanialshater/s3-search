"""S6-18: the terms-facet histogram (``bincount_rows``) as one Triton kernel.

The torch path finds every matched (query, doc) with ``nonzero`` over the [Q, N] mask, then ``bincount``s
``q * n + code``. Both read the device to size their results (two host syncs), and the nonzero writes every match
out. The kernel reads the mask and the codes once: each program takes one query and one block of docs, and adds 1 to
``counts[q, code]`` for each matched doc with an integer atomic add. Integer adds give the same total in any order, so
the counts equal the torch path's exactly (summed in int32, at most N, returned as int64 as ``bincount`` does), and
run to run.

Used on cuda when ``S3_TRITON_FACETS`` is "1" and ``triton_gate.supported`` (Triton imports, at the tested version, on a GPU class that passed); the torch path stays the default until it is
measured faster on the A100.
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

BLOCK = 1024


def wanted(b) -> bool:
    return os.environ.get("S3_TRITON_FACETS", "0") == "1" and triton_gate.supported(b)


if triton is not None:
    @triton.jit
    def _hist_kernel(mask_ptr, codes_ptr, out_ptr, N, n_values, mask_stride, BLOCK: tl.constexpr):
        q = tl.program_id(0)
        blk = tl.program_id(1)
        d = blk * BLOCK + tl.arange(0, BLOCK)
        inb = d < N
        m = tl.load(mask_ptr + q.to(tl.int64) * mask_stride + d, mask=inb, other=0)
        c = tl.load(codes_ptr + d, mask=inb, other=0).to(tl.int64)
        take = inb & (m != 0)
        tl.atomic_add(out_ptr + q.to(tl.int64) * n_values + c, tl.full([BLOCK], 1, tl.int32), mask=take)


def bincount_rows(mask, codes, n_values: int):
    """``int64[Q, n_values]``: per row of ``mask`` (bool [Q, N]), the count of each code among its true docs."""
    import torch
    Q, N = mask.shape
    out = torch.zeros(Q * n_values, dtype=torch.int32, device=mask.device)  # int32 atomics; counts <= N
    if Q == 0 or N == 0:
        return out.view(Q, n_values).to(torch.int64)
    m = mask.contiguous().view(torch.uint8)
    grid = (Q, triton.cdiv(N, BLOCK))
    _hist_kernel[grid](m, codes.contiguous(), out, N, n_values, m.stride(0), BLOCK=BLOCK)
    return out.view(Q, n_values).to(torch.int64)
