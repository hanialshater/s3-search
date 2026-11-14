"""S6-22: the fused lexical path. One pass per (query, tile of docs) gives a bare lexical query's page input, hits total
and terms-facet counts, with the same bits as today's path (docs/proposals/2026-11-14-s6-22-sparse-fused-lexical.md).

Today's path for a batch of bare lexical roots is several stages:
- ``ops.bm25f``: csr_expand, gathers, the float32 weights, the Triton segmented sum into a float64 accumulator, a
  float32 row and a match count;
- the executor: msm, boost, the live mask, the selects;
- the page: a masked top-k and count_nonzero;
- the facets: bincount_rows.

Each stage reads and writes [Q, N] or [P, F] arrays. Here one Triton program per (query, 2,048-doc tile):

1. For each of the query's terms in order, it finds the term's postings inside the tile (two binary searches over the
   term's sorted doc ids), computes each posting's weight in float32 with the torch path's exact operations, and adds
   it to the tile's float64 accumulator in posting order, with a barrier between terms. The operations are:
   - norm = (1 - b_f) + b_f * rel_len, then the 1e-6 floor;
   - (w_f * tf_f) / norm, summed over the fields left to right;
   - idf * ptf * (k1 + 1) / (k1 + ptf).
   FMA contraction is off and the divisions round to nearest, so each float32 op rounds as torch's does. The sum is
   ``_segsum_kernel``'s: the same additions in the same order, so the same bits.
2. It reads the tile back and computes match = (count >= msm) & live.
3. It writes one float32 row: score * boost where matched, else -inf. That is the masked top-k's input, the same
   values today's ``b.topk(score, k, match)`` builds.
4. It adds the tile's matches to the query's total, and each matched doc's facet code to its counts, with integer
   atomic adds. Integers give the same result in any order.

The accumulators live in a [Q, N] scratch that each program zeroes for its own tile only, so they stay in L2 while
the program runs. There is no full-length zeroing pass.

``reference`` computes the same outputs with backend operations (any device). It is what the executor uses when
``S3_TRITON_FUSED_LEXICAL=reference``, and what the kernel's bitwise tests compare against: it is today's path's own
operations, in today's order.
"""
from __future__ import annotations

import os

import numpy as np

from . import triton_gate

try:
    import triton
    import triton.language as tl
except Exception:  # noqa: BLE001
    triton = None
    tl = None

TILE = 2048
BLOCK = 256
ENV = "S3_TRITON_FUSED_LEXICAL"


def mode(b) -> str | None:
    """``"kernel"`` (cuda, the gate passes, ``S3_TRITON_FUSED_LEXICAL=1``), ``"reference"``
    (``S3_TRITON_FUSED_LEXICAL=reference``, any device, for tests), or None: today's path. Off by default."""
    v = os.environ.get(ENV, "0")
    if v == "reference":
        return "reference"
    if v == "1" and triton is not None and triton_gate.supported(b):
        return "kernel"
    return None


if triton is not None:
    @triton.jit
    def _fused_kernel(docs_ptr, tf_ptr, rel_ptr, row_idf_ptr, row_start_ptr, row_len_ptr, q_row_lo_ptr, q_row_hi_ptr,
                      msm_ptr, boost_ptr, live_ptr, w_ptr, omb_ptr, bb_ptr, codes_ptr, fc_off_ptr, fc_nv_ptr,
                      acc_ptr, cnt_ptr, out_ptr, total_ptr, fc_ptr, N, c1, c2,
                      F: tl.constexpr, NFACET: tl.constexpr, TILE: tl.constexpr, BLOCK: tl.constexpr):
        g = tl.program_id(0)
        tile = tl.program_id(1)
        d0 = tile * TILE
        d1 = d0 + TILE
        base = g.to(tl.int64) * N
        tl_idx = tl.arange(0, TILE)
        dd = d0 + tl_idx
        inb_t = dd < N
        tl.store(acc_ptr + base + dd, tl.zeros([TILE], tl.float64), mask=inb_t)    # this tile only
        tl.store(cnt_ptr + base + dd, tl.zeros([TILE], tl.int32), mask=inb_t)
        tl.debug_barrier()
        r_lo = tl.load(q_row_lo_ptr + g)
        r_hi = tl.load(q_row_hi_ptr + g)
        lanes = tl.arange(0, BLOCK)
        for r in range(r_lo, r_hi):
            start = tl.load(row_start_ptr + r)
            n = tl.load(row_len_ptr + r)
            idf = tl.load(row_idf_ptr + r)
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
                ptf = tl.zeros([BLOCK], tl.float32)
                for f in tl.static_range(F):
                    tfv = tl.load(tf_ptr + e.to(tl.int64) * F + f, mask=ok, other=0).to(tl.float32)
                    rel = tl.load(rel_ptr + d.to(tl.int64) * F + f, mask=ok, other=1.0)
                    norm = tl.load(omb_ptr + f) + tl.load(bb_ptr + f) * rel
                    norm = tl.where(norm > 1e-6, norm, 1e-6)
                    x = tl.math.div_rn(tl.load(w_ptr + f) * tfv, norm)
                    if f == 0:
                        ptf = x
                    else:
                        ptf = ptf + x
                imp = tl.math.div_rn(idf * ptf * c1, c2 + ptf)
                h = (ptf > 0).to(tl.int32)
                off = base + d
                cur = tl.load(acc_ptr + off, mask=ok, other=0.0)
                tl.store(acc_ptr + off, cur + imp.to(tl.float64), mask=ok)
                c = tl.load(cnt_ptr + off, mask=ok, other=0)
                tl.store(cnt_ptr + off, c + h, mask=ok)
            tl.debug_barrier()                              # this term's adds land before the next term's
        acc = tl.load(acc_ptr + base + dd, mask=inb_t, other=0.0)
        cnt = tl.load(cnt_ptr + base + dd, mask=inb_t, other=0)
        live = tl.load(live_ptr + dd, mask=inb_t, other=0)
        msm = tl.load(msm_ptr + g)
        boost = tl.load(boost_ptr + g)
        match = inb_t & (cnt >= msm) & (live != 0)
        s32 = acc.to(tl.float32)
        tl.store(out_ptr + base + dd, tl.where(match, s32 * boost, float("-inf")), mask=inb_t)
        tl.atomic_add(total_ptr + g, tl.sum(match.to(tl.int32), axis=0))
        for j in tl.static_range(NFACET):
            code = tl.load(codes_ptr + j * N + dd, mask=inb_t, other=0)
            fo = tl.load(fc_off_ptr + j) + g.to(tl.int64) * tl.load(fc_nv_ptr + j)
            tl.atomic_add(fc_ptr + fo + code, tl.full([TILE], 1, tl.int32), mask=match)


def _rows(part, terms):
    """Per query its term rows in order, as the kernel's flat arrays; ``None`` if a query has no rows."""
    ptr = part.host_term_ptr
    row_start, row_len, row_term, q_lo, q_hi = [], [], [], [], []
    for ts in terms:
        q_lo.append(len(row_start))
        for t in ts:
            row_start.append(int(ptr[t]))
            row_len.append(int(ptr[t + 1] - ptr[t]))
            row_term.append(int(t))
        q_hi.append(len(row_start))
    return row_start, row_len, row_term, q_lo, q_hi


def run(b, part, terms, w, bb, k1: float, idf, msm, boost, live, facet_codes=(), facet_sizes=()):
    """For a chunk of bare lexical queries on one segment (``part.identity``), return the arrays below.

    The inputs: ``terms`` (per query, its term ids in query order), ``msm`` and ``boost`` (per query), ``live``
    (bool [N]), and ``facet_codes`` (int [N] per terms facet, 0 = missing) with ``facet_sizes`` (each facet's number
    of codes).

    The outputs:
    - ``scores`` float32 [Q, N]: score * boost where matched, else -inf;
    - ``totals`` int64 [Q];
    - ``facets``: one int64 [Q, size] per facet.
    """
    if mode(b) == "kernel":
        return _run_kernel(b, part, terms, w, bb, k1, idf, msm, boost, live, facet_codes, facet_sizes)
    return reference(b, part, terms, w, bb, k1, idf, msm, boost, live, facet_codes, facet_sizes)


def _run_kernel(b, part, terms, w, bb, k1, idf, msm, boost, live, facet_codes, facet_sizes):
    import torch
    dev = b.device
    Q, N = len(terms), part.N
    F = int(part.tf.shape[1])
    row_start, row_len, row_term, q_lo, q_hi = _rows(part, terms)
    nf = len(facet_codes)
    sizes = [int(s) for s in facet_sizes]
    offs = [int(x) for x in np.cumsum([0] + [Q * s for s in sizes])[:-1]] if nf else []
    R = len(row_start)
    # S6-24: every small per-call array in two host-to-device copies (one int64, one float32), not one per array
    ints = [row_start or [0], row_len or [0], q_lo, q_hi, [int(m) for m in msm], offs or [0], sizes or [1],
            row_term or [0]]
    lens = [len(x) for x in ints]
    host_i = np.fromiter((v for x in ints for v in x), np.int64, sum(lens))
    w32 = np.asarray(w, np.float32)
    bb32 = np.asarray(bb, np.float32)
    host_f = np.concatenate([w32, bb32, np.float32(1.0) - bb32,          # (1 - b) in float32, as the torch path
                             np.asarray([float(x) for x in boost], np.float32)])
    dev_i = torch.from_numpy(host_i).to(dev)
    dev_f = torch.from_numpy(host_f).to(dev)
    at = np.cumsum([0] + lens)
    rs, rl, qlo, qhi, msm_d, off_d, nv_d, rt = (dev_i[at[k]:at[k + 1]] for k in range(len(ints)))
    w_d, bb_d, omb_d, boost_d = dev_f[:F], dev_f[F:2 * F], dev_f[2 * F:3 * F], dev_f[3 * F:]
    row_idf = (idf[rt] if R else torch.zeros(1, dtype=torch.float32, device=dev)).contiguous()
    acc = torch.empty(Q * N, dtype=torch.float64, device=dev)
    cnt = torch.empty(Q * N, dtype=torch.int32, device=dev)
    out = torch.empty(Q * N, dtype=torch.float32, device=dev)
    total = torch.zeros(Q, dtype=torch.int32, device=dev)
    fc = torch.zeros(max(1, sum(Q * s for s in sizes)), dtype=torch.int32, device=dev)
    codes = (torch.stack([c.to(torch.int32) for c in facet_codes], 0).contiguous() if nf
             else torch.zeros((1, 1), dtype=torch.int32, device=dev))
    c1 = float(np.float32(k1 + 1.0))
    c2 = float(np.float32(k1))
    grid = (Q, triton.cdiv(N, TILE))
    _fused_kernel[grid](part.doc_ids, part.tf, part.rel_len, row_idf, rs, rl, qlo, qhi, msm_d, boost_d,
                        live.view(torch.uint8), w_d, omb_d, bb_d, codes, off_d, nv_d, acc, cnt, out, total, fc, N,
                        c1, c2, F=F, NFACET=nf, TILE=TILE, BLOCK=BLOCK, enable_fp_fusion=False)
    facets = [fc[o:o + Q * s].view(Q, s).to(torch.int64) for o, s in zip(offs, sizes)]
    return out.view(Q, N), total.to(torch.int64), facets


def reference(b, part, terms, w, bb, k1, idf, msm, boost, live, facet_codes=(), facet_sizes=()):
    """The same outputs from today's operations, in today's order (``ops.bm25f``, then the executor's msm, boost, live
    and selects, then the masked top-k's -inf, count_nonzero and bincount_rows)."""
    from ..ops import lexical as L
    Q = len(terms)
    S, NM = L.bm25f(b, part, terms, w, bb, k1, idf)
    msm_a = b.asarray(np.asarray([int(m) for m in msm], np.int64).reshape(Q, 1), "int64")
    boost_a = b.asarray(np.asarray([float(x) for x in boost], np.float32).reshape(Q, 1), "float32")
    M = NM >= msm_a
    raw = b.where(M, S * boost_a, 0.0)
    match = M & live.reshape(1, -1)
    score = b.where(match, raw, 0.0)
    scores = b.where(match, score, float("-inf"))
    totals = b.astype(b.count_nonzero(match, axis=1), "int64")
    facets = [b.bincount_rows(codes, match, int(n)) for codes, n in zip(facet_codes, facet_sizes)]
    return scores, totals, facets
