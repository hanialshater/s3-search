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

TILE = 1024        # S6-32: from the A100 sweep (arch-sweep1): 14% under (2048, 256, 4) at batch 64 without facets,
BLOCK = 128        # 4% under with facets, 3% and 6% under at batch 32; every shape gave the same bits
NUM_WARPS = 4


def _shape():
    """(TILE, BLOCK, num_warps), from S3_TRITON_FUSED_TILE, _BLOCK and _WARPS for tuning (S6-30). A tile is a doc
    range one program owns, so its size never changes the order of any key's adds, nor the bits."""
    return (int(os.environ.get("S3_TRITON_FUSED_TILE", TILE)), int(os.environ.get("S3_TRITON_FUSED_BLOCK", BLOCK)),
            int(os.environ.get("S3_TRITON_FUSED_WARPS", NUM_WARPS)))
ENV = "S3_TRITON_FUSED_LEXICAL"
HIST24_ENV = "S3_FUSED_HIST24"          # item 2 (opt-in): the top-k's first radix pass, histogrammed by the kernel


def hist24_on() -> bool:
    return os.environ.get(HIST24_ENV, "0") == "1"


def hist24_of(scores):
    """The first-digit histogram the kernel wrote for these score rows (``S3_FUSED_HIST24=1``), or None. It is taken
    (removed): the radix select clears it as it uses it."""
    h = getattr(scores, "_s3_hist24", None)               # set on the launch's own score-rows tensor
    if h is not None:
        del scores._s3_hist24
    return h


def mode(b) -> str | None:
    """``"kernel"`` (cuda, the gate passes, ``S3_TRITON_FUSED_LEXICAL=1``), ``"reference"``
    (``S3_TRITON_FUSED_LEXICAL=reference``, any device, for tests), or None: the unfused path. On by default since
    S6-26 (the founder's decision); ``S3_TRITON_FUSED_LEXICAL=0`` turns it off."""
    v = os.environ.get(ENV, "1")
    if v == "reference":
        return "reference"
    if v == "1" and triton is not None and triton_gate.supported(b):
        return "kernel"
    return None


if triton is not None:
    @triton.jit
    def _fused_kernel(docs_ptr, tf_ptr, norm_ptr, row_idf_ptr, row_start_ptr, row_len_ptr, q_row_lo_ptr, q_row_hi_ptr,
                      msm_ptr, boost_ptr, live_ptr, w_ptr, codes_ptr, fc_off_ptr, fc_nv_ptr,
                      acc_ptr, cnt_ptr, out_ptr, total_ptr, fc_ptr, N, c1, c2, imp_ptr, hist_ptr,
                      F: tl.constexpr, NFACET: tl.constexpr, TILE: tl.constexpr, BLOCK: tl.constexpr,
                      IMPACT: tl.constexpr, HIST24: tl.constexpr):
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
                if IMPACT:                                  # S6-33: the posting's weight, computed once per w
                    imp = tl.load(imp_ptr + e, mask=ok, other=0.0)
                    h = (imp > 0).to(tl.int32)              # imp > 0 exactly where ptf > 0 (checked at build)
                else:
                    ptf = tl.zeros([BLOCK], tl.float32)
                    for f in tl.static_range(F):
                        tfv = tl.load(tf_ptr + e.to(tl.int64) * F + f, mask=ok, other=0).to(tl.float32)
                        norm = tl.load(norm_ptr + d.to(tl.int64) * F + f, mask=ok, other=1.0)   # S6-28: per index
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
        val = tl.where(match, s32 * boost, float("-inf"))
        tl.store(out_ptr + base + dd, val, mask=inb_t)
        if HIST24:
            # Item 2: triton_topk._hist_kernel's first pass (SHIFT 24) over this tile's stored values, so the top-k
            # need not read the rows for it. The key is the order-preserving 32-bit pattern of val + 0.0 (-0.0 as
            # +0.0, as triton_topk.topk does before its select). Only non-zero bins are added: the matched scores'
            # top digits are a handful, and every unmatched -inf goes to bin 0.
            bits = val.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
            bits = tl.where(bits == 0x80000000, 0, bits)
            key = tl.where(bits >= 0x80000000, 0xFFFFFFFF - bits, bits + 0x80000000)
            digit = (key >> 24).to(tl.int32)
            hh = tl.histogram(tl.where(inb_t, digit, 0), 256)
            bins = tl.arange(0, 256)
            hh = hh - tl.where(bins == 0, tl.sum((inb_t == 0).to(tl.int32), axis=0), 0)
            tl.atomic_add(hist_ptr + g * 256 + bins, hh, mask=hh != 0)
        tl.atomic_add(total_ptr + g, tl.sum(match.to(tl.int32), axis=0))
        for j in tl.static_range(NFACET):
            code = tl.load(codes_ptr + j * N + dd, mask=inb_t, other=0)
            fo = tl.load(fc_off_ptr + j) + g.to(tl.int64) * tl.load(fc_nv_ptr + j)
            tl.atomic_add(fc_ptr + fo + code, tl.full([TILE], 1, tl.int32), mask=match)


def pack(parts, dtype, align):
    """``parts`` one after another in one array, each starting at a multiple of ``align`` elements (16 bytes), and
    their start offsets. Triton specialises a pointer argument on 16-byte alignment, so views at unaligned offsets
    would compile the kernel again for each new alignment (a ~0.5 s stall a time: the L4 hostpath profile)."""
    at, n = [], 0
    for x in parts:
        n += (-n) % align
        at.append(n)
        n += len(x)
    out = np.zeros(n, dtype)
    for a, x in zip(at, parts):
        out[a:a + len(x)] = x
    return out, at


if triton is not None:
    @triton.jit
    def _impact_kernel(docs_ptr, tf_ptr, norm_ptr, term_ptr, idf_ptr, w_ptr, imp_ptr, bad_ptr, P, c1, c2,
                       F: tl.constexpr, BLOCK: tl.constexpr):
        """Every posting's weight, with the fused kernel's own operations in its order; ``bad`` counts postings whose
        weight is not > 0 where ptf > 0 (or the reverse), where the impact path's hit test would differ."""
        e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = e < P
        d = tl.load(docs_ptr + e, mask=ok, other=0)
        t = tl.load(term_ptr + e, mask=ok, other=0)
        idf = tl.load(idf_ptr + t, mask=ok, other=0.0)
        ptf = tl.zeros([BLOCK], tl.float32)
        for f in tl.static_range(F):
            tfv = tl.load(tf_ptr + e.to(tl.int64) * F + f, mask=ok, other=0).to(tl.float32)
            norm = tl.load(norm_ptr + d.to(tl.int64) * F + f, mask=ok, other=1.0)
            x = tl.math.div_rn(tl.load(w_ptr + f) * tfv, norm)
            if f == 0:
                ptf = x
            else:
                ptf = ptf + x
        imp = tl.math.div_rn(idf * ptf * c1, c2 + ptf)
        tl.store(imp_ptr + e, imp, mask=ok)
        tl.atomic_add(bad_ptr, tl.sum((ok & ((imp > 0) != (ptf > 0))).to(tl.int32), axis=0))


def impacts(b, part, w, bb, k1, idf):
    """S6-33: the per-posting weights for one field-weight vector ``w`` (and b, k1, idf), float32 [P], kept on the part.
    They are computed by ``_impact_kernel`` with the fused kernel's exact operations, so the fused kernel can load
    them instead of recomputing: the same bits. None when the hit test would differ for some posting
    (``imp > 0`` against ``ptf > 0``), or when ``S3_TRITON_FUSED_IMPACTS`` is not "1" (off by default). The cost is
    4 bytes per posting per cached w."""
    import torch
    if os.environ.get("S3_TRITON_FUSED_IMPACTS", "0") != "1":
        return None
    docs32, norm = lean_inputs(b, part, bb)
    cache = part.__dict__.setdefault("_fused_impacts", {})
    key = (np.asarray(w, np.float32).tobytes(), np.asarray(bb, np.float32).tobytes(), float(k1), idf.data_ptr(),
           part.doc_ids.data_ptr(), norm.data_ptr())
    if key in cache:
        return cache[key]
    if "terms" not in cache or cache["terms"][0] != part.doc_ids.data_ptr():
        cache.clear()
        t = torch.as_tensor(np.asarray(part.term_of_posting, np.int32), device=b.device)
        cache["terms"] = (part.doc_ids.data_ptr(), t)
    P = int(part.doc_ids.shape[0])
    imp = torch.empty(P, dtype=torch.float32, device=b.device)
    bad = torch.zeros(1, dtype=torch.int32, device=b.device)
    w32 = b.asarray(np.asarray(w, np.float32), "float32")
    BLK = 1024
    _impact_kernel[(triton.cdiv(P, BLK),)](docs32, part.tf.contiguous(), norm, cache["terms"][1], idf.contiguous(), w32,
                                           imp, bad, P, float(np.float32(k1 + 1.0)), float(np.float32(k1)),
                                           F=int(part.tf.shape[1]), BLOCK=BLK, enable_fp_fusion=False)
    cache[key] = imp if int(bad.item()) == 0 else None
    return cache[key]


def lean_inputs(b, part, bb):
    """S6-28 (shortlist row 5), kept on the part and computed once per b vector:
    - its doc ids as int32, half the bytes of the postings reads and binary searches;
    - the per-(doc, field) BM25 norm table, the torch path's exact float32 operations done once per index rather than
      per posting: norm = (1 - b) + b * rel_len (with 1 - b computed first), then the 1e-6 floor.
    The same values reach the kernel, so the same bits."""
    import torch
    cache = part.__dict__.setdefault("_fused_lean", {})
    dk = (part.doc_ids.data_ptr(), tuple(part.doc_ids.shape))          # a relayout gives new arrays: new entries
    if cache.get("docs_key") != dk:
        cache.clear()
        cache["docs_key"] = dk
        cache["docs32"] = part.doc_ids.to(torch.int32).contiguous()
    key = (np.asarray(bb, np.float32).tobytes(), part.rel_len.data_ptr(), tuple(part.rel_len.shape))
    got = cache.get(key)
    if got is None:
        bb32 = b.asarray(np.asarray(bb, np.float32), "float32")
        norm = (1.0 - bb32) + bb32 * part.rel_len
        norm = torch.where(norm > 1e-6, norm, torch.tensor(1e-6, dtype=torch.float32)).contiguous()
        got = cache[key] = (cache["docs32"], norm)
    return got


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
    host_i, at = pack(ints, np.int64, 2)
    w32 = np.asarray(w, np.float32)
    bb32 = np.asarray(bb, np.float32)
    floats = [w32, bb32, np.float32(1.0) - bb32,                          # (1 - b) in float32, as the torch path
              np.asarray([float(x) for x in boost], np.float32)]
    host_f, fat = pack(floats, np.float32, 4)
    dev_i = torch.from_numpy(host_i).to(dev)
    dev_f = torch.from_numpy(host_f).to(dev)
    vi = [dev_i[a:a + len(x)] for a, x in zip(at, ints)]
    vf = [dev_f[a:a + len(x)] for a, x in zip(fat, floats)]
    return _launch(b, part, k1, idf, Q, R, vi, vf, live, facet_codes, sizes, offs, bb32, w32)


def _launch(b, part, k1, idf, Q, R, vi, vf, live, facet_codes, sizes, offs, bb, w):
    """The kernel on device inputs: ``vi`` the int64 views (row_start, row_len, q_lo, q_hi, msm, facet offsets, facet
    sizes, row_term), ``vf`` the float32 views (w, b, 1 - b, boost); ``bb`` and ``w`` on the host (the norm table's
    and the impacts' cache keys)."""
    import torch
    dev = b.device
    N = part.N
    F = int(part.tf.shape[1])
    rs, rl, qlo, qhi, msm_d, off_d, nv_d, rt = vi
    w_d, bb_d, omb_d, boost_d = vf
    nf = len(facet_codes)
    row_idf = (idf[rt] if R else torch.zeros(1, dtype=torch.float32, device=dev)).contiguous()
    docs32, norm = lean_inputs(b, part, bb)                 # S6-28: int32 doc ids, the per-index norm table
    acc = torch.empty(Q * N, dtype=torch.float64, device=dev)
    cnt = torch.empty(Q * N, dtype=torch.int32, device=dev)
    out = torch.empty(Q * N, dtype=torch.float32, device=dev)
    total = torch.zeros(Q, dtype=torch.int32, device=dev)
    fc = torch.zeros(max(1, sum(Q * s for s in sizes)), dtype=torch.int32, device=dev)
    codes = (torch.stack([c.to(torch.int32) for c in facet_codes], 0).contiguous() if nf
             else torch.zeros((1, 1), dtype=torch.int32, device=dev))
    c1 = float(np.float32(k1 + 1.0))
    c2 = float(np.float32(k1))
    tile, block, warps = _shape()
    grid = (Q, triton.cdiv(N, tile))
    imp = impacts(b, part, w, bb, k1, idf)                  # S6-33 (opt-in): the per-posting weights, or None
    h24 = hist24_on()
    hist = torch.zeros(Q * 256 if h24 else 1, dtype=torch.int32, device=dev)
    _fused_kernel[grid](docs32, part.tf.contiguous(), norm, row_idf, rs, rl, qlo, qhi, msm_d, boost_d,
                        live.contiguous().view(torch.uint8), w_d, codes, off_d, nv_d, acc, cnt, out, total, fc, N,
                        c1, c2, imp if imp is not None else out, hist, F=F, NFACET=nf, TILE=tile, BLOCK=block,
                        IMPACT=imp is not None, HIST24=h24, num_warps=warps, enable_fp_fusion=False)
    facets = [fc[o:o + Q * s].view(Q, s).to(torch.int64) for o, s in zip(offs, sizes)]
    rows = out.view(Q, N)
    if h24:
        rows._s3_hist24 = hist                              # hist24_of(rows) takes it; a derived tensor has none
    return rows, total.to(torch.int64), facets


def run_prepared(b, part, k1, idf, chunk, host_i, host_f, dev_i, dev_f, live, facet_codes):
    """One chunk of ``s3_hostpath.HostIndex.prepare``'s output. ``host_i`` and ``host_f`` are its int64 and float32
    blocks as numpy arrays; ``dev_i`` and ``dev_f`` are the same blocks on the device, or None off the kernel path.
    Gives ``run``'s outputs: the kernel reads the device views directly, and the reference rebuilds the terms from
    the host blocks."""
    F = int(part.tf.shape[1])
    at, ln, fa = chunk["int_at"], chunk["int_len"], chunk["float_at"]
    Q = len(chunk["positions"])
    hi = [host_i[a:a + n] for a, n in zip(at, ln)]
    qlo, qhi = hi[2], hi[3]
    R = int(qhi[-1]) if Q else 0
    sizes = [int(x) for x in hi[6]] if chunk["facets"] else []
    offs = [int(x) for x in hi[5]] if chunk["facets"] else []
    if mode(b) == "kernel":
        vi = [dev_i[a:a + n] for a, n in zip(at, ln)]
        vf = [dev_f[a:a + n] for a, n in zip(fa, (F, F, F, Q))]
        return _launch(b, part, k1, idf, Q, R, vi, vf, live, facet_codes, sizes, offs, host_f[fa[1]:fa[1] + F],
                       host_f[fa[0]:fa[0] + F])
    rt = hi[7]
    terms = [[int(t) for t in rt[qlo[g]:qhi[g]]] for g in range(Q)]
    w = host_f[fa[0]:fa[0] + F]
    bb = host_f[fa[1]:fa[1] + F]
    boost = host_f[fa[3]:fa[3] + Q]
    return reference(b, part, terms, w, bb, k1, idf, [int(m) for m in hi[4]], [float(x) for x in boost], live,
                     facet_codes, sizes)


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
