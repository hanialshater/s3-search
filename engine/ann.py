"""Approximate knn with cuVS (S9-1): CAGRA and IVF-PQ, built per segment on the GPU, filtered with cuVS's bitset
prefilter. Optional: without cuVS, CuPy or a CUDA index, ``available()`` is False and the caller runs exact knn
and says so (``knn.<field>.ran``).

Indexes are built lazily per (segment, field, method) on first use and kept on the segment's ``Part``; a merged
segment is a new ``Part``, so its index is built afresh there (rebuilt on merge). Distances are squared euclidean on
the L2-normalised vectors (the same order as cosine, and as inner product for unit vectors). The candidates' scores
are then the exact similarities, so a hit's ``_score`` is exact; only which docs are found is approximate.
"""
from __future__ import annotations

import numpy as np

METHODS = ("exact", "cagra", "ivf_pq")
SEARCH_PARAMS = {"cagra": {"itopk_size", "search_width"}, "ivf_pq": {"n_probes"}}


def available(b) -> bool:
    if b.name != "cuda":
        return False
    try:
        import cupy  # noqa: F401
        from cuvs.neighbors import cagra, ivf_pq  # noqa: F401
        return True
    except Exception:
        return False


def ran(b, method: str) -> str:
    """The method that runs for a requested one on this backend."""
    return method if method != "exact" and available(b) else "exact"


def _index(part, field, method):
    cache = part.__dict__.setdefault("_ann", {})
    key = (field, method)
    if key not in cache:
        import cupy as cp
        from cuvs.neighbors import cagra, ivf_pq
        data = cp.from_dlpack(part.vectors[field].contiguous())
        if method == "cagra":
            cache[key] = cagra.build(cagra.IndexParams(metric="sqeuclidean"), data)
        else:
            n_lists = max(1, min(1024, int(np.sqrt(max(part.n, 1)))))
            cache[key] = ivf_pq.build(ivf_pq.IndexParams(n_lists=n_lists, metric="sqeuclidean"), data)
    return cache[key]


def search_part(b, part, field, method, params, q, k, valid_local):
    """Candidate local doc ids ``int64[Q, k]`` (-1 pads) of one segment; ``valid_local`` (bool[n]) is the prefilter."""
    import cupy as cp
    import torch
    from cuvs.neighbors import cagra, filters, ivf_pq
    idx = _index(part, field, method)
    qc = cp.from_dlpack(q.to(torch.float32).contiguous())
    bits = cp.asarray(np.ascontiguousarray(b.pack_bits(valid_local), np.uint32))   # LSB first, 1 = included
    flt = filters.from_bitset(bits)
    if method == "cagra":
        itopk = max(int(params.get("itopk_size", 64)), -(-k // 32) * 32)
        sp = cagra.SearchParams(itopk_size=itopk, search_width=int(params.get("search_width", 1)))
        _, nb = cagra.search(sp, idx, qc, k, filter=flt)
    else:
        sp = ivf_pq.SearchParams(n_probes=int(params.get("n_probes", 20)))
        _, nb = ivf_pq.search(sp, idx, qc, k, filter=flt)
    nb = torch.from_dlpack(cp.asarray(nb).astype(cp.int64))
    return torch.where((nb >= 0) & (nb < part.n), nb, torch.full_like(nb, -1))


def candidates(b, st, field, method, params, q, k, valid):
    """bool[N]: the slots any segment's index returns among its ``valid`` docs, for one query vector ``q`` (f32[D]).
    Deleted and filtered-out docs are excluded by the prefilter, so each segment returns up to ``k`` valid docs."""
    m = b.full(st.N, False, "bool")
    q2 = q.reshape(1, -1)
    for part in st.parts:
        if not len(part.live_local):
            continue
        sl = b.asarray(part.live_slot, "int64")
        vl = b.full(part.n, False, "bool")
        vl[b.asarray(part.live_local, "int64")] = valid[sl]
        kk = min(int(k), int(b.count_nonzero(vl)))
        if kk == 0:
            continue
        loc = search_part(b, part, field, method, params, q2, kk, vl)[0]
        loc = loc[loc >= 0]
        slot_of = b.asarray(part.slot_of_local, "int64")
        s = slot_of[loc]
        m[s[s >= 0]] = True
    return m & valid
