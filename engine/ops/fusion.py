"""Reciprocal rank fusion (the prototype's fusion stage)."""
from __future__ import annotations


def rrf(b, ranked_ids, N: int, k: int = 60):
    """Fuse ranked lists ``ranked_ids`` (each ``int32[Q,K]``, -1 pads, best first) into dense buffers.

    Returns ``(rrf f32[Q,N], ranks list of f32[Q,N])``: the RRF score ``sum 1/(k + rank)`` (rank from 1) and, per list,
    each doc's 1-based rank (0 when absent).
    """
    Q = ranked_ids[0].shape[0] if ranked_ids else 0
    score = b.zeros(Q * N, "float32")
    ranks = []
    rows = b.arange(Q)
    for ids in ranked_ids:
        ids = b.astype(ids, "int64")
        K = ids.shape[1]
        ok = ids >= 0
        r = b.astype(b.arange(K), "float32") + 1.0
        r_full = r.reshape(1, K) + b.zeros((Q, K), "float32")
        flat = (rows.reshape(Q, 1) * N + ids)[ok]
        rb = b.zeros(Q * N, "float32")
        rb[flat] = r_full[ok]
        score = score + b.segment_sum(flat, 1.0 / (k + r_full[ok]), Q * N)
        ranks.append(rb.reshape(Q, N))
    return score.reshape(Q, N), ranks
