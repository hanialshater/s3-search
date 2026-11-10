"""Phrases over positions (S7-1): ``match_phrase`` with ``slop``, as Lucene matches it.

A phrase is a list of slots, each with its query offset and its alternative term ids (one term, or several for a
synonym). For a document and field, slot j contributes the values ``v = position - offset_j``. The phrase matches
where some choice of one value per slot spans at most ``slop``: ``max(v) - min(v) <= slop`` (Lucene's
``matchLength``; with slop 0 every slot sits at the same ``v``, i.e. the terms are adjacent and in order).

Everything runs on the backend: per slot, CSR expansion of the postings and their positions into sorted keys
``doc << 26 | v``; then, for each candidate start ``W`` (every key of every slot), a ``searchsorted`` per slot finds
the first value at or after ``W`` in the same document. Positions are in ``engine.build``'s layout
(``(field << 24) | position``).
"""
from __future__ import annotations

POS_BITS = 24
V_BITS = 26                     # v = position - offset + 2**24 fits in 26 bits
SHIFT = 1 << POS_BITS


def _slot_keys(b, st, field_i: int, term_ids, offset: int):
    """Sorted int64 keys ``doc << 26 | (position - offset + 2**24)`` of one slot (all its alternatives)."""
    if not term_ids:
        return b.zeros(0, "int64")
    pos_ptr, positions = st.positions()
    _, entry = b.csr_expand(st.term_ptr, list(term_ids), [0] * len(term_ids))
    docs = st.doc_ids[entry]
    doc_of, pidx = b.csr_expand(pos_ptr, entry, docs)
    code = b.astype(positions[pidx], "int64")
    keep = (code >> POS_BITS) == field_i
    p = code[keep] & (SHIFT - 1)
    keys = (doc_of[keep] << V_BITS) | (p - offset + SHIFT)
    return b.unique(keys) if len(term_ids) > 1 else keys      # one term: already sorted by (doc, position)


def phrase(b, st, field_i: int, slots, offsets, slop: int):
    """``(freq f32[N], match bool[N])``. ``slots``: per slot, its term ids (an empty slot: nothing matches).
    ``freq`` is the phrase frequency: with slop 0 the number of occurrences; with slop > 0, the sum of
    ``1 / (1 + matchLength)`` over matching starts (an approximation of Lucene's sloppy frequency)."""
    N = st.N
    freq = b.zeros(N, "float32")
    match = b.full(N, False, "bool")
    if not slots or any(len(s) == 0 for s in slots):
        return freq, match
    keys = [_slot_keys(b, st, field_i, s, o) for s, o in zip(slots, offsets)]
    if any(k.shape[0] == 0 for k in keys):
        return freq, match
    W = b.unique(b.concatenate(keys))
    ok = b.full(W.shape[0], True, "bool")
    end = W
    for k in keys:
        i = b.searchsorted(k, W)
        n = k.shape[0]
        ic = b.where(i < n, i, n - 1)
        kv = k[ic]
        ok = ok & (i < n) & ((kv >> V_BITS) == (W >> V_BITS)) & ((kv - W) <= slop)
        end = b.where(kv > end, kv, end)
    hits = W[ok]
    if hits.shape[0] == 0:
        return freq, match
    docs = hits >> V_BITS
    spread = b.astype(end[ok] - hits, "float32")
    freq = b.segment_sum(docs, 1.0 / (1.0 + spread), N)
    match[docs] = True
    return freq, match


def phrase_scores(b, st, field_i: int, freq, idf_sum: float, k1: float, bpar: float):
    """BM25 with the phrase frequency as tf and the sum of the slots' idf: as Lucene scores a phrase."""
    norm = (1.0 - bpar) + bpar * st.rel_len[:, field_i]
    norm = b.where(norm > 1e-6, norm, 1e-6)
    ptf = freq / norm
    return b.where(freq > 0, idf_sum * ptf * (k1 + 1.0) / (k1 + ptf), 0.0)
