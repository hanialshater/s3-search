"""BM25F, exhaustive over the postings of the query terms (the prototype's ``Index.score_into``)."""
from __future__ import annotations

MAX_POSTINGS_PER_CHUNK = 30_000_000


def bm25f(b, st, terms, w, bb, k1: float, idf):
    """Score every document for a batch of queries.

    ``terms``: per query, its distinct term ids. ``w``, ``bb``: float32[F] field weights and b per field (0 weight:
    field not queried). ``idf``: float32[V] (global) or a per-field column. Returns ``(scores f32[Q,N],
    nmatch i32[Q,N])``; nmatch counts the query's terms found in a queried field.

        pseudo_tf = sum_f w_f * tf_f / (1 - b_f + b_f * len_f / avg_f)
        score     = sum_t idf_t * pseudo_tf * (k1 + 1) / (k1 + pseudo_tf)
    """
    Q, N = len(terms), st.N
    scores = b.zeros(Q * N, "float32")
    nmatch = b.zeros(Q * N, "int32")
    owners = [q for q, ts in enumerate(terms) for _ in ts]
    rows = [t for ts in terms for t in ts]
    if not rows or N == 0:
        return scores.reshape(Q, N), nmatch.reshape(Q, N)
    host_ptr = st.host_term_ptr
    lens = [int(host_ptr[t + 1] - host_ptr[t]) for t in rows]
    chunk, start, acc = [], 0, 0
    groups = []
    for i, n in enumerate(lens):
        if acc + n > MAX_POSTINGS_PER_CHUNK and chunk:
            groups.append(chunk)
            chunk, acc = [], 0
        chunk.append(i)
        acc += n
    groups.append(chunk)
    w = b.asarray(w, "float32")
    bb = b.asarray(bb, "float32")
    for g in groups:
        q_of, entry = b.csr_expand(st.term_ptr, [rows[i] for i in g], [owners[i] for i in g])
        t_rep = b.csr_expand(st.term_ptr, [rows[i] for i in g], [rows[i] for i in g])[0]
        docs = st.doc_ids[entry]
        tf = b.astype(st.tf[entry], "float32")
        norm = (1.0 - bb) + bb * st.rel_len[docs]
        norm = b.where(norm > 1e-6, norm, 1e-6)
        ptf = b.sum(w * tf / norm, axis=1)
        imp = idf[t_rep] * ptf * (k1 + 1.0) / (k1 + ptf)
        keys = q_of * N + docs
        scores = scores + b.segment_sum(keys, imp, Q * N)
        hit = ptf > 0
        nmatch = nmatch + b.segment_count(keys[hit], Q * N)
    return scores.reshape(Q, N), nmatch.reshape(Q, N)
