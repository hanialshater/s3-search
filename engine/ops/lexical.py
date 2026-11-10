"""BM25F, exhaustive over the postings of the query terms (the prototype's ``Index.score_into``)."""
from __future__ import annotations

MAX_POSTINGS_PER_CHUNK = 30_000_000


def sum_lr(x, axis: int):
    """The sum over a short axis (fields, groups), added left to right with elementwise adds. A library's
    reduction kernel may order a float32 sum differently by version and device (torch-cpu 2.14 and 2.15 differ);
    elementwise IEEE adds in a fixed order give the same bits on every backend."""
    n = x.shape[axis]
    pick = (lambda i: x[:, i]) if axis == 1 else (lambda i: x[i])
    out = pick(0)
    for i in range(1, n):
        out = out + pick(i)
    return out


def bm25f_groups(b, st, groups, modes, w, bb, k1: float, idf, group_idf=None):
    """Score clause groups (S7-2, S7-6): each group is a list of alternative term ids that count as one clause.

    ``modes[g]``: ``"sum"``: the BM25F scores of the group's terms add (expansions of a fuzzy term); ``"synonym"``:
    the group is one term (Lucene's SynonymQuery): pseudo-tfs add before saturation, with ``group_idf[g]`` (the
    caller passes the group's lowest idf, i.e. the highest df); ``"const"``: 1.0 where any term is present (a prefix). ``idf``: float32[V] or a field column.
    Returns ``(scores f32[G,N], present bool[G,N])``: a doc's score per group, and whether the group matched."""
    G, N = len(groups), st.N
    scores = b.zeros(G * N, "float32")
    present = b.zeros(G * N, "int32")
    owners = [g for g, ts in enumerate(groups) for _ in ts]
    rows = [t for ts in groups for t in ts]
    if not rows or N == 0:
        return scores.reshape(G, N), present.reshape(G, N) > 0
    w, bb = b.asarray(w, "float32"), b.asarray(bb, "float32")
    q_of, entry = b.csr_expand(st.term_ptr, rows, owners)
    t_rep = b.csr_expand(st.term_ptr, rows, rows)[0]
    if getattr(st, "post_live", None) is not None:          # a segment of a multi-segment index: dead copies out
        keep = st.post_live[entry]
        q_of, entry, t_rep = q_of[keep], entry[keep], t_rep[keep]
    docs = st.doc_ids[entry]
    tf = b.astype(st.tf[entry], "float32")
    norm = (1.0 - bb) + bb * st.rel_len[docs]
    norm = b.where(norm > 1e-6, norm, 1e-6)
    ptf = sum_lr(w * tf / norm, 1)
    keys = q_of * N + docs
    hit = ptf > 0
    present = present + b.segment_count(keys[hit], G * N)
    mode_of = b.asarray([{"sum": 0, "synonym": 1, "const": 2}[m] for m in modes], "int64")[q_of]
    s_sum = (mode_of == 0) & hit
    if "sum" in modes:
        imp = idf[t_rep] * ptf * (k1 + 1.0) / (k1 + ptf)
        scores = scores + b.segment_sum(keys[s_sum], imp[s_sum], G * N)
    s_syn = (mode_of == 1) & hit
    if "synonym" in modes:
        P = b.segment_sum(keys[s_syn], ptf[s_syn], G * N)
        gi = b.asarray(group_idf, "float32").reshape(G, 1) + b.zeros((G, N), "float32")
        syn = b.where(P > 0, gi.reshape(-1) * P * (k1 + 1.0) / (k1 + P), 0.0)
        scores = scores + syn
    present = present.reshape(G, N) > 0
    const = b.asarray([m == "const" for m in modes], "bool").reshape(G, 1)
    scores = b.where(const & present, 1.0, scores.reshape(G, N))
    return scores, present


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
        if getattr(st, "post_live", None) is not None:      # a segment of a multi-segment index: dead copies out
            keep = st.post_live[entry]
            q_of, entry, t_rep = q_of[keep], entry[keep], t_rep[keep]
        docs = st.doc_ids[entry]
        tf = b.astype(st.tf[entry], "float32")
        norm = (1.0 - bb) + bb * st.rel_len[docs]
        norm = b.where(norm > 1e-6, norm, 1e-6)
        ptf = sum_lr(w * tf / norm, 1)
        imp = idf[t_rep] * ptf * (k1 + 1.0) / (k1 + ptf)
        keys = q_of * N + docs
        scores = scores + b.segment_sum(keys, imp, Q * N)
        hit = ptf > 0
        nmatch = nmatch + b.segment_count(keys[hit], Q * N)
    return scores.reshape(Q, N), nmatch.reshape(Q, N)


def bm25f_multi(b, parts, terms_per_part, w, bb, k1: float, idfs, N: int):
    """``bm25f`` over several segments (S8-2c): every segment's entries (postings of live docs, mapped to slots)
    are gathered first and summed by one ``segment_sum`` into one buffer, so a query costs one pass over the slot
    space however many segments there are. A doc lives in one segment, so its addends keep their order and the
    scores equal a single segment's."""
    Q = len(terms_per_part[0]) if terms_per_part else 0
    w, bb = b.asarray(w, "float32"), b.asarray(bb, "float32")
    keys_all, imp_all, hit_all = [], [], []
    for st, terms, idf in zip(parts, terms_per_part, idfs):
        owners = [q for q, ts in enumerate(terms) for _ in ts]
        rows = [t for ts in terms for t in ts]
        if not rows:
            continue
        q_of, entry = b.csr_expand(st.term_ptr, rows, owners)
        t_rep = b.csr_expand(st.term_ptr, rows, rows)[0]
        if getattr(st, "post_live", None) is not None:
            keep = st.post_live[entry]
            q_of, entry, t_rep = q_of[keep], entry[keep], t_rep[keep]
        docs = st.doc_ids[entry]
        tf = b.astype(st.tf[entry], "float32")
        norm = (1.0 - bb) + bb * st.rel_len[docs]
        norm = b.where(norm > 1e-6, norm, 1e-6)
        ptf = sum_lr(w * tf / norm, 1)
        keys_all.append(q_of * N + docs)
        imp_all.append(idf[t_rep] * ptf * (k1 + 1.0) / (k1 + ptf))
        hit_all.append(ptf > 0)
    if not keys_all or N == 0:
        return b.zeros((Q, N), "float32"), b.zeros((Q, N), "int32")
    keys = b.concatenate(keys_all) if len(keys_all) > 1 else keys_all[0]
    imp = b.concatenate(imp_all) if len(imp_all) > 1 else imp_all[0]
    hit = b.concatenate(hit_all) if len(hit_all) > 1 else hit_all[0]
    scores = b.segment_sum(keys, imp, Q * N)
    nmatch = b.segment_count(keys[hit], Q * N)
    return scores.reshape(Q, N), nmatch.reshape(Q, N)
