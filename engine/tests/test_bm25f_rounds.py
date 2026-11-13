"""S6-13: bm25f sums by rounds of distinct keys; bit for bit what the sort-based segment_sum gave, on every device."""
import numpy as np
import pytest

from engine import ops
from engine.backend import available_devices, get_backend
from engine.index_state import IndexState
from engine.ops import lexical as L
from referee import checks, loader


def _bm25f_segment_sum(b, st, terms, w, bb, k1, idf):
    """The previous bm25f (one segment_sum and one segment_count per group), as the reference."""
    Q, N = len(terms), st.N
    scores, nmatch = b.zeros(Q * N, "float32"), b.zeros(Q * N, "int32")
    owners = [q for q, ts in enumerate(terms) for _ in ts]
    rows = [t for ts in terms for t in ts]
    if not rows:
        return scores.reshape(Q, N), nmatch.reshape(Q, N)
    lens = [int(st.host_term_ptr[t + 1] - st.host_term_ptr[t]) for t in rows]
    groups, chunk, acc = [], [], 0
    for i, n in enumerate(lens):
        if acc + n > L.MAX_POSTINGS_PER_CHUNK and chunk:
            groups.append(chunk)
            chunk, acc = [], 0
        chunk.append(i)
        acc += n
    groups.append(chunk)
    w, bb = b.asarray(w, "float32"), b.asarray(bb, "float32")
    for g in groups:
        q_of, entry = b.csr_expand(st.term_ptr, [rows[i] for i in g], [owners[i] for i in g])
        t_rep = b.csr_expand(st.term_ptr, [rows[i] for i in g], [rows[i] for i in g])[0]
        docs = st.doc_ids[entry]
        tf = b.astype(st.tf[entry], "float32")
        norm = (1.0 - bb) + bb * st.rel_len[docs]
        norm = b.where(norm > 1e-6, norm, 1e-6)
        ptf = L.sum_lr(w * tf / norm, 1)
        imp = idf[t_rep] * ptf * (k1 + 1.0) / (k1 + ptf)
        keys = q_of * N + docs
        scores = scores + b.segment_sum(keys, imp, Q * N)
        nmatch = nmatch + b.segment_count(keys[ptf > 0], Q * N)
    return scores.reshape(Q, N), nmatch.reshape(Q, N)


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("accumulate", ["float64", "float32"])
@pytest.mark.parametrize("chunk", [None, 40])                 # one group, and groups that split queries' terms
def test_rounds_equal_segment_sum_bit_for_bit(device, accumulate, chunk, monkeypatch):
    from engine import api
    if chunk is not None:
        monkeypatch.setattr(L, "MAX_POSTINGS_PER_CHUNK", chunk)
    ds = loader.load_dataset("toy")
    b = get_backend(device, accumulate)
    ix = api.build_index(ds.catalogue, ds.mapping, device=device, accumulate=accumulate)
    st = ix._st
    part = st.parts[0]
    texts = list(dict.fromkeys(list(ds.fit["query"]) + list(ds.test_queries["query"])))[:60]
    from engine.analysis import analyze
    terms = [sorted({part.vocab[t] for t in analyze(x) if t in part.vocab}) for x in texts] + [[]]
    F = len(st.text_fields)
    w = np.linspace(0.5, 3.0, F).astype(np.float32)
    bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)
    got = L.bm25f(b, part, terms, w, bb, 1.2, part.idf)
    want = _bm25f_segment_sum(b, part, terms, w, bb, 1.2, part.idf)
    assert np.array_equal(b.to_numpy(got[0]), b.to_numpy(want[0]))
    assert np.array_equal(b.to_numpy(got[1]), b.to_numpy(want[1]))
    assert b.to_numpy(got[1]).max() > 1                         # some docs match several terms: rounds were summed
