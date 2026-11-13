"""S6-16: BM25F is batch-invariant bit for bit even when a batch's postings are cut into chunks.

plat-m8 saw 1 of 1,516 depth-2,000 responses differ from its unloaded replay. The cause was in the engine: bm25f cut
a batch's postings into chunks of MAX_POSTINGS_PER_CHUNK at any term, so in a full batch a query's terms could be
split over two chunks, and the sum of its float32 partials could round differently from the query alone (one float64
sum of all its terms). Chunks now end only between queries.
"""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend
from engine.ops import lexical as L
from referee import loader


@pytest.mark.parametrize("device", available_devices())
@pytest.mark.parametrize("limit", [10**9, 500, 150, 70, 25])
def test_batched_scores_equal_alone_bit_for_bit_whatever_the_chunking(device, limit, monkeypatch):
    from engine import api
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    st = ix._st
    part, b = st.parts[0], st.b
    rng = np.random.default_rng(0)
    common = np.argsort(-np.diff(part.host_term_ptr))[:12]          # frequent terms: docs hold several of a query's
    terms = [sorted(rng.choice(common, 5, replace=False).tolist()) for _ in range(64)] + [[]]
    F = len(st.text_fields)
    w = np.linspace(0.7, 2.3, F).astype(np.float32)
    bb = np.full(F, 0.75, np.float32)
    monkeypatch.setattr(L, "MAX_POSTINGS_PER_CHUNK", limit)
    got_s, got_n = (b.to_numpy(x) for x in L.bm25f(b, part, terms, w, bb, 1.2, part.idf))
    for q, ts in enumerate(terms):
        s, n = (b.to_numpy(x)[0] for x in L.bm25f(b, part, [ts], w, bb, 1.2, part.idf))
        assert np.array_equal(got_s[q], s), (limit, q)
        assert np.array_equal(got_n[q], n), (limit, q)
