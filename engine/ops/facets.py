"""Terms facets over the full match set (the prototype's ``gpu_facets``)."""
from __future__ import annotations


def facet_terms(b, codes, mask, n_values: int):
    """``int64[Q, n_values]``: per query, the count of each code over its matched docs (code 0, missing, included)."""
    return b.bincount_rows(codes, mask, n_values)


def top_buckets(b, counts, size: int):
    """Per row: the ``size`` codes with the highest count (> 0), ties by code ascending (= key ascending, since codes
    follow sorted values); code 0 (missing) excluded. Returns host lists of (code, count)."""
    Q, n = counts.shape
    valid = (counts > 0) & (b.arange(n) > 0)
    idx, val = b.topk(b.astype(counts, "float32"), size, valid)
    idx, val = b.to_numpy(idx), b.to_numpy(val)
    return [[(int(c), int(v)) for c, v in zip(idx[q], val[q]) if c >= 0] for q in range(Q)]
