"""Terms facets over the full match set (the prototype's ``gpu_facets``)."""
from __future__ import annotations


def facet_terms(b, codes, mask, n_values: int):
    """``int64[Q, n_values]``: per query, the count of each code over its matched docs (code 0, missing, included)."""
    return b.bincount_rows(codes, mask, n_values)


def top_buckets(b, counts, size: int):
    """Per row: ``(buckets, with_value)``. ``buckets``: the ``size`` codes with the highest count (> 0), ties by code
    ascending (= key ascending, since codes follow sorted values), code 0 (missing) excluded, as host (code, count)
    pairs. ``with_value``: the row's docs that have a value. One transfer to the host for all of it."""
    buf, k = top_buckets_device(b, counts, size)
    return top_buckets_parse(b.to_numpy(buf), k)


def top_buckets_device(b, counts, size: int):
    """``top_buckets`` before its transfer (S6-34): the device buffer ``float64[Q, 2k + 1]`` (codes, counts, with_value)
    and k, so a caller can bring several of them to the host in one transfer."""
    Q, n = counts.shape
    valid = (counts > 0) & (b.arange(n) > 0)
    idx, val = b.topk(b.astype(counts, "float32"), size, valid)
    with_value = b.sum(counts[:, 1:], axis=1).reshape(Q, 1)
    return b.concatenate([b.astype(idx, "float64"), b.astype(val, "float64"), b.astype(with_value, "float64")],
                         axis=1), idx.shape[1]


def top_buckets_parse(host, k: int):
    """``top_buckets``'s result from a host copy of ``top_buckets_device``'s buffer."""
    return [([(int(c), int(v)) for c, v in zip(host[q, :k], host[q, k:2 * k]) if c >= 0], int(host[q, -1]))
            for q in range(host.shape[0])]
