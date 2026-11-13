"""Pagination: sort the match set by the sort keys, ties by doc id ascending, apply ``search_after``, then from/size.

One query at a time (the executor loops over the batch); arrays stay on the device until the page is cut, and the
page comes to the host in one transfer (doc ids, key values and any extra columns stacked as float64, which holds
float32 scores, doc ids and int32 values exactly).
"""
from __future__ import annotations


def after_mask(b, keys, after, tiekey=None):
    """``bool[N]``: entries strictly after ``after`` (one value per key, then the tie key) in the sort order. The tie
    key is ``tiekey`` (the docs' first_offset, ascending in doc order) or the doc id."""
    N = keys[0][0].shape[0]
    doc = b.arange(N) if tiekey is None else tiekey
    out = b.full(N, False, "bool")
    eq = b.full(N, True, "bool")
    for (key, desc), v in zip(keys, after[:-1]):
        later = (key < v) if desc else (key > v)
        out = out | (eq & later)
        eq = eq & (key == v)
    return out | (eq & (doc > int(after[-1])))


def page(b, keys, mask, from_: int, size: int, after=None, extra=(), tiekey=None, collapse=None):
    """``keys``: list of ``(array[N], descending)``; the first may be the float32 score. ``extra``: arrays[N] to read
    at the page's docs (e.g. the score). Returns host ``(doc_ids list[int], key_values list[list], extra_values
    list[list])``."""
    ids, kv, ev = page_arrays(b, keys, mask, from_, size, after, extra, tiekey, collapse)
    return ids.tolist(), kv.tolist(), ev.tolist()


def page_arrays(b, keys, mask, from_: int, size: int, after=None, extra=(), tiekey=None, collapse=None):
    """``page`` as host arrays (S6-7): ``(doc_ids int64[h], key_values float64[h, len(keys)], extra_values
    float64[h, len(extra)])``."""
    nk, ne = len(keys), len(extra)

    def empty():
        return (b.to_numpy(b.zeros(0, "int64")), b.to_numpy(b.zeros((0, nk), "float64")),
                b.to_numpy(b.zeros((0, ne), "float64")))
    # collapse: int64[N] group codes (S7-3): only the first doc of each group in sort order is kept, and
    # from/size count groups.
    valid = mask if after is None else mask & after_mask(b, keys, after, tiekey)
    need = from_ + size
    if need == 0:
        return empty()
    if collapse is None and len(keys) == 1 and keys[0][1] and str(keys[0][0].dtype).endswith("float32"):
        idx, _ = b.topk(keys[0][0].reshape(1, -1), need, valid.reshape(1, -1))
        sel = b.astype(idx[0][from_:need], "int64")
    else:
        (cand,) = b.nonzero(valid)                  # ascending doc ids: the final tie-break
        for key, desc in reversed(keys):
            order = b.argsort(key[cand], descending=desc)
            cand = cand[order]
        if collapse is not None:                    # the first of each group, in sort order
            g = collapse[cand]
            o = b.argsort(g)                         # stable: within a group, sort order is kept
            gs = g[o]
            first = b.concatenate([b.full(1, True, "bool"), gs[1:] != gs[:-1]]) if gs.shape[0] else gs > 0
            firsts = o[first]
            cand = cand[firsts[b.argsort(firsts)]]   # back in sort order
        sel = cand[from_:need]
    if sel.shape[0] == 0:
        return empty()
    safe = b.where(sel >= 0, sel, 0)
    cols = [b.astype(sel, "float64")] + [b.astype(k[safe], "float64") for k, _ in keys] + \
        [b.astype(e[safe], "float64") for e in extra]
    host = b.to_numpy(b.stack(cols, 1))             # the one transfer
    host = host[host[:, 0] >= 0]
    return host[:, 0].astype("int64"), host[:, 1:1 + nk], host[:, 1 + nk:]
