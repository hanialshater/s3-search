"""Pagination: sort the match set by the sort keys, ties by doc id ascending, apply ``search_after``, then from/size.

One query at a time (the executor loops over the batch); arrays stay on the device until the page is cut.
"""
from __future__ import annotations


def after_mask(b, keys, after):
    """``bool[N]``: entries strictly after ``after`` (one value per key, then a doc id) in the sort order."""
    N = keys[0][0].shape[0]
    doc = b.arange(N)
    out = b.full(N, False, "bool")
    eq = b.full(N, True, "bool")
    for (key, desc), v in zip(keys, after[:-1]):
        later = (key < v) if desc else (key > v)
        out = out | (eq & later)
        eq = eq & (key == v)
    return out | (eq & (doc > int(after[-1])))


def page(b, keys, mask, from_: int, size: int, after=None):
    """``keys``: list of ``(array[N], descending)``; the first may be the float32 score. Returns host
    ``(doc_ids int64[n], key_values list[list])`` for the page."""
    valid = mask if after is None else mask & after_mask(b, keys, after)
    need = from_ + size
    if need == 0:
        return [], []
    if len(keys) == 1 and keys[0][1] and str(keys[0][0].dtype).endswith("float32"):
        idx, val = b.topk(keys[0][0].reshape(1, -1), need, valid.reshape(1, -1))
        idx = b.to_numpy(idx)[0]
        idx = idx[idx >= 0][from_:need].astype("int64")
    else:
        (cand,) = b.nonzero(valid)                  # ascending doc ids: the final tie-break
        for key, desc in reversed(keys):
            order = b.argsort(key[cand], descending=desc)
            cand = cand[order]
        idx = b.to_numpy(cand[from_:need]).astype("int64")
    if len(idx) == 0:
        return [], []
    sel = b.asarray(idx, "int64")
    cols = [b.to_numpy(key[sel]).tolist() for key, _ in keys]
    return idx.tolist(), [list(row) for row in zip(*cols)] if cols else [[] for _ in idx]
