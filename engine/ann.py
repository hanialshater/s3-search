"""Approximate knn with cuVS (S9-1): CAGRA and IVF-PQ, built per segment on the GPU, filtered with cuVS's bitset
prefilter. Optional: without cuVS, CuPy or a CUDA index the method runs as exact, and the response says so
(``knn.<field>.ran``, with ``reason``).

Indexes are built lazily per (segment, field, method) on first use and kept on the segment's ``Part``; a merged
segment is a new ``Part``, so its index is built afresh there (rebuilt on merge). Distances are squared euclidean on
the L2-normalised vectors (the same order as cosine, and as inner product for unit vectors). The candidates' scores
are then the exact similarities, so a hit's ``_score`` is exact; only which docs are found is approximate.

**Version tolerance.** cuVS's Python API moves between releases (RAPIDS 26.6's ``ivf_pq.search`` refuses the
``filter=`` keyword of 24.x). The filter argument of each ``search`` is discovered from the installed function (its
signature, else its docstring's parameter list), never assumed: ``filter`` when present, else the one parameter whose
name contains "filter". When none is found and a filter is needed, or a build or search fails for any reason, the
clause runs exact and ``reason`` names why; a search never crashes on cuVS's API. A search with no filter argument
(cuVS 26.06's ``ivf_pq.search``) is post-filtered when a segment has a filter or deletes (``search_part``), and
``ran`` says so (``"ivf_pq+postfilter"``).
"""
from __future__ import annotations

import inspect
import re

import numpy as np

METHODS = ("exact", "cagra", "ivf_pq")
SEARCH_PARAMS = {"cagra": {"itopk_size", "search_width"}, "ivf_pq": {"n_probes"}}


class Fallback(Exception):
    """cuVS could not answer this clause; the message is the reason the response gives."""


def _unavailable(b) -> str | None:
    if b.name != "cuda":
        return f"not a cuda index (device {b.name})"
    try:
        import cupy  # noqa: F401
    except Exception as e:
        return f"CuPy not importable: {type(e).__name__}"
    try:
        from cuvs.neighbors import cagra, ivf_pq  # noqa: F401
    except Exception as e:
        return f"cuVS not importable: {type(e).__name__}"
    return None


def available(b) -> bool:
    return _unavailable(b) is None


def status(b, method: str) -> tuple[str, str | None]:
    """(the method that will be tried, the reason it is exact instead) for a requested method on this backend."""
    if method == "exact":
        return "exact", None
    why = _unavailable(b)
    return ("exact", why) if why else (method, None)


def ran(b, method: str) -> str:
    return status(b, method)[0]


def param_names(fn) -> list[str]:
    """The parameter names of an installed function: its signature, else the numpydoc ``name : type`` lines of its
    docstring (Cython functions may have no signature)."""
    try:
        names = list(inspect.signature(fn).parameters)
        if names:
            return names
    except (TypeError, ValueError):
        pass
    return re.findall(r"^\s*([A-Za-z_]\w*)\s*:", fn.__doc__ or "", re.M)


def filter_arg(fn) -> str | None:
    names = param_names(fn)
    if "filter" in names:
        return "filter"
    hits = [n for n in names if "filter" in n.lower()]
    return hits[0] if len(hits) == 1 else None


_FILTER_ARG: dict = {}


def _search_filter_arg(method: str):
    if method not in _FILTER_ARG:
        from cuvs.neighbors import cagra, ivf_pq
        _FILTER_ARG[method] = filter_arg((cagra if method == "cagra" else ivf_pq).search)
    return _FILTER_ARG[method]


def _version() -> str:
    try:
        import cuvs
        return getattr(cuvs, "__version__", "?")
    except Exception:
        return "?"


def _index(part, field, method):
    cache = part.__dict__.setdefault("_ann", {})
    key = (field, method)
    if key not in cache:
        import cupy as cp
        from cuvs.neighbors import cagra, ivf_pq
        data = cp.from_dlpack(part.vectors[field].contiguous())
        if method == "cagra":
            cache[key] = cagra.build(cagra.IndexParams(metric="sqeuclidean"), data)
        else:
            n_lists = max(1, min(1024, int(np.sqrt(max(part.n, 1)))))
            cache[key] = ivf_pq.build(ivf_pq.IndexParams(n_lists=n_lists, metric="sqeuclidean"), data)
    return cache[key]


MAX_OVERFETCH = 8192


def _prefilter(method):
    """``(argument name, constructor)`` to pass the bitset to ``<method>.search``, or None. Both are discovered from
    the installed cuVS (the search's parameters; ``cuvs.neighbors.filters.from_bitset``), never from a version."""
    name = _search_filter_arg(method)
    if name is None:
        return None
    try:
        from cuvs.neighbors import filters
    except Exception:
        return None
    make = getattr(filters, "from_bitset", None)
    return (name, make) if callable(make) else None


def overfetch(k: int, n: int, n_valid: int) -> int:
    """The candidates to ask for when the filter is applied after the search: ``ceil(k / max(selectivity, 0.01))
    * 2``, at most the segment's docs and ``MAX_OVERFETCH``."""
    sel = max(n_valid / max(n, 1), 0.01)
    return int(min(n, MAX_OVERFETCH, max(k, int(np.ceil(k / sel)) * 2)))


def _exact_part(b, part, field, q, k, valid_local):
    from . import ops
    sims = ops.dense_scores(b, part.vectors[field], q)
    idx, _ = b.topk(sims, k, valid_local.reshape(1, -1))
    return b.astype(idx, "int64")


def raw_search(b, part, field, method, params, q, k, prefilter):
    """cuVS's ``search`` on one segment: local ids ``int64[Q, k]`` on the device. ``prefilter``: None or
    ``((argument name, constructor), valid_local)``. Raises ``Fallback`` when cuVS fails."""
    import cupy as cp
    import torch
    from cuvs.neighbors import cagra, ivf_pq
    try:
        idx = _index(part, field, method)
    except Exception as e:
        raise Fallback(f"cuVS {method} build failed (cuvs {_version()}): {type(e).__name__}: {str(e)[:160]}")
    qc = cp.from_dlpack(q.to(torch.float32).contiguous())
    kw = {}
    if prefilter is not None:
        (name, make), valid_local = prefilter
        bits = cp.asarray(np.ascontiguousarray(b.pack_bits(valid_local), np.uint32))   # LSB first, 1 = included
        kw[name] = make(bits)
    if method == "cagra":
        itopk = max(int(params.get("itopk_size", 64)), -(-k // 32) * 32)
        sp = cagra.SearchParams(itopk_size=itopk, search_width=int(params.get("search_width", 1)))
        fn = cagra.search
    else:
        sp = ivf_pq.SearchParams(n_probes=int(params.get("n_probes", 20)))
        fn = ivf_pq.search
    try:
        _, nb = fn(sp, idx, qc, k, **kw)
        return torch.from_dlpack(cp.asarray(nb).astype(cp.int64))
    except Exception as e:
        raise Fallback(f"cuVS {method}.search failed (cuvs {_version()}): {type(e).__name__}: {str(e)[:160]}")


def search_part(b, part, field, method, params, q, k, valid_local):
    """``(candidate local doc ids int64[Q, k] with -1 pads, how)`` for one segment; ``valid_local`` (bool[n]) is the
    filter with the deletes. ``how``: "cuvs" (prefiltered, or nothing to filter), "postfilter" (the search takes no
    filter: ``overfetch`` candidates, the failing ones dropped), or "exact" (a post-filter that left fewer than
    ``k`` docs: this segment answered exactly). Raises ``Fallback`` when cuVS fails. Backend operations only, so
    the rule is tested on CPU with a stand-in ``raw_search``."""
    n_valid = int(b.count_nonzero(valid_local))
    pre, how, kk = None, "cuvs", k
    if n_valid < part.n:
        pf = _prefilter(method)
        if pf is not None:
            pre = (pf, valid_local)
        else:
            how, kk = "postfilter", overfetch(k, part.n, n_valid)
    nb = raw_search(b, part, field, method, params, q, kk, pre)
    nb = b.where((nb >= 0) & (nb < part.n), nb, -1)
    if how == "postfilter":
        ok = (nb >= 0) & valid_local[b.where(nb >= 0, nb, 0)]
        if int(b.count_nonzero(ok[0])) < k:
            return _exact_part(b, part, field, q, k, valid_local), "exact"
        nb = b.where(ok, nb, -1)
    return nb, how


def label(method, hows) -> tuple[str, str | None]:
    """The response's ``(ran, reason)`` from the segments' ``how``s."""
    hows = set(hows) or {"cuvs"}
    if hows == {"exact"}:
        return "exact", f"filter: {method} had no filter argument and too few post-filtered candidates passed"
    if "postfilter" in hows or "exact" in hows:
        why = (f"{sum(1 for h in hows if h == 'exact')} segment(s) answered exact (too few candidates passed the "
               "filter)") if "exact" in hows else None
        return f"{method}+postfilter", why
    return method, None


def candidates(b, st, field, method, params, q, k, valid):
    """``(bool[N], hows)``: the slots any segment returns among its ``valid`` docs for one query vector ``q``
    (f32[D]), and how each segment answered (``search_part``). Each segment returns up to ``k`` valid docs.
    Raises ``Fallback`` when any segment cannot be answered (the caller then runs the clause exact)."""
    m = b.full(st.N, False, "bool")
    q2 = q.reshape(1, -1)
    hows = []
    for part in st.parts:
        if not len(part.live_local):
            continue
        sl = b.asarray(part.live_slot, "int64")
        vl = b.full(part.n, False, "bool")
        vl[b.asarray(part.live_local, "int64")] = valid[sl]
        kk = min(int(k), int(b.count_nonzero(vl)))
        if kk == 0:
            continue
        loc, how = search_part(b, part, field, method, params, q2, kk, vl)
        hows.append(how)
        loc = loc[0]
        loc = loc[loc >= 0]
        slot_of = b.asarray(part.slot_of_local, "int64")
        s = slot_of[loc]
        m[s[s >= 0]] = True
    return m & valid, hows
