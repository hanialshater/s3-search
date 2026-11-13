"""Metric and bucket aggregations (S7-4) over a match mask, on the backend, as OpenSearch defines them.

Each aggregation brings its numbers to the host in one transfer. Values are read as float64 (a float field's float32
values are exact in float64, as OpenSearch reads its doc values as doubles).

- min / max / avg: ``{"value": x}``, ``None`` when no doc has a value; sum: ``{"value": x}`` (0.0 for none);
  value_count: the docs with a value; cardinality: distinct values (exact); stats: ``{count, min, max, avg, sum}``.
- range: buckets sorted by (from, to), ``from`` inclusive and ``to`` exclusive; key ``"<from>-<to>"`` with ``*`` for an
  open end, unless the request names one.
- histogram: bucket key ``floor((v - offset) / interval) * interval + offset``, ascending; with ``min_doc_count`` 0
  (the default) the empty buckets between the first and the last are returned.
"""
from __future__ import annotations

import math

import numpy as np

INF = float("inf")


def metric(b, kind: str, vals, present) -> dict:
    v = b.astype(vals, "float64")
    count = b.sum(b.astype(present, "float64"))
    total = b.sum(b.where(present, v, 0.0))
    mn = _reduce_min(b, b.where(present, v, INF))
    mx = -_reduce_min(b, b.where(present, -v, INF))
    host = b.to_numpy(b.stack([b.astype(x, "float64").reshape(1) for x in (count, total, mn, mx)], 0)).ravel()
    n, s, lo, hi = int(host[0]), float(host[1]), float(host[2]), float(host[3])
    if kind == "value_count":
        return {"value": n}
    if kind == "sum":
        return {"value": s}
    if kind == "min":
        return {"value": lo if n else None}
    if kind == "max":
        return {"value": hi if n else None}
    if kind == "avg":
        return {"value": s / n if n else None}
    if kind == "stats":
        return {"count": n, "min": lo if n else None, "max": hi if n else None, "avg": s / n if n else None,
                "sum": s}
    raise ValueError(kind)


def _reduce_min(b, x):
    """The minimum of a 1-D array (+inf when empty), on the device."""
    if x.shape[0] == 0:
        return b.full(1, INF, "float64").reshape(())
    return b.amin(x).reshape(())


def keyword_metric(b, kind: str, codes, mask, n_values: int) -> dict:
    """value_count or cardinality over codes (0 = missing)."""
    counts = b.bincount(b.astype(codes[mask], "int64"), n_values)
    host = b.to_numpy(b.stack([b.sum(counts[1:]), b.count_nonzero(counts[1:])], 0)).ravel()
    return {"value": int(host[0] if kind == "value_count" else host[1])}


def _fmt(x: float) -> str:
    return repr(float(x))


def range_agg(b, vals, present, ranges) -> dict:
    v = b.astype(vals, "float64")
    order = sorted(range(len(ranges)), key=lambda i: (-INF if ranges[i][0] is None else ranges[i][0],
                                                      INF if ranges[i][1] is None else ranges[i][1]))
    cols = []
    for i in order:
        lo, hi, _ = ranges[i]
        m = present
        if lo is not None:
            m = m & (v >= lo)
        if hi is not None:
            m = m & (v < hi)
        cols.append(b.astype(b.count_nonzero(m), "float64").reshape(1))
    counts = b.to_numpy(b.concatenate(cols, 0)).ravel() if cols else []
    buckets = []
    for i, n in zip(order, counts):
        lo, hi, key = ranges[i]
        bk = {"key": key if key is not None else f"{'*' if lo is None else _fmt(lo)}-{'*' if hi is None else _fmt(hi)}"}
        if lo is not None:
            bk["from"] = float(lo)
        if hi is not None:
            bk["to"] = float(hi)
        bk["doc_count"] = int(n)
        buckets.append(bk)
    return {"buckets": buckets}


def histogram(b, vals, present, interval: float, offset: float, min_doc_count: int) -> dict:
    v = b.astype(vals, "float64")[present]
    if v.shape[0] == 0:
        return {"buckets": []}
    k = b.astype(_floor(b, (v - offset) / interval), "int64")
    kmin = _reduce_min(b, b.astype(k, "float64"))
    kmax = -_reduce_min(b, -b.astype(k, "float64"))
    lims = b.to_numpy(b.stack([kmin.reshape(1), kmax.reshape(1)], 0)).ravel()
    k0, k1 = int(lims[0]), int(lims[1])
    counts = b.to_numpy(b.bincount(k - k0, k1 - k0 + 1))
    out = []
    for i, n in enumerate(counts.tolist()):
        if n >= max(min_doc_count, 0) and (n > 0 or min_doc_count == 0):
            out.append({"key": (k0 + i) * interval + offset, "doc_count": int(n)})
    return {"buckets": out}


def _floor(b, x):
    """floor() through the backend: integer part, minus 1 where a negative value had a fraction."""
    t = b.astype(b.astype(x, "int64"), "float64")
    return b.where(t > x, t - 1.0, t)
