"""The freshness hook for the steward's B1-4 harness (milestone S2's gate: visible within 1 s at p99, results equal
to a full rebuild).

    report = run(store, replica, n_changes=10_000)

writes ``n_changes`` hot-field patches (one field of one live document each, in batches of ``batch``) to the store,
while the replica's poller (started here at ``interval`` unless it is already running) makes them visible. A change's
latency is the time from ``store.apply`` returning to the end of the first poll whose version covers it (the moment
``Index.patch`` and ``set_version`` are done, so the next search sees it). Then, unless ``check_rebuild=False``, it
builds a fresh segment from the store, loads it, and compares every query's response on the live index with the
rebuild's: totals, ids in order, scores (``rtol=1e-5``) and sort values (all but the last element, the internal doc
id). The harness owns the thresholds; this function only measures.
"""
from __future__ import annotations

import bisect
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataplane import _engine
from dataplane.indexer import build


@dataclass
class FreshnessReport:
    n_changes: int
    batches: int
    seconds: float                     # wall time of the write phase
    p50_ms: float
    p99_ms: float
    max_ms: float
    version: int                       # the store head the replica reached
    equal_to_rebuild: bool | None      # None when not checked
    queries: int = 0
    mismatches: list[str] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("n_changes", "batches", "seconds", "p50_ms", "p99_ms", "max_ms",
                                           "version", "equal_to_rebuild", "queries", "mismatches")}
        return d


def _values(rng: np.random.Generator, ftype: str, n: int) -> list:
    if ftype == "float":
        return [round(float(x), 2) for x in rng.uniform(0.5, 500.0, n)]
    if ftype in ("integer", "long"):
        return [int(x) for x in rng.integers(0, 1000, n)]
    return [bool(x) for x in rng.random(n) < 0.5]


def default_queries(mapping) -> list[dict]:
    """Requests whose results move with the hot fields: sorts and ranges on each (§5 subset)."""
    qs = []
    for f in mapping.hot:
        t = mapping.fields[f].type
        qs.append({"query": {"match_all": {}}, "sort": [{f: {"order": "asc"}}], "size": 100})
        qs.append({"query": {"match_all": {}}, "sort": [{f: {"order": "desc"}}], "size": 100})
        if t == "boolean":
            qs.append({"query": {"bool": {"filter": [{"term": {f: True}}]}}, "size": 100})
        else:
            mid = 250.0 if t == "float" else 500
            qs.append({"query": {"bool": {"filter": [{"range": {f: {"gte": mid}}}]}},
                       "sort": [{f: {"order": "asc"}}], "size": 100})
    return qs


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-6) or (a != a and b != b)
        except (TypeError, ValueError):
            return False
    return a == b


def compare_responses(live: dict, fresh: dict) -> str | None:
    """None when equal (as defined in the module docstring); else what differs."""
    lt, ft = live["hits"]["total"], fresh["hits"]["total"]
    if lt != ft:
        return f"total {lt} != {ft}"
    lh, fh = live["hits"]["hits"], fresh["hits"]["hits"]
    lid, fid = [h["_id"] for h in lh], [h["_id"] for h in fh]
    if lid != fid:
        i = next((i for i, (a, b) in enumerate(zip(lid, fid)) if a != b), min(len(lid), len(fid)))
        return f"ids differ from rank {i}: {lid[i:i + 3]} != {fid[i:i + 3]}"
    for i, (a, b) in enumerate(zip(lh, fh)):
        if not _close(float(a["_score"]), float(b["_score"])):
            return f"rank {i} ({a['_id']}): _score {a['_score']} != {b['_score']}"
        sa, sb = list(a.get("sort", []))[:-1], list(b.get("sort", []))[:-1]
        if len(sa) != len(sb) or not all(_close(x, y) for x, y in zip(sa, sb)):
            return f"rank {i} ({a['_id']}): sort {sa} != {sb}"
    if live.get("aggregations") != fresh.get("aggregations"):
        return "aggregations differ"
    return None


def run(store, replica, n_changes: int = 10_000, batch: int = 100, interval: float = 0.005, seed: int = 0,
        fields: list[str] | None = None, queries: list[dict] | None = None, check_rebuild: bool = True,
        rebuild_dir: str | Path | None = None, rate: float | None = None, timeout: float = 60.0) -> FreshnessReport:
    m = store.mapping
    fields = list(fields or m.hot)
    if not fields:
        raise ValueError("the mapping has no hot fields")
    ids = replica.live_ids()
    if not ids:
        raise ValueError("the replica holds no live documents")
    rng = np.random.default_rng(seed)
    events: list[tuple[int, float]] = []

    def record(v: int, t: float) -> None:
        events.append((v, t))

    replica.on_visible(record)
    own_poller = replica._thread is None
    if own_poller:
        replica.start(interval)
    applied: list[tuple[int, int, float]] = []    # (last offset, n changes, t apply returned)
    t0 = time.perf_counter()
    try:
        left = n_changes
        while left > 0:
            k = min(batch, left)
            pick = rng.integers(0, len(ids), k)
            fpick = rng.integers(0, len(fields), k)
            changes = []
            for f_i in range(len(fields)):
                sel = np.nonzero(fpick == f_i)[0]
                vals = _values(rng, m.fields[fields[f_i]].type, len(sel))
                changes += [(int(j), {"op": "patch", "_id": ids[pick[j]], "fields": {fields[f_i]: v}})
                            for j, v in zip(sel, vals)]
            changes.sort(key=lambda c: c[0])
            _, last = store.apply(c for _, c in changes)
            applied.append((last, k, time.perf_counter()))
            left -= k
            if rate:
                time.sleep(k / rate)
        seconds = time.perf_counter() - t0
        final = applied[-1][0] if applied else store.head()
        deadline = time.perf_counter() + timeout
        while not (events and events[-1][0] >= final) and time.perf_counter() < deadline:
            time.sleep(interval / 2)
    finally:
        if own_poller:
            replica.stop()
        replica.off_visible(record)
    if not (events and events[-1][0] >= final):
        raise TimeoutError(f"the replica reached {replica.version}, not {final}, within {timeout} s")
    events = list(events)
    versions = [v for v, _ in events]
    lat = []
    for last, k, t in applied:
        i = bisect.bisect_left(versions, last)
        lat += [max(0.0, (events[i][1] - t) * 1000.0)] * k
    a = np.asarray(lat)
    rep = FreshnessReport(n_changes, len(applied), round(seconds, 3), float(np.percentile(a, 50)),
                          float(np.percentile(a, 99)), float(a.max()), replica.version, None, latencies_ms=lat)
    if check_rebuild:
        queries = queries if queries is not None else default_queries(m)
        with tempfile.TemporaryDirectory() as tmp:
            seg = build(store, None, rebuild_dir or tmp)
            fresh = _engine.hooks().index_from_segment(seg, replica.device)
            live = replica.index
            for i, q in enumerate(queries):
                why = compare_responses(live.search(q), fresh.search(q))
                if why:
                    rep.mismatches.append(f"query {i}: {why}")
        rep.queries = len(queries)
        rep.equal_to_rebuild = not rep.mismatches
    return rep
