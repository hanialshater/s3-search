"""Counters, gauges and histograms in Prometheus text format (stdlib, thread-safe)."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Iterable

# milliseconds
LATENCY_BUCKETS = (0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 5000)
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _labels(labels: dict[str, str] | None) -> tuple:
    return tuple(sorted((labels or {}).items()))


def _escape(v: object) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(name: str, labels: tuple, value: float) -> str:
    if labels:
        body = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
        return f"{name}{{{body}}} {value:g}"
    return f"{name} {value:g}"


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._help: dict[str, tuple[str, str]] = {}
        self._counters: dict[str, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, dict[tuple, float]] = defaultdict(dict)
        self._hist: dict[str, dict[tuple, list]] = defaultdict(dict)   # name -> labels -> [counts..., sum, n]
        self._buckets: dict[str, tuple] = {}

    def describe(self, name: str, kind: str, text: str, buckets: Iterable[float] | None = None) -> None:
        self._help[name] = (kind, text)
        if buckets is not None:
            self._buckets[name] = tuple(buckets)

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._counters[name][_labels(labels)] += value

    def set(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._gauges[name][_labels(labels)] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            buckets = self._buckets.get(name, LATENCY_BUCKETS)
            row = self._hist[name].setdefault(_labels(labels), [0] * len(buckets) + [0.0, 0])
            for i, edge in enumerate(buckets):
                if value <= edge:
                    row[i] += 1
            row[-2] += value
            row[-1] += 1

    def value(self, name: str, **labels: str) -> float:
        """Current value of a counter or gauge (for tests and /readyz)."""
        with self._lock:
            if name in self._counters:
                return self._counters[name].get(_labels(labels), 0.0)
            return self._gauges.get(name, {}).get(_labels(labels), 0.0)

    def hist_count(self, name: str, **labels: str) -> int:
        with self._lock:
            row = self._hist.get(name, {}).get(_labels(labels))
            return int(row[-1]) if row else 0

    def render(self) -> str:
        out: list[str] = []
        with self._lock:
            names = sorted(set(self._counters) | set(self._gauges) | set(self._hist))
            for name in names:
                kind, text = self._help.get(name, ("untyped", ""))
                if name in self._hist:
                    kind = "histogram"
                if text:
                    out.append(f"# HELP {name} {text}")
                out.append(f"# TYPE {name} {kind}")
                for labels, v in sorted(self._counters.get(name, {}).items()):
                    out.append(_fmt(name, labels, v))
                for labels, v in sorted(self._gauges.get(name, {}).items()):
                    out.append(_fmt(name, labels, v))
                buckets = self._buckets.get(name, LATENCY_BUCKETS)
                for labels, row in sorted(self._hist.get(name, {}).items()):
                    for edge, c in zip(buckets, row):
                        out.append(_fmt(f"{name}_bucket", labels + (("le", f"{edge:g}"),), c))
                    out.append(_fmt(f"{name}_bucket", labels + (("le", "+Inf"),), row[-1]))
                    out.append(_fmt(f"{name}_sum", labels, row[-2]))
                    out.append(_fmt(f"{name}_count", labels, row[-1]))
        return "\n".join(out) + "\n"


def standard() -> Metrics:
    """A registry with every serving metric described, so the names are the contract the benchmark reads."""
    m = Metrics()
    m.describe("s3_http_requests_total", "counter", "HTTP requests by route and status")
    m.describe("s3_search_requests_total", "counter", "Search requests by index and outcome (ok, error)")
    m.describe("s3_cache_hits_total", "counter", "Query cache hits by index")
    m.describe("s3_cache_misses_total", "counter", "Query cache misses by index")
    m.describe("s3_cache_entries", "gauge", "Entries in the query cache by index")
    m.describe("s3_batches_total", "counter", "Batches sent to Index.search_batch by index")
    m.describe("s3_batch_size", "histogram", "Requests per batch", BATCH_BUCKETS)
    m.describe("s3_queue_wait_ms", "histogram", "Time a request waited in the scheduler queue (ms)")
    m.describe("s3_stage_ms", "histogram", "Engine stage time from took_ms (ms) by stage")
    m.describe("s3_request_ms", "histogram", "Serving time per search request incl. queue (ms)")
    m.describe("s3_errors_total", "counter", "Named errors by type")
    m.describe("s3_index_version", "gauge", "Version (store offset) of the live segment by index")
    m.describe("s3_index_docs", "gauge", "Documents in the live segment by index")
    m.describe("s3_index_ready", "gauge", "1 when the index has a loaded segment")
    m.describe("s3_swaps_total", "counter", "Segment swaps by index and kind (load, reindex, rollback)")
    m.describe("s3_patches_total", "counter", "Hot-field patches applied to the live index")
    m.describe("s3_queue_depth", "gauge", "Requests waiting in the scheduler by index")
    m.describe("s3_index_lag", "gauge", "Store head minus the highest offset fully visible in the live index, by index")
    m.describe("s3_index_pending_upserts", "gauge", "Upserts stored but not yet visible (waiting for a refresh), by index")
    m.describe("s3_refresh_ms", "histogram", "Time a refresh took (ms), by index")
    m.describe("s3_bulk_requests_total", "counter", "_bulk requests by index and refresh mode")
    m.describe("s3_bulk_items_total", "counter", "_bulk actions by index, op and outcome (ok, error)")
    m.describe("s3_bulk_rejected_total", "counter", "_bulk actions rejected, by index and named reason")
    m.describe("s3_bulk_items_per_second", "gauge", "Actions per second of the last _bulk request, by index")
    m.describe("s3_bulk_ms", "histogram", "Time per _bulk request (ms)", (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000))
    return m
