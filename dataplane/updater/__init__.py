"""The updater (docs/INTERFACES.md §3, §0): a replica's live ``Index``, kept fresh from the change log.

``Replica(store, segment_dir, device)`` loads a segment (checking ``device_nbytes + work_nbytes <= budget``) and
catches up with the log. ``poll()`` reads ``store.changes(after=index.version)`` and makes them visible:
``patch`` -> ``Index.patch`` (hot columns, in place), ``delete`` -> ``Index.delete`` (a tombstone: the live bit),
``upsert`` -> counted as pending (visible after the next build and swap). ``swap(new_segment)`` checks the double
buffer fits (``current + new device_nbytes + work_nbytes <= budget``, else ``OverBudget``), loads the second
``Index``, replays the log after its version, and swaps one reference; searches in flight finish on the old one.

Within a poll, changes are coalesced: the last value of a (field, id) wins, a deleted id is never patched, and each
field is patched with one ``Index.patch`` call. A patch for an id the index does not hold live (an upsert since the
segment, or a re-insert after a delete) waits for the next build, like the upsert it follows.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dataplane import _engine
from dataplane.indexer import MANIFEST, latest_segment, read_manifest

try:
    from engine.backend import OverBudget
except ImportError:  # pragma: no cover  (engine.backend exists from the skeleton on)
    class OverBudget(MemoryError):
        pass

DEFAULT_BATCH = 32   # queries per batch for work_nbytes (§4: Q * N * 4 * 2)


def work_nbytes(n_docs: int, batch: int = DEFAULT_BATCH) -> int:
    return batch * n_docs * 4 * 2


def memory_budget(device: str) -> int:
    """``Backend.memory_budget()`` (§0), with the same rule computed here while the backends are not built yet:
    ``S3_SEARCH_DEVICE_BUDGET`` if set, else 50% of available RAM on CPU devices and 80% of free CUDA memory."""
    try:
        from engine.backend import get_backend
        return int(get_backend(device).memory_budget())
    except (ImportError, NotImplementedError, AttributeError):
        pass
    env = os.environ.get("S3_SEARCH_DEVICE_BUDGET")
    if env:
        return int(env)
    if device == "cuda":
        import torch
        free, _ = torch.cuda.mem_get_info()
        return int(free * 0.8)
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 // 2
    except OSError:
        pass
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") // 2


def _resolve_segment(path: str | Path) -> Path:
    p = Path(path)
    if (p / MANIFEST).exists():
        return p
    seg = latest_segment(p)
    if seg is None:
        raise FileNotFoundError(f"no segment in {p}")
    return seg


def _segment_ids(segment: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_table(segment / "ids.parquet", columns=["_id"]).column("_id").to_pylist()


@dataclass
class PollReport:
    version: int          # the index's version after the poll
    changes: int          # change records read
    patched: int          # (field, id) values patched in place
    deleted: int          # ids whose live bit was cleared
    pending_upserts: int  # upserts (and patches behind them) waiting for the next build, since the segment


@dataclass
class SwapReport:
    segment: Path
    old_version: int
    new_version: int      # after replay
    replayed: int         # change records replayed onto the new index


class _Live:
    """One loaded index with what the updater knows about it."""

    def __init__(self, index, segment: Path, ids: list[str], manifest: dict):
        self.index = index
        self.segment = segment
        self.manifest = manifest
        self.live = set(ids)          # ids the index holds with the live bit set
        self.n_docs = len(ids)
        self.deleted = 0
        self.pending = 0
        self.loaded_at = time.time()


class Replica:
    def __init__(self, store, segment_dir: str | Path, device: str = "cpu", budget: int | None = None,
                 batch: int = DEFAULT_BATCH, page: int = 10_000):
        self.store = store
        self.device = device
        self.batch = batch
        self.page = page
        self._budget = budget
        self._lock = threading.RLock()        # one writer to the live index at a time (poll, swap)
        self._listeners: list[Callable[[int, float], None]] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_error: Exception | None = None   # the background poller's last failure, if any
        seg = _resolve_segment(segment_dir)
        man = read_manifest(seg)
        need = int(man.get("device_nbytes", 0)) + work_nbytes(int(man.get("n_docs", 0)), batch)
        if need > self.budget:
            raise OverBudget(f"loading {seg.name} needs {need} bytes on {device}; the budget is {self.budget}")
        self._cur = self._load(seg, man)
        self._replay(self._cur)

    # ---- state ------------------------------------------------------------------------------------------------------
    @property
    def budget(self) -> int:
        return self._budget if self._budget is not None else memory_budget(self.device)

    @property
    def index(self):
        """The live index. Callers that search hold this reference for the whole request."""
        return self._cur.index

    @property
    def version(self) -> int:
        return int(self._cur.index.version)

    @property
    def segment(self) -> Path:
        return self._cur.segment

    def search(self, request: dict) -> dict:
        return self._cur.index.search(request)

    def search_batch(self, requests: list[dict]) -> list[dict]:
        return self._cur.index.search_batch(requests)

    def stats(self) -> dict:
        c = self._cur
        return {"segment": c.segment.name, "version": self.version, "segment_version": int(c.manifest.get("version", 0)),
                "n_docs": c.n_docs, "live": len(c.live), "deleted": c.deleted, "pending_upserts": c.pending,
                "loaded_at": c.loaded_at, "device": self.device, "device_nbytes": int(c.manifest.get("device_nbytes", 0))}

    def live_ids(self) -> list[str]:
        return sorted(self._cur.live)

    def on_visible(self, fn: Callable[[int, float], None]) -> None:
        """Call ``fn(version, perf_counter())`` after each poll that moved the version (freshness measurement)."""
        self._listeners.append(fn)

    def off_visible(self, fn: Callable[[int, float], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ---- loading and replay -----------------------------------------------------------------------------------------
    def _load(self, seg: Path, man: dict) -> _Live:
        index = _engine.hooks().index_from_segment(seg, self.device)
        return _Live(index, seg, _segment_ids(seg), man)

    def _apply(self, cur: _Live, recs: list[dict]) -> tuple[int, int]:
        patches: dict[str, dict[str, object]] = {}
        dead: list[str] = []
        live = cur.live
        for r in recs:
            op, _id = r["op"], r["_id"]
            if op == "patch":
                if _id in live:
                    for f, v in r["fields"].items():
                        patches.setdefault(f, {})[_id] = v
                else:
                    cur.pending += 1
            elif op == "delete":
                if _id in live:
                    live.discard(_id)
                    dead.append(_id)
                    for p in patches.values():
                        p.pop(_id, None)
            else:
                cur.pending += 1
        index = cur.index
        if dead:
            index.delete(dead)
            cur.deleted += len(dead)
        n = 0
        for f, vals in patches.items():
            if vals:
                index.patch(f, list(vals), list(vals.values()))
                n += len(vals)
        return n, len(dead)

    def _replay(self, cur: _Live) -> PollReport:
        changes = patched = deleted = 0
        while True:
            recs = self.store.changes(after=int(cur.index.version), limit=self.page, docs=False)
            if not recs:
                break
            p, d = self._apply(cur, recs)
            cur.index.set_version(recs[-1]["offset"])
            changes += len(recs)
            patched += p
            deleted += d
            if len(recs) < self.page:
                break
        return PollReport(int(cur.index.version), changes, patched, deleted, cur.pending)

    # ---- the two operations -----------------------------------------------------------------------------------------
    def poll(self) -> PollReport:
        with self._lock:
            before = self.version
            rep = self._replay(self._cur)
        if rep.version != before:
            t = time.perf_counter()
            for fn in list(self._listeners):
                fn(rep.version, t)
        return rep

    def swap(self, new_segment: str | Path) -> SwapReport:
        seg = _resolve_segment(new_segment)
        man = read_manifest(seg)
        cur_bytes = int(self._cur.manifest.get("device_nbytes", 0))
        new_bytes = int(man.get("device_nbytes", 0))
        need = cur_bytes + new_bytes + work_nbytes(max(int(man.get("n_docs", 0)), self._cur.n_docs), self.batch)
        if need > self.budget:
            raise OverBudget(f"swapping to {seg.name} needs {need} bytes on {self.device} (double buffer); "
                             f"the budget is {self.budget}")
        nxt = self._load(seg, man)
        first = self._replay(nxt)                 # the bulk of the replay, outside the lock
        with self._lock:
            last = self._replay(nxt)              # what arrived meanwhile
            old = self._cur
            self._cur = nxt                       # the swap: one reference
        t = time.perf_counter()
        for fn in list(self._listeners):
            fn(self.version, t)
        return SwapReport(seg, int(old.index.version), int(nxt.index.version), first.changes + last.changes)

    # ---- a background poller ----------------------------------------------------------------------------------------
    def start(self, interval: float = 0.05) -> "Replica":
        if self._thread is not None:
            return self
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                try:
                    self.poll()
                    self.last_error = None
                except Exception as e:  # keep polling; the next poll retries from the same offset
                    self.last_error = e
                self._stop.wait(interval)

        self._thread = threading.Thread(target=run, name="replica-poll", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def __enter__(self) -> "Replica":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
