"""The updater (docs/INTERFACES.md §3, §0, §4 "Multi-segment indexes"): a replica's live ``Index``, a base segment
plus delta segments, kept fresh from the change log (S8; S2 had one segment and in-place patches only).

Every refresh (0.5 s by default) turns what the store logged into one delta step (``delta.apply_delta``): patches in
place, upserts as a small delta segment that tombstones their older copies, deletes as tombstones. A merger thread
runs the tiered merge policy (``indexer.TieredMergePolicy``) with the snapshot, merge, replay and replace protocol
(``delta.merge_once``) and writes commit points. ``swap`` replaces the whole index (a rebuild or a rollback) after the
double-buffer budget check (``current + new device_nbytes + work_nbytes <= budget``, else ``OverBudget``).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from dataplane import _engine
from dataplane.indexer import MANIFEST, TieredMergePolicy, latest_segment, read_manifest
from dataplane.indexer.commit import gc_commits, latest_commit, read_commit
from dataplane.updater.delta import DEFAULT_REFRESH, MergeReport, apply_delta, merge_once

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
    deleted: int          # keys deleted or tombstoned
    pending_upserts: int  # upserts not yet visible (0 with delta segments: a refresh makes them visible)
    added: int = 0        # docs in the delta segment this refresh added
    segment_id: str | None = None


@dataclass
class SwapReport:
    segment: Path
    old_version: int
    new_version: int      # after replay
    replayed: int         # change records replayed onto the new index


def serving_refresh(mi) -> dict:
    """Serving's ``Ports.refresh`` hook (serving runs it under the index's admin lock): one delta step from the
    oldest pending upsert (serving has already applied the patches and deletes after it; re-applying them is
    harmless), then one tiered merge if one is due."""
    policy = getattr(mi, "merge_policy", None) or TieredMergePolicy()
    with mi._store_lock:
        store = mi._get_store()
        index = mi.index
        after = (mi.pending_since - 1) if mi.pending_since else int(index.version)
        rep = apply_delta(index, store, min(after, int(index.version)))
        mi.applied_offset = max(int(getattr(mi, "applied_offset", 0)), rep.version)
        mi.pending_upserts, mi.pending_since = 0, None
        merged = merge_once(index, store, policy)
    mi.epoch += 1
    mi.cache.clear()
    return {"version": int(index.version), "segment_id": rep.segment_id, "added": rep.added,
            "segments": len(index.segments), "merged": None if merged is None else merged.inputs}


def _resolve(path: str | Path) -> tuple[str, Path]:
    """``("commit", file)``, ``("segment", dir)``: what ``path`` names (a commit file, a segment, or a directory
    whose newest commit, else newest segment, is used)."""
    p = Path(path)
    if p.is_file() and p.name.startswith("commit-"):
        return "commit", p
    if (p / MANIFEST).exists():
        return "segment", p
    c = latest_commit(p)
    if c is not None:
        return "commit", c
    return "segment", _resolve_segment(p)


class Replica:
    """A replica's live ``Index`` over a base segment plus delta segments (S8), kept fresh from the store's log.

    - ``refresh()`` (and ``poll()``, the S2 name for the same step): one delta step, ``delta.apply_delta``.
    - ``merge()``: one tiered merge (``delta.merge_once``) under the budget check.
    - ``commit()``: ``Index.save_commit`` into ``commit_dir``, then ``gc_commits``.
    - ``start(interval)``: a refresher thread every ``interval`` (default 0.5 s) and a merger thread that merges when
      the policy says so and commits every ``commit_every`` seconds (with ``commit_dir``).
    - ``swap(path)``: a whole new index (a rebuild, or a rollback to a commit or segment), caught up, then one
      reference swap after the double-buffer budget check.

    Loading a commit: a commit holds deletes, not hot-column patches, so the patches logged after the oldest
    segment's version and up to the commit's are replayed, each only onto keys whose live copy is in a segment built
    before that patch (a segment built after it already holds it, or a newer value). Then the log after the commit
    is applied as a refresh.
    """

    def __init__(self, store, segment_dir: str | Path, device: str = "cpu", budget: int | None = None,
                 batch: int = DEFAULT_BATCH, page: int = 10_000, refresh_interval: float = DEFAULT_REFRESH,
                 merge_policy: TieredMergePolicy | None = None, commit_dir: str | Path | None = None,
                 commit_every: float = 5.0, keep_commits: int = 3):
        self.store = store
        self.device = device
        self.batch = batch
        self.page = page
        self.refresh_interval = refresh_interval
        self.merge_policy = merge_policy or TieredMergePolicy()
        self.commit_dir = Path(commit_dir) if commit_dir is not None else None
        self.commit_every = commit_every
        self.keep_commits = keep_commits
        self._budget = budget
        self._lock = threading.RLock()        # one writer to the live index at a time (refresh, merge, swap)
        self._listeners: list[Callable[[int, float], None]] = []
        self._threads: list[threading.Thread] = []
        self._thread: threading.Thread | None = None   # the refresher (kept for S2 callers that look at it)
        self._stop = threading.Event()
        self.last_error: Exception | None = None       # the background threads' last failure, if any
        self.merges: list[MergeReport] = []
        self._committed = -1
        kind, path = _resolve(segment_dir)
        self._kind, self._path = kind, path
        nbytes, n = self._footprint(kind, path)
        need = nbytes + work_nbytes(n, batch)
        if need > self.budget:
            raise OverBudget(f"loading {path.name} needs {need} bytes on {device}; the budget is {self.budget}")
        self._index, self._live = self._load(kind, path)
        self._loaded_at = time.time()
        self.refresh()

    # ---- state ------------------------------------------------------------------------------------------------------
    @property
    def budget(self) -> int:
        return self._budget if self._budget is not None else memory_budget(self.device)

    @property
    def index(self):
        """The live index. Refreshes and merges change it in place; ``swap`` replaces it."""
        return self._index

    @property
    def version(self) -> int:
        return int(self._index.version)

    @property
    def segment(self) -> Path:
        """What this replica was loaded or last swapped from (a segment directory or a commit file)."""
        return self._path

    def search(self, request: dict) -> dict:
        return self._index.search(request)

    def search_batch(self, requests: list[dict]) -> list[dict]:
        return self._index.search_batch(requests)

    def stats(self) -> dict:
        segs = self._index.segments
        n = sum(s["n_docs"] for s in segs)
        live = sum(s["live"] for s in segs)
        return {"segment": self._path.name, "version": self.version, "segments": len(segs), "n_docs": n,
                "live": live, "deleted": n - live, "pending_upserts": 0, "loaded_at": self._loaded_at,
                "device": self.device, "device_nbytes": int(self._index.device_nbytes), "merges": len(self.merges)}

    def live_ids(self) -> list[str]:
        return sorted(self._live)

    def on_visible(self, fn: Callable[[int, float], None]) -> None:
        """Call ``fn(version, perf_counter())`` after each step that moved the version (freshness measurement)."""
        self._listeners.append(fn)

    def off_visible(self, fn: Callable[[int, float], None]) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def _notify(self) -> None:
        t = time.perf_counter()
        for fn in list(self._listeners):
            fn(self.version, t)

    # ---- loading ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _footprint(kind: str, path: Path) -> tuple[int, int]:
        if kind == "segment":
            man = read_manifest(path)
            return int(man.get("device_nbytes", 0)), int(man.get("n_docs", 0))
        c = read_commit(path, verify=False)
        mans = [read_manifest(path.parent / s) for s in c["segments"]]
        return sum(int(m.get("device_nbytes", 0)) for m in mans), sum(int(m.get("n_docs", 0)) for m in mans)

    def _load(self, kind: str, path: Path):
        from engine.api import Index
        if kind == "segment":
            index = _engine.hooks().index_from_segment(path, self.device)
            return index, set(_segment_ids(path))
        index = Index.from_commit(path, device=self.device)
        c = read_commit(path)
        owner: dict[str, int] = {}
        for sid in c["segments"]:
            d = path.parent / sid
            ids = _segment_ids(d)
            v = int(read_manifest(d).get("version", 0))
            bits = c["live"].get(sid)
            if bits is None:
                alive = range(len(ids))
            else:
                words = np.asarray(bits, np.uint32)
                alive = [i for i in range(len(ids)) if (int(words[i >> 5]) >> (i & 31)) & 1]
            for i in alive:
                owner[ids[i]] = v
        self._replay_patches(index, owner, upto=int(c["version"]))
        return index, set(owner)

    def _replay_patches(self, index, owner: dict[str, int], upto: int) -> int:
        """The commit hole (§3): patches in (oldest segment version, upto] onto keys whose live copy predates them."""
        if not owner:
            return 0
        after = min(owner.values())
        runs: dict[str, dict[str, object]] = {}
        while after < upto:
            recs = self.store.changes(after, limit=self.page, docs=False)
            if not recs:
                break
            for r in recs:
                if r["offset"] > upto:
                    break
                if r["op"] == "patch" and r["_id"] in owner and r["offset"] > owner[r["_id"]]:
                    for f, v in r["fields"].items():
                        runs.setdefault(f, {})[r["_id"]] = v
            after = recs[-1]["offset"]
        n = 0
        for f, kv in runs.items():
            index.patch(f, list(kv), list(kv.values()))
            n += len(kv)
        return n

    # ---- the steps --------------------------------------------------------------------------------------------------
    def refresh(self) -> PollReport:
        with self._lock:
            before = self.version
            rep = apply_delta(self._index, self.store, before)
            self._live.difference_update(rep.removed)
            self._live.update(rep.upserted)
        if rep.version != before:
            self._notify()
        return PollReport(rep.version, rep.records, rep.patched, rep.deleted, 0, rep.added, rep.segment_id)

    poll = refresh

    def merge(self) -> MergeReport | None:
        work = work_nbytes(int(self._index.n_docs), self.batch)
        rep = merge_once(self._index, self.store, self.merge_policy, lock=self._lock, budget=self.budget, work=work)
        if rep is not None:
            self.merges.append(rep)
        return rep

    def commit(self) -> Path | None:
        if self.commit_dir is None:
            return None
        with self._lock:
            if self.version == self._committed:
                return latest_commit(self.commit_dir)
            p = Path(self._index.save_commit(self.commit_dir))
            self._committed = self.version
        gc_commits(self.commit_dir, keep=self.keep_commits)
        return p

    def swap(self, new_segment: str | Path) -> SwapReport:
        kind, path = _resolve(new_segment)
        nbytes, n = self._footprint(kind, path)
        need = int(self._index.device_nbytes) + nbytes + work_nbytes(max(n, int(self._index.n_docs)), self.batch)
        if need > self.budget:
            raise OverBudget(f"swapping to {path.name} needs {need} bytes on {self.device} (double buffer); "
                             f"the budget is {self.budget}")
        index, live = self._load(kind, path)
        first = apply_delta(index, self.store, int(index.version))   # the bulk of the replay, outside the lock
        live.difference_update(first.removed)
        live.update(first.upserted)
        with self._lock:
            last = apply_delta(index, self.store, int(index.version))   # what arrived meanwhile
            live.difference_update(last.removed)
            live.update(last.upserted)
            old = self._index
            self._index, self._live, self._kind, self._path = index, live, kind, path
            self._loaded_at = time.time()
        self._notify()
        return SwapReport(path, int(old.version), int(index.version), first.records + last.records)

    # ---- background threads -----------------------------------------------------------------------------------------
    def start(self, interval: float | None = None, merges: bool = True) -> "Replica":
        """A refresher every ``interval`` seconds (default ``refresh_interval``) and, with ``merges``, a merger."""
        if self._thread is not None:
            return self
        self._stop.clear()
        interval = self.refresh_interval if interval is None else interval

        def guard(fn):
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - keep going; the next step retries from the same offset
                self.last_error = e

        def refresher():                     # a fixed cadence: the refresh's own cost does not stretch the period
            due = time.monotonic()
            while not self._stop.is_set():
                guard(self.refresh)
                due = max(due + interval, time.monotonic())
                self._stop.wait(max(0.0, due - time.monotonic()))

        def merger():
            last_commit = time.monotonic()
            while not self._stop.is_set():
                merged = []
                guard(lambda: merged.append(self.merge()))
                if self.commit_dir is not None and (merged and merged[0] is not None
                                                    or time.monotonic() - last_commit >= self.commit_every):
                    guard(self.commit)
                    last_commit = time.monotonic()
                if not (merged and merged[0] is not None):
                    self._stop.wait(max(interval, 0.05))

        self._thread = threading.Thread(target=refresher, name="replica-refresh", daemon=True)
        self._threads = [self._thread]
        if merges:
            self._threads.append(threading.Thread(target=merger, name="replica-merge", daemon=True))
        for t in self._threads:
            t.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        for t in self._threads:
            t.join()
        self._threads, self._thread = [], None

    def __enter__(self) -> "Replica":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
