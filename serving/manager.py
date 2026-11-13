"""Indexes, their segments and the admin operations (docs/INTERFACES.md §7).

On disk, per index:  <data_dir>/<name>/mapping.json
                                       /store.sqlite            (the data plane's store; absent on a read-only replica)
                                       /segments/seg-<version>-<hex>/   (kept for rollback, newest ``keep_segments``)

A ``ManagedIndex`` holds one live ``Index`` reference. A swap loads the new index first and then replaces the
reference, so a search never sees a half-loaded index and one in flight finishes on the old one. The query cache key
carries (segment id, epoch), the epoch moving on every swap, patch and delete, so stale entries are unreachable.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from . import bulk as bulk_mod
from .cache import LRUCache, request_key
from .config import ServingConfig
from .errors import (IndexExists, InvalidRequestBody, NotReady, UnknownIndex, UnknownSegment)
from .metrics import Metrics
from .ports import Ports, read_manifest
from .scheduler import BatchScheduler, search_batch

log = logging.getLogger("serving")
INTERVAL_RE = re.compile(r"^(-1|\d+(?:\.\d+)?)(ms|s|m|h)?$")
DEFAULT_SETTINGS = {"refresh_interval": "500ms"}


def parse_interval(text) -> float | None:
    """OpenSearch interval ("1s", "500ms", "2m", "-1") to seconds; None disables the background refresh."""
    m = INTERVAL_RE.match(str(text).strip())
    if not m:
        raise InvalidRequestBody(f"refresh_interval {text!r} is not like 1s, 500ms, 2m or -1", path="index.refresh_interval")
    if m.group(1) == "-1":
        return None
    return float(m.group(1)) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600, None: 1}[m.group(2)]


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
VALID_OPS = ("upsert", "patch", "delete")


class ManagedIndex:
    def __init__(self, name: str, root: Path, mapping: dict, cfg: ServingConfig, ports: Ports, metrics: Metrics):
        self.name, self.root, self.mapping = name, Path(root), mapping
        self.cfg, self.ports, self.metrics = cfg, ports, metrics
        self.segments_dir = self.root / "segments"
        self.store_path = self.root / "store.sqlite"
        self.index: Any = None
        self.segment_id: str | None = None
        self.epoch = 0
        self.applied_offset = 0
        self.pending_upserts = 0
        self.pending_since: int | None = None     # offset of the oldest upsert not yet visible in the live index
        self.settings = {**DEFAULT_SETTINGS, "refresh_interval": cfg.refresh_interval}
        self._load_settings()
        self.refreshed_upto = 0                   # every change up to this offset is visible (for refresh=wait_for)
        self._refresh_cv = threading.Condition()
        self._refresh_stop = threading.Event()
        self._refresh_wake = threading.Event()
        self._last_refresh_cost = 0.0
        self._refresher: threading.Thread | None = None
        self.rankers: dict[str, Any] = {}        # re-registered on every new Index, so a swap keeps them
        self._admin = threading.RLock()          # one admin operation at a time
        self._store_lock = threading.RLock()     # the store is one SQLite connection
        self._store: Any = None
        self.cache = LRUCache(cfg.cache_size)
        self.scheduler = BatchScheduler(lambda: self.index, name=name, max_batch=cfg.max_batch,
                                        max_wait_ms=cfg.max_wait_ms, max_queue=cfg.max_queue, metrics=metrics,
                                        columns=cfg.columns)

    # -- state ------------------------------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.index is not None

    def close(self) -> None:
        self._refresh_stop.set()
        self._refresh_wake.set()
        if self._refresher is not None:
            self._refresher.join(5)
        self.scheduler.close()

    # -- settings and the background refresh ------------------------------------------------------------------------
    def _load_settings(self) -> None:
        p = self.root / "settings.json"
        if p.exists():
            try:
                self.settings.update(json.loads(p.read_text()))
            except ValueError:
                pass

    def update_settings(self, body: dict) -> dict:
        """``{"index": {"refresh_interval": "5s"}}`` (or the flat form). Only refresh_interval is known."""
        if not isinstance(body, dict):
            raise InvalidRequestBody("settings body is an object", path="")
        flat = body.get("index", body)
        if not isinstance(flat, dict):
            raise InvalidRequestBody('"index" must be an object', path="index")
        flat = {k.removeprefix("index."): v for k, v in flat.items()}
        unknown = sorted(set(flat) - set(DEFAULT_SETTINGS))
        if unknown:
            raise InvalidRequestBody(f"unknown setting(s) {unknown}; known: {sorted(DEFAULT_SETTINGS)}",
                                     path=f"index.{unknown[0]}")
        if "refresh_interval" in flat:
            parse_interval(flat["refresh_interval"])
            self.settings["refresh_interval"] = str(flat["refresh_interval"])
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "settings.json").write_text(json.dumps(self.settings))
        self._refresh_wake.set()
        return {"acknowledged": True, "index": dict(self.settings)}

    def start_refresher(self) -> None:
        if self._refresher is None:
            self._refresher = threading.Thread(target=self._refresh_loop, name=f"refresh-{self.name}", daemon=True)
            self._refresher.start()

    def _refresh_loop(self) -> None:
        """Make pending upserts visible every ``refresh_interval`` (never while there is nothing to show). With the
        full-rebuild backend a refresh costs seconds, so the gap is also at least 4x the last refresh's cost (at most
        20% of the time spent rebuilding); a delta refresh from the data plane (``Ports.refresh``) is cheap and keeps the
        configured interval."""
        while not self._refresh_stop.is_set():
            interval = parse_interval(self.settings["refresh_interval"])
            wait = 1.0 if interval is None else interval
            if self.ports.refresh is None:
                wait = max(wait, 4 * self._last_refresh_cost)
            self._refresh_wake.wait(wait)
            self._refresh_wake.clear()
            if self._refresh_stop.is_set():
                break
            if interval is not None and self.ready and self.pending_upserts:
                try:
                    self.refresh()
                except Exception:  # noqa: BLE001 - try again next interval
                    log.exception("background refresh of %s failed", self.name)

    def refresh(self) -> dict:
        """Make every stored change visible: a delta refresh when the data plane provides one (``Ports.refresh``), else a
        full rebuild from the store and a swap (the reindex path)."""
        with self._admin:
            if not self.ready:
                raise NotReady(f"index {self.name!r} is not loaded")
            if not self.pending_upserts:
                self._mark_refreshed()
                return {"refreshed": False, "version": self.index.version, "segment_id": self.segment_id}
            t0 = time.perf_counter()
            if self.ports.refresh is not None:
                out = self.ports.refresh(self) or {}
                self._publish_gauges()
            else:
                out = self.reindex()
            self._last_refresh_cost = time.perf_counter() - t0
            self.metrics.observe("s3_refresh_ms", self._last_refresh_cost * 1000.0, index=self.name)
            self._mark_refreshed()
            return {"refreshed": True, **out}

    def _mark_refreshed(self) -> None:
        with self._refresh_cv:
            self.refreshed_upto = max(self.refreshed_upto, self.applied_offset)
            self._refresh_cv.notify_all()

    def after_write(self, refresh: str, last_offset: int | None) -> dict:
        """The ``refresh`` query parameter: ``false`` (default) leaves it to the interval, ``true`` refreshes before
        returning, ``wait_for`` returns once a refresh covers the write (forcing one if the interval is off)."""
        self._publish_gauges()
        if last_offset is None or refresh in ("", "false"):
            return {}
        if refresh == "true":
            self.refresh()
            return {"forced_refresh": True}
        # wait_for: with the interval off nothing would ever refresh, so force one; else wait for the next scheduled one
        interval = parse_interval(self.settings["refresh_interval"])
        if interval is None:
            self.refresh()
            return {"forced_refresh": True}
        self._refresh_wake.set()
        patience = 3 * interval + 4 * self._last_refresh_cost + 5
        with self._refresh_cv:
            if self._refresh_cv.wait_for(lambda: self.refreshed_upto >= last_offset or self.pending_since is None,
                                         timeout=patience):
                return {"forced_refresh": False}
        self.refresh()
        return {"forced_refresh": True}

    def lag(self) -> int:
        """Store head minus the highest offset that is entirely visible in the live index."""
        store = self._get_store()
        if store is None or not self.ready:
            return 0
        with self._store_lock:
            head = store.head()
        visible = (self.pending_since - 1) if self.pending_since else max(self.applied_offset, self.index.version)
        return max(0, head - visible)

    def _get_store(self, create: bool = False):
        with self._store_lock:
            if self._store is None:
                if not create and not self.store_path.exists():
                    return None
                self._store = self.ports.open_store(self.store_path, self.mapping)
            return self._store

    def _publish_gauges(self) -> None:
        self.metrics.set("s3_index_ready", 1 if self.ready else 0, index=self.name)
        if self.ready:
            self.metrics.set("s3_index_version", getattr(self.index, "version", 0), index=self.name)
            self.metrics.set("s3_index_docs", getattr(self.index, "n_docs", 0), index=self.name)
        self.metrics.set("s3_cache_entries", len(self.cache), index=self.name)
        if self.ready:
            self.metrics.set("s3_index_pending_upserts", self.pending_upserts, index=self.name)
            self.metrics.set("s3_index_lag", self.lag(), index=self.name)

    # -- segments ---------------------------------------------------------------------------------------------------
    def list_segments(self) -> list[dict]:
        out = []
        if self.segments_dir.is_dir():
            for d in self.segments_dir.iterdir():
                try:
                    m = read_manifest(d)
                except (OSError, ValueError):
                    continue
                out.append({"segment_id": m.get("segment_id", d.name), "version": m.get("version", 0),
                            "created": m.get("created", ""), "n_docs": m.get("n_docs"), "_dir": d})
        out.sort(key=lambda s: (s["version"], s["created"], s["segment_id"]))
        return out

    def versions(self) -> list[dict]:
        return [{"segment_id": s["segment_id"], "version": s["version"], "created": s["created"],
                 "live": s["segment_id"] == self.segment_id} for s in self.list_segments()]

    def _segment_dir(self, segment_id: str) -> Path:
        for s in self.list_segments():
            if s["segment_id"] == segment_id:
                return s["_dir"]
        raise UnknownSegment(f"no kept segment {segment_id!r} for index {self.name!r}", path="segment_id")

    def _prune(self, protect: set[str]) -> None:
        segs = self.list_segments()
        keep = {s["segment_id"] for s in segs[-self.cfg.keep_segments:]} | protect
        for s in segs:
            if s["segment_id"] not in keep:
                shutil.rmtree(s["_dir"], ignore_errors=True)

    # -- loading and swapping ---------------------------------------------------------------------------------------
    def _load(self, seg_dir: Path):
        return self.ports.load_index(seg_dir, self.cfg.resolved_device())

    def _swap_in(self, new_index: Any, segment_id: str, kind: str) -> None:
        """Replay the log onto the new index, then replace the reference (searches in flight finish on the old one)."""
        with self._store_lock:
            store = self._get_store()
            if store is not None:
                self._replay(new_index, after=getattr(new_index, "version", 0))
        for rname, model in self.rankers.items():
            new_index.add_ranker(rname, model)
        self.index, self.segment_id = new_index, segment_id
        self.epoch += 1
        self.cache.clear()
        self.metrics.inc("s3_swaps_total", index=self.name, kind=kind)
        self._publish_gauges()
        self.start_refresher()

    def load_existing(self, segment_id: str | None = None) -> str:
        """Load a kept segment (newest when ``segment_id`` is None); used at start and by replicas."""
        with self._admin:
            segs = self.list_segments()
            if not segs:
                raise NotReady(f"index {self.name!r} has no segment under {self.segments_dir}")
            seg = segs[-1] if segment_id is None else next(
                (s for s in segs if s["segment_id"] == segment_id), None)
            if seg is None:
                raise UnknownSegment(f"no segment {segment_id!r} for index {self.name!r}", path="segment_id")
            self._swap_in(self._load(seg["_dir"]), seg["segment_id"], "load")
            return seg["segment_id"]

    # -- admin ------------------------------------------------------------------------------------------------------
    def create(self, source: dict | None) -> dict:
        with self._admin:
            if not source or not source.get("path"):
                raise InvalidRequestBody('"source.path" is required', path="source.path")
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "mapping.json").write_text(json.dumps(self.mapping))
            with self._store_lock:
                store = self._get_store(create=True)
                self.ports.ingest(source["path"], self.mapping, store, self.root / "rejected.jsonl")
                seg_dir = self.ports.build_segment(store, self.mapping, self.segments_dir)
            seg_dir = Path(seg_dir)
            self._swap_in(self._load(seg_dir), read_manifest(seg_dir)["segment_id"], "load")
            return {"name": self.name, "version": self.index.version, "segment_id": self.segment_id}

    def build_only(self, source: dict) -> dict:
        """What the indexer Job does: ingest and build, without loading an index (no search memory used)."""
        with self._admin:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "mapping.json").write_text(json.dumps(self.mapping))
            with self._store_lock:
                store = self._get_store(create=True)
                self.ports.ingest(source["path"], self.mapping, store, self.root / "rejected.jsonl")
                seg_dir = Path(self.ports.build_segment(store, self.mapping, self.segments_dir))
            m = read_manifest(seg_dir)
            self._prune({m["segment_id"]})
            return {"segment_id": m["segment_id"], "version": m["version"], "n_docs": m.get("n_docs")}

    def reindex(self) -> dict:
        with self._admin:
            with self._store_lock:
                store = self._get_store()
                if store is None:
                    raise InvalidRequestBody("this index has no store to rebuild from", path="")
                seg_dir = Path(self.ports.build_segment(store, self.mapping, self.segments_dir))
            seg_id = read_manifest(seg_dir)["segment_id"]
            self._swap_in(self._load(seg_dir), seg_id, "reindex")
            self._prune({seg_id})
            return {"segment_id": self.segment_id, "version": self.index.version}

    def rollback(self, segment_id: str) -> dict:
        with self._admin:
            seg_dir = self._segment_dir(segment_id)
            self._swap_in(self._load(seg_dir), segment_id, "rollback")
            return {"segment_id": self.segment_id, "version": self.index.version}

    def add_ranker(self, name: str, path: str) -> dict:
        """Load a trained ranker file (under the data dir) and register it for ``rescore.s3_ranker``."""
        if not isinstance(name, str) or not name or not isinstance(path, str) or not path:
            raise InvalidRequestBody('body is {"name": "...", "path": "..."}', path="name")
        if self.ports.load_ranker is None:
            raise InvalidRequestBody("no ranker loader configured", path="path")
        base = Path(self.cfg.data_dir).resolve()
        full = (base / path).resolve()
        if base not in full.parents:
            raise InvalidRequestBody("ranker files must live under the serving data dir", path="path")
        if not full.is_file():
            raise InvalidRequestBody(f"no ranker file {path!r} under the data dir", path="path")
        with self._admin:
            if not self.ready:
                raise NotReady(f"index {self.name!r} is not loaded")
            try:
                model = self.ports.load_ranker(full)
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidRequestBody(f"not a ranker file: {exc}", path="path") from None
            self.index.add_ranker(name, model)
            self.rankers[name] = model
            self.epoch += 1
            self.cache.clear()
        return {"name": name, "rankers": sorted(self.rankers)}

    def field_capabilities(self) -> dict:
        """Per field (sub-fields as ``f.keyword``): type and what it can do, for the demo page and other clients."""
        hot = set(((self.mapping.get("_meta") or {}).get("engine") or {}).get("hot") or [])
        out: dict = {}
        scalar = ("keyword", "float", "integer", "long", "boolean")

        def add(fname: str, prop: dict) -> None:
            t = prop.get("type")
            cap = {"type": t, "searchable": t == "text", "filterable": t in scalar, "facetable": t == "keyword",
                   "sortable": t in scalar, "dense": t == "knn_vector", "hot": fname in hot}
            if t == "knn_vector":
                cap["dimension"] = prop.get("dimension")
            out[fname] = cap
            for sub, sp in (prop.get("fields") or {}).items():
                add(f"{fname}.{sub}", sp)

        for fname, prop in (self.mapping.get("properties") or {}).items():
            add(fname, prop)
        return out

    def add_docs(self, changes: list[dict]) -> dict:
        if not isinstance(changes, list) or not changes:
            raise InvalidRequestBody("body must be a non-empty list of change records", path="")
        for i, ch in enumerate(changes):
            if not isinstance(ch, dict) or ch.get("op") not in VALID_OPS or not isinstance(ch.get("_id"), str):
                raise InvalidRequestBody(f'need {{"op": upsert|patch|delete, "_id": str, ...}}', path=f"[{i}]")
            if ch["op"] == "upsert" and not isinstance(ch.get("doc"), dict):
                raise InvalidRequestBody('"doc" must be an object', path=f"[{i}].doc")
            if ch["op"] == "patch" and not isinstance(ch.get("fields"), dict):
                raise InvalidRequestBody('"fields" must be an object', path=f"[{i}].fields")
        if not self.ready:
            raise NotReady(f"index {self.name!r} is not loaded")
        with self._admin:
            with self._store_lock:
                store = self._get_store()
                if store is None:
                    raise InvalidRequestBody("this index has no store to write to", path="")
                ready, rejected = self.ports.prepare(changes, self.mapping)
                if not ready:
                    return {"first_offset": None, "last_offset": None, "rejected": rejected,
                            "applied": {"patched": 0, "deleted": 0, "pending_upserts": self.pending_upserts}}
                first, last = store.apply(ready)
                applied = self._replay(self.index, after=self.applied_offset)
            self.epoch += 1
            self.cache.clear()
            self._publish_gauges()
            return {"first_offset": first, "last_offset": last, "rejected": rejected, "applied": applied}

    # -- _bulk ------------------------------------------------------------------------------------------------------
    def bulk_chunk(self, actions: list) -> tuple[list[dict], int | None]:
        """Plan, apply (one atomic store batch) and replay one chunk of actions. ``(items, last_offset or None)``."""
        if not self.ready:
            raise NotReady(f"index {self.name!r} is not loaded")
        if self.ports.apply_changes is None:
            raise InvalidRequestBody("no apply_changes port configured", path="")
        m = self.mapping
        meta = (m.get("_meta") or {}).get("engine") or {}
        id_field, hot = meta.get("id_field", "_id"), set(meta.get("hot") or [])
        with self._admin:
            with self._store_lock:
                store = self._get_store()
                if store is None:
                    raise InvalidRequestBody("this index has no store to write to", path="")
                ids = bulk_mod.chunk_ids(actions, id_field)
                existing = dict(zip(ids, store.get(ids))) if ids else {}
                plan = bulk_mod.plan_chunk(actions, self.name, hot, id_field, existing)
                changes = [p.change for p in plan if p.change is not None]
                report = self.ports.apply_changes(changes, m, store) if changes else None
                if report is not None and report.accepted:
                    self._replay(self.index, after=self.applied_offset)
            if report is not None and report.accepted:
                self.epoch += 1
                self.cache.clear()
        last = report.last_offset if report is not None and report.accepted else None
        return bulk_mod.finish(plan, report, self.name), last

    def bulk(self, lines, refresh: str = "false") -> tuple[int, dict]:
        """Stream actions in chunks. Returns ``(http_status, response)``; every action has an item."""
        t0 = time.perf_counter()
        items: list[dict] = []
        last_offset: int | None = None
        failure: dict | None = None
        chunk: list = []

        def flush() -> bool:
            nonlocal last_offset, failure
            if not chunk:
                return True
            try:
                got, last = self.bulk_chunk(chunk)
            except Exception as exc:  # noqa: BLE001 - this chunk was not applied; say so per action and stop
                from .errors import classify
                status, body = classify(exc)
                err = {**body["error"], "type": body["error"]["type"]}
                for a in chunk:
                    items.append({a.op: {"_index": self.name, "_id": a.meta.get("_id"), "status": status, "error": err}})
                failure = {"status": status, **body["error"]}
                chunk.clear()
                return False
            items.extend(got)
            chunk.clear()
            if last is not None:
                last_offset = last
            return True

        fmt: bulk_mod.BulkFormatError | None = None
        try:
            for act in lines:
                chunk.append(act)
                if len(chunk) >= self.cfg.bulk_chunk and not flush():
                    break
        except bulk_mod.BulkFormatError as exc:
            fmt = exc
        if failure is None:
            flush()
        took = (time.perf_counter() - t0) * 1000.0
        errors = any("error" in next(iter(i.values())) for i in items)
        m = self.metrics
        m.inc("s3_bulk_requests_total", index=self.name, refresh=refresh or "false")
        for it in items:
            op, body = next(iter(it.items()))
            m.inc("s3_bulk_items_total", index=self.name, op=op, outcome="error" if "error" in body else "ok")
            if "error" in body:
                m.inc("s3_bulk_rejected_total", index=self.name,
                      reason=body["error"].get("s3_reason") or body["error"]["type"])
        m.observe("s3_bulk_ms", took, index=self.name)
        if took > 0:
            m.set("s3_bulk_items_per_second", len(items) * 1000.0 / took, index=self.name)
        out: dict = {"took": int(took), "errors": errors, "items": items}
        status = 200
        if fmt is not None:
            status, out["error"] = 400, {"type": "illegal_argument_exception", "reason": fmt.reason, "line": fmt.line,
                                         "path": ""}
            out["errors"] = True
        elif failure is not None:
            status, out["error"] = failure["status"], {k: v for k, v in failure.items() if k != "status"}
            out["errors"] = True
        if last_offset is not None and failure is None:
            out.update(self.after_write(refresh, last_offset))
        else:
            self._publish_gauges()
        return status, out

    def _replay(self, index: Any, after: int) -> dict:
        """Apply the store's changes after ``after`` to ``index``: patch, delete; upserts wait for a rebuild."""
        store = self._get_store()
        patched = deleted = pending = 0
        first_pending: int | None = None
        last = after
        runs: dict[str, dict[str, Any]] = {}

        def tolerant(fn, ids: list, *rest) -> None:
            """Ids that are pending upserts are not in the segment yet (the engine raises NoSuchKey): they get their
            patch or delete from the store at the next refresh, so skip them one by one."""
            try:
                fn(ids, *rest)
            except Exception as exc:  # noqa: BLE001
                if not any(c.__name__ in ("NoSuchKey", "KeyError") for c in type(exc).__mro__):
                    raise
                for i, _id in enumerate(ids):
                    try:
                        fn([_id], *[[r[i]] for r in rest])
                    except Exception as exc2:  # noqa: BLE001
                        if not any(c.__name__ in ("NoSuchKey", "KeyError") for c in type(exc2).__mro__):
                            raise

        def flush() -> None:
            nonlocal patched
            for fld, kv in runs.items():
                tolerant(lambda ids, vals, f=fld: index.patch(f, ids, vals), list(kv.keys()), list(kv.values()))
                patched += len(kv)
                self.metrics.inc("s3_patches_total", len(kv))
            runs.clear()

        while True:
            try:
                batch = store.changes(last, 10_000, docs=False)
            except TypeError:
                batch = store.changes(last, 10_000)
            if not batch:
                break
            for ch in batch:
                last = ch["offset"]
                if ch["op"] == "patch":
                    for fld, val in ch["fields"].items():
                        runs.setdefault(fld, {})[ch["_id"]] = val
                elif ch["op"] == "delete":
                    flush()
                    tolerant(index.delete, [ch["_id"]])
                    deleted += 1
                else:
                    pending += 1
                    if first_pending is None:
                        first_pending = ch["offset"]
        flush()
        if last != after and hasattr(index, "set_version"):
            index.set_version(last)
        self.applied_offset = last
        # on the live index upserts accumulate; a freshly loaded index starts from those after its own segment
        self.pending_upserts = self.pending_upserts + pending if index is self.index else pending
        if index is self.index:
            self.pending_since = self.pending_since or first_pending
        else:
            self.pending_since = first_pending
        if not self.pending_upserts:
            self.pending_since = None
        return {"patched": patched, "deleted": deleted, "pending_upserts": self.pending_upserts}

    # -- search -----------------------------------------------------------------------------------------------------
    def _key(self, request: dict) -> tuple:
        return (self.segment_id, self.epoch, request_key(request))

    def search(self, request: dict) -> tuple[dict, dict]:
        """One request through the cache and the scheduler. Returns ``(response, info)``."""
        if not self.ready:
            raise NotReady(f"index {self.name!r} is not loaded")
        t0 = time.perf_counter()
        key = self._key(request)
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.inc("s3_cache_hits_total", index=self.name)
            return cached, {"cache": "hit", "queue_ms": 0.0, "serve_ms": (time.perf_counter() - t0) * 1000.0}
        self.metrics.inc("s3_cache_misses_total", index=self.name)
        fut = self.scheduler.submit(request)
        resp, wait = fut.result(timeout=self.cfg.request_timeout_s)
        self.cache.put(key, resp)
        self._observe(resp, t0)
        return resp, {"cache": "miss", "queue_ms": wait, "serve_ms": (time.perf_counter() - t0) * 1000.0}

    def msearch(self, requests: list[dict]) -> list[dict]:
        """The whole list as one ``search_batch`` (cache hits are answered from the cache, the rest batched)."""
        if not self.ready:
            raise NotReady(f"index {self.name!r} is not loaded")
        t0 = time.perf_counter()
        index = self.index
        keys = [self._key(r) for r in requests]
        out: list[dict | None] = [self.cache.get(k) for k in keys]
        miss = [i for i, r in enumerate(out) if r is None]
        self.metrics.inc("s3_cache_hits_total", len(requests) - len(miss), index=self.name)
        self.metrics.inc("s3_cache_misses_total", len(miss), index=self.name)
        if miss:
            self.metrics.inc("s3_batches_total", index=self.name)
            self.metrics.observe("s3_batch_size", len(miss), index=self.name)
            resp = search_batch(index, [requests[i] for i in miss], self.cfg.columns)
            for i, r in zip(miss, resp):
                out[i] = r
                self.cache.put(keys[i], r)
                self._observe(r, t0)
        return out  # type: ignore[return-value]

    def _observe(self, resp: dict, t0: float) -> None:
        self.metrics.observe("s3_request_ms", (time.perf_counter() - t0) * 1000.0, index=self.name)
        for stage, ms in (resp.get("took_ms") or {}).items():
            self.metrics.observe("s3_stage_ms", float(ms), index=self.name, stage=stage)
        self.metrics.set("s3_cache_entries", len(self.cache), index=self.name)

    # -- _source ----------------------------------------------------------------------------------------------------
    def attach_source(self, response: dict, spec: Any) -> None:
        """Add ``_source`` to each hit from the store. ``spec``: True, a list of fields or {includes, excludes}."""
        hits = (response.get("hits") or {}).get("hits") or []
        if not hits:
            return
        with self._store_lock:
            store = self._get_store()
            if store is None:
                raise InvalidRequestBody("_source needs the store, which this replica does not have", path="_source")
            docs = store.get([h["_id"] for h in hits])
        vec_fields = {f for f, p in (self.mapping.get("properties") or {}).items() if p.get("type") == "knn_vector"}
        inc = exc = None
        if isinstance(spec, list):
            inc = set(spec)
        elif isinstance(spec, dict):
            inc = set(spec["includes"]) if spec.get("includes") else None
            exc = set(spec.get("excludes") or [])
        for h, doc in zip(hits, docs):
            if doc is None:
                continue
            src = {k: v for k, v in doc.items() if k not in vec_fields}
            if inc is not None:
                src = {k: v for k, v in src.items() if k in inc}
            if exc:
                src = {k: v for k, v in src.items() if k not in exc}
            h["_source"] = src

    def info(self) -> dict:
        return {"name": self.name, "ready": self.ready, "segment_id": self.segment_id,
                "version": getattr(self.index, "version", None), "n_docs": getattr(self.index, "n_docs", None),
                "device": self.cfg.resolved_device(), "rankers": sorted(self.rankers), "pending_upserts": self.pending_upserts,
                "cache": {"entries": len(self.cache), "hits": self.cache.hits, "misses": self.cache.misses}}


class IndexManager:
    def __init__(self, cfg: ServingConfig, ports: Ports, metrics: Metrics):
        self.cfg, self.ports, self.metrics = cfg, ports, metrics
        self.root = Path(cfg.data_dir)
        self._indexes: dict[str, ManagedIndex] = {}
        self._lock = threading.Lock()

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._indexes)

    def get(self, name: str) -> ManagedIndex:
        with self._lock:
            try:
                return self._indexes[name]
            except KeyError:
                raise UnknownIndex(f"no index named {name!r}", path="name") from None

    def _mapping_on_disk(self, name: str) -> dict:
        p = self.root / name / "mapping.json"
        if not p.exists():
            raise UnknownIndex(f"no index named {name!r} under {self.root}", path="name")
        return json.loads(p.read_text())

    def open_existing(self, name: str, segment_id: str | None = None) -> ManagedIndex:
        """Register and load an index that is already on disk (start-up preload, a replica pod)."""
        mi = ManagedIndex(name, self.root / name, self._mapping_on_disk(name), self.cfg, self.ports, self.metrics)
        mi.load_existing(segment_id)
        for rname, rpath in self.cfg.rankers.items():
            mi.add_ranker(rname, rpath)
        with self._lock:
            self._indexes[name] = mi
        return mi

    def create(self, name: str, mapping: dict, source: dict | None) -> dict:
        if not NAME_RE.match(name):
            raise InvalidRequestBody("index names are lowercase letters, digits, '.', '_' and '-'", path="name")
        if not isinstance(mapping, dict) or not isinstance(mapping.get("properties"), dict):
            raise InvalidRequestBody('"mapping" must be an OpenSearch mapping body with "properties"', path="mapping")
        with self._lock:
            if name in self._indexes or (self.root / name / "mapping.json").exists():
                raise IndexExists(f"index {name!r} exists", path="name")
            mi = ManagedIndex(name, self.root / name, mapping, self.cfg, self.ports, self.metrics)
            self._indexes[name] = mi
        try:
            return mi.create(source)
        except BaseException:
            with self._lock:
                self._indexes.pop(name, None)
            mi.close()
            shutil.rmtree(mi.root, ignore_errors=True)  # a failed create leaves nothing behind
            raise

    def ready(self) -> bool:
        with self._lock:
            return bool(self._indexes) and all(i.ready for i in self._indexes.values())

    def close(self) -> None:
        with self._lock:
            for i in self._indexes.values():
                i.close()
