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
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .cache import LRUCache, request_key
from .config import ServingConfig
from .errors import (IndexExists, InvalidRequestBody, NotReady, UnknownIndex, UnknownSegment)
from .metrics import Metrics
from .ports import Ports, read_manifest
from .scheduler import BatchScheduler

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
        self._admin = threading.RLock()          # one admin operation at a time
        self._store_lock = threading.RLock()     # the store is one SQLite connection
        self._store: Any = None
        self.cache = LRUCache(cfg.cache_size)
        self.scheduler = BatchScheduler(lambda: self.index, name=name, max_batch=cfg.max_batch,
                                        max_wait_ms=cfg.max_wait_ms, max_queue=cfg.max_queue, metrics=metrics)

    # -- state ------------------------------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.index is not None

    def close(self) -> None:
        self.scheduler.close()

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
        self.index, self.segment_id = new_index, segment_id
        self.epoch += 1
        self.cache.clear()
        self.metrics.inc("s3_swaps_total", index=self.name, kind=kind)
        self._publish_gauges()

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

    def _replay(self, index: Any, after: int) -> dict:
        """Apply the store's changes after ``after`` to ``index``: patch, delete; upserts wait for a rebuild."""
        store = self._get_store()
        patched = deleted = pending = 0
        last = after
        runs: dict[str, dict[str, Any]] = {}

        def flush() -> None:
            nonlocal patched
            for fld, kv in runs.items():
                index.patch(fld, list(kv.keys()), list(kv.values()))
                patched += len(kv)
                self.metrics.inc("s3_patches_total", len(kv))
            runs.clear()

        while True:
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
                    index.delete([ch["_id"]])
                    deleted += 1
                else:
                    pending += 1
        flush()
        if last != after and hasattr(index, "set_version"):
            index.set_version(last)
        self.applied_offset = last
        # on the live index upserts accumulate; a freshly loaded index starts from those after its own segment
        self.pending_upserts = self.pending_upserts + pending if index is self.index else pending
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
            resp = index.search_batch([requests[i] for i in miss])
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
                "device": self.cfg.resolved_device(), "pending_upserts": self.pending_upserts,
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
