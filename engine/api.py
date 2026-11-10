"""The engine's API (docs/INTERFACES.md §5). The referee's contract tests are written against exactly this surface.

    build_index(catalogue, mapping, device="cpu") -> Index
    Index.search(request) -> response;  Index.search_batch(requests) -> [response]
    Index.from_segment(path, device) -> Index;  Index.save(path) -> path
    available_devices() -> ["cpu", ...];  UnsupportedQuery, InvalidRequest, InvalidMapping
"""
from __future__ import annotations

import functools
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .backend import available_devices, get_backend
from .build import build_segment
from .errors import (CorruptSegment, DeviceUnavailable, InvalidMapping, InvalidRequest, NoSuchKey, NotHot,
                     OverBudget, UnsupportedQuery)
from .index_state import IndexState
from .plan.execute import Executor
from .plan.plan import plan as make_plan
from .segment import SegmentArrays, read_segment, write_segment

__all__ = ["build_index", "Index", "available_devices", "UnsupportedQuery", "InvalidRequest", "InvalidMapping",
           "OverBudget", "DeviceUnavailable", "NoSuchKey", "CorruptSegment",
           "NotHot", "save_ranker", "load_ranker", "RANKER_FORMAT"]


def _locked(fn):
    @functools.wraps(fn)
    def run(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)
    return run


class Index:
    """An ordered list of segments (one, or a base plus deltas: S8) loaded on one device, searchable, with the
    updater's hooks (patch, delete, set_version) and the multi-segment hooks (add_segment, replace_segments,
    remove_segment, snapshot_segments, save_commit).

    Thread-safe: every public method that reads or changes the index takes one re-entrant lock per Index, so a
    search sees a patch, delete, add or replace either entirely or not at all. Searches on one Index therefore run
    one batch at a time (serving batches requests with search_batch).
    """

    def __init__(self, seg, device: str = "cpu", accumulate: str | None = None, version: int | None = None):
        """``seg``: a SegmentArrays, or a list of them (oldest first; without a commit, a key live in several
        keeps its newest copy). ``accumulate``: "float64" (the default reference) or "float32" (§0)."""
        self._st = IndexState(seg, get_backend(device, accumulate), version=version)
        self.rankers: dict = {}
        self._lock = threading.RLock()

    # ---- construction
    @classmethod
    def from_segment(cls, path, device: str = "cpu", verify: bool = True, accumulate: str | None = None) -> "Index":
        return cls(read_segment(path, verify=verify), device, accumulate)

    @classmethod
    def from_segments(cls, paths, device: str = "cpu", verify: bool = True, accumulate: str | None = None) -> "Index":
        return cls([read_segment(p, verify=verify) for p in paths], device, accumulate)

    @classmethod
    def from_commit(cls, path, device: str = "cpu", verify: bool = True, accumulate: str | None = None) -> "Index":
        """The index a commit names: its segments (directories named by segment_id, next to the commit file) with
        the commit's live bitmaps, at the commit's version."""
        from .commit import read_commit
        path = Path(path)
        c = read_commit(path, verify=verify)
        segs = []
        for sid in c["segments"]:
            seg = read_segment(path.parent / sid, verify=verify)
            if sid in c["live"]:
                seg.live = np.asarray(c["live"][sid], np.uint32)
            segs.append(seg)
        ix = cls.__new__(cls)
        ix._st = IndexState(segs, get_backend(device, accumulate), version=int(c["version"]), newest_wins=False)
        ix.rankers, ix._lock = {}, threading.RLock()
        return ix

    @_locked
    def save(self, path) -> Path:
        """Write the index as one segment (``path``: the segment directory, new or empty): its live docs, with
        hot-field patches, at the current ``version`` (a merge of all its segments)."""
        from .build import merge_segments
        st = self._st
        seg = merge_segments([p.snapshot() for p in st.parts], version=st.version)
        return write_segment(seg, path)

    @_locked
    def save_commit(self, out_dir) -> Path:
        """Write ``commit-<version>.json`` in ``out_dir`` (§4's format and checksum), with a delete file for each
        segment that has deletes; a segment not yet on disk there is written to ``out_dir/<segment_id>``."""
        from .commit import write_commit
        from .index_state import _pack
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        live = {}
        for p in self._st.parts:
            d = out / p.segment_id
            if not (d / "manifest.json").exists():
                seg = p.seg
                written = write_segment(_copy_for_write(seg), d)
                if written.name != p.segment_id:      # write_segment derives the id from the content
                    p.segment_id = written.name
            if not p.local_live.all():
                live[p.segment_id] = _pack(p.local_live)
        return write_commit(out, self._st.version, [p.segment_id for p in self._st.parts], live)

    # ---- segments (S8)
    @property
    def segments(self) -> list[dict]:
        return [{"segment_id": p.segment_id, "version": p.version, "n_docs": p.n, "live": int(p.local_live.sum()),
                 "device_nbytes": p.device_nbytes} for p in self._st.parts]

    def _segment_arrays(self, seg):
        return seg if isinstance(seg, SegmentArrays) else read_segment(seg)

    def add_segment(self, seg, version: int, delete_ids=()) -> None:
        """Atomically: tombstone ``delete_ids`` and the new segment's ids in the older segments (unknown ids are
        ignored), add the segment, update the global statistics, set the version. Incremental (S8-2b): the new
        segment's arrays go to the device before the lock is taken; under it, only the delta's docs, terms and
        values are touched (edited keys keep their slot, new keys take new slots at the end)."""
        from .index_state import Part
        part = Part(self._segment_arrays(seg), self._st.b)      # outside the lock: the device upload
        with self._lock:
            st = self._st
            st.check_part(part)
            st.add_part(part, delete_ids)
            st.version = int(version)

    def replace_segments(self, old_ids, new=None, version: int | None = None, delete_ids=()) -> None:
        """Atomically replace the segments ``old_ids`` by ``new`` (their merge, or None to drop them).
        ``delete_ids`` are tombstoned in ``new``; docs of ``new`` that are no longer live in the old segments (changed
        since the snapshot) stay dead. The merged docs keep their slots, so the order and statistics are those of
        the live set."""
        from .index_state import Part
        part = Part(self._segment_arrays(new), self._st.b) if new is not None else None
        with self._lock:
            st = self._st
            old = [st.part(i) for i in old_ids]
            if part is not None:
                st.check_part(part)
                owned = {q.ids[loc] for q in old for loc in q.live_local.tolist()}
                clash = [i for i in part.ids[np.flatnonzero(part.local_live)].tolist()
                         if i in st.pos and i not in owned]
                if clash:
                    raise ValueError(f"_id {clash[0]!r} is live in a segment that is not replaced")
            if part is None and len(old) == len(st.parts):
                raise ValueError("an index needs at least one segment")
            st.replace_parts(old, part, delete_ids)
            if version is not None:
                st.version = int(version)

    def remove_segment(self, segment_id: str) -> None:
        self.replace_segments([segment_id], None)

    @_locked
    def snapshot_segments(self, segment_ids=None):
        """``(segments, version)``: SegmentArrays of the named segments (all by default) as they are now, sharing
        their immutable arrays and copying their live bits and hot columns, for ``engine.build.merge_segments``."""
        st = self._st
        parts = st.parts if segment_ids is None else [st.part(i) for i in segment_ids]
        return [p.snapshot() for p in parts], st.version

    # ---- properties
    @property
    def version(self) -> int:
        return self._st.version

    @property
    def n_docs(self) -> int:
        return self._st.n_live

    @property
    def device(self) -> str:
        return self._st.b.name

    @property
    def mapping(self) -> dict:
        return self._st.mapping.raw

    @property
    def device_nbytes(self) -> int:
        return self._st.device_nbytes

    @property
    def segment_id(self) -> str:
        return self._st.parts[0].segment_id if len(self._st.parts) == 1 else ""

    # ---- search
    def search(self, request: dict) -> dict:
        return self.search_batch([request])[0]

    @_locked
    def search_batch(self, requests: list[dict]) -> list[dict]:
        if not isinstance(requests, (list, tuple)):
            raise InvalidRequest("$", "search_batch takes a list of requests")
        plans, parse_ms = [], []
        for r in requests:
            t0 = time.perf_counter()
            plans.append(make_plan(r, self._st))
            parse_ms.append((time.perf_counter() - t0) * 1000.0)
        out = Executor(self._st, self.rankers).run(plans) if plans else []
        for resp, ms in zip(out, parse_ms):
            resp["took_ms"] = {"parse": round(ms, 3), **resp["took_ms"]}
            resp["took_ms"]["total"] = round(resp["took_ms"]["total"] + ms, 3)
        return out

    # ---- the updater's hooks (docs/INTERFACES.md §3, §4)
    @_locked
    def patch(self, field: str, ids: Sequence[str], values: Sequence) -> None:
        """Hot fields only (NotHot otherwise); KeyError for unknown ids; deleted ids are skipped. ``None`` or NaN
        makes the value missing."""
        if field not in self._st.mapping.hot:
            raise NotHot(f"{field!r} is not a hot field (_meta.engine.hot: {self._st.mapping.hot})")
        missing = [i for i in ids if i not in self._st.known]
        if missing:
            raise NoSuchKey(f"unknown _id(s): {missing[:5]}")
        if len(ids) != len(values):
            raise ValueError("ids and values differ in length")
        self._st.patch(field, list(ids), list(values))

    @_locked
    def delete(self, ids: Sequence[str]) -> None:
        """KeyError for ids the segment never held; already-deleted ids are a no-op."""
        missing = [i for i in ids if i not in self._st.known]
        if missing:
            raise NoSuchKey(f"unknown _id(s): {missing[:5]}")
        self._st.delete(list(ids))

    @_locked
    def set_version(self, version: int) -> None:
        self._st.version = int(version)

    # ---- ranking
    def add_ranker(self, name: str, model) -> None:
        """Register a model with ``predict(X: float32[n, n_features]) -> float32[n]`` for ``rescore.s3_ranker``."""
        self.rankers[name] = model

    def feature_names(self) -> list[str]:
        from .rank.features import feature_names_for
        return feature_names_for(self._st)

    @_locked
    def candidate_features(self, request: dict, window: int = 1000):
        """The top ``window`` candidates of ``request`` and their ranker features: ``(ids list[str], X f32[n, F])``."""
        from .rank.features import candidate_features
        st, b = self._st, self._st.b
        p = make_plan(request, st)
        ex = Executor(st, self.rankers)
        from .plan.execute import Timer
        ex._prefetch([p], Timer(b))
        ctx = {"explain": None}
        score, match = ex.eval(p.root, ctx)
        match = match & st.live
        idx, _ = b.topk(b.where(match, score, 0.0).reshape(1, -1), min(window, st.N), match.reshape(1, -1))
        X, cand = candidate_features(ex, ctx, idx)
        cand_h = b.to_numpy(cand)[0]
        ok = cand_h >= 0
        return [str(st.ids[d]) for d in cand_h[ok]], np.asarray(b.to_numpy(X)[0][ok], np.float32)


def _copy_for_write(seg):
    """A shallow copy whose manifest write_segment may update without touching the loaded segment's."""
    import copy
    c = copy.copy(seg)
    c.manifest = dict(seg.manifest)
    return c


def build_index(catalogue: pd.DataFrame, mapping: dict, device: str = "cpu", accumulate: str | None = None) -> Index:
    if not isinstance(catalogue, pd.DataFrame):
        raise TypeError("catalogue: a pandas DataFrame")
    return Index(build_segment(catalogue, mapping), device, accumulate)


# ---- rankers on disk (docs/INTERFACES.md §5): JSON, no pickle
RANKER_FORMAT, RANKER_FORMAT_VERSION = "s3-ranker", 1


def save_ranker(model, path, feature_names: Sequence[str] | None = None) -> Path:
    """Write a ranker as JSON: ``{"format": "s3-ranker", "format_version": 1, "kind": "gbdt",
    "feature_names": [...] | null, "model": GBDT.to_dict()}``. Written to ``<path>.tmp`` and renamed."""
    import json
    import os
    from .rank.gbdt import GBDT
    if not isinstance(model, GBDT):
        raise TypeError("save_ranker: v1 stores engine.rank.gbdt.GBDT models only")
    path = Path(path)
    doc = {"format": RANKER_FORMAT, "format_version": RANKER_FORMAT_VERSION, "kind": "gbdt",
           "feature_names": list(feature_names) if feature_names is not None else None, "model": model.to_dict()}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":")))
    os.replace(tmp, path)
    return path


def load_ranker(path):
    """Read a ranker written by ``save_ranker``, or a bare ``GBDT.to_dict()`` JSON (serving's interim format).
    Returns the model (``predict``, ``contributions``); the stored ``feature_names`` are on ``model.feature_names``."""
    import json
    from .rank.gbdt import GBDT
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: not a ranker")
    if "format" in doc:
        if doc.get("format") != RANKER_FORMAT or doc.get("format_version") != RANKER_FORMAT_VERSION \
                or doc.get("kind") != "gbdt":
            raise ValueError(f"{path}: format {doc.get('format')} v{doc.get('format_version')} "
                             f"kind {doc.get('kind')}; expected {RANKER_FORMAT} v{RANKER_FORMAT_VERSION} gbdt")
        model, names = GBDT.from_dict(doc["model"]), doc.get("feature_names")
    elif {"params", "trees", "edges", "base"} <= set(doc):
        model, names = GBDT.from_dict(doc), None
    else:
        raise ValueError(f"{path}: not a ranker")
    model.feature_names = names
    return model
