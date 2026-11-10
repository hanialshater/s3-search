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
from .segment import NumericColumn, read_segment, write_segment

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
    """One segment loaded on one device, searchable, with the updater's hooks (patch, delete, set_version).

    Thread-safe: search, search_batch, patch, delete, set_version, save and candidate_features take one re-entrant
    lock per Index, so a search sees a patch or delete call either entirely or not at all. Searches on one Index
    therefore run one batch at a time (serving batches requests with search_batch).
    """

    def __init__(self, seg, device: str = "cpu"):
        self._st = IndexState(seg, get_backend(device))
        self.rankers: dict = {}
        self._lock = threading.RLock()

    # ---- construction
    @classmethod
    def from_segment(cls, path, device: str = "cpu", verify: bool = True) -> "Index":
        return cls(read_segment(path, verify=verify), device)

    @_locked
    def save(self, path) -> Path:
        """Write the index as a segment (``path``: the segment directory, new or empty), including deletes and
        hot-field patches applied since it was loaded, at the current ``version``."""
        st = self._st
        seg = st.seg
        b = st.b
        seg.live = b.pack_bits(st.live) if st.b.name == "cpu" else b.pack_bits(b.to_numpy(st.live))
        for f in seg.numerics:
            seg.numerics[f] = NumericColumn(values=np.asarray(b.to_numpy(st.numerics[f])),
                                            missing=b.pack_bits(b.to_numpy(st.missing[f])))
        seg.manifest["version"] = st.version
        return write_segment(seg, path)

    # ---- properties
    @property
    def version(self) -> int:
        return self._st.version

    @property
    def n_docs(self) -> int:
        return self._st.N

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
        return self._st.manifest.get("segment_id", "")

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
        missing = [i for i in ids if i not in self._st.pos]
        if missing:
            raise NoSuchKey(f"unknown _id(s): {missing[:5]}")
        if len(ids) != len(values):
            raise ValueError("ids and values differ in length")
        self._st.patch(field, list(ids), list(values))

    @_locked
    def delete(self, ids: Sequence[str]) -> None:
        """KeyError for ids the segment never held; already-deleted ids are a no-op."""
        missing = [i for i in ids if i not in self._st.pos]
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


def build_index(catalogue: pd.DataFrame, mapping: dict, device: str = "cpu") -> Index:
    if not isinstance(catalogue, pd.DataFrame):
        raise TypeError("catalogue: a pandas DataFrame")
    return Index(build_segment(catalogue, mapping), device)


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
