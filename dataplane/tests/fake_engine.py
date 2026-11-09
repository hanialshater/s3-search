"""A small stand-in for the engine, following docs/INTERFACES.md §4 and §5, for the data plane's tests while
``engine.build``, ``engine.segment`` and ``engine.api`` are being written. It covers only what the data plane
touches: numeric columns (the hot fields), the id table, the live bitmap, the manifest with checksums, and a search
over ``match_all``/``term``/``range``/``bool.filter`` with ``sort``, ``from`` and ``size``.

Remove it once the integration tests (``test_integration.py``) run against the real engine.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataplane._engine import EngineHooks
from dataplane.mapping import as_mapping

NUM_DTYPES = {"float": "float32", "integer": "int32", "long": "int64", "boolean": "bool"}


class NotHot(ValueError):
    pass


@dataclass
class FakeSegment:
    manifest: dict
    ids: np.ndarray
    first_offset: np.ndarray
    numerics: dict          # field -> (values, missing bool[N])
    live: np.ndarray        # bool[N] (packed on write)


def build_segment(catalogue: pd.DataFrame, mapping: dict) -> FakeSegment:
    m = as_mapping(mapping)
    n = len(catalogue)
    ids = catalogue[m.id_field].astype(str).to_numpy(dtype=object)
    first = (catalogue["_first_offset"].to_numpy(dtype=np.int64) if "_first_offset" in catalogue
             else np.arange(n, dtype=np.int64))
    nums = {}
    for name, f in m.fields.items():
        if f.type in NUM_DTYPES:
            col = catalogue[name] if name in catalogue else pd.Series([None] * n)
            missing = col.isna().to_numpy()
            dt = NUM_DTYPES[f.type]
            vals = np.zeros(n, dtype=dt)
            if (~missing).any():
                vals[~missing] = col[~missing].to_numpy().astype(dt)
            nums[name] = (vals, missing)
    manifest = {"format": "s3seg", "format_version": 1, "version": 0, "n_docs": n, "mapping": m.raw,
                "numeric_fields": {k: NUM_DTYPES[m.fields[k].type] for k in nums}, "hot": list(m.hot)}
    return FakeSegment(manifest, ids, first, nums, np.ones(n, dtype=bool))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_segment(seg: FakeSegment, path, exact: bool = False) -> Path:
    """``path`` is the parent directory (the segment goes in ``seg-<version>-<8 hex>/``), or with ``exact`` the
    segment directory itself: the interfaces leave this open, and the indexer accepts both."""
    tmp = Path(path) / ".writing"
    tmp.mkdir(parents=True)
    pq.write_table(pa.table({"_id": pa.array(list(seg.ids), pa.string()),
                             "first_offset": pa.array(seg.first_offset, pa.int64())}), tmp / "ids.parquet")
    np.save(tmp / "live.npy", np.packbits(seg.live, bitorder="little").view(np.uint8))
    (tmp / "num").mkdir()
    nbytes = 0
    for name, (vals, missing) in seg.numerics.items():
        np.save(tmp / "num" / f"{name}.npy", vals)
        np.save(tmp / "num" / f"{name}.missing.npy", np.packbits(missing, bitorder="little"))
        nbytes += vals.nbytes + missing.nbytes
    files = {str(p.relative_to(tmp)): {"sha256": _sha(p), "bytes": p.stat().st_size}
             for p in sorted(tmp.rglob("*")) if p.is_file()}
    checksum = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    man = dict(seg.manifest, files=files, checksum=checksum, nbytes=nbytes, device_nbytes=nbytes + seg.live.nbytes,
               segment_id=f"seg-{seg.manifest['version']}-{checksum[:8]}", created="2026-11-09T00:00:00Z")
    (tmp / "manifest.json").write_text(json.dumps(man))
    final = Path(path) if exact else Path(path) / man["segment_id"]
    if exact:
        for p in tmp.iterdir():
            p.rename(final / p.name)
        tmp.rmdir()
    else:
        tmp.rename(final)
    return final


class FakeIndex:
    def __init__(self, path: Path, device: str):
        path = Path(path)
        self.manifest = json.loads((path / "manifest.json").read_text())
        for rel, meta in self.manifest["files"].items():
            assert _sha(path / rel) == meta["sha256"], f"corrupt {rel}"
        self.version = int(self.manifest["version"])
        self.n_docs = int(self.manifest["n_docs"])
        self.device = device
        self.mapping = self.manifest["mapping"]
        self.device_nbytes = int(self.manifest["device_nbytes"])
        self.hot = set(self.manifest["hot"])
        self.ids = pq.read_table(path / "ids.parquet").column("_id").to_pylist()
        self.pos = {d: i for i, d in enumerate(self.ids)}
        self.live = np.unpackbits(np.load(path / "live.npy"), count=self.n_docs, bitorder="little").astype(bool)
        self.cols = {}
        for name in self.manifest["numeric_fields"]:
            vals = np.load(path / "num" / f"{name}.npy").copy()
            miss = np.unpackbits(np.load(path / "num" / f"{name}.missing.npy"), count=self.n_docs,
                                 bitorder="little").astype(bool)
            self.cols[name] = (vals, miss)
        self.patch_calls = 0

    @classmethod
    def from_segment(cls, path, device: str = "cpu") -> "FakeIndex":
        return cls(path, device)

    def patch(self, field: str, ids, values) -> None:
        if field not in self.hot:
            raise NotHot(field)
        vals, miss = self.cols[field]
        rows = np.fromiter((self.pos[i] for i in ids), dtype=np.int64, count=len(ids))
        vals[rows] = np.asarray(values).astype(vals.dtype)
        miss[rows] = False
        self.patch_calls += 1

    def delete(self, ids) -> None:
        self.live[[self.pos[i] for i in ids]] = False

    def set_version(self, version: int) -> None:
        self.version = int(version)

    # ---- search ------------------------------------------------------------------------------------------------------
    def _mask(self, q: dict) -> np.ndarray:
        (kind, body), = q.items()
        if kind == "match_all":
            return np.ones(self.n_docs, bool)
        if kind == "term":
            (f, v), = body.items()
            vals, miss = self.cols[f]
            return (vals == v) & ~miss
        if kind == "range":
            (f, r), = body.items()
            vals, miss = self.cols[f]
            m = ~miss
            for op, fn in (("gte", np.greater_equal), ("gt", np.greater), ("lte", np.less_equal), ("lt", np.less)):
                if op in r:
                    m &= fn(vals, r[op])
            return m
        if kind == "bool":
            m = np.ones(self.n_docs, bool)
            for c in body.get("filter", []) + body.get("must", []):
                m &= self._mask(c)
            for c in body.get("must_not", []):
                m &= ~self._mask(c)
            return m
        raise ValueError(f"fake engine: unsupported query {kind}")

    def search(self, request: dict) -> dict:
        mask = self._mask(request.get("query", {"match_all": {}})) & self.live
        docs = np.nonzero(mask)[0]
        keys = []
        for s in request.get("sort", []):
            (f, spec), = (s.items() if isinstance(s, dict) else [(s, {})])
            if f == "_score":
                continue
            keys.append((f, spec.get("order", "asc")))
        order = sorted(docs, key=lambda d: tuple(-float(vals_of(self, f, d)) if dr == "desc"
                                                 else float(vals_of(self, f, d)) for f, dr in keys) + (int(d),))
        start, size = request.get("from", 0), request.get("size", 10)
        hits = [{"_id": self.ids[d], "_score": 1.0,
                 "sort": [vals_of(self, f, d) for f, _ in keys] + [int(d)]} for d in order[start:start + size]]
        return {"hits": {"total": int(len(docs)), "hits": hits}, "took_ms": {"total": 0.0}}

    def search_batch(self, requests):
        return [self.search(r) for r in requests]


def vals_of(index: FakeIndex, f: str, d: int):
    v = index.cols[f][0][d]
    return v.item() if hasattr(v, "item") else v


def hooks(exact: bool = False) -> EngineHooks:
    return EngineHooks(build_segment, lambda seg, path: write_segment(seg, path, exact=exact), FakeIndex.from_segment)
