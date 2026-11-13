"""The indexer (docs/INTERFACES.md §3): segments from the store, in the format of §4.

``build(store, mapping, out_dir)`` takes ``store.snapshot()`` (head), hands it to ``engine.build.build_segment``, sets
the segment's ``version`` to that head offset, and writes it with ``engine.segment.write_segment`` into a temporary
directory, renamed into ``out_dir/<segment_id>`` only when complete (a reader never sees half a segment).

**Merges and deletes.** v1 has one segment per index, so every build is a full merge: deleted documents are gone,
upserts are in, statistics (df, avg_len, N) are recomputed, and doc ids are the snapshot's row order (ascending
``first_offset``), so surviving documents keep their relative order. Between builds, deletes are tombstones (live bits)
and hot fields are patched in place by the updater. ``MergePolicy`` says when a rebuild is due; ``gc`` keeps the
newest segments (for rollback, §7) and removes the rest.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from dataplane import _engine
from dataplane.mapping import Mapping, as_mapping

MANIFEST = "manifest.json"


def read_manifest(segment: str | Path) -> dict:
    with open(Path(segment) / MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def _set_version(seg, version: int) -> None:
    manifest = getattr(seg, "manifest", None)
    if isinstance(manifest, dict):
        manifest["version"] = int(version)
    elif isinstance(seg, dict) and isinstance(seg.get("manifest"), dict):
        seg["manifest"]["version"] = int(version)
    else:
        raise TypeError("build_segment returned no manifest dict to carry the version (§4)")


def attach_encoders(seg, store, mapping: Mapping) -> list[str]:
    """Put the store's fitted encoders (``intake.encoders.fitted_arrays``) into the segment, for the vector fields
    whose vectors they produced, so the segment answers ``neural`` queries (§4: ``vec/<f>.encoder.*``, and
    ``vector_fields.<f>.encoder``). ``build_segment`` stores an encoder only when it fits one itself; the store's
    vectors come in as given vectors, so the indexer adds the one intake fitted. Returns the fields given one."""
    from dataplane.intake.encoders import fitted_arrays
    done = []
    vf_meta = seg.manifest.get("vector_fields", {})
    for vf in mapping.vector_fields:
        arrays = fitted_arrays(store, vf)
        if arrays is None or vf not in getattr(seg, "vectors", {}) or vf not in vf_meta:
            continue
        if int(arrays["dim"]) != int(vf_meta[vf]["dim"]):
            raise ValueError(f"the store's encoder for {vf!r} gives {int(arrays['dim'])} dims, not {vf_meta[vf]['dim']}")
        seg.encoders[vf] = arrays
        vf_meta[vf]["encoder"] = True
        vf_meta[vf]["model"] = store.meta(f"encoder.{vf}") or vf_meta[vf].get("model")
        done.append(vf)
    return done


def build(store,mapping: dict | Mapping | None, out_dir: str | Path, device_budget: int | None = None) -> Path:
    """Build a segment of the store at its head; returns the segment directory.

    ``device_budget`` (bytes): refuse a segment whose ``device_nbytes`` would not fit (``engine.backend.OverBudget``),
    before any replica tries to load it.
    """
    m = as_mapping(mapping) if mapping is not None else store.mapping
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snap = store.snapshot()
    version = snap.attrs["version"]
    h = _engine.hooks()
    models = {vf: store.meta(f"encoder.{vf}") for vf in m.vector_fields if store.meta(f"encoder.{vf}")}
    seg = h.build_segment(snap, m.raw, version=version, vector_models=models)
    _set_version(seg, version)
    attach_encoders(seg, store, m)
    tmp = out / f".tmp-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        written = Path(h.write_segment(seg, tmp) or tmp)
        if not (written / MANIFEST).exists():
            raise RuntimeError(f"write_segment wrote no {MANIFEST} in {written}")
        manifest = read_manifest(written)
        if manifest.get("version") != version:
            raise RuntimeError(f"segment version {manifest.get('version')} != store head {version}")
        if device_budget is not None and manifest.get("device_nbytes", 0) > device_budget:
            from engine.backend import OverBudget
            raise OverBudget(f"segment needs {manifest['device_nbytes']} device bytes; the budget is {device_budget}")
        seg_id = manifest.get("segment_id") or f"seg-{version}-{uuid.uuid4().hex[:8]}"
        final = out / seg_id
        if final.exists():   # the same content built twice: keep the first
            return final
        written.rename(final)
        return final
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def list_segments(out_dir: str | Path) -> list[tuple[Path, dict]]:
    """Complete segments under ``out_dir``, oldest first (by version, then ``created``, then name)."""
    out = Path(out_dir)
    if not out.exists():
        return []
    segs = []
    for p in out.iterdir():
        if p.is_dir() and not p.name.startswith(".") and (p / MANIFEST).exists():
            try:
                segs.append((p, read_manifest(p)))
            except (OSError, json.JSONDecodeError):
                continue
    segs.sort(key=lambda s: (s[1].get("version", 0), s[1].get("created", ""), s[0].stat().st_mtime, s[0].name))
    return segs


def latest_segment(out_dir: str | Path) -> Path | None:
    segs = list_segments(out_dir)
    return segs[-1][0] if segs else None


def gc(out_dir: str | Path, keep: int = 3, protect: tuple[str | Path, ...] = ()) -> list[Path]:
    """Remove all but the newest ``keep`` segments (and never one in ``protect``, e.g. the live one).
    Returns what was removed."""
    keep_names = {Path(p).name for p in protect}
    segs = list_segments(out_dir)
    removed = []
    for p, _ in segs[:max(0, len(segs) - keep)]:
        if p.name in keep_names:
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p)
    return removed


@dataclass
class MergePolicy:
    """When a rebuild (a full merge) is due, from a replica's ``stats()``: too many tombstones (they cost scoring
    work and skew df), upserts waiting to become visible, or a segment older than ``max_age_s`` with any of those."""
    max_deleted_ratio: float = 0.10
    max_pending: int = 1000
    max_age_s: float = 60.0

    def due(self, stats: dict) -> bool:
        n = max(1, stats.get("n_docs", 0))
        if stats.get("deleted", 0) / n > self.max_deleted_ratio:
            return True
        waiting = stats.get("pending_upserts", 0) + stats.get("deleted", 0)
        if stats.get("pending_upserts", 0) >= self.max_pending:
            return True
        return waiting > 0 and time.time() - stats.get("loaded_at", time.time()) > self.max_age_s


@dataclass
class TieredMergePolicy:
    """Which segments to merge (S8-1 proposal, section 6). Segments are dicts with ``segment_id``, ``live`` and
    ``n_docs`` (``Index.segments``). Tiers by live docs: ``[floor * factor**k, floor * factor**(k+1))``, the first
    tier being everything under ``floor * factor``. A tier holding ``merge_factor`` or more segments merges its
    smallest ``merge_factor``; a segment whose deleted share exceeds ``max_deleted_ratio`` is rewritten alone; above
    ``max_segments`` the smallest merge first, so query fan-out stays bounded. ``plan`` returns at most one merge,
    the most urgent (the updater runs one at a time)."""
    floor: int = 1_000
    factor: int = 10
    merge_factor: int = 10
    max_deleted_ratio: float = 0.25
    max_segments: int = 16

    def tier(self, live: int) -> int:
        k, bound = 0, self.floor * self.factor
        while live >= bound:
            k += 1
            bound *= self.factor
        return k

    def plan(self, segments: list[dict]) -> list[str] | None:
        segs = sorted(segments, key=lambda s: (s["live"], s["segment_id"]))
        if len(segs) < 2 and not any(self._expunge(s) for s in segs):
            return None
        tiers: dict[int, list[dict]] = {}
        for s in segs:
            tiers.setdefault(self.tier(s["live"]), []).append(s)
        for k in sorted(tiers):
            if len(tiers[k]) >= self.merge_factor:
                return [s["segment_id"] for s in tiers[k][:self.merge_factor]]
        if len(segs) > self.max_segments:
            return [s["segment_id"] for s in segs[:len(segs) - self.max_segments + 1]]
        worst = max(segs, key=lambda s: 1 - s["live"] / max(1, s["n_docs"]))
        return [worst["segment_id"]] if self._expunge(worst) else None

    def _expunge(self, s: dict) -> bool:
        return s["n_docs"] > 0 and 1 - s["live"] / s["n_docs"] > self.max_deleted_ratio


def reindex(store,out_dir: str | Path, replica=None, keep: int = 3) -> Path:
    """Build from the store and, given a replica, swap it onto the new segment; then gc. (§7 ``_reindex``.)"""
    seg = build(store, None, out_dir)
    if replica is not None:
        replica.swap(seg)
    gc(out_dir, keep=keep, protect=(seg,))
    return seg
