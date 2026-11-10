"""Delta segments and merges (S8; docs/INTERFACES.md §3 and §4 "Multi-segment indexes"; the S8-1 ruling).

``apply_delta(index, store, after)``: everything the store logged after ``after`` becomes visible in one step.
- One read transaction (``Store.delta``) gives the head H, the records in ``(after, H]`` and the live docs at H of
  the keys upserted in that range.
- Patches to keys *not* upserted in the range go to ``Index.patch`` in place (their live copy is in an older
  segment); patches to upserted keys are already in their docs.
- If any key was upserted: one ``Index.add_segment(delta, version=H, delete_ids=...)``, which tombstones the older
  copies of the delta's keys and of the deleted keys atomically. Else the deletes go to ``Index.delete`` and the
  version is set.
- Re-applying records the index already has is harmless (patches in order end at the latest value; deletes of dead
  keys are no-ops), so serving can pass the offset of its oldest pending upsert.

``merge_once(index, store, policy, ...)``: one tiered merge with the snapshot, merge, replay and replace protocol.
- Under the caller's lock: ``Index.snapshot_segments(ids)`` gives the inputs at version V.
- Outside it: ``engine.build.merge_segments(..., version=V)``.
- Under the lock again: the records in ``(V, index.version]`` for the merged keys are replayed onto the merged arrays
  (patches into its hot columns; upserted or deleted keys become ``delete_ids``), then one
  ``Index.replace_segments``. A search sees the inputs or the merged segment, never both.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass

import numpy as np

from dataplane import _engine
from dataplane.mapping import as_mapping

DEFAULT_REFRESH = 0.5      # seconds (the S8-1 ruling)


def _missing_key(exc: BaseException) -> bool:
    return any(c.__name__ in ("NoSuchKey", "KeyError") for c in type(exc).__mro__)


def _tolerant(fn, ids: list, *cols) -> int:
    """``fn(ids, *cols)``, skipping ids the index does not know (NoSuchKey) one by one. Returns ids applied."""
    if not ids:
        return 0
    try:
        fn(ids, *cols)
        return len(ids)
    except Exception as e:  # noqa: BLE001
        if not _missing_key(e):
            raise
    n = 0
    for k, i in enumerate(ids):
        try:
            fn([i], *[[c[k]] for c in cols])
            n += 1
        except Exception as e:  # noqa: BLE001
            if not _missing_key(e):
                raise
    return n


@dataclass
class DeltaReport:
    version: int           # the index's version after the step
    records: int           # change records read
    added: int             # docs in the delta segment (0: no delta)
    deleted: int           # keys tombstoned or deleted
    patched: int           # (field, id) values patched in place
    segment_id: str | None
    upserted: list         # keys live in the delta
    removed: list          # keys whose last op in the range was a delete
    seconds: float


def build_delta(store, docs, version: int):
    """The delta segment's arrays (not written to disk: ``Index.save_commit`` writes it with the next commit)."""
    from dataplane.indexer import attach_encoders
    m = store.mapping
    models = {vf: store.meta(f"encoder.{vf}") for vf in m.vector_fields if store.meta(f"encoder.{vf}")}
    seg = _engine.hooks().build_segment(docs, m.raw, version=version, vector_models=models)
    seg.manifest["version"] = int(version)
    attach_encoders(seg, store, m)
    return seg


def apply_delta(index, store, after: int) -> DeltaReport:
    t0 = time.perf_counter()
    head, recs, docs = store.delta(after)
    if not recs:
        return DeltaReport(int(index.version), 0, 0, 0, 0, None, [], [], time.perf_counter() - t0)
    id_field = store.mapping.id_field
    upserted = docs[id_field].tolist() if len(docs) else []
    up = set(upserted)
    last_op: dict[str, str] = {}
    patches: dict[str, dict[str, object]] = {}
    for r in recs:
        op, k = r["op"], r["_id"]
        if op == "patch":
            if k not in up:
                for f, v in r["fields"].items():
                    patches.setdefault(f, {})[k] = v
        else:
            last_op[k] = op
            if op == "delete":
                for p in patches.values():
                    p.pop(k, None)
    removed = [k for k, op in last_op.items() if op == "delete"]
    patched = 0
    for f, kv in patches.items():
        patched += _tolerant(lambda ids, vals, f=f: index.patch(f, ids, vals), list(kv), list(kv.values()))
    seg_id = None
    if upserted:
        seg = build_delta(store, docs, head)
        index.add_segment(seg, version=head, delete_ids=removed)
        seg_id = seg.manifest.get("segment_id")
        deleted = len(removed)
    else:
        deleted = _tolerant(index.delete, removed)
        index.set_version(head)
    return DeltaReport(int(index.version), len(recs), len(upserted), deleted, patched, seg_id, upserted, removed,
                       time.perf_counter() - t0)


# ---- merges -------------------------------------------------------------------------------------------------------

def _clear_bit(words: np.ndarray, d: int) -> None:
    words[d >> 5] &= np.uint32(~(1 << (d & 31)) & 0xFFFFFFFF)


def patch_arrays(seg, field: str, ids: list[str], values: list, pos: dict[str, int] | None = None) -> int:
    """Set hot ``field`` of ``ids`` in a segment's in-memory arrays (before the engine holds them)."""
    pos = pos if pos is not None else {k: i for i, k in enumerate(seg.ids.tolist())}
    col = seg.numerics[field]
    n = 0
    for k, v in zip(ids, values):
        d = pos.get(k)
        if d is None:
            continue
        col.values[d] = np.asarray(v).astype(col.values.dtype)
        _clear_bit(col.missing, d)
        n += 1
    return n


@dataclass
class MergeReport:
    inputs: list[str]
    output: str
    n_docs: int
    version: int
    replayed: int
    seconds: float


def merge_once(index, store, policy, lock=None, budget: int | None = None, work: int = 0) -> MergeReport | None:
    """Plan one merge with ``policy`` and run it; None when nothing is due or the budget refuses it. ``lock``: the
    caller's writer lock (refreshes take it too); None when the caller already holds what it needs."""
    from engine.build import merge_segments
    lock = lock if lock is not None else contextlib.nullcontext()
    segs = {s["segment_id"]: s for s in index.segments}
    plan = policy.plan(list(segs.values()))
    if not plan:
        return None
    if budget is not None:
        need = int(index.device_nbytes) + sum(int(segs[i]["device_nbytes"]) for i in plan) + work
        if need > budget:
            return None
    t0 = time.perf_counter()
    with lock:
        snaps, v = index.snapshot_segments(plan)
    merged = merge_segments(snaps, store.mapping.raw, version=v)
    with lock:
        recs = [r for r in store.changes(v, limit=max(1, int(index.version) - v), docs=False)
                if r["offset"] <= int(index.version)] if int(index.version) > v else []
        pos = {k: i for i, k in enumerate(merged.ids.tolist())}
        dead: list[str] = []
        for r in recs:
            k = r["_id"]
            if k not in pos:
                continue
            if r["op"] == "patch":
                for f, val in r["fields"].items():
                    patch_arrays(merged, f, [k], [val], pos)
            else:
                dead.append(k)
        index.replace_segments(plan, merged, delete_ids=dead)
    return MergeReport(plan, merged.manifest.get("segment_id", ""), len(pos), v, len(recs), time.perf_counter() - t0)
