"""The key-value store and its change log (docs/INTERFACES.md §3): the source of truth, on SQLite (stdlib).

One file, one writer (any number of readers; each thread gets its own connection, WAL mode). A batch of changes
commits in one transaction and gets consecutive offsets from ``head() + 1``; a refused change refuses its whole batch,
so offsets never have gaps.

Change records (the log):
    {"offset": 7, "op": "upsert", "_id": "B1", "doc": {...whole doc...}, "ts": "..."}
    {"offset": 8, "op": "patch",  "_id": "B1", "fields": {"price": 8.49}, "ts": "..."}
    {"offset": 9, "op": "delete", "_id": "B2", "ts": "..."}

Vectors are kept as float32 bytes beside the JSON (the API hands them out as lists of float32 values).
The store trusts intake (``dataplane.intake``) to validate upserts; it checks ids, ops, vector shapes and, for
patches, that the fields are hot, the values fit and the key is live.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from dataplane.mapping import ID_MAX_BYTES, BadValue, Mapping, as_mapping, coerce, is_missing, json_safe

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, value BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS docs (
    id TEXT PRIMARY KEY, doc TEXT NOT NULL, vecf TEXT NOT NULL, vec BLOB,
    first_offset INTEGER NOT NULL, last_offset INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS docs_first ON docs (first_offset);
CREATE TABLE IF NOT EXISTS log (
    offset INTEGER PRIMARY KEY, op TEXT NOT NULL, id TEXT NOT NULL, body TEXT, vecf TEXT, vec BLOB, ts TEXT NOT NULL);
"""
OPS = ("upsert", "patch", "delete")
FIRST, LAST = "_first_offset", "_last_offset"   # snapshot columns (names starting "_" cannot be mapped fields)


class StoreError(ValueError):
    """A change the store refuses; the whole batch is rolled back and nothing is logged."""


class NotHot(StoreError):
    """A patch names a field that is not in ``_meta.engine.hot``."""


class NoSuchKey(StoreError, KeyError):
    """A patch names a key that is not live."""

    def __str__(self) -> str:  # KeyError quotes its argument; keep the message readable
        return ValueError.__str__(self)


class BadChange(StoreError):
    """A malformed change: unknown op, missing or over-long id, a bad value."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Store:
    def __init__(self, path: str | Path, mapping: dict | Mapping):
        self.path = Path(path)
        self.mapping = as_mapping(mapping)
        self._vec_dims = {n: self.mapping.fields[n].dim for n in self.mapping.vector_fields}
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        conn = self._conn()
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT value FROM meta WHERE key = 'mapping'").fetchone()
        if row is None:
            with conn:
                conn.execute("INSERT INTO meta VALUES ('mapping', ?)", (json.dumps(self.mapping.raw, sort_keys=True),))
                conn.execute("INSERT INTO meta VALUES ('mapping_sha256', ?)", (self.mapping.sha256,))
        elif json.loads(row[0]) != json.loads(json.dumps(self.mapping.raw)):
            raise ValueError(f"{self.path} was created with a different mapping; v1 changes mappings by a new store")

    # ---- connections ------------------------------------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    def close(self) -> None:
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except sqlite3.ProgrammingError:  # closed from its own thread already
                    pass
            self._conns.clear()
        self._local = threading.local()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- meta -------------------------------------------------------------------------------------------------------
    def meta(self, key: str) -> str | None:
        row = self._conn().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def blob(self, key: str) -> bytes | None:
        row = self._conn().execute("SELECT value FROM blobs WHERE key = ?", (key,)).fetchone()
        return None if row is None else bytes(row[0])

    def set_blob(self, key: str, value: bytes) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO blobs VALUES (?, ?)", (key, value))

    # ---- vectors ----------------------------------------------------------------------------------------------------
    def _split(self, doc: dict) -> tuple[dict, str, bytes | None]:
        """A doc to (JSON part, present vector fields, their float32 bytes)."""
        plain, names, parts = {}, [], []
        for k, v in doc.items():
            if k in self._vec_dims:
                if is_missing(v):
                    continue
                a = np.asarray(v, dtype=np.float32)
                if a.shape != (self._vec_dims[k],):
                    raise BadChange(f"{k}: wants {self._vec_dims[k]} dims, got shape {list(a.shape)}")
                names.append(k)
                parts.append(a)
            else:
                plain[k] = None if is_missing(v) else v
        order = sorted(range(len(names)), key=names.__getitem__)
        blob = b"".join(parts[i].tobytes() for i in order) if parts else None
        return plain, ",".join(names[i] for i in order), blob

    def _vectors(self, vecf: str | None, blob: bytes | None) -> dict[str, np.ndarray]:
        if not vecf:
            return {}
        out, pos = {}, 0
        flat = np.frombuffer(blob, dtype=np.float32)
        for n in vecf.split(","):
            d = self._vec_dims[n]
            out[n] = flat[pos:pos + d]
            pos += d
        return out

    def _doc(self, text: str, vecf: str | None, blob: bytes | None, lists: bool) -> dict:
        doc = json.loads(text)
        for n, a in self._vectors(vecf, blob).items():
            doc[n] = a.tolist() if lists else a
        return doc

    # ---- writes -----------------------------------------------------------------------------------------------------
    def head(self) -> int:
        return self._head(self._conn())

    @staticmethod
    def _head(c: sqlite3.Connection) -> int:
        return c.execute("SELECT COALESCE(MAX(offset), 0) FROM log").fetchone()[0]

    def apply(self, changes: Iterable[dict]) -> tuple[int, int]:
        """Commit a batch atomically. Returns ``(first, last)`` offsets; ``(0, 0)`` for an empty batch."""
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            head = self._head(c)
            off = head
            hot = set(self.mapping.hot)
            id_field = self.mapping.id_field
            for ch in changes:
                op = ch.get("op")
                if op not in OPS:
                    raise BadChange(f"unknown op {op!r}")
                _id = ch.get("_id")
                if _id is None and op == "upsert":
                    _id = (ch.get("doc") or {}).get(id_field)
                if not isinstance(_id, str) or not _id:
                    raise BadChange(f"{op}: missing _id")
                if len(_id.encode("utf-8")) > ID_MAX_BYTES:
                    raise BadChange(f"{op}: _id over {ID_MAX_BYTES} bytes")
                off += 1
                ts = ch.get("ts") or utc_now()
                if op == "upsert":
                    doc = dict(ch.get("doc") or {})
                    doc[id_field] = _id
                    plain, vecf, blob = self._split(doc)
                    text = json.dumps(plain, separators=(",", ":"), ensure_ascii=False, default=json_safe)
                    row = c.execute("SELECT first_offset FROM docs WHERE id = ?", (_id,)).fetchone()
                    first = off if row is None else row[0]
                    c.execute("INSERT OR REPLACE INTO docs VALUES (?, ?, ?, ?, ?, ?)",
                              (_id, text, vecf, blob, first, off))
                    c.execute("INSERT INTO log VALUES (?, 'upsert', ?, ?, ?, ?, ?)", (off, _id, text, vecf, blob, ts))
                elif op == "patch":
                    fields = ch.get("fields")
                    if not isinstance(fields, dict) or not fields:
                        raise BadChange(f"patch {_id!r}: needs a non-empty 'fields' object")
                    cold = sorted(set(fields) - hot)
                    if cold:
                        raise NotHot(f"patch {_id!r}: {', '.join(cold)} not hot (hot: {', '.join(sorted(hot))})")
                    vals = {}
                    for k, v in fields.items():
                        if is_missing(v):
                            raise BadChange(f"patch {_id!r}: {k} is null; a patch sets values, an upsert removes them")
                        try:
                            vals[k] = coerce(self.mapping.fields[k], v)
                        except BadValue as e:
                            raise BadChange(f"patch {_id!r}: {e.detail}") from None
                    row = c.execute("SELECT doc FROM docs WHERE id = ?", (_id,)).fetchone()
                    if row is None:
                        raise NoSuchKey(f"patch {_id!r}: no such live key")
                    doc = json.loads(row[0])
                    doc.update(vals)
                    c.execute("UPDATE docs SET doc = ?, last_offset = ? WHERE id = ?",
                              (json.dumps(doc, separators=(",", ":"), ensure_ascii=False), off, _id))
                    c.execute("INSERT INTO log VALUES (?, 'patch', ?, ?, NULL, NULL, ?)",
                              (off, _id, json.dumps(vals, separators=(",", ":")), ts))
                else:
                    c.execute("DELETE FROM docs WHERE id = ?", (_id,))
                    c.execute("INSERT INTO log VALUES (?, 'delete', ?, NULL, NULL, NULL, ?)", (off, _id, ts))
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        return (0, 0) if off == head else (head + 1, off)

    # ---- reads ------------------------------------------------------------------------------------------------------
    def get(self, ids: Sequence[str]) -> list[dict | None]:
        c = self._conn()
        found: dict[str, dict] = {}
        ids = list(ids)
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = f"SELECT id, doc, vecf, vec FROM docs WHERE id IN ({','.join('?' * len(chunk))})"
            for _id, text, vecf, blob in c.execute(q, chunk):
                found[_id] = self._doc(text, vecf, blob, lists=True)
        return [found.get(i) for i in ids]

    def changes(self, after: int, limit: int = 10_000, docs: bool = True) -> list[dict]:
        """Change records with ``offset > after``, ascending, at most ``limit``. ``docs=False`` leaves upserts'
        ``doc`` out (the updater does not need it)."""
        c = self._conn()
        cols = "offset, op, id, body, vecf, vec, ts" if docs else "offset, op, id, CASE op WHEN 'patch' THEN body END, NULL, NULL, ts"
        out = []
        for off, op, _id, body, vecf, blob, ts in c.execute(
                f"SELECT {cols} FROM log WHERE offset > ? ORDER BY offset LIMIT ?", (after, limit)):
            rec = {"offset": off, "op": op, "_id": _id}
            if op == "upsert" and docs:
                rec["doc"] = self._doc(body, vecf, blob, lists=True)
            elif op == "patch":
                rec["fields"] = json.loads(body)
            rec["ts"] = ts
            out.append(rec)
        return out

    def count(self) -> int:
        return self._conn().execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def snapshot(self, at: int | None = None) -> pd.DataFrame:
        """Live documents as of offset ``at`` (default: head), one row each, ordered by ``first_offset``.

        Columns: the id field, every mapped top-level field (text and keyword: str or None; float: float64 with
        NaN for missing; integer, long, boolean: int32, int64, bool, or pandas' nullable Int32, Int64, boolean
        when a value is missing; knn_vector: float32 arrays or None), then ``_first_offset`` and ``_last_offset``
        (int64). Unmapped fields are not columns. ``df.attrs["version"]`` is the offset the snapshot is at.
        """
        c = self._conn()
        c.execute("BEGIN")   # one read transaction: head and rows agree
        try:
            head = self._head(c)
            at = head if at is None else at
            if at < 0 or at > head:
                raise ValueError(f"snapshot at {at}: the store's head is {head}")
            if at == head:
                rows = [(self._doc(text, vecf, blob, lists=False), first, last) for text, vecf, blob, first, last in
                        c.execute("SELECT doc, vecf, vec, first_offset, last_offset FROM docs ORDER BY first_offset")]
            else:
                rows = self._replay(c, at)
        finally:
            c.execute("COMMIT")
        df = self._frame(rows)
        df.attrs["version"] = at
        return df

    def _replay(self, c: sqlite3.Connection, at: int) -> list[tuple[dict, int, int]]:
        state: dict[str, list] = {}
        for off, op, _id, body, vecf, blob in c.execute(
                "SELECT offset, op, id, body, vecf, vec FROM log WHERE offset <= ? ORDER BY offset", (at,)):
            if op == "upsert":
                prev = state.get(_id)
                state[_id] = [self._doc(body, vecf, blob, lists=False), prev[1] if prev else off, off]
            elif op == "patch":
                s = state[_id]
                s[0].update(json.loads(body))
                s[2] = off
            else:
                state.pop(_id, None)
        return sorted((tuple(v) for v in state.values()), key=lambda r: r[1])

    def _frame(self, rows: list[tuple[dict, int, int]]) -> pd.DataFrame:
        m = self.mapping
        cols: dict[str, object] = {m.id_field: [r[0].get(m.id_field) for r in rows]}
        for name, f in m.fields.items():
            if name == m.id_field:
                continue
            vals = [r[0].get(name) for r in rows]
            if f.type in ("text", "keyword"):
                cols[name] = pd.Series(vals, dtype=object)
            elif f.type == "float":
                cols[name] = pd.Series([np.nan if v is None else v for v in vals], dtype="float64")
            elif f.type in ("integer", "long", "boolean"):
                np_t, pd_t = {"integer": ("int32", "Int32"), "long": ("int64", "Int64"),
                              "boolean": ("bool", "boolean")}[f.type]
                cols[name] = pd.Series(vals, dtype=pd_t if any(v is None for v in vals) else np_t)
            else:
                arr = np.empty(len(vals), dtype=object)
                for i, v in enumerate(vals):
                    arr[i] = v
                cols[name] = pd.Series(arr, dtype=object)
        cols[FIRST] = np.array([r[1] for r in rows], dtype=np.int64)
        cols[LAST] = np.array([r[2] for r in rows], dtype=np.int64)
        return pd.DataFrame(cols)
