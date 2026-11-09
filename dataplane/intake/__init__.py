"""Intake (docs/INTERFACES.md §2): batch files and the change feed, into the store.

    read_batch(path, mapping)            .parquet or .jsonl; one source doc per row (a Rejection for a malformed line)
    validate(doc, mapping)               the coerced doc, or a Rejection
    enrich(docs, mapping, encoder)       adds vectors for the fields ``_meta.engine.embeddings`` names
    ingest(path, mapping, store, rejected, encoder=None)   read -> validate -> enrich -> store.apply (one batch)
    apply_changes(changes, mapping, store, ...)            the change feed: upsert, patch and delete records

Rejected records go to ``rejected.jsonl``, one ``{"record": ..., "rejection": ...}`` per line.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from dataplane.intake.encoders import Encoder, HashingEncoder, get_encoder, register  # noqa: F401 (re-exported)
from dataplane.mapping import (ID_MAX_BYTES, BadValue, Mapping, as_mapping, coerce, is_missing, json_safe)

# The closed set of rejection reasons (additive). The first six are the architect's draft; the last four were added
# by the dataplane-engineer for malformed input and the change feed (docs/INTERFACES.md §2, Changes).
REASONS = ("missing_id", "duplicate_id_in_batch", "bad_type", "dims_mismatch", "unknown_field", "too_long",
           "malformed", "bad_op", "not_hot", "no_such_key")


@dataclass
class Rejection:
    _id: str | None
    reason: str
    field: str | None
    detail: str

    def __post_init__(self):
        assert self.reason in REASONS, self.reason

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntakeReport:
    accepted: int
    rejected: int
    first_offset: int
    last_offset: int
    reasons: dict[str, int] = field(default_factory=dict)   # rejection counts by reason (additive)


@dataclass
class ChangeReport:
    accepted: int
    first_offset: int
    last_offset: int
    rejected: list[tuple[dict, Rejection]] = field(default_factory=list)


# ---- reading ----------------------------------------------------------------------------------------------------------

def read_batch(path: str | Path, mapping: dict | Mapping | None = None, batch_rows: int = 4096
               ) -> Iterator[dict | Rejection]:
    """Source documents from a ``.parquet`` or ``.jsonl`` file, in file order. A JSONL line that is not a JSON
    object comes out as a ``Rejection`` (reason ``malformed``) whose ``raw`` attribute holds the line."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows):
            yield from batch.to_pylist()
    elif suffix in (".jsonl", ".ndjson", ".json"):
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    doc, err = None, f"line {n}: {e.msg} at column {e.colno}"
                else:
                    err = None if isinstance(doc, dict) else f"line {n}: a record is a JSON object"
                if err:
                    rej = Rejection(None, "malformed", None, err)
                    rej.raw = line.rstrip("\n")
                    yield rej
                else:
                    yield doc
    else:
        raise ValueError(f"{path}: intake reads .parquet or .jsonl files")


# ---- validation -------------------------------------------------------------------------------------------------------

def _doc_id(doc: dict, m: Mapping) -> str | Rejection:
    v = doc.get(m.id_field)
    if is_missing(v) and m.id_field != "_id":
        v = doc.get("_id")
    if is_missing(v) or v == "":
        return Rejection(None, "missing_id", m.id_field, f"no {m.id_field!r}")
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (str, int, np.integer)):
        return Rejection(None, "bad_type", m.id_field, f"ids are strings, got {type(v).__name__}")
    v = str(v)
    if len(v.encode("utf-8")) > ID_MAX_BYTES:
        return Rejection(v[:64], "too_long", m.id_field, f"id is over {ID_MAX_BYTES} bytes")
    return v


def validate(doc: Any, mapping: dict | Mapping) -> dict | Rejection:
    m = as_mapping(mapping)
    if not isinstance(doc, dict):
        return Rejection(None, "malformed", None, f"a record is an object, got {type(doc).__name__}")
    _id = _doc_id(doc, m)
    if isinstance(_id, Rejection):
        return _id
    out: dict[str, Any] = {m.id_field: _id}
    for k, v in doc.items():
        if k == m.id_field or k == "_id":
            continue
        f = m.fields.get(k)
        if f is None:
            if m.dynamic == "strict":
                return Rejection(_id, "unknown_field", k, f"{k!r} is not mapped and the mapping is strict")
            if not is_missing(v):
                out[k] = json_safe(v)   # kept in the source, not indexed (OpenSearch's dynamic: false)
            continue
        if is_missing(v):
            continue
        try:
            out[k] = coerce(f, v)
        except BadValue as e:
            return Rejection(_id, e.reason, k, e.detail)
    return out


# ---- enrichment -------------------------------------------------------------------------------------------------------

_DEFAULTS: dict[tuple[str, int], Encoder] = {}


def resolve_encoder(encoder: Encoder | dict | None, field_name: str, model: str, dim: int) -> Encoder:
    if isinstance(encoder, dict):
        encoder = encoder.get(field_name)
    if encoder is None:
        encoder = _DEFAULTS.get((model, dim))
        if encoder is None:
            encoder = _DEFAULTS[(model, dim)] = get_encoder(model, dim)
    if int(encoder.dim) != dim:
        raise ValueError(f"encoder {encoder.name!r} gives {encoder.dim} dims; {field_name!r} is mapped with {dim}")
    return encoder


def enrich(docs: list[dict], mapping: dict | Mapping, encoder: Encoder | dict | None = None) -> list[dict]:
    """Add the vector of each ``_meta.engine.embeddings`` field a doc lacks, from its input fields (joined by a
    space). A doc with no input text, or whose vector is zero, stays without one. Works in place; returns ``docs``."""
    m = as_mapping(mapping)
    for vf, spec in m.embeddings.items():
        need = [d for d in docs if is_missing(d.get(vf))]
        texts = [" ".join(str(d[i]) for i in spec["inputs"] if not is_missing(d.get(i))) for d in need]
        pairs = [(d, t) for d, t in zip(need, texts) if t.strip()]
        if not pairs:
            continue
        enc = resolve_encoder(encoder, vf, spec["model"], m.fields[vf].dim)
        vecs = np.asarray(enc.encode([t for _, t in pairs]), dtype=np.float32)
        if vecs.shape != (len(pairs), m.fields[vf].dim):
            raise ValueError(f"encoder {enc.name!r} returned shape {vecs.shape}")
        norms = np.linalg.norm(vecs, axis=1)
        for (d, _), v, n in zip(pairs, vecs, norms):
            if n > 0 and np.isfinite(n):
                d[vf] = (v / n).astype(np.float32)
    return docs


def encoder_names(mapping: dict | Mapping, encoder: Encoder | dict | None = None) -> dict[str, str]:
    m = as_mapping(mapping)
    return {vf: resolve_encoder(encoder, vf, s["model"], m.fields[vf].dim).name for vf, s in m.embeddings.items()}


def _check_encoders(store, mapping: Mapping, encoder) -> dict[str, str]:
    """One store, one encoder per vector field: vectors from two encoders are not comparable."""
    names = encoder_names(mapping, encoder)
    for vf, name in names.items():
        have = store.meta(f"encoder.{vf}")
        if have is not None and have != name:
            raise ValueError(f"{vf!r} in this store was enriched by {have!r}; refusing to mix in {name!r}")
    return names


def _record_encoders(store, names: dict[str, str]) -> None:
    for vf, name in names.items():
        if store.meta(f"encoder.{vf}") is None:
            store.set_meta(f"encoder.{vf}", name)


# ---- ingest -----------------------------------------------------------------------------------------------------------

def _rejected_path(rejected: str | Path) -> Path:
    p = Path(rejected)
    return p / "rejected.jsonl" if p.is_dir() else p


def _write_rejection(fh, record: Any, rej: Rejection) -> None:
    fh.write(json.dumps({"record": json_safe(record), "rejection": rej.to_dict()}, ensure_ascii=False) + "\n")


def ingest(path: str | Path, mapping: dict | Mapping, store, rejected: str | Path,
           encoder: Encoder | dict | None = None, chunk: int = 2000) -> IntakeReport:
    """A batch file into the store as one atomic batch of upserts. Within the file, the first valid record of an id
    wins; later ones are rejected (``duplicate_id_in_batch``)."""
    m = as_mapping(mapping)
    names = _check_encoders(store, m, encoder)
    reasons: Counter = Counter()
    accepted = 0
    rpath = _rejected_path(rejected)
    rpath.parent.mkdir(parents=True, exist_ok=True)
    with open(rpath, "a", encoding="utf-8") as rfh:

        def reject(record, rej):
            reasons[rej.reason] += 1
            _write_rejection(rfh, record, rej)

        def upserts() -> Iterator[dict]:
            nonlocal accepted
            seen: set[str] = set()
            buf: list[dict] = []

            def flush():
                nonlocal accepted
                enrich(buf, m, encoder)
                accepted += len(buf)
                out = [{"op": "upsert", "_id": d[m.id_field], "doc": d} for d in buf]
                buf.clear()
                return out

            for item in read_batch(path, m):
                if isinstance(item, Rejection):
                    reject(getattr(item, "raw", None), item)
                    continue
                doc = validate(item, m)
                if isinstance(doc, Rejection):
                    reject(item, doc)
                    continue
                _id = doc[m.id_field]
                if _id in seen:
                    reject(item, Rejection(_id, "duplicate_id_in_batch", m.id_field, "an earlier record has this id"))
                    continue
                seen.add(_id)
                buf.append(doc)
                if len(buf) >= chunk:
                    yield from flush()
            if buf:
                yield from flush()

        first, last = store.apply(upserts())
    if accepted:
        _record_encoders(store, names)
    return IntakeReport(accepted, sum(reasons.values()), first, last, dict(reasons))


# ---- the change feed --------------------------------------------------------------------------------------------------

def apply_changes(changes: Iterable[dict], mapping: dict | Mapping, store, encoder: Encoder | dict | None = None,
                  rejected: str | Path | None = None) -> ChangeReport:
    """Change records without offsets (§3: upsert with the whole doc, patch with hot fields, delete) into the store
    as one atomic batch. Records that would be refused are rejected one by one instead (reasons ``bad_op``,
    ``missing_id``, ``not_hot``, ``no_such_key``, and the validation reasons for upserts and patch values)."""
    m = as_mapping(mapping)
    changes = list(changes)
    names = _check_encoders(store, m, encoder)
    hot = set(m.hot)
    patch_ids = [c.get("_id") for c in changes if isinstance(c, dict) and c.get("op") == "patch"
                 and isinstance(c.get("_id"), str)]
    live: dict[str, bool] = {}
    if patch_ids:
        uniq = list(dict.fromkeys(patch_ids))
        live = {i: d is not None for i, d in zip(uniq, store.get(uniq))}
    accepted: list[dict] = []
    upsert_docs: list[dict] = []
    bad: list[tuple[dict, Rejection]] = []
    for ch in changes:
        if not isinstance(ch, dict):
            bad.append((ch, Rejection(None, "malformed", None, "a change record is an object")))
            continue
        op = ch.get("op")
        _id = ch.get("_id")
        if op not in ("upsert", "patch", "delete"):
            bad.append((ch, Rejection(_id if isinstance(_id, str) else None, "bad_op", "op",
                                      f"op is upsert, patch or delete; got {op!r}")))
            continue
        if op == "upsert":
            src = dict(ch.get("doc") or {})
            if isinstance(_id, str) and _id:
                if not is_missing(src.get(m.id_field)) and str(src[m.id_field]) != _id:
                    bad.append((ch, Rejection(_id, "bad_type", m.id_field, "the doc's id differs from _id")))
                    continue
                src[m.id_field] = _id
            doc = validate(src, m)
            if isinstance(doc, Rejection):
                bad.append((ch, doc))
                continue
            _id = doc[m.id_field]
            live[_id] = True
            upsert_docs.append(doc)
            accepted.append({"op": "upsert", "_id": _id, "doc": doc})
            continue
        if not isinstance(_id, str) or not _id:
            bad.append((ch, Rejection(None, "missing_id", "_id", f"a {op} needs an _id")))
            continue
        if len(_id.encode("utf-8")) > ID_MAX_BYTES:
            bad.append((ch, Rejection(_id[:64], "too_long", "_id", f"id is over {ID_MAX_BYTES} bytes")))
            continue
        if op == "delete":
            live[_id] = False
            accepted.append({"op": "delete", "_id": _id})
            continue
        fields = ch.get("fields")
        if not isinstance(fields, dict) or not fields:
            bad.append((ch, Rejection(_id, "malformed", "fields", "a patch needs a non-empty fields object")))
            continue
        rej = None
        vals = {}
        for k, v in fields.items():
            if k not in hot:
                rej = Rejection(_id, "not_hot", k, f"{k!r} is not hot (hot: {', '.join(m.hot)})")
                break
            if is_missing(v):
                rej = Rejection(_id, "bad_type", k, "a patch sets values; null is not one")
                break
            try:
                vals[k] = coerce(m.fields[k], v)
            except BadValue as e:
                rej = Rejection(_id, e.reason, k, e.detail)
                break
        if rej is None and not live.get(_id, False):
            rej = Rejection(_id, "no_such_key", "_id", "a patch needs a live key")
        if rej is not None:
            bad.append((ch, rej))
            continue
        accepted.append({"op": "patch", "_id": _id, "fields": vals})
    enrich(upsert_docs, m, encoder)
    first, last = store.apply(accepted)
    if upsert_docs:
        _record_encoders(store, names)
    if rejected is not None and bad:
        rpath = _rejected_path(rejected)
        rpath.parent.mkdir(parents=True, exist_ok=True)
        with open(rpath, "a", encoding="utf-8") as fh:
            for rec, rej in bad:
                _write_rejection(fh, rec, rej)
    return ChangeReport(len(accepted), first, last, bad)
