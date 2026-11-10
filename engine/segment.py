"""The segment format (docs/INTERFACES.md §4): a directory of ``.npy`` arrays and ``.parquet`` string tables, described
by ``manifest.json`` with a sha256 per file and a checksum over them."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT, FORMAT_VERSION = "s3seg", 1


from .errors import CorruptSegment  # noqa: E402,F401  (re-exported)


@dataclass
class TextArrays:
    fields: list[str]
    vocab: list[str]                  # sorted; row = term id
    term_ptr: np.ndarray              # int64[V+1]
    doc_ids: np.ndarray               # int32[P]
    tf: np.ndarray                    # uint8[P, F]
    field_len: np.ndarray             # int32[N, F]
    df: np.ndarray                    # int32[V]
    df_field: np.ndarray              # int32[V, F]
    avg_len: np.ndarray               # float32[F]
    pos_ptr: np.ndarray | None = None     # int64[P+1], optional (S7-1): positions of posting p are
    positions: np.ndarray | None = None   # positions[pos_ptr[p]:pos_ptr[p+1]], uint32 (field << 24) | position


@dataclass
class KeywordColumn:
    codes: np.ndarray                 # int32[N], 0 = missing
    values: list[str]                 # values[0] == "", the rest sorted


@dataclass
class NumericColumn:
    values: np.ndarray                # float32 | int32 | int64 | bool [N]
    missing: np.ndarray               # uint32 bitmap


@dataclass
class SegmentArrays:
    manifest: dict
    ids: np.ndarray                   # object[N]
    first_offset: np.ndarray          # int64[N]
    text: TextArrays
    keywords: dict[str, KeywordColumn] = field(default_factory=dict)
    numerics: dict[str, NumericColumn] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    live: np.ndarray | None = None    # uint32 bitmap
    encoders: dict[str, dict] = field(default_factory=dict)   # vector field -> encoder arrays (terms, idf, components, dim)
    source: dict[str, np.ndarray] = field(default_factory=dict)  # S7-5, optional: text field -> object[N] raw text
    present: dict[str, np.ndarray] = field(default_factory=dict)  # optional: field -> uint32 bitmap, a stored value

    @property
    def n_docs(self) -> int:
        return len(self.ids)


def device_nbytes(seg: SegmentArrays) -> int:
    """Bytes the engine moves to the device for this segment (the string tables stay on the host), plus derived
    arrays (idf, relative lengths, a bool live mask, missing masks)."""
    t = seg.text
    n, F, V = seg.n_docs, len(t.fields), len(t.vocab)
    total = t.term_ptr.nbytes + t.doc_ids.nbytes + t.tf.nbytes + n * F * 4 + V * 4 * (1 + F)
    total += sum(k.codes.nbytes for k in seg.keywords.values())
    total += sum(c.values.nbytes + n for c in seg.numerics.values())
    total += sum(v.nbytes for v in seg.vectors.values()) + n
    if t.positions is not None:
        total += t.pos_ptr.nbytes + t.positions.nbytes
    return int(total)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _checksum(files: dict) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _strings(path: Path, col: str, values) -> None:
    pd.DataFrame({col: pd.Series(list(values), dtype="string")}).to_parquet(path, index=False)


def write_segment(seg: SegmentArrays, path) -> Path:
    """Write ``seg`` into the segment directory ``path`` itself (not its parent). ``path`` must not exist or must be
    an empty directory (``FileExistsError`` otherwise). The files are written to a sibling ``<name>.tmp`` and renamed
    into place. ``manifest["version"]`` is written as the caller left it, and ``segment_id`` is derived from it and the
    content: ``seg-<version>-<first 8 hex of the checksum>``."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path}: the segment directory must not exist or be empty")
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "text").mkdir(parents=True)
    (tmp / "kw").mkdir()
    (tmp / "num").mkdir()
    (tmp / "vec").mkdir()
    arrays: dict[str, np.ndarray] = {}
    pd.DataFrame({"_id": pd.Series(list(seg.ids), dtype="string"),
                  "first_offset": seg.first_offset.astype(np.int64)}).to_parquet(tmp / "ids.parquet", index=False)
    arrays["live.npy"] = seg.live
    t = seg.text
    _strings(tmp / "text/vocab.parquet", "term", t.vocab)
    for name in ("term_ptr", "doc_ids", "tf", "field_len", "df", "df_field", "avg_len"):
        arrays[f"text/{name}.npy"] = getattr(t, name)
    if t.positions is not None:
        arrays["text/pos_ptr.npy"] = t.pos_ptr
        arrays["text/positions.npy"] = t.positions
    for f, col in seg.keywords.items():
        arrays[f"kw/{f}.codes.npy"] = col.codes
        _strings(tmp / f"kw/{f}.values.parquet", "value", col.values)
    for f, col in seg.numerics.items():
        arrays[f"num/{f}.npy"] = col.values
        arrays[f"num/{f}.missing.npy"] = col.missing
    for f, v in seg.vectors.items():
        arrays[f"vec/{f}.npy"] = v
    for f, enc in seg.encoders.items():
        _strings(tmp / f"vec/{f}.encoder.terms.parquet", "term", enc["terms"])
        arrays[f"vec/{f}.encoder.idf.npy"] = np.asarray(enc["idf"], np.float32)
        arrays[f"vec/{f}.encoder.components.npy"] = np.asarray(enc["components"], np.float32)
    if seg.present:
        (tmp / "present").mkdir()
        for f, bits in seg.present.items():
            arrays[f"present/{f}.npy"] = np.asarray(bits, np.uint32)
    if seg.source:
        (tmp / "src").mkdir()
        for f, col in seg.source.items():
            pd.DataFrame({"text": pd.Series([None if v is None else str(v) for v in col], dtype="string")}
                         ).to_parquet(tmp / f"src/{f}.parquet", index=False)
    for rel, a in arrays.items():
        np.save(tmp / rel, np.ascontiguousarray(a), allow_pickle=False)
    files = {}
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            rel = p.relative_to(tmp).as_posix()
            entry = {"sha256": _sha(p), "bytes": p.stat().st_size}
            if rel in arrays:
                entry.update(dtype=str(arrays[rel].dtype), shape=list(arrays[rel].shape))
            files[rel] = entry
    m = dict(seg.manifest)
    checksum = _checksum(files)
    version = int(m.get("version", 0))
    if seg.source:
        m["source_fields"] = list(seg.source)
    if seg.present:
        m["present_fields"] = list(seg.present)
    m.update(format=FORMAT, format_version=FORMAT_VERSION, n_docs=seg.n_docs, files=files, version=version,
             nbytes=sum(e["bytes"] for e in files.values()), device_nbytes=device_nbytes(seg),
             checksum=checksum, segment_id=f"seg-{version}-{checksum[:8]}")
    with open(tmp / "manifest.json", "w") as f:
        json.dump(m, f, indent=1, sort_keys=True, default=str)
    if path.exists():
        path.rmdir()
    os.replace(tmp, path)
    seg.manifest = m
    return path


def read_manifest(path) -> dict:
    with open(Path(path) / "manifest.json") as f:
        return json.load(f)


def read_segment(path, verify: bool = True, mmap: bool = False) -> SegmentArrays:
    path = Path(path)
    try:
        m = read_manifest(path)
    except (OSError, json.JSONDecodeError) as e:
        raise CorruptSegment(f"{path}: no readable manifest.json ({e})") from e
    if m.get("format") != FORMAT or m.get("format_version") != FORMAT_VERSION:
        raise CorruptSegment(f"{path}: format {m.get('format')} v{m.get('format_version')}, expected {FORMAT} v1")
    files = m.get("files", {})
    if verify:
        if _checksum(files) != m.get("checksum"):
            raise CorruptSegment(f"{path}: manifest checksum mismatch")
        for rel, e in files.items():
            p = path / rel
            if not p.is_file() or _sha(p) != e["sha256"]:
                raise CorruptSegment(f"{path}: {rel} is missing or its sha256 does not match")

    def arr(rel):
        return np.load(path / rel, mmap_mode="r" if mmap else None, allow_pickle=False)

    def strings(rel, col):
        return pd.read_parquet(path / rel)[col].astype(object).fillna("").tolist()

    ids = pd.read_parquet(path / "ids.parquet")
    text = TextArrays(fields=list(m["text_fields"]), vocab=strings("text/vocab.parquet", "term"),
                      **{k: arr(f"text/{k}.npy") for k in ("term_ptr", "doc_ids", "tf", "field_len", "df",
                                                            "df_field", "avg_len")})
    if "text/positions.npy" in files:          # optional (S7-1): older segments have none
        text.pos_ptr, text.positions = arr("text/pos_ptr.npy"), arr("text/positions.npy")
    seg = SegmentArrays(manifest=m, ids=ids["_id"].astype(object).to_numpy(),
                        first_offset=ids["first_offset"].to_numpy(np.int64), text=text, live=arr("live.npy"))
    for f in m.get("keyword_fields", []):
        seg.keywords[f] = KeywordColumn(arr(f"kw/{f}.codes.npy"), strings(f"kw/{f}.values.parquet", "value"))
    for f in m.get("numeric_fields", {}):
        seg.numerics[f] = NumericColumn(arr(f"num/{f}.npy"), arr(f"num/{f}.missing.npy"))
    for f, spec in m.get("vector_fields", {}).items():
        seg.vectors[f] = arr(f"vec/{f}.npy")
        if spec.get("encoder"):
            seg.encoders[f] = {"terms": strings(f"vec/{f}.encoder.terms.parquet", "term"),
                               "idf": arr(f"vec/{f}.encoder.idf.npy"),
                               "components": arr(f"vec/{f}.encoder.components.npy"), "dim": spec["dim"]}
    for f in m.get("present_fields", []):                # optional: which docs have a value (exists)
        seg.present[f] = arr(f"present/{f}.npy")
    for f in m.get("source_fields", []):                 # optional (S7-5)
        col = pd.read_parquet(path / f"src/{f}.parquet")["text"]
        seg.source[f] = np.array([None if v is pd.NA or v is None else str(v) for v in col.tolist()], dtype=object)
    return seg
