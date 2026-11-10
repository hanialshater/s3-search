"""Commit points, the engine's own reader and writer (docs/INTERFACES.md §4, "Multi-segment indexes (S8)").

The format and the checksum are the data plane's (``dataplane/indexer/commit.py``); the engine may not import the
data plane, so it carries this reader and writer of the same bytes, and a shared test vector checks both sides
(``engine/tests/test_s8_segments.py::test_commit_checksum_vector``).

    commit-<version>.json   {"format": "s3commit", "format_version": 1, "version": int,
                             "segments": [segment_id, ...], "deletes": {segment_id: "del-<version>-<segment_id>.npy"},
                             "checksum": hex}
    del-<version>-<segment_id>.npy   uint32 [ceil(n_docs / 32)], the segment's live bitmap

The checksum: sha256 of the canonical JSON (sorted keys, separators "," and ":") of every field but ``checksum``,
with one more key, ``"delete_sha256": {segment_id: sha256 of that delete file}``, hashed but not stored.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .errors import CorruptSegment

FORMAT, FORMAT_VERSION = "s3commit", 1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def commit_checksum(commit: dict, out_dir) -> str:
    out = Path(out_dir)
    body = {k: v for k, v in commit.items() if k not in ("checksum", "live")}
    body["delete_sha256"] = {seg: _sha(out / name) for seg, name in sorted(commit.get("deletes", {}).items())}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def delete_file_name(version: int, segment_id: str) -> str:
    return f"del-{int(version)}-{segment_id}.npy"


def write_commit(out_dir, version: int, segments: list[str], live: dict | None = None) -> Path:
    """``live``: the uint32 live bitmap of each segment that has deletes. Delete files first, then the commit, each
    to a temporary name and renamed."""
    out = Path(out_dir)
    path = out / f"commit-{int(version)}.json"
    if path.exists():
        raise FileExistsError(path)
    deletes = {}
    for seg, bits in sorted((live or {}).items()):
        name = delete_file_name(version, seg)
        tmp = out / f".{name}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, np.ascontiguousarray(bits, dtype=np.uint32), allow_pickle=False)
        os.replace(tmp, out / name)
        deletes[seg] = name
    commit = {"format": FORMAT, "format_version": FORMAT_VERSION, "version": int(version),
              "segments": list(segments), "deletes": deletes}
    commit["checksum"] = commit_checksum(commit, out)
    tmp = out / f".{path.name}.tmp"
    tmp.write_text(json.dumps(commit, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, path)
    return path


def read_commit(path, verify: bool = True) -> dict:
    """The commit, plus ``"live": {segment_id: uint32 bitmap}``. Raises ``CorruptSegment`` when it does not check out."""
    path = Path(path)
    out = path.parent
    try:
        commit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CorruptSegment(f"{path}: {e}") from None
    if commit.get("format") != FORMAT or commit.get("format_version") != FORMAT_VERSION:
        raise CorruptSegment(f"{path}: not an {FORMAT} v{FORMAT_VERSION} commit")
    if verify:
        if any(not (out / n).exists() for n in commit.get("deletes", {}).values()):
            raise CorruptSegment(f"{path}: a delete file is missing")
        if commit_checksum(commit, out) != commit.get("checksum"):
            raise CorruptSegment(f"{path}: checksum mismatch")
    commit["live"] = {seg: np.load(out / name, allow_pickle=False) for seg, name in commit.get("deletes", {}).items()}
    return commit
