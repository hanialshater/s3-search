"""Commit points (docs/INTERFACES.md §4, "Multi-segment indexes (S8)"): which segments make up an index at a version,
and each segment's live bitmap at that version. Segment files never change; deletes live in the commit's files.

    commit-<version>.json   {"format": "s3commit", "format_version": 1, "version": int,
                             "segments": [segment_id, ...],                       # the index's order
                             "deletes": {segment_id: "del-<version>-<segment_id>.npy"},   # only segments with deletes
                             "checksum": hex}
    del-<version>-<segment_id>.npy   uint32 [ceil(n_docs / 32)], the segment's live bitmap (§4's layout)

**The checksum** (``commit_checksum``): sha256 of the canonical JSON (sorted keys, separators ``,`` and ``:``) of the
commit's fields other than ``checksum``, with one more key, ``"delete_sha256": {segment_id: sha256 of that delete
file}``, which is hashed but not stored. This is the data plane's reading of the ruling's "the canonical JSON of the
other fields plus each delete file's sha256"; ``Index.save_commit`` and ``Index.from_commit`` must use the same bytes.

A commit is written to a temporary name and renamed, delete files first, so a reader never sees half a commit. A
replica loads the newest commit (``latest_commit``) and replays the store's log after its version.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

FORMAT, FORMAT_VERSION = "s3commit", 1
_COMMIT = re.compile(r"commit-(\d+)\.json")


class CorruptCommit(ValueError):
    """A commit whose format, delete files or checksum do not check out."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def commit_checksum(commit: dict, out_dir: str | Path) -> str:
    out = Path(out_dir)
    body = {k: v for k, v in commit.items() if k != "checksum"}
    body["delete_sha256"] = {seg: _sha(out / name) for seg, name in sorted(commit.get("deletes", {}).items())}
    return hashlib.sha256(_canonical(body)).hexdigest()


def delete_file_name(version: int, segment_id: str) -> str:
    return f"del-{int(version)}-{segment_id}.npy"


def write_commit(out_dir: str | Path, version: int, segments: list[str], live: dict[str, np.ndarray] | None = None
                 ) -> Path:
    """Write ``commit-<version>.json``. ``live``: the uint32 live bitmap of each segment that has deletes (a segment
    left out is wholly live). Refuses to overwrite an existing commit of that version."""
    out = Path(out_dir)
    path = out / f"commit-{int(version)}.json"
    if path.exists():
        raise FileExistsError(path)
    if len(set(segments)) != len(segments):
        raise ValueError("a segment is listed twice")
    deletes = {}
    for seg, bits in sorted((live or {}).items()):
        if seg not in segments:
            raise ValueError(f"live bitmap for {seg!r}, which the commit does not list")
        bits = np.ascontiguousarray(bits, dtype=np.uint32)
        name = delete_file_name(version, seg)
        tmp = out / f".{name}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, bits, allow_pickle=False)
        os.replace(tmp, out / name)
        deletes[seg] = name
    commit = {"format": FORMAT, "format_version": FORMAT_VERSION, "version": int(version),
              "segments": list(segments), "deletes": deletes}
    commit["checksum"] = commit_checksum(commit, out)
    tmp = out / f".{path.name}.tmp"
    tmp.write_text(json.dumps(commit, sort_keys=True, indent=1) + "\n")
    os.replace(tmp, path)
    return path


def read_commit(path: str | Path, verify: bool = True) -> dict:
    """The commit dict, plus ``"live": {segment_id: uint32 bitmap}`` for segments with deletes. With ``verify``, the
    format, the checksum, the delete files' dtype and, where the segment is present, their length are checked."""
    path = Path(path)
    out = path.parent
    try:
        commit = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CorruptCommit(f"{path}: {e}") from None
    if commit.get("format") != FORMAT or commit.get("format_version") != FORMAT_VERSION:
        raise CorruptCommit(f"{path}: not an {FORMAT} v{FORMAT_VERSION} commit")
    if verify:
        missing = [n for n in commit.get("deletes", {}).values() if not (out / n).exists()]
        if missing:
            raise CorruptCommit(f"{path}: missing delete files {missing}")
        if commit_checksum(commit, out) != commit.get("checksum"):
            raise CorruptCommit(f"{path}: checksum mismatch")
    live = {}
    for seg, name in commit.get("deletes", {}).items():
        bits = np.load(out / name, allow_pickle=False)
        if verify:
            if bits.dtype != np.uint32 or bits.ndim != 1:
                raise CorruptCommit(f"{path}: {name} is {bits.dtype}{list(bits.shape)}, not uint32[W]")
            man = out / seg / "manifest.json"
            if man.exists():
                n = int(json.loads(man.read_text()).get("n_docs", 0))
                if bits.shape[0] != (n + 31) // 32:
                    raise CorruptCommit(f"{path}: {name} has {bits.shape[0]} words for {n} docs")
        live[seg] = bits
    commit["live"] = live
    return commit


def list_commits(out_dir: str | Path) -> list[Path]:
    """Commit files under ``out_dir``, oldest version first."""
    out = Path(out_dir)
    if not out.exists():
        return []
    found = [(int(m.group(1)), p) for p in out.iterdir() if (m := _COMMIT.fullmatch(p.name))]
    return [p for _, p in sorted(found)]


def latest_commit(out_dir: str | Path) -> Path | None:
    commits = list_commits(out_dir)
    return commits[-1] if commits else None


def gc_commits(out_dir: str | Path, keep: int = 3) -> list[Path]:
    """Keep the newest ``keep`` commits; remove older commits, delete files no kept commit names, and segment
    directories no kept commit names. Returns what was removed. Does nothing when there is no commit."""
    out = Path(out_dir)
    commits = list_commits(out)
    if not commits:
        return []
    kept, dropped = commits[-max(1, keep):], commits[:-max(1, keep)]
    names_segs, names_dels = set(), set()
    for p in kept:
        c = json.loads(p.read_text())
        names_segs.update(c.get("segments", []))
        names_dels.update(c.get("deletes", {}).values())
    removed = []
    for p in dropped:
        p.unlink()
        removed.append(p)
    for p in out.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_file() and p.name.startswith("del-") and p.name not in names_dels:
            p.unlink()
            removed.append(p)
        elif p.is_dir() and (p / "manifest.json").exists() and p.name not in names_segs:
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p)
    return removed
