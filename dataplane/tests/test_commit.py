from __future__ import annotations

import json

import numpy as np
import pytest

from dataplane.indexer import build, read_manifest
from dataplane.indexer.commit import (CorruptCommit, commit_checksum, gc_commits, latest_commit, list_commits,
                                      read_commit, write_commit)
from dataplane.tests.conftest import load


def live_bits(n: int, dead: list[int]) -> np.ndarray:
    alive = np.ones(n, bool)
    alive[dead] = False
    words = np.zeros((n + 31) // 32, np.uint32)
    for d in np.nonzero(alive)[0]:
        words[d >> 5] |= np.uint32(1) << np.uint32(d & 31)
    return words


@pytest.fixture
def two_segments(tmp_path, store, catalogue):
    load(store, catalogue.head(100))
    a = build(store, None, tmp_path / "segs")
    load(store, catalogue.iloc[100:140])
    b = build(store, None, tmp_path / "segs")
    return tmp_path / "segs", a, b


def test_commit_round_trip_in_the_ruled_format(two_segments):
    out, a, b = two_segments
    bits = live_bits(read_manifest(a)["n_docs"], [3, 40])
    p = write_commit(out, 140, [a.name, b.name], {a.name: bits})
    assert p.name == "commit-140.json"
    raw = json.loads(p.read_text())
    assert set(raw) == {"format", "format_version", "version", "segments", "deletes", "checksum"}
    assert (raw["format"], raw["format_version"], raw["version"]) == ("s3commit", 1, 140)
    assert raw["deletes"] == {a.name: f"del-140-{a.name}.npy"} and raw["segments"] == [a.name, b.name]
    assert raw["checksum"] == commit_checksum(raw, out)
    c = read_commit(p)
    assert np.array_equal(c["live"][a.name], bits) and b.name not in c["live"]
    assert not [x for x in out.iterdir() if x.name.startswith(".")]          # no temp files left


def test_corruption_is_caught(two_segments):
    out, a, b = two_segments
    n = read_manifest(a)["n_docs"]
    p = write_commit(out, 140, [a.name, b.name], {a.name: live_bits(n, [1])})
    np.save(out / f"del-140-{a.name}.npy", live_bits(n, [2]))                 # a delete file changed after the commit
    with pytest.raises(CorruptCommit, match="checksum"):
        read_commit(p)
    raw = json.loads(p.read_text())
    raw["segments"] = [b.name, a.name]                                        # the order is part of the checksum
    np.save(out / f"del-140-{a.name}.npy", live_bits(n, [1]))
    p.write_text(json.dumps(raw))
    with pytest.raises(CorruptCommit, match="checksum"):
        read_commit(p)
    p2 = write_commit(out, 141, [a.name], {a.name: np.zeros(1, np.uint32)})   # wrong length for the segment
    with pytest.raises(CorruptCommit, match="words"):
        read_commit(p2)
    with pytest.raises(FileExistsError):
        write_commit(out, 141, [a.name])
    with pytest.raises(ValueError):
        write_commit(out, 142, [a.name], {b.name: np.zeros(2, np.uint32)})


def test_gc_keeps_what_the_kept_commits_name(two_segments, store, catalogue):
    out, a, b = two_segments
    n = read_manifest(a)["n_docs"]
    write_commit(out, 100, [a.name], {a.name: live_bits(n, [0])})
    write_commit(out, 140, [a.name, b.name], {a.name: live_bits(n, [0, 5])})
    load(store, catalogue.iloc[140:150])
    c = build(store, None, out)
    write_commit(out, 150, [c.name])
    removed = {p.name for p in gc_commits(out, keep=2)}
    assert removed == {"commit-100.json", f"del-100-{a.name}.npy"}
    assert [p.name for p in list_commits(out)] == ["commit-140.json", "commit-150.json"]
    assert latest_commit(out).name == "commit-150.json" and a.exists() and b.exists()
    removed = {p.name for p in gc_commits(out, keep=1)}
    assert removed == {"commit-140.json", f"del-140-{a.name}.npy", a.name, b.name} and c.exists()


def test_commit_checksum_vector(tmp_path):
    """The shared vector (engine/tests/test_s8_segments.py::test_commit_checksum_vector, recorded in §4): the
    data plane's and the engine's commit writers hash the same bytes."""
    import hashlib
    from engine.commit import commit_checksum as engine_checksum
    bits = np.array([0xFFFFFFFE, 0x00000003], dtype=np.uint32)
    with open(tmp_path / "del-7-seg-5-aaaa0000.npy", "wb") as fh:
        np.save(fh, bits, allow_pickle=False)
    commit = {"format": "s3commit", "format_version": 1, "version": 7, "segments": ["seg-5-aaaa0000", "seg-7-bbbb1111"],
              "deletes": {"seg-5-aaaa0000": "del-7-seg-5-aaaa0000.npy"}}
    file_sha = hashlib.sha256((tmp_path / "del-7-seg-5-aaaa0000.npy").read_bytes()).hexdigest()
    body = dict(commit, delete_sha256={"seg-5-aaaa0000": file_sha})
    want = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert commit_checksum(commit, tmp_path) == want == engine_checksum(commit, tmp_path)
