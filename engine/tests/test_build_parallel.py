"""The build speed-ups (2026-11-11) leave the segment identical: distinct values analysed once, the analysis in forked
workers, the vectorised missing checks and keyword columns. Here: a forced parallel build equals the serial one file
for file, with missing values, repeated values, non-string values and every column kind."""
import hashlib
import json

import numpy as np
import pandas as pd

import engine.build as B
from engine.segment import write_segment
from engine.tests.conftest import toy_catalogue, toy_mapping


def digest(seg, path):
    out = write_segment(seg, path)
    man = json.loads((out / "manifest.json").read_text())
    return man["checksum"], {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob("*.npy")}


def catalogue():
    c = toy_catalogue(400, 3)
    for col, every in (("product_brand", 3), ("product_color", 4), ("product_description", 2), ("price", 5)):
        c.loc[c.index[::every], col] = None
    c = c.astype({"product_bullet_point": object})
    c.loc[c.index[::7], "product_bullet_point"] = 12.5                  # a non-string value in a text field
    c.loc[c.index[1::9], "product_title"] = ""                          # empty and stopword-only values
    c.loc[c.index[2::9], "product_title"] = "the and of"
    c.loc[c.index[3::11], "category"] = "x" * 300                       # over ignore_above
    return c


def test_parallel_build_equals_serial(tmp_path, monkeypatch):
    m = toy_mapping()
    m["properties"]["category"] = {"type": "keyword", "ignore_above": 256}
    c = catalogue()
    monkeypatch.setenv("S3_BUILD_WORKERS", "1")
    serial = digest(B.build_segment(c, m), tmp_path / "a")
    monkeypatch.setenv("S3_BUILD_WORKERS", "3")
    monkeypatch.setattr(B, "PARALLEL_MIN_CHARS", 0)
    assert B._pool_ok(3, 10)
    parallel = digest(B.build_segment(c, m), tmp_path / "b")
    assert serial == parallel


def test_missing_mask_matches_the_scalar_rule():
    vals = [None, np.nan, pd.NA, pd.NaT, "", "x", 0, 1.5, [1], (), np.array([1]), {"a": 1}, float("nan")]
    s = pd.Series(vals, dtype=object)
    assert B._missing_mask(s).tolist() == [B._is_missing(v) for v in vals]
    t = pd.Series(["a", None, "b"], dtype="string")
    assert B._missing_mask(t).tolist() == [False, True, False]
    assert B._strings(t, B._missing_mask(t)).tolist() == ["a", "", "b"]
