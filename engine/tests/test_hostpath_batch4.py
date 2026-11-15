"""R3c: s3_hostpath.batch4 (Rust) writes the bytes serving.binproto.batch4 (the platform's Python reference) writes.

The checks:
- every golden4 frame (serving/native_front/tests/golden4.txt), rebuilt from make_vectors' own messages;
- the platform's unit case, which includes NaN, Unicode, an interior NUL, a whole reply, an error and a call;
- random shells, with every float edge, big ints, astral and control characters, non-str keys, tuples and nesting,
  and random columns, with ids of mixed widths and field sorts with and without missing values;
- what the writer does not take raises, and the seam's guard then gives the Python bytes.

They skip without the built extension, which only Colab has (colab/hostpath_ci.py)."""
from __future__ import annotations

import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

s3_hostpath = pytest.importorskip("s3_hostpath")
from serving import binproto  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BATCH4_ENV = "S3_HOSTPATH_BATCH4"


@pytest.fixture(autouse=True)
def _rust(monkeypatch):
    monkeypatch.setenv(BATCH4_ENV, "1")


def _same(msgs):
    want = binproto.batch4(msgs)
    got = s3_hostpath.batch4(msgs)
    assert got == want, _diff(got, want)
    return got


def _diff(a, b):
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"lengths {len(a)} vs {len(b)}; first difference at byte {i}: {a[i:i + 40]!r} vs {b[i:i + 40]!r}"


_GOLDEN4_CHECK = r"""
import json, random, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/serving/native_front/tests")
import numpy as np
from serving import binproto, encode as enc
import make_vectors as mv
import s3_hostpath
sub_in = []                                   # per batch: whether a response reached flush_response with a subnormal
real_flush = enc.flush_response
def tiny(a):
    a = np.asarray(a)
    return a.dtype.kind == "f" and bool(np.any((a != 0) & (np.abs(a) < np.finfo(a.dtype).tiny)))
def spy(resp):
    c = resp.get("hits", {}).get("columns") if isinstance(resp, dict) else None
    if c is not None and (tiny(np.asarray(c["_score"], np.float32)) or any(tiny(x["values"]) for x in c["sort"])):
        sub_in[-1] = True
    return real_flush(resp)
enc.flush_response = spy
seen, ref = [], binproto.batch4
def cap(msgs):
    seen.append((list(msgs), ref(msgs)))
    sub_in.append(False)
    return seen[-1][1]
binproto.batch4 = cap
sub_in.append(False)
rows = mv.golden4(random.Random(10))
sub_in.pop()
binproto.batch4 = ref
print(json.dumps({"rows": rows, "rust_equal": [s3_hostpath.batch4(m) == f for m, f in seen],
                  "rust_hex": [s3_hostpath.batch4(m)[4:].hex() for m, _ in seen], "subnormal_input": sub_in,
                  "fp_clean": mv._fp_mode_clean(), "torch_loaded": "torch" in sys.modules,
                  "numpy": np.__version__, "python": sys.version.split()[0]}))
"""


def test_every_golden4_frame():
    """golden4.txt (the platform's, PR #45; its environment in golden.env.json) against Python and Rust here, in a
    fresh process without torch, in a clean FP mode, as make_vectors requires:
    - Rust equals Python on every input;
    - Rust equals the file on every line Python reproduces;
    - Python reproduces all 40 lines and Rust equals the file on all 40. (arch-hp17 and hp18 on the Colab CPU had line
      33 differ, and still did with exact casts in plat-r9: golden4's float pool used pow, log and exp, whose last bit
      differs between macOS and glibc; make_vectors now draws golden4 from floats_exact, exact operations only, with
      the float32 scores by exact scalar casts: the steward's ruling, lab 706a7b9.) A differing line is printed.
    """
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", _GOLDEN4_CHECK, str(ROOT)], capture_output=True, text=True,
                       env={**os.environ, "S3_HOSTPATH_BATCH4": "1"})
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["torch_loaded"] is False and got["fp_clean"] is True
    lines = (ROOT / "serving/native_front/tests/golden4.txt").read_text().splitlines()
    assert len(got["rows"]) == len(lines) == len(got["subnormal_input"]) == 40
    assert all(got["rust_equal"])                              # Rust equals Python on every input
    bad = [i for i, (row, line) in enumerate(zip(got["rows"], lines)) if row != line]
    for i in bad:                                              # what differs, for the platform engineer
        a, b = got["rows"][i].split(" "), lines[i].split(" ")
        part = "payload" if a[0] != b[0] else "bodies"
        print(f"\ngolden4 line {i} differs here ({part}; python {got['python']}, numpy {got['numpy']}; "
              f"subnormal input {got['subnormal_input'][i]})")
    assert bad == [], bad                                      # all 40 lines, on every platform
    for i in range(40):
        assert got["rust_hex"][i] == lines[i].split(" ")[0], i
    print("\ngolden4: 40 of 40 lines reproduced (fresh process, no torch, clean FP mode); rust equals python and the file")


def test_subnormals_pass_through_both_writers_alike():
    """The writers do not flush (the GPU process does, once, before them: PR #45); given subnormals, Rust writes
    exactly what Python writes."""
    sub32 = np.array([1e-40, -1e-40, 1.0, 0.0, -0.0, np.float32(1.4e-45)], np.float32)
    sub64 = np.array([5e-324, -2.2e-308, 3.5, 0.0, -0.0, 1e-310])
    ids = np.array([b"a", b"b", b"c", b"d", b"e", b"f"], "S1")
    _same([("search", 1, "ok", _resp(ids, sub32, [{"values": sub64, "missing": None},
                                                   {"values": sub32.astype(np.float64), "missing": None}]), ("s", 1), 0.0, 1),
           ("search", 2, "ok", _resp(np.array([b"a\x00b"], "S3"), sub32[:1], [{"values": sub64[:1], "missing": [True]}]),
            ("s", 1), 0.0, 1)])


def _resp(ids, sc, sort, total=None, aggs=False):
    r = {"took_ms": {"parse": 0.1}, "hits": {"total": total or len(ids), "columns": {"_id": ids, "_score": sc, "sort": sort}}}
    if aggs:
        r["aggregations"] = {"c": {"buckets": [{"key": "a\"b", "doc_count": 3}]}}
    return r


def test_the_platforms_unit_case():
    a_ids = np.array([b"B001", b"B002", b"B003"], "S4")
    a_sc = np.array([3.5, 2.0, 1.0], np.float32)
    msgs = [
        ("search", 11, "ok", _resp(a_ids, a_sc, [{"values": a_sc.astype(np.float64), "missing": None}]), ("seg", 1), 0.5, 4, 2.0),
        ("search", 12, "ok", _resp(np.array(['x"y'.encode(), "é中".encode()], "S6"), np.array([np.nan, 1e-7], np.float32),
                                   [{"values": np.array([1.5, -0.0]), "missing": [False, True]},
                                    {"values": np.array([7, 2**62], np.int64), "missing": None}], aggs=True), ("seg", 1), 0.4, 4),
        ("search", 13, "ok", _resp(np.array([b"q"], "S1"), np.array([9.0], np.float32),
                                   [{"values": np.array([3], np.int64), "missing": None},
                                    {"values": np.array([2.25]), "missing": None}]), ("seg", 1), 0.3, 4),
        ("search", 14, "ok", _resp(np.array([b"a\x00b", b"c"], "S3"), np.array([1.0, 2.0], np.float32), []), ("seg", 1), 0.2, 4),
        ("search", 15, "error", 400, {"error": {"type": "invalid_request", "path": "", "reason": "no"}}),
        ("call", 16, 200, b'{"status":"ok"}', "application/json", {}),
        ("search", 17, "ok", {"took_ms": 1, "hits": {"total": 0, "hits": []}}, ("seg", 2), 0.1),
        ("search", 18, "ok", _resp(np.array([], "S1"), np.array([], np.float32), []), ("seg", 2), 0.1, 2, None),
        ("search", 20, "ok", _resp(np.array([b"u"], "S1"), np.array([1.0], np.float32),
                                   [{"values": np.array(["k"], object), "missing": None}]), ("seg", 2), 0.1, 2),
        ("search", 21, "ok", _resp(np.array([b"v"], "S1"), [0.25], [{"values": np.array([5], np.uint32), "missing": [True]}]),
         ("seg", 2), 1 / 3, 2, 2 / 3),
    ]
    _same(msgs)
    _same(msgs[:1])
    _same([])


FLOATS = [0.0, -0.0, 1.0, -1.0, 0.1, 1e16, 1e15, 9999999999999998.0, 1e-4, 1e-5, 1e-40, 5e-324, 1.7976931348623157e308,
          float("nan"), float("inf"), float("-inf"), 2.5, 123456.789, 1 / 3]


TIES = [0.0625, 0.0005, 0.0015, 1.0005, 2.675, 0.125, 0.3125, -0.0625, 1e-9, -1e-9, 4.5e12 + 0.0625, 123.4565]


def _value(rng, depth=0):
    k = rng.random()
    if depth > 3 or k < 0.3:
        c = rng.randrange(8)
        if c == 0:
            return rng.choice(FLOATS + [rng.uniform(-1e6, 1e6), rng.random() * 10 ** rng.randint(-30, 30)])
        if c == 1:
            return rng.choice([0, -1, 2**63 - 1, -2**63, 2**64, -2**100, 10**30, rng.randint(-10**6, 10**6)])
        if c == 2:
            return rng.choice([True, False, None])
        return "".join(rng.choice(['a', '"', '\\', '\n', '\t', '\x00', '\x01', '\x1f', '\x7f', 'é', '中', '\U0001f600', ' ',
                                   '/', ' ', '\x08', '\x0c', '\r'])
                       for _ in range(rng.randint(0, 8)))
    if k < 0.6:
        seq = [_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
        return tuple(seq) if rng.random() < 0.3 else seq
    d = {}
    for _ in range(rng.randint(0, 4)):
        key = rng.choice(["k", "hits", "é", "a\"b", 1, 2.5, True, None, -0.0, "columns", "\U0001f600"])
        d[key] = _value(rng, depth + 1)
    return d


def _layout(rng, a):
    """The same values, sometimes as an array the buffer fast path does not take (strided, byte-swapped) or a list."""
    r = rng.random()
    if r < 0.1:
        big = np.empty(2 * len(a), a.dtype)
        big[::2] = a
        return big[::2]                                     # not C-contiguous
    if r < 0.2 and a.dtype.kind in "fiu" and a.dtype.itemsize > 1:
        return a.astype(a.dtype.newbyteorder(">"))          # not native byte order
    return a


def _random_reply(rng, rid):
    h = rng.randint(1, 20)
    w = rng.randint(1, 12)
    ids = _layout(rng, np.array([bytes(rng.choice(b"ab0Z\"\\") for _ in range(rng.randint(1, w))) for _ in range(h)], f"S{w}"))
    with np.errstate(over="ignore"):
        sc = np.array([rng.choice(FLOATS[:13]) for _ in range(h)], np.float64).astype(np.float32)
    sort = []
    for c in rng.sample(["score", "f8", "i8", "f8m", "i8m", "u4", "i1", "u8", "f4", "f2"], rng.randint(0, 3)):
        miss = [rng.random() < 0.3 for _ in range(h)] if c.endswith("m") else None
        if miss is not None and rng.random() < 0.5:
            miss = _layout(rng, np.array(miss))
        if c == "score":
            sort.append({"values": _layout(rng, sc.astype(np.float64)), "missing": None})
        elif c.startswith("f8"):
            sort.append({"values": _layout(rng, np.array([rng.choice(FLOATS) for _ in range(h)])), "missing": miss})
        elif c in ("u4", "i1", "u8"):
            dt, hi = {"u4": (np.uint32, 2**32), "i1": (np.int8, 128), "u8": (np.uint64, 2**64)}[c]
            lo = -128 if c == "i1" else 0
            sort.append({"values": _layout(rng, np.array([rng.randrange(lo, hi) for _ in range(h)], dt)), "missing": None})
        elif c in ("f4", "f2"):                             # not float64: the reply goes whole
            sort.append({"values": np.array([1.5] * h, c), "missing": None})
        else:
            sort.append({"values": _layout(rng, np.array([rng.randint(-2**62, 2**62) for _ in range(h)], np.int64)),
                         "missing": miss})
    hits = {"total": _value(rng, 2), "columns": {"_id": ids, "_score": sc, "sort": sort}}
    if rng.random() < 0.3:
        hits = {"max_score": rng.choice(FLOATS), **hits}
    if rng.random() < 0.1:
        hits["hits"] = "old"
    resp = {"took_ms": {"parse": rng.random(), "search": rng.choice(FLOATS)}, "hits": hits}
    for _ in range(rng.randint(0, 3)):
        resp[rng.choice(["aggregations", "timed_out", 7, "é"])] = _value(rng)
    rest = [rng.choice([1, 64, True, None, 2.5])][:rng.randint(0, 1)]
    if rest and rng.random() < 0.5:
        rest.append(rng.choice([None, 0.0005, 1.23456, 2.5e-4, 1e20, *TIES]))
    if rng.random() < 0.2:
        hits["columns"]["_score"] = _layout(rng, sc) if rng.random() < 0.7 else sc.tolist()
    q = rng.choice(FLOATS[:13] + TIES + [rng.random() * 10 ** rng.randint(-6, 6), 3, True, 2**70])
    return ("search", rid, "ok", resp, (rng.choice(["seg", "é", 3, 2**70, 1.5]), rng.randint(0, 9)), q, *rest)


def test_random_batches():
    rng = random.Random(31)
    for k in range(300):
        msgs = []
        for j in range(rng.randint(0, 16)):
            r = rng.random()
            if r < 0.8:
                msgs.append(_random_reply(rng, k * 100 + j))
            elif r < 0.9:
                msgs.append(("search", k * 100 + j, "error", rng.choice([400, 404, 500]), _value(rng)))
            else:
                msgs.append(("call", k * 100 + j, 200, rng.choice([b"", b"{}", bytes(range(256))]),
                             rng.choice(["application/json", "text/plain; charset=é"]), {}))
        _same(msgs)


def test_float_text_on_the_platforms_repr_vectors():
    """Every float of serving/native_front/tests/pyrepr_vectors.txt, written by the Rust JSON writer (an error body)."""
    import struct
    fl = [struct.unpack("<d", bytes.fromhex(line.split(" ")[0]))[0]
          for line in (ROOT / "serving/native_front/tests/pyrepr_vectors.txt").read_text().splitlines() if line]
    assert len(fl) > 1000
    for k in range(0, len(fl), 500):
        _same([("search", 1, "error", 400, {"x": fl[k:k + 500], "k": {str(i): f for i, f in enumerate(fl[k:k + 50])}})])
    msgs = [("search", i, "ok", _resp(np.array([b"a"], "S1"), np.array([1.0], np.float32), []), ("s", 1), q, 1, q)
            for i, q in enumerate(fl[:3000] + TIES)]
    _same(msgs)


def test_what_it_does_not_take_raises_and_the_guard_gives_the_python_bytes():
    from serving.hostpath import _Guarded
    ids, sc = np.array([b"a"], "S1"), np.array([1.0], np.float32)
    lone = ("search", 1, "ok", {"took_ms": 1, "x": "\ud800", "hits": {"total": 1, "columns": {"_id": ids, "_score": sc, "sort": []}}},
            ("s", 1), 0.0, 1)
    want = binproto.batch4([lone])
    with pytest.raises(Exception):
        s3_hostpath.batch4([lone])
    g = _Guarded(s3_hostpath, "s3_hostpath")
    assert g.batch4([lone]) == want and g.fallbacks == 1
    bad = ("search", 2, "ok", {"took_ms": np.int64(1), "hits": {"total": 1, "columns": {"_id": ids, "_score": sc, "sort": []}}},
           ("s", 1), 0.0, 1)
    with pytest.raises(TypeError):
        binproto.batch4([bad])
    with pytest.raises(TypeError):
        s3_hostpath.batch4([bad])


def test_the_switch_uses_the_python_writer(monkeypatch):
    monkeypatch.setenv(BATCH4_ENV, "0")
    msgs = [("call", 1, 200, b"{}", "application/json", {})]
    assert s3_hostpath.batch4(msgs) == binproto.batch4(msgs)


def _speed_msgs(kind, n=64, h=10):
    rng = np.random.default_rng(1)
    msgs = []
    for i in range(n):
        if kind == "call":
            msgs.append(("call", i, 200, b"{}", "application/json", {}))
            continue
        if kind == "error":
            msgs.append(("search", i, "error", 400, {"error": {"type": "invalid_request", "path": "", "reason": "x"}}))
            continue
        ids = np.array([b"B0%08d" % rng.integers(10**8) for _ in range(h)], "S10")
        s = np.sort(rng.random(h).astype(np.float32))[::-1].copy()
        r = {"took_ms": {"parse": 0.012, "search": 1.234},
             "hits": {"total": {"value": 1234, "relation": "eq"}, "max_score": float(s[0]),
                      "columns": {"_id": ids, "_score": s, "sort": []}}}
        if kind == "faceted":
            r["aggregations"] = {"brand": {"buckets": [{"key": f"b{k}", "doc_count": 10 - k} for k in range(10)]}}
        msgs.append(("search", i, "ok", r, ("seg", 3), 0.512, 64, 2.25))
    return msgs


def test_batch4_speed_is_reported():
    """Not a pass mark: the per-reply time of each writer on batches of 64 replies, printed (faceted: 10 hits and a
    terms aggregation of 10 buckets; deep: 10 hits; h1: 1 hit; error and call: no columns)."""
    import time
    if hasattr(s3_hostpath, "_native"):
        s3_hostpath._native.batch4_profile()
    for kind, h in [("faceted", 10), ("deep", 10), ("deep", 1), ("error", 0), ("call", 0)]:
        msgs = _speed_msgs(kind, h=h)
        assert s3_hostpath.batch4(msgs) == binproto.batch4(msgs)
        out = {}
        for name, fn in [("python", binproto.batch4), ("rust", s3_hostpath.batch4), ("native", s3_hostpath._batch4)]:
            args = (msgs, s3_hostpath._ctx()) if name == "native" else (msgs,)
            for _ in range(20):
                fn(*args)
            t = time.perf_counter()
            for _ in range(200):
                fn(*args)
            out[name] = (time.perf_counter() - t) / 200 / len(msgs) * 1e6
        prof = dict(s3_hostpath._native.batch4_profile()) if hasattr(s3_hostpath, "_native") else {}
        n = 2 * 220 * len(msgs) + 1
        print("\n  rust phases per reply: " + ", ".join(f"{k} {v / n * 1e6:.1f} us" for k, v in prof.items()))
        print(f"batch4 per reply ({kind}, 64 x {h} hits): python {out['python']:.1f} us, rust {out['rust']:.1f} us"
              f" (the extension alone {out['native']:.1f} us)")
