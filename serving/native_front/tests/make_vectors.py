"""Writes the S3B2 test vectors the native front must reproduce byte for byte (S6-22s):

    python serving/native_front/tests/make_vectors.py

- pyrepr_vectors.txt: ``hexbits json`` per line, json being ``json.dumps(x)`` (``float.__repr__``; NaN, Infinity,
  -Infinity): every exponent, the positional/exponent boundaries, subnormals, +-0.0, NaN, +-inf, float32 scores.
- escape_vectors.txt: ``hex(utf8) hex(json)`` per line, json being ``encode_basestring_ascii`` of the decoded string:
  quotes, backslashes, every control character, 0x7F, the BMP, astral characters, interior NULs.
- golden3.txt: ``hex(kind 3 frame payload) hex(body)`` per line: whole replies, body being serving.encode.encode.
- golden4.txt: kind 4 frames (S3B3) and their bodies.
golden3 and golden4 are written from responses passed through serving.encode.flush_response first, as the GPU process
does before any writer (the steward's ruling, lab f134183): subnormal floats are signed zero in the frames and bodies.
The inputs themselves are drawn with float arithmetic, so the generator runs only in a process with flush-to-zero and
denormals-are-zero off (it refuses otherwise) and records where it ran in golden.env.json. The files before the flush
are kept as golden3.pre-flush.txt and golden4.pre-flush.txt; golden4 before its libm-free pool (floats_exact) as
golden4.pre-exact.txt.
Deterministic (seeded); regenerate only when the protocol changes."""
import json
import random
import struct
import sys
from json.encoder import encode_basestring_ascii as esc
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))


def floats(rng):
    out = [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 5e-324, -5e-324, 2.2250738585072014e-308,
           2.225073858507201e-308, 1.7976931348623157e308, -1.7976931348623157e308, 1.0, -1.0, 0.5, 0.1, 0.2, 0.3]
    for e in range(-330, 310):                               # powers of ten and their neighbours
        for m in (1.0, 9.999999999999999, 1.5, 2.5, 3.0, 9.9, 1.0000000000000002):
            try:
                x = float(f"{m}e{e}")
            except OverflowError:
                continue
            out += [x, -x, np.nextafter(x, np.inf), np.nextafter(x, -np.inf)]
    for x in (1e-4, 1e-5, 1e15, 1e16, 1e17, 9.999e-5, 1.0001e-4, 9.99e15, 1.234e16, 123456789012345.6,
              1234567890123456.7, 12345678901234567.0, 0.00012345, 0.000012345):
        out += [x, -x, np.nextafter(x, np.inf), np.nextafter(x, -np.inf)]
    for _ in range(6000):                                    # random bit patterns (finite)
        bits = rng.getrandbits(64)
        x = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if x == x and abs(x) != float("inf"):
            out.append(x)
    for _ in range(6000):                                    # random magnitudes
        out.append(rng.uniform(-1, 1) * 10 ** rng.uniform(-30, 30))
    for _ in range(4000):                                    # float32 scores widened, as the engine sends them
        out.append(float(np.float32(rng.gammavariate(2, 2) * 10 ** rng.uniform(-6, 3))))
    for _ in range(2000):                                    # integral values
        out.append(float(rng.randint(-10**17, 10**17)))
    return out


def floats_exact(rng):
    """golden4's float pool, made with exact operations only (no libm): the same specials, powers of ten parsed from
    text and their neighbours, random bit patterns, random magnitudes as a uniform times a power of two, float32-sized
    scores the same way, explicit subnormal bit patterns of both widths, and integral values. ``floats`` uses ``**`` and
    ``gammavariate`` (pow, log, exp), whose last bit differs between macOS and glibc: golden4 line 33 did not reproduce
    on the Colab CPU (arch-hp17, hp18, plat-r9) even with exact casts (the steward, lab 706a7b9)."""
    out = [0.0, -0.0, float("nan"), float("inf"), float("-inf"), 5e-324, -5e-324, 2.2250738585072014e-308,
           2.225073858507201e-308, 1.7976931348623157e308, -1.7976931348623157e308, 1.0, -1.0, 0.5, 0.1, 0.2, 0.3]
    for e in range(-330, 310, 3):                            # powers of ten (correctly rounded parses) and neighbours
        for m in (1.0, 9.999999999999999, 1.5, 3.0):
            try:
                x = float(f"{m}e{e}")
            except OverflowError:
                continue
            out += [x, -x, float(np.nextafter(x, np.inf)), float(np.nextafter(x, -np.inf))]
    for _ in range(3000):                                    # random bit patterns (finite)
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if x == x and abs(x) != float("inf"):
            out.append(x)
    for _ in range(3000):                                    # random magnitudes: a uniform times a power of two (exact)
        out.append(rng.uniform(-1, 1) * 2.0 ** rng.randint(-100, 100))
    for _ in range(3000):                                    # float32-sized scores, positive
        out.append(rng.random() * 2.0 ** rng.randint(-20, 10))
    for _ in range(300):                                     # subnormals by bits: float32-range and float64
        out.append(struct.unpack("<f", struct.pack("<I", rng.getrandbits(23) | (rng.getrandbits(1) << 31)))[0])
        out.append(struct.unpack("<d", struct.pack("<Q", rng.getrandbits(52) | (rng.getrandbits(1) << 63)))[0])
    for _ in range(1000):                                    # integral values
        out.append(float(rng.randint(-10**17, 10**17)))
    return out


def strings(rng):
    pool = (["\"", "\\", "\x7f", "\x00", " ", "~", "é", "ß", "中", " ", "�", "😀", "\U0010ffff", "/"]
            + [chr(c) for c in range(0x20)])
    out = ["", "B07ABC1234", "a\"b", "a\\b", "\x00x", "x\x00y", "tab\there", "\U0001f600end"]
    for _ in range(3000):
        n = rng.randint(1, 24)
        s = "".join(rng.choice(pool) if rng.random() < 0.3 else chr(rng.choice([rng.randint(0x21, 0x7e),
                    rng.randint(0xa0, 0xd7ff), rng.randint(0xe000, 0xfffd), rng.randint(0x10000, 0x10ffff)]))
                    for _ in range(n))
        out.append(s)
    return [s for s in out if not s.endswith("\x00")]       # S<w> ids never end in NUL (numpy strips them)


def _fp_mode_clean() -> bool:
    """Whether subnormals survive a float32 widening and a multiply here (no flush-to-zero, no denormals-are-zero)."""
    b = np.frombuffer(struct.pack("<I", 1), np.float32)
    return bool(float(b.astype(np.float64)[0]) != 0.0 and (b * np.float32(1.0)).view(np.uint32)[0] == 1)


def golden(rng):
    from serving import binproto
    from serving.encode import encode, flush_response
    rows = []
    strs = strings(random.Random(7))[1:200]
    fl = floats(random.Random(8))
    for k in range(60):
        h = rng.randint(1, 40)
        ids = [rng.choice(strs) or "x" for _ in range(h)]
        w = max(len(s.encode("utf-8")) for s in ids)
        idarr = np.array([s.encode("utf-8") for s in ids], f"S{w}")
        sc = np.array([rng.choice(fl) for _ in range(h)], np.float64).astype(np.float32)
        sort = []
        for kind in rng.sample(["score", "f8", "i8"], rng.randint(0, 3)):
            if kind == "score":
                sort.append({"values": sc.astype(np.float64), "missing": None})
            elif kind == "f8":
                sort.append({"values": np.array([rng.choice(fl) for _ in range(h)], np.float64), "missing": None})
            else:
                sort.append({"values": np.array([rng.randint(-2**62, 2**62) for _ in range(h)], np.int64), "missing": None})
        resp = {"took_ms": {"parse": 0.1}, "hits": {"total": h + k, "columns": {"_id": idarr, "_score": sc, "sort": sort}}}
        if k % 5 == 0:
            resp["aggregations"] = {"c": {"buckets": [{"key": "a\"b", "doc_count": 3}]}}
        resp = flush_response(resp)
        f = binproto.columnar3(k, resp, {"version": "seg:1", "queue_ms": 0.5, "batch": 3})
        assert f is not None and f[4] == 3
        rows.append(f"{f[4:].hex()} {encode(resp).hex()}")
    return rows


def _f32_exact(values):
    """float32 scores from float64 values, each as exact float32 bits: one scalar IEEE conversion per value (struct's
    C cast, round to nearest even; out of range is +-inf), never numpy's vectorised cast. On the Colab CPU numpy's
    float64-to-float32 cast gave a different subnormal for one value (golden4 line 33, arch-hp17 and hp18), so the
    subnormal inputs are now bit patterns made the same way everywhere (the steward, lab 706a7b9)."""
    out = []
    for x in values:
        try:
            out.append(struct.unpack("<I", struct.pack("<f", x))[0])
        except OverflowError:                                   # beyond float32's range after rounding: +-inf
            out.append(0x7F800000 | (0x80000000 if x < 0 else 0))
    return np.array(out, np.uint32).view(np.float32)


def golden4(rng):
    """Kind 4 frames (S3B3): random batches of replies (columnar with any ids and floats, field sorts with missing
    values, an id with an interior NUL, whole, error, call), each line ``hex(payload) id:status:hex(body) ...``."""
    from serving import binproto
    from serving.encode import encode, flush_response
    strs = strings(random.Random(17))[1:200]
    fl = floats_exact(random.Random(18))
    rows = []
    for k in range(40):
        msgs, want = [], []
        for j in range(rng.randint(1, 12)):
            rid = k * 100 + j
            kind = rng.random()
            if kind < 0.75:
                h = rng.randint(1, 30)
                ids = [rng.choice(strs) or "x" for _ in range(h)]
                if rng.random() < 0.05:
                    ids[0] = "a\x00b"
                w = max(len(s.encode("utf-8")) for s in ids)
                idarr = np.array([s.encode("utf-8") for s in ids], f"S{w}")
                sc = _f32_exact([rng.choice(fl) for _ in range(h)])
                sort = []
                for c in rng.sample(["score", "f8", "i8", "f8m", "i8m"], rng.randint(0, 3)):
                    miss = None
                    if c.endswith("m"):
                        miss = [rng.random() < 0.3 for _ in range(h)]
                    if c == "score":
                        sort.append({"values": sc.astype(np.float64), "missing": None})
                    elif c.startswith("f8"):
                        sort.append({"values": np.array([rng.choice(fl) for _ in range(h)], np.float64), "missing": miss})
                    else:
                        sort.append({"values": np.array([rng.randint(-2**62, 2**62) for _ in range(h)], np.int64), "missing": miss})
                resp = {"took_ms": {"parse": 0.1}, "hits": {"total": h + j, "columns": {"_id": idarr, "_score": sc, "sort": sort}}}
                resp = flush_response(resp)
                msgs.append(("search", rid, "ok", resp, ("seg", 1), 0.5, 3, 1.25))
                want.append((rid, 200, encode(resp)))
            elif kind < 0.9:
                body = {"error": {"type": "invalid_request", "path": "", "reason": "x"}}
                msgs.append(("search", rid, "error", 400, body))
                want.append((rid, 400, json.dumps(body, separators=(",", ":")).encode()))
            else:
                msgs.append(("call", rid, 200, b'{"status":"ok"}', "application/json", {}))
                want.append((rid, 200, b'{"status":"ok"}'))
        f = binproto.batch4(msgs)
        assert f[4] == 4
        rows.append(f[4:].hex() + "".join(f" {i}:{st}:{b.hex()}" for i, st, b in want))
    return rows


# Where golden4.txt was checked to reproduce all 40 lines (Python and the Rust writer), recorded by hand after each check.
GOLDEN4_REPRODUCED_ON = [
    {"platform": "macOS-26.6.2-x86_64 (the generating host)", "python": "3.11.5", "numpy": "2.4.6", "check": "make_vectors.py, golden4 unchanged on regeneration"},
    {"platform": "Colab CPU, Linux x86_64 (glibc)", "python": "3.13", "check": "plat-r10 at 640b9a3: engine/tests/test_hostpath_batch4.py 8 passed (all 40 lines, Python and Rust) and the native front's cargo test 5/5"},
]


def main():
    if not _fp_mode_clean():
        sys.exit("make_vectors: flush-to-zero or denormals-are-zero is on in this process (is torch loaded?); run it in a clean one")
    rng = random.Random(22)
    with open(HERE / "pyrepr_vectors.txt", "w") as f:
        for x in floats(rng):
            f.write(f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016x} {json.dumps(x)}\n")
    with open(HERE / "escape_vectors.txt", "w") as f:
        for s in strings(rng):
            f.write(f"{s.encode('utf-8').hex()} {esc(s).encode().hex()}\n")
    (HERE / "golden3.txt").write_text("\n".join(golden(random.Random(9))) + "\n")
    (HERE / "golden4.txt").write_text("\n".join(golden4(random.Random(10))) + "\n")
    import platform
    env = {"generated_by": "serving/native_front/tests/make_vectors.py", "python": platform.python_version(),
           "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(),
           "torch_loaded": "torch" in sys.modules, "fp_mode_clean": bool(_fp_mode_clean()),
           "note": "golden3 and golden4 come from responses passed through serving.encode.flush_response (lab f134183)",
           "golden4_inputs": ("floats_exact(random.Random(18)): exact operations only (no pow, log or exp), float32 scores "
                              "by exact scalar casts (lab 706a7b9)"),
           "golden4_reproduced_on": GOLDEN4_REPRODUCED_ON}
    (HERE / "golden.env.json").write_text(json.dumps(env, indent=1) + "\n")
    print("written")


if __name__ == "__main__":
    main()
