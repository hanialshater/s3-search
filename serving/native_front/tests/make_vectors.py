"""Writes the S3B2 test vectors the native front must reproduce byte for byte (S6-22s):

    python serving/native_front/tests/make_vectors.py

- pyrepr_vectors.txt: ``hexbits json`` per line, json being ``json.dumps(x)`` (``float.__repr__``; NaN, Infinity,
  -Infinity): every exponent, the positional/exponent boundaries, subnormals, +-0.0, NaN, +-inf, float32 scores.
- escape_vectors.txt: ``hex(utf8) hex(json)`` per line, json being ``encode_basestring_ascii`` of the decoded string:
  quotes, backslashes, every control character, 0x7F, the BMP, astral characters, interior NULs.
- golden3.txt: ``hex(kind 3 frame payload) hex(body)`` per line: whole replies, body being serving.encode.encode.
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


def golden(rng):
    from serving import binproto
    from serving.encode import encode
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
        f = binproto.columnar3(k, resp, {"version": "seg:1", "queue_ms": 0.5, "batch": 3})
        assert f is not None and f[4] == 3
        rows.append(f"{f[4:].hex()} {encode(resp).hex()}")
    return rows


def golden4(rng):
    """Kind 4 frames (S3B3): random batches of replies (columnar with any ids and floats, field sorts with missing
    values, an id with an interior NUL, whole, error, call), each line ``hex(payload) id:status:hex(body) ...``."""
    from serving import binproto
    from serving.encode import encode
    strs = strings(random.Random(17))[1:200]
    fl = floats(random.Random(18))
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
                with np.errstate(over="ignore"):
                    sc = np.array([rng.choice(fl) for _ in range(h)], np.float64).astype(np.float32)
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


def main():
    rng = random.Random(22)
    with open(HERE / "pyrepr_vectors.txt", "w") as f:
        for x in floats(rng):
            f.write(f"{struct.unpack('<Q', struct.pack('<d', x))[0]:016x} {json.dumps(x)}\n")
    with open(HERE / "escape_vectors.txt", "w") as f:
        for s in strings(rng):
            f.write(f"{s.encode('utf-8').hex()} {esc(s).encode().hex()}\n")
    (HERE / "golden3.txt").write_text("\n".join(golden(random.Random(9))) + "\n")
    (HERE / "golden4.txt").write_text("\n".join(golden4(random.Random(10))) + "\n")
    print("written")


if __name__ == "__main__":
    main()
