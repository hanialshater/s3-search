"""R4c/R4d: where s3_hostpath.batch4's time goes, per phase, on CPU (development only, never a score).

    python native/s3_hostpath/bench_batch4_phases.py [--reps=300] [--out=batch4-phases.json]

batch4 (native/s3_hostpath/src/reply.rs) writes one kind 4 frame for a batch's replies, holding the GIL throughout.
Its phases (``batch4_profile()``) are:
- meta: the record's meta JSON (version, queue_ms, batch, batch_ms);
- columns: reading the hit columns (ids, scores, sort values) through the buffer protocol and copying them;
- shell: the response JSON without its hits (total, aggregations, took_ms), written from the Python dict;
- frame: laying out the frame from what the other phases collected.
Only "columns" and "frame" could leave the GIL (R4d): the buffers copied under it, the layout done by the Loop's
writer. The meta and the shell read Python objects.

The replies are shaped like the engine's (hits="columns", ids="bytes"; serving/gpu.py ``_answer``'s message tuple):
- deep: 2,000 hits, ESCI-like 10-byte ids, float32 scores, sort [the score widened, an int64 doc index], total;
- faceted: 10 hits and two terms aggregations of 10 buckets each (brand and colour).
Batches of 10 and 64 replies (plat-m40's R4 batches were 10 to 12). The Python reference writer
(serving.binproto.batch4) and flush_response are timed beside it. Prints and writes JSON.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np


def _opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


REPS = int(_opt("reps", "300"))
OUT = _opt("out", "batch4-phases.json")


def deep(i, rng):
    h = 2000
    ids = np.array([b"B0%08d" % ((i * 7919 + k * 104729) % 10**8) for k in range(h)], dtype="S10")
    s = np.sort(rng.random(h, dtype=np.float32) * 20)[::-1].astype(np.float32)
    return {"hits": {"total": int(rng.integers(2000, 400000)), "columns": {
        "_id": ids, "_score": s,
        "sort": [{"values": s.astype(np.float64), "missing": None},
                 {"values": rng.integers(0, 1_200_000, h).astype(np.int64), "missing": None}]}},
        "took_ms": {"parse": 0.31, "retrieve": 2.517, "filter": 0.064, "page": 0.57, "total": 4.674}}


def faceted(i, rng):
    r = deep(i, rng)
    c = r["hits"]["columns"]
    r["hits"]["columns"] = {"_id": c["_id"][:10], "_score": c["_score"][:10],
                            "sort": [{"values": v["values"][:10], "missing": None} for v in c["sort"]]}
    words = ["Amazon Basics", "Anker", "Apple", "Belkin", "Logitech", "Samsung", "Sony", "UGREEN", "JBL", "Bose",
             "Black", "White", "Silver", "Blue", "Red", "Gray", "Pink", "Green", "Gold", "Clear"]
    r["aggregations"] = {f: {"doc_count_error_upper_bound": 0, "sum_other_doc_count": int(rng.integers(0, 5000)),
                             "buckets": [{"key": words[(j + 10 * n) % 20], "doc_count": int(rng.integers(1, 900))}
                                         for j in range(10)]}
                         for n, f in enumerate(("brand", "color"))}
    r["took_ms"]["aggs"] = 0.42
    return r


def msgs_of(resps, base):
    t = time.perf_counter()
    return [("search", base + k, "ok", r, ("seg-200-0cf5b2ae", 1), 3.512 + k / 7, len(resps), 5.183, t, None)
            for k, r in enumerate(resps)]


def main():
    import s3_hostpath
    from s3_hostpath import _native
    from serving import binproto
    from serving.encode import flush_response
    rng = np.random.default_rng(0)
    out = {"kind": "batch4-phases", "label": "development only, never a score", "reps": REPS,
           "python": sys.version.split()[0], "numpy": np.__version__, "cases": {}}
    for kind, make in (("deep", deep), ("faceted", faceted)):
        for bs in (10, 64):
            batches = [msgs_of([make(b * bs + k, rng) for k in range(bs)], b * bs) for b in range(8)]
            for m in batches[:2]:                                   # warm, and the bytes are the reference's
                assert s3_hostpath.batch4(m) == binproto.batch4(m)
            _native.batch4_profile()                                # reset the phase totals
            t0 = time.perf_counter()
            for r in range(REPS):
                s3_hostpath.batch4(batches[r % 8])
            wall = time.perf_counter() - t0
            ph = dict(_native.batch4_profile())
            n = REPS * bs
            us = {k: round(v / n * 1e6, 3) for k, v in ph.items()}
            rest = round(us["total"] - sum(us[k] for k in ("meta", "columns", "shell", "frame")), 3)
            py_reps = max(10, REPS // 10)
            t0 = time.perf_counter()
            for r in range(py_reps):
                binproto.batch4(batches[r % 8])
            py_us = (time.perf_counter() - t0) / (py_reps * bs) * 1e6
            t0 = time.perf_counter()
            for r in range(REPS):
                for m in batches[r % 8]:
                    flush_response(m[3])
            fl_us = (time.perf_counter() - t0) / n * 1e6
            frame_bytes = len(s3_hostpath.batch4(batches[0]))
            case = {"us_per_reply": us, "unattributed_us": rest, "wall_us_per_reply": round(wall / n * 1e6, 3),
                    "python_batch4_us_per_reply": round(py_us, 3), "flush_response_us_per_reply": round(fl_us, 3),
                    "frame_bytes_per_reply": frame_bytes // bs,
                    "share": {k: round(us[k] / us["total"], 3) for k in ("meta", "columns", "shell", "frame")}}
            out["cases"][f"{kind}_batch{bs}"] = case
            print(f"[batch4-phases] {kind} batch {bs}: {json.dumps(case)}", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
