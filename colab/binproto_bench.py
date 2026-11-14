"""S6-22s: the kind-1 checks' cost per reply, the single-pass forms in serving/binproto.py against the numpy forms
they replaced (serving/tests/test_binproto.py keeps those as the reference). A 2,000-hit reply like the depth-2,000
ESCI ones: ids S10 (ASINs), float32 scores, the score widened as the first sort column and an int64 second.
Prints one JSON line. Development only; CPU only.

    python colab/binproto_bench.py [REPEATS]
"""
import json
import sys
import time

import numpy as np

from serving import binproto
from serving.tests.test_binproto import _plain_ids_ref, _safe_ref


def reply(h=2000, seed=0):
    rng = np.random.default_rng(seed)
    letters = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", np.uint8)
    ids = np.frombuffer(rng.choice(letters, size=(h, 10)).astype(np.uint8).tobytes(), "S10")
    sc = np.sort(rng.gamma(2.0, 2.0, h).astype(np.float32))[::-1].copy()
    return {"took_ms": {"parse": 0.1, "retrieve": 2.0}, "hits": {"total": 185504, "columns": {
        "_id": ids, "_score": sc, "sort": [{"values": sc.astype(np.float64), "missing": None},
                                           {"values": rng.integers(0, 1_215_854, h).astype(np.int64), "missing": None}]}}}


def per_call_us(fn, n):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    r = reply()
    c = r["hits"]["columns"]
    ids, s64, v2 = c["_id"], c["_score"].astype(np.float64), c["sort"][1]["values"]
    new = per_call_us(lambda: (binproto._plain_ids(ids), binproto._safe(s64)), n)
    old = per_call_us(lambda: (_plain_ids_ref(ids), _safe_ref(s64)), n)
    meta = {"version": "seg:1", "queue_ms": 0.0, "batch": 1}
    col_new = per_call_us(lambda: binproto.columnar(1, r, meta), n)
    saved_p, saved_s = binproto._plain_ids, binproto._safe
    binproto._plain_ids, binproto._safe = _plain_ids_ref, _safe_ref
    try:
        col_old = per_call_us(lambda: binproto.columnar(1, r, meta), n)
    finally:
        binproto._plain_ids, binproto._safe = saved_p, saved_s
    print(json.dumps({"hits": len(ids), "repeats": n, "numpy": np.__version__,
                      "checks_us": {"new": round(new, 2), "old": round(old, 2), "ratio_old_over_new": round(old / new, 2)},
                      "columnar_us": {"new": round(col_new, 2), "old": round(col_old, 2),
                                      "saved_pct": round(100 * (1 - col_new / col_old), 1)}}))


if __name__ == "__main__":
    main()
