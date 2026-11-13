"""Replay queries at a steady rate against a serving endpoint and count failures. Standard library only.

    python -m deploy.e2e.replay --url http://toy.default.svc --index toy --qps 50 --duration 60

Exit status 0 only if every query got HTTP 200 with a well-formed body. The kind end-to-end runs this in a pod inside
the cluster while the operator rolls a new segment (the S4 prediction: zero failed queries at 50 QPS).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WORDS = ["shirt", "jacket", "jeans", "sneaker", "boot", "backpack", "mug", "kettle", "lamp", "chair",
         "red", "blue", "black", "white", "green", "grey", "durable", "northwind", "acme", "summit"]


def one(url: str, index: str, i: int, field: str) -> tuple[bool, str]:
    q = {"query": {"match": {field: f"{WORDS[i % len(WORDS)]} {WORDS[(i * 7 + 3) % len(WORDS)]}"}}, "size": 5}
    req = urllib.request.Request(f"{url}/indexes/{index}/_search", data=json.dumps(q).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.load(r)
            if r.status == 200 and "hits" in body and "total" in body["hits"]:
                return True, ""
            return False, f"bad body {str(body)[:120]}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001 - a refused or reset connection is a failed query
        return False, f"{type(e).__name__}: {e}"


def replay(url: str, index: str, qps: float, duration: float, field: str = "product_title", workers: int = 32) -> dict:
    fails: list[str] = []
    lat: list[float] = []
    lock = threading.Lock()
    counter = itertools.count()

    def task(i: int) -> None:
        t0 = time.perf_counter()
        ok, why = one(url, index, i, field)
        with lock:
            lat.append((time.perf_counter() - t0) * 1000)
            if not ok:
                fails.append(why)

    start = time.perf_counter()
    n = int(qps * duration)
    with ThreadPoolExecutor(workers) as ex:
        for _ in range(n):
            i = next(counter)
            ex.submit(task, i)
            delay = start + (i + 1) / qps - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    lat.sort()
    pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 2) if lat else None
    return {"sent": n, "failed": len(fails), "failures": sorted(set(fails))[:5], "p50_ms": pct(0.5), "p99_ms": pct(0.99),
            "seconds": round(time.perf_counter() - start, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--index", default="toy")
    ap.add_argument("--qps", type=float, default=50)
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--field", default="product_title")
    a = ap.parse_args(argv)
    out = replay(a.url.rstrip("/"), a.index, a.qps, a.duration, a.field)
    print(json.dumps(out))
    return 0 if out["failed"] == 0 and out["sent"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
