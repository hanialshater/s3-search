"""S6-22s: where a request's time goes, from the native load client's per-request records (fair_bench --client=native).

    python colab/tail_breakdown.py RECORDS.csv [OUT.json]

Each line: i, latency ms (from the due time), send lag ms, status, GPU process, batch size, and the server's
X-S3-Queue-Ms (waiting in the GPU process's queue), X-S3-Serve-Ms (the front, from the request's arrival to its reply)
and X-S3-Batch-Ms (that batch's search_batch wall time). Per answered request:

    client+network = latency - serve          (the send lag, the wire both ways, reading the body)
    front          = serve - queue - batch    (the front's parse, the GPU process's reply path and its encoding)
    queue          = queue
    gpu batch      = batch

Reports each component at p50, p90, p99 and p99.9; for the slowest 1% of requests, the share each component is of
their latency and which dominates, by GPU process and batch size; and a 100 ms time series of the latency p99 and
the queue wait (to tell periodic stalls from load). Development only."""
import csv
import json
import sys
from collections import Counter, defaultdict

import numpy as np

PARTS = ("client_network", "front", "queue", "gpu_batch")


def rows(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["status"] != "200" or not r["lat_ms"] or not r["serve_ms"]:
                continue
            lat, serve = float(r["lat_ms"]), float(r["serve_ms"])
            q = float(r["queue_ms"] or 0.0)
            b = float(r["batch_ms"]) if r["batch_ms"] else float("nan")
            out.append({"i": int(r["i"]), "lat": lat, "lag": float(r["lag_ms"]), "gpu": r["gpu"],
                        "batch": int(r["batch"] or 0), "client_network": lat - serve,
                        "front": serve - q - (0.0 if np.isnan(b) else b), "queue": q, "gpu_batch": b})
    return out


def summary(recs, rate=None):
    lat = np.array([r["lat"] for r in recs])
    pct = {k: {f"p{q}": float(np.nanpercentile([r[k] for r in recs], q)) for q in (50, 90, 99, 99.9)}
           for k in ("lat",) + PARTS}
    cut = np.percentile(lat, 99)
    slow = [r for r in recs if r["lat"] >= cut]
    dom = Counter(max(PARTS, key=lambda k: -1 if np.isnan(r[k]) else r[k]) for r in slow)
    share = {k: float(np.nanmean([r[k] / r["lat"] for r in slow])) for k in PARTS}
    by_gpu = Counter(r["gpu"] for r in slow)
    all_gpu = Counter(r["gpu"] for r in recs)
    batch_slow = float(np.median([r["batch"] for r in slow])) if slow else None
    batch_all = float(np.median([r["batch"] for r in recs])) if recs else None
    series = []
    if rate:
        buckets = defaultdict(list)
        for r in recs:
            buckets[int(r["i"] / rate * 10)].append(r)       # 100 ms of due time
        for k in sorted(buckets):
            b = buckets[k]
            series.append([k / 10, float(np.percentile([r["lat"] for r in b], 99)),
                           float(np.percentile([r["queue"] for r in b], 99)), len(b)])
    return {"answered": len(recs), "percentiles_ms": pct, "slowest_1pct": {
        "n": len(slow), "latency_cut_ms": float(cut), "dominant": dict(dom), "mean_share": share,
        "by_gpu_process": dict(by_gpu), "all_by_gpu_process": dict(all_gpu),
        "median_batch": batch_slow, "median_batch_all": batch_all}, "series_100ms": series}


def main():
    path = sys.argv[1]
    recs = rows(path)
    rate = None
    try:
        rate = float(path.rsplit("-", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        pass
    s = summary(recs, rate)
    s["records"] = path
    if len(sys.argv) > 2:
        json.dump(s, open(sys.argv[2], "w"), indent=1)
    p = s["percentiles_ms"]
    print(f"{path}: {s['answered']} answered")
    for k in ("lat",) + PARTS:
        print(f"  {k:15s} " + "  ".join(f"{q} {v:8.2f}" for q, v in p[k].items()))
    sl = s["slowest_1pct"]
    print(f"  slowest 1% (>= {sl['latency_cut_ms']:.1f} ms): dominant {sl['dominant']}, mean share "
          + ", ".join(f"{k} {v:.0%}" for k, v in sl["mean_share"].items())
          + f"; median batch {sl['median_batch']} (all {sl['median_batch_all']})")


if __name__ == "__main__":
    main()
