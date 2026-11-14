"""S6-22s: the steward's send-schedule precondition (fairness-rulings, lab 2db4dfc; the per-millisecond rule of ca05113
withdrawn) for one rung's per-request records.

    python colab/send_schedule.py RECORDS.csv RATE [OUT.json]

A rung's sends match the schedule when their send lag over every request has a p99 of at most 5 ms and a maximum of
at most 20 ms; every native rung must meet it to count. Also reported, for information only: the busiest millisecond
and the p99.9 of sends per millisecond (each send at its due time i / RATE plus its lag). Development only."""
import csv
import json
import sys
from collections import Counter

import numpy as np


def check(path, rate):
    per_ms, lags = Counter(), []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["lag_ms"] in ("", "nan"):
                continue
            lag = float(r["lag_ms"])
            lags.append(lag)
            per_ms[int(int(r["i"]) / rate * 1000.0 + lag)] += 1
    counts = np.array(list(per_ms.values()))
    out = {"records": path, "rate": rate, "sends": len(lags), "lag_p99_ms": float(np.percentile(lags, 99)),
           "lag_max_ms": float(max(lags)), "max_sends_in_a_ms": int(counts.max()),
           "p999_sends_per_ms": float(np.percentile(counts, 99.9)), "rule": "lag p99 <= 5 ms and lag max <= 20 ms"}
    out["holds"] = out["lag_p99_ms"] <= 5.0 and out["lag_max_ms"] <= 20.0   # the steward's definition (lab 2db4dfc)
    return out


if __name__ == "__main__":
    res = check(sys.argv[1], float(sys.argv[2]))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(res))
    sys.exit(0 if res["holds"] else 1)
