"""S6-22s: the steward's send-schedule precondition (fairness-rulings, lab ca05113) for one rung's per-request records.

    python colab/send_schedule.py RECORDS.csv RATE [OUT.json]

A rung's figures stand only when the client's arrivals matched the schedule: send lag p99 at most 5 ms, and at most 4
sends in any millisecond above the scheduled rate per millisecond (RATE / 1000). Each request's send time is its due
time i / RATE plus its recorded lag. Prints and writes the lag percentiles, the busiest millisecond, the p99.9 of sends
per millisecond and whether the precondition holds. Development only."""
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
    limit = rate / 1000.0 + 4
    out = {"records": path, "rate": rate, "sends": len(lags), "lag_p99_ms": float(np.percentile(lags, 99)),
           "lag_max_ms": float(max(lags)), "max_sends_in_a_ms": int(counts.max()),
           "p999_sends_per_ms": float(np.percentile(counts, 99.9)), "limit_sends_per_ms": limit}
    out["holds"] = out["lag_p99_ms"] <= 5.0 and out["max_sends_in_a_ms"] <= limit
    return out


if __name__ == "__main__":
    res = check(sys.argv[1], float(sys.argv[2]))
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(res))
    sys.exit(0 if res["holds"] else 1)
