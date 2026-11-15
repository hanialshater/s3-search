"""S6-22s: the steward's amended client-equivalence check (fairness-rulings, lab 9826ffb), from one fair_bench results
JSON run with --clients=aiohttp,native,native,aiohttp --warm-rung=1 and both clients' per-request records.

    python colab/client_gate.py RESULTS.json --e2e-rate=R [OUT.json]

Per rate, between the clients' rungs (the mean of each client's two rungs):
  - server serve time (X-S3-Serve-Ms): p50 within 10%, p99 within max(25%, 10 ms);
  - native overhead (latency - serve) p99 at most 5 ms, and native send lag p99 at most 5 ms;
  - achieved within 0.5%, 0 errors and 0 timeouts on every rung, equal sample index sets;
  - the end-to-end gate (latency p50 within 10%, p99 within max(25%, 10 ms) and no wider than the same-client replicate
    spread plus that bound) only at --e2e-rate, filed beforehand, and only if aiohttp's send lag p99 there is at most 5 ms.
The unloaded replay's result for the run is reported too (0 differences required). Development only."""
import csv
import json
import sys

import numpy as np


def rung_records(path):
    lat, serve, lag = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["lag_ms"] not in ("", "nan"):
                lag.append(float(r["lag_ms"]))
            if r["status"] == "200" and r["lat_ms"] and r["serve_ms"]:
                lat.append(float(r["lat_ms"]))
                serve.append(float(r["serve_ms"]))
    lat, serve = np.array(lat), np.array(serve)
    return {"n": int(len(lat)), "serve_p50": float(np.percentile(serve, 50)), "serve_p99": float(np.percentile(serve, 99)),
            "overhead_p99": float(np.percentile(lat - serve, 99)), "lag_p99": float(np.percentile(lag, 99)),
            "lat_p50": float(np.percentile(lat, 50)), "lat_p99": float(np.percentile(lat, 99))}


def within(a, b, rel, floor=0.0):
    return abs(a - b) <= max(rel * max(a, b), floor)


def gate(results_path, e2e_rate):
    d = json.load(open(results_path))
    e = d["engines"]["engine"]
    runs = e["runs"]
    out = {"results": results_path, "e2e_rate": e2e_rate, "rates": {}, "unloaded": {
        "checked": e["unloaded_check"].get("checked"), "differ": len(e["unloaded_check"].get("mismatch", [])),
        "error": e["unloaded_check"].get("error")}}
    for rate in sorted({r["target_qps"] for r in runs}):
        rr = [r for r in runs if r["target_qps"] == rate]
        by = {"aiohttp": [], "native": []}
        for r in rr:
            m = rung_records(r["records"])
            m.update({"achieved": r["achieved_qps"], "errors": r["errors"], "timeouts": r["timeouts"],
                      "samples": sorted(s[0] for s in r["samples"])})
            by[r["client"]].append(m)
        a, n = by["aiohttp"], by["native"]
        mean = lambda xs, k: float(np.mean([x[k] for x in xs]))
        g = {
            "serve_p50": within(mean(a, "serve_p50"), mean(n, "serve_p50"), 0.10),
            "serve_p99": within(mean(a, "serve_p99"), mean(n, "serve_p99"), 0.25, 10.0),
            "native_overhead_p99_le_5ms": max(x["overhead_p99"] for x in n) <= 5.0,
            "native_lag_p99_le_5ms": max(x["lag_p99"] for x in n) <= 5.0,
            "achieved_within_0.5pct": all(abs(x["achieved"] - rate) <= 0.005 * rate for x in a + n),
            "no_errors_or_timeouts": all(x["errors"] == 0 and x["timeouts"] == 0 for x in a + n),
            "same_sample_sets": len({tuple(x["samples"]) for x in a + n}) == 1,
        }
        if e2e_rate is not None and rate == e2e_rate:
            a_lag_ok = max(x["lag_p99"] for x in a) <= 5.0
            spread = max(abs(a[0]["lat_p99"] - a[1]["lat_p99"]), abs(n[0]["lat_p99"] - n[1]["lat_p99"])) if len(a) > 1 and len(n) > 1 else 0.0
            ap, np99 = mean(a, "lat_p99"), mean(n, "lat_p99")
            bound = max(0.25 * max(ap, np99), 10.0)
            g["e2e_aiohttp_lag_p99_le_5ms"] = a_lag_ok
            g["e2e_p50"] = within(mean(a, "lat_p50"), mean(n, "lat_p50"), 0.10)
            g["e2e_p99"] = abs(ap - np99) <= bound and abs(ap - np99) <= spread + bound
        out["rates"][str(int(rate))] = {"aiohttp": [{k: v for k, v in x.items() if k != "samples"} for x in a],
                                        "native": [{k: v for k, v in x.items() if k != "samples"} for x in n],
                                        "gates": g, "pass": all(g.values())}
    out["pass"] = all(v["pass"] for v in out["rates"].values()) and out["unloaded"]["differ"] == 0 \
        and not out["unloaded"]["error"]
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    e2e = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--e2e-rate=")), None)
    res = gate(args[0], e2e)
    if len(args) > 1:
        json.dump(res, open(args[1], "w"), indent=1)
    for rate, v in res["rates"].items():
        print(f"{rate} QPS: pass {v['pass']}  " + "  ".join(f"{k}={'ok' if ok else 'FAIL'}" for k, ok in v["gates"].items()))
    print(f"unloaded {res['unloaded']}; overall pass {res['pass']}")
    sys.exit(0 if res["pass"] else 1)
