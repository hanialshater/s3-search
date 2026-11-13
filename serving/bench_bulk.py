"""Measure ``POST _bulk`` on a dataset over HTTP, with the server pinned to 2 cores (S8-6: all 48,639 ESCI-sample
products in at most 15 s).

    S3_SEARCH_SAMPLE=/path/to/data/esci-sample python -m serving.bench_bulk [--dataset esci-sample] [--cores 0,1]
                                                                            [--seed 200] [--chunk 2000] [--refresh]

Steps: build the NDJSON body in a file (not timed), start ``python -m serving`` under ``taskset``, create an index from a
small seed, then send the rest of the catalogue in one ``_bulk`` request and time it (request sent to response read).
``--refresh`` then times a ``POST _refresh`` (the full rebuild, until delta segments land). Prints one JSON line.
"""
from __future__ import annotations

import argparse
import http.client
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esci-sample")
    ap.add_argument("--cores", default="0,1")
    ap.add_argument("--seed", type=int, default=200, help="documents used to create the index (not part of the timing)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--refresh-param", default="", help="false|true|wait_for: the ?refresh= of the bulk request")
    ap.add_argument("--interval", default="-1", help="index.refresh_interval of the server (-1 off)")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    from referee import loader
    ds = loader.load_dataset(a.dataset)
    cat = ds.catalogue
    id_field = (ds.mapping.get("_meta", {}).get("engine", {}) or {}).get("id_field", "product_id")
    cols = [c for c in cat.columns if c in ds.mapping["properties"] or c == id_field]
    recs = cat[cols].to_dict("records")
    clean = lambda r: {k: v for k, v in r.items() if v is not None and v == v}  # noqa: E731  (drop None and NaN)
    work = Path(tempfile.mkdtemp(prefix="s3-bulk-"))
    seed_path, body_path = work / "seed.jsonl", work / "body.ndjson"
    seed_path.write_text("\n".join(json.dumps(clean(r), default=str) for r in recs[:a.seed]) + "\n")
    rest = recs[a.seed:]
    with open(body_path, "w") as f:
        for r in rest:
            f.write(json.dumps({"index": {"_id": str(r[id_field])}}) + "\n")
            f.write(json.dumps(clean(r), default=str) + "\n")
    size = body_path.stat().st_size
    port = free_port()
    cmd = ["taskset", "-c", a.cores, sys.executable, "-m", "serving", "--data-dir", str(work / "data"), "--port", str(port)]
    srv = subprocess.Popen(cmd, env={**__import__("os").environ, "S3_SEARCH_DEVICE": "cpu", "S3_SERVING_BULK_CHUNK": str(a.chunk),
                                     "S3_SERVING_REFRESH_INTERVAL": a.interval},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                http.client.HTTPConnection("127.0.0.1", port, timeout=1).request("GET", "/healthz")
                break
            except OSError:
                time.sleep(0.1)
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=3600)
        t = time.perf_counter()
        c.request("PUT", "/indexes/bench", json.dumps({"mapping": ds.mapping, "source": {"path": str(seed_path)}}))
        r = c.getresponse()
        create = r.read()
        assert r.status == 200, create
        t_create = time.perf_counter() - t
        with open(body_path, "rb") as f:
            t = time.perf_counter()
            c.request("POST", "/indexes/bench/_bulk" + (f"?refresh={a.refresh_param}" if a.refresh_param else ""), f, {"Content-Type": "application/x-ndjson", "Content-Length": str(size)})
            r = c.getresponse()
            raw = r.read()
            t_bulk = time.perf_counter() - t
        out = json.loads(raw)
        items = out["items"]
        by = {}
        for it in items:
            (op, b), = it.items()
            k = f"{op}:{b['status']}"
            by[k] = by.get(k, 0) + 1
        res = {"dataset": a.dataset, "docs_in_body": len(rest), "seed": a.seed, "http_status": r.status, "items": len(items),
               "errors": out["errors"], "by_status": by, "bulk_seconds": round(t_bulk, 2),
               "docs_per_second": round(len(items) / t_bulk), "seed_create_seconds": round(t_create, 2),
               "body_mb": round(size / 1e6, 1), "cores": a.cores, "chunk": a.chunk}
        c.request("GET", "/metrics")
        m = c.getresponse().read().decode()
        res["lag_after_bulk"] = next((float(x.split()[-1]) for x in m.splitlines() if x.startswith("s3_index_lag{")), None)
        res["forced_refresh"] = out.get("forced_refresh")
        res["swaps"] = {x.split("{")[1].split("}")[0]: float(x.split()[-1]) for x in m.splitlines() if x.startswith("s3_swaps_total{")}
        res["refresh_ms_count"] = next((float(x.split()[-1]) for x in m.splitlines() if x.startswith("s3_refresh_ms_count")), 0)
        c.request("GET", "/indexes/bench/versions")
        res["segments_on_disk"] = len(json.loads(c.getresponse().read()))
        c.request("GET", "/indexes/bench")
        res["index_info"] = {k: v for k, v in json.loads(c.getresponse().read()).items() if k in ("version", "n_docs", "pending_upserts")}
        last = rest[-1]
        q = {"query": {"match": {"product_title": str(last.get("product_title", ""))[:60]}}, "size": 5}
        c.request("POST", "/indexes/bench/_search", json.dumps(q))
        hits = json.loads(c.getresponse().read())["hits"]["hits"]
        res["last_doc_visible"] = any(h["_id"] == str(last[id_field]) for h in hits)
        if a.refresh:
            t = time.perf_counter()
            c.request("POST", "/indexes/bench/_refresh", b"")
            rr = c.getresponse()
            rr.read()
            res["refresh_seconds"] = round(time.perf_counter() - t, 2)
            res["refresh_status"] = rr.status
        print(json.dumps(res))
        return 0 if r.status == 200 else 1
    finally:
        srv.terminate()
        srv.wait(10)
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
