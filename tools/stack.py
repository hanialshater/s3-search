"""A real serving replica (``python -m serving`` in a subprocess) over a slice of ESCI-sample, for the demo's tests and
for measuring the page. No fakes: this is the engine and the platform engineer's server.

    python -m tools.stack [--products N] [--fields] [--port P]      # prints the base URL; Ctrl-C stops it

``fields=True`` adds synthetic ``price`` (float), ``stock`` and ``category`` columns, so the page's numeric controls
(sort, range, histogram) can be exercised; ESCI-sample as served has none of them, and the page must work without.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ["apparel", "home", "electronics", "toys", "sports", "beauty"]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def catalogue(n: int | None = None, fields: bool = False):
    """(catalogue DataFrame, mapping) from the named dataset esci-sample, the first ``n`` products, optionally with the
    synthetic numeric and category columns."""
    import numpy as np
    from referee import loader
    ds = loader.load_dataset("esci-sample")
    cat, mapping = ds.catalogue.copy(), copy.deepcopy(ds.mapping)
    if n:
        # keep products that have judgments first, so queries find something, then fill with the rest
        cat = cat.iloc[:n].reset_index(drop=True)
    if fields:
        rng = np.random.default_rng(7)
        cat["price"] = np.round(rng.uniform(1, 500, len(cat)), 2)
        cat["stock"] = (rng.integers(0, 4, len(cat)) * rng.integers(0, 40, len(cat))).astype(np.int32)
        cat["category"] = rng.choice(CATEGORIES, len(cat))
        mapping["properties"].update({"price": {"type": "float"}, "stock": {"type": "integer"},
                                      "category": {"type": "keyword"}})
        mapping["_meta"]["engine"]["hot"] = ["price", "stock"]
    return cat, mapping


class Stack:
    def __init__(self, n_products: int | None = 4000, fields: bool = False, name: str = "esci-sample",
                 port: int | None = None, cores: str | None = None):
        self.name, self.port = name, port or free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.work = Path(tempfile.mkdtemp(prefix="tools-stack-"))
        cat, mapping = catalogue(n_products, fields)
        cat.to_parquet(self.work / "catalogue.parquet", index=False)
        cmd = [sys.executable, "-m", "serving", "--data-dir", str(self.work / "serving"), "--port", str(self.port),
               "--device", "cpu"]
        if cores:
            cmd = ["taskset", "-c", cores] + cmd
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "S3_SEARCH_DEVICE": "cpu"},
                                     stdout=subprocess.DEVNULL, stderr=open(self.work / "serving.log", "w"))
        try:
            for _ in range(300):
                try:
                    self.call("GET", "/healthz")
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError("serving did not start: " + (self.work / "serving.log").read_text()[-500:])
            self.call("PUT", f"/indexes/{name}", {"mapping": mapping, "source": {"path": str(self.work / "catalogue.parquet")}})
        except BaseException:
            self.close()
            raise

    def call(self, method: str, path: str, body=None, timeout: float = 600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")

    def search(self, request: dict) -> dict:
        return self.call("POST", f"/indexes/{self.name}/_search", request)

    def close(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.stack", description=__doc__.split("\n\n")[0])
    ap.add_argument("--products", type=int, default=None, help="first N products (default: all of esci-sample)")
    ap.add_argument("--fields", action="store_true", help="add synthetic price, stock, category")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    with Stack(args.products, args.fields, port=args.port) as st:
        print(st.base, flush=True)
        try:
            st.proc.wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
