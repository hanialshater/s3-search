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


# The same kinds of synthetic data as the steward's s9-v1 suite (referee/conformance_s9.py): 70% of products around
# 20 city centres (Gaussian, about 30 km), 30% uniform over inhabited latitudes; 0 to 5 nested offers per product from
# 40 sellers. Deterministic: fixed seed. Coordinates are multiples of Lucene's quantum so they are stored losslessly.
CITIES = [(40.71, -74.01), (34.05, -118.24), (41.88, -87.63), (29.76, -95.37), (47.61, -122.33), (25.76, -80.19),
          (51.51, -0.13), (48.86, 2.35), (52.52, 13.40), (40.42, -3.70), (41.90, 12.50), (35.68, 139.69),
          (37.57, 126.98), (1.35, 103.82), (-33.87, 151.21), (-23.55, -46.63), (19.43, -99.13), (28.61, 77.21),
          (30.04, 31.24), (-26.20, 28.05)]
SELLERS = [f"seller-{i:02d}" for i in range(40)]
GEO_SEED = 20261110


def _locations_and_offers(n: int):
    import numpy as np
    lat_q, lon_q = 180.0 / 2 ** 32, 360.0 / 2 ** 32
    rng = np.random.default_rng(GEO_SEED)
    locs, offers = [], []
    for _ in range(n):
        if rng.random() < 0.7:
            lat0, lon0 = CITIES[int(rng.integers(len(CITIES)))]
            lat, lon = lat0 + rng.normal(0, 0.27), lon0 + rng.normal(0, 0.35)
        else:
            lat, lon = rng.uniform(-50, 65), rng.uniform(-179.9, 179.9)
        lat, lon = float(np.clip(lat, -89.9, 89.9)), float(np.clip(lon, -179.99, 179.99))
        locs.append({"lat": float(np.floor(lat / lat_q) * lat_q), "lon": float(np.floor(lon / lon_q) * lon_q)})
        offers.append([{"seller": SELLERS[int(rng.integers(len(SELLERS)))], "price": round(float(rng.uniform(1, 500)), 2),
                        "stock": int(rng.integers(0, 50))} for _ in range(int(rng.integers(0, 6)))])
    return locs, offers


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def catalogue(n: int | None = None, fields: bool = False, geo: bool = False):
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
    if geo:  # synthetic locations and nested offers (S9-7); serving's intake must accept geo_point and nested
        cat["location"], cat["offers"] = _locations_and_offers(len(cat))
        mapping["properties"]["location"] = {"type": "geo_point"}
        mapping["properties"]["offers"] = {"type": "nested", "properties": {
            "seller": {"type": "keyword"}, "price": {"type": "float"}, "stock": {"type": "integer"}}}
    return cat, mapping


class Stack:
    def __init__(self, n_products: int | None = 4000, fields: bool = False, geo: bool = False, name: str = "esci-sample",
                 port: int | None = None, cores: str | None = None):
        self.name = name
        self.work = Path(tempfile.mkdtemp(prefix="tools-stack-"))
        cat, mapping = catalogue(n_products, fields, geo)
        cat.to_parquet(self.work / "catalogue.parquet", index=False)
        self.proc = None
        try:
            self._start(port, cores)
            self.call("PUT", f"/indexes/{name}", {"mapping": mapping, "source": {"path": str(self.work / "catalogue.parquet")}})
        except BaseException:
            self.close()
            raise

    def _start(self, port, cores, attempts: int = 5, deadline_s: float = 180.0) -> None:
        """Start serving and wait until it answers. A free port found by binding and closing can be taken by another
        process before serving binds it (or by a neighbour's test): then serving exits at once, and the wait notices
        that (it watches the process, not a fixed count of tries) and starts again on a new port."""
        last = ""
        for _ in range(1 if port else attempts):
            self.port = port or free_port()
            self.base = f"http://127.0.0.1:{self.port}"
            cmd = [sys.executable, "-m", "serving", "--data-dir", str(self.work / "serving"), "--port", str(self.port),
                   "--device", "cpu"]
            if cores:
                cmd = ["taskset", "-c", cores] + cmd
            log = self.work / "serving.log"
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, "S3_SEARCH_DEVICE": "cpu"},
                                         stdout=subprocess.DEVNULL, stderr=open(log, "w"))
            t0 = time.time()
            while time.time() - t0 < deadline_s:
                if self.proc.poll() is not None:  # exited: most likely the port was taken
                    last = log.read_text()[-500:]
                    break
                try:
                    self.call("GET", "/healthz", timeout=5)
                    if self.proc.poll() is None:  # the answer came from our process, not a stranger on the port
                        return
                except OSError:
                    pass
                time.sleep(0.1)
            else:
                self.proc.kill()
                raise RuntimeError(f"serving did not answer within {deadline_s:.0f} s: " + log.read_text()[-500:])
        raise RuntimeError("serving did not start: " + last)

    def call(self, method: str, path: str, body=None, timeout: float = 600):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"null")

    def search(self, request: dict) -> dict:
        return self.call("POST", f"/indexes/{self.name}/_search", request)

    def close(self) -> None:
        if self.proc is None:
            return
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
    ap.add_argument("--geo", action="store_true", help="add synthetic location (geo_point) and offers (nested)")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    with Stack(args.products, args.fields, args.geo, port=args.port) as st:
        print(st.base, flush=True)
        try:
            st.proc.wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
