"""S6-7: worker processes behind one port (serving/workers.py), with the real engine on the toy dataset."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("engine.api")
REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method, url, obj=None, timeout=30):
    data = None if obj is None else json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from referee import loader
    ds = loader.load_dataset("toy")
    d = tmp_path_factory.mktemp("serve")
    (d / "mapping.json").write_text(json.dumps(ds.mapping))
    ds.catalogue.to_parquet(d / "products.parquet")
    subprocess.run([sys.executable, "-m", "serving", "build", "--name", "toy", "--mapping", str(d / "mapping.json"),
                    "--source", str(d / "products.parquet"), "--data-dir", str(d), "--termination-log",
                    str(d / "build.json")], check=True, cwd=REPO, capture_output=True)
    return d


def _start(data_dir, workers, port, extra_env=None):
    env = {**os.environ, "S3_SERVING_CACHE": "0", "S3_SERVING_DRAIN_SECONDS": "0.2", **(extra_env or {})}
    proc = subprocess.Popen([sys.executable, "-m", "serving", "--data-dir", str(data_dir), "--port", str(port),
                             "--preload", "toy", "--workers", str(workers)],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    t_end = time.time() + 120
    while time.time() < t_end:
        try:
            if _http("GET", f"http://127.0.0.1:{port}/readyz", timeout=2)[0] == 200:
                return proc
        except OSError:
            pass
        assert proc.poll() is None, proc.stdout.read().decode()[-3000:]
        time.sleep(0.2)
    proc.kill()
    raise AssertionError("not ready")


def _hits(body: bytes) -> dict:
    return {k: v for k, v in json.loads(body).items() if k != "took_ms"}


def _stop(proc):
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(30)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_workers_share_the_port_and_answer_the_same(built):
    port = _free_port()
    single = _start(built, 1, port)
    url = f"http://127.0.0.1:{port}/indexes/toy/_search"
    reqs = [{"query": {"match": {"product_title": w}}, "size": 50} for w in ("lamp", "shoes", "black", "table")]
    try:
        want = [_hits(_http("POST", url, r)[1]) for r in reqs]
    finally:
        assert _stop(single) == 0
    port = _free_port()
    proc = _start(built, 3, port)
    url = f"http://127.0.0.1:{port}/indexes/toy/_search"
    try:
        t_end = time.time() + 120                      # workers 1 and 2 start accepting once their preload is done
        while json.loads(_http("GET", f"http://127.0.0.1:{port}/readyz")[1]).get("workers_ready") != 3:
            assert time.time() < t_end, "the workers did not all load"
            time.sleep(0.2)
        rz = json.loads(_http("GET", f"http://127.0.0.1:{port}/readyz")[1])
        assert rz["workers"] == 3 and rz["ready"] is True and rz["worker"] in (0, 1, 2)
        seen = set()

        def one(i):
            st, body, h = _http("POST", url, reqs[i % len(reqs)])
            return i, st, body, h.get("X-S3-Worker")

        with ThreadPoolExecutor(24) as ex:              # a new connection per request: accepts spread over workers
            for i, st, body, w in ex.map(one, range(240)):
                assert st == 200
                assert _hits(body) == want[i % len(reqs)]
                seen.add(w)
        assert len(seen) >= 2, seen
        st, body, _ = _http("POST", f"http://127.0.0.1:{port}/indexes/toy/_reindex", {})
        assert st == 405 and json.loads(body)["error"]["type"] == "method_not_allowed"
        st, body, _ = _http("POST", f"http://127.0.0.1:{port}/indexes/toy/_docs", [])
        assert st == 405
        assert _http("GET", f"http://127.0.0.1:{port}/indexes/toy/_mapping")[0] == 200
    finally:
        assert _stop(proc) == 0
    out = proc.stdout.read().decode()
    assert out.count("preloaded") >= 3, out[-3000:]


def _children(pid: int) -> list[int]:
    """Child pids, with the standard tools only (pgrep on Linux and macOS; no psutil in CI)."""
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def test_a_dead_worker_is_started_again(built):
    port = _free_port()
    proc = _start(built, 2, port)
    try:
        kids = _children(proc.pid)
        assert len(kids) == 2
        os.kill(kids[1], signal.SIGKILL)
        t_end = time.time() + 60
        while time.time() < t_end:
            now = _children(proc.pid)
            if len(now) == 2 and kids[1] not in now:
                break
            time.sleep(0.2)
        else:
            raise AssertionError("the worker was not started again")
        assert _http("GET", f"http://127.0.0.1:{port}/healthz")[0] == 200
    finally:
        assert _stop(proc) == 0


def test_columnar_and_dict_serving_send_the_same_bytes(built):
    """S3_SERVING_COLUMNS=1 (columnar hits, serving/encode.py) and =0 (the dict form, json.dumps): the same bytes but
    for took_ms, at depth, with sorts, aggregations, _source and _msearch."""
    reqs = [{"query": {"match": {"product_title": "lamp"}}, "size": 2000},
            {"query": {"match_all": {}}, "sort": [{"price": "asc"}], "size": 40, "aggs": {"c": {"terms": {"field": "category"}}}},
            {"query": {"match_all": {}}, "size": 5, "_source": True},
            {"query": {"match": {"product_title": "black shoes"}}, "size": 30, "collapse": {"field": "product_brand.keyword"}}]
    got = {}
    for columns in ("0", "1"):
        port = _free_port()
        proc = _start(built, 1, port, {"S3_SERVING_COLUMNS": columns})
        try:
            base = f"http://127.0.0.1:{port}/indexes/toy"
            got[columns] = [_http("POST", f"{base}/_search", r) for r in reqs] + [_http("POST", f"{base}/_msearch", reqs)]
        finally:
            assert _stop(proc) == 0
    for (s0, b0, _), (s1, b1, _) in zip(got["0"], got["1"]):
        assert s0 == s1
        j0, j1 = json.loads(b0), json.loads(b1)
        if isinstance(j0, list):
            j0, j1 = [{k: v for k, v in r.items() if k != "took_ms"} for r in j0], [{k: v for k, v in r.items() if k != "took_ms"} for r in j1]
            assert j0 == j1
        else:
            assert s0 == 200, b0
            t0, t1 = j0.pop("took_ms"), j1.pop("took_ms")
            assert j0 == j1
            assert b0.replace(json.dumps(t0, separators=(",", ":")).encode(), b"") == \
                b1.replace(json.dumps(t1, separators=(",", ":")).encode(), b"")
