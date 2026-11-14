"""S6-10s: fronts and GPU processes (serving/front.py, serving/gpu.py) answer as one process does, with the real engine
on the toy dataset."""
import json
import os
from pathlib import Path
import urllib.request
import stat
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("engine.api")
from .test_workers import _free_port, _hits, _http, _stop, built  # noqa: E402,F401
from .test_workers import REPO  # noqa: E402,F401
import subprocess  # noqa: E402
import sys  # noqa: E402

REQS = [{"query": {"match": {"product_title": "lamp"}}, "size": 2000},
        {"query": {"match_all": {}}, "sort": [{"price": "asc"}], "size": 40, "aggs": {"c": {"terms": {"field": "category"}}}},
        {"query": {"bool": {"must": [{"match": {"product_title": "black"}}], "filter": [{"range": {"price": {"lt": 50}}}]}}, "size": 25},
        {"query": {"match_all": {}}, "size": 5, "_source": True},
        {"query": {"match": {"product_title": "black shoes"}}, "size": 30, "collapse": {"field": "product_brand.keyword"}},
        {"query": {"match": {"product_title": "lamp"}}, "size": 5, "highlight": {"fields": {"product_title": {}}}},
        {"query": {"match": {"product_title": "lamp"}}, "size": 3, "explain": True},
        {"query": {"match": {"no_such_field": "lamp"}}},
        {"query": {"wildcard": {"product_title": "la*"}}}]


def _start(data_dir, port, args, extra_env=None):
    env = {**os.environ, "S3_SERVING_CACHE": "0", "S3_SERVING_DRAIN_SECONDS": "0.2", **(extra_env or {})}
    proc = subprocess.Popen([sys.executable, "-m", "serving", "--data-dir", str(data_dir), "--port", str(port),
                             "--preload", "toy", *args], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
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


def _answers(port):
    base = f"http://127.0.0.1:{port}/indexes/toy"
    out = [_http("POST", f"{base}/_search", r)[:2] for r in REQS]
    out.append(_http("POST", f"{base}/_msearch", REQS[:4])[:2])
    out.append(_http("POST", f"{base}/_msearch", REQS[:2] + REQS[7:8])[:2])
    out.append(_http("GET", f"{base}/_mapping")[:2])
    return out


def _same(a, b):
    for (s0, b0), (s1, b1) in zip(a, b):
        assert s0 == s1, (b0[:300], b1[:300])
        j0, j1 = json.loads(b0), json.loads(b1)
        if isinstance(j0, list):
            j0 = [{k: v for k, v in r.items() if k != "took_ms"} for r in j0]
            j1 = [{k: v for k, v in r.items() if k != "took_ms"} for r in j1]
            assert j0 == j1
        elif "took_ms" in j0:
            t0, t1 = j0.pop("took_ms"), j1.pop("took_ms")
            assert j0 == j1
            assert b0.replace(json.dumps(t0, separators=(",", ":")).encode(), b"") == \
                b1.replace(json.dumps(t1, separators=(",", ":")).encode(), b"")
        else:
            assert b0 == b1


@pytest.fixture(scope="module")
def single(built):
    port = _free_port()
    p = _start(built, port, [])
    try:
        return _answers(port)
    finally:
        assert _stop(p) == 0


def test_fronts_and_one_gpu_process_answer_as_one_process(built, single, tmp_path):
    import shutil
    data = tmp_path / "data"                                # this test writes: its own copy of the index and store
    shutil.copytree(built, data)
    port = _free_port()
    import tempfile                                          # a short path: a Unix socket path is limited to ~104 bytes
    state = tempfile.mkdtemp(prefix="s3t-")                  # the supervisor's state directory, named so we can look in it
    p = _start(data, port, ["--fronts", "2", "--gpu-processes", "1"], {"S3_SERVING_STATE_DIR": str(state)})
    try:
        rz = json.loads(_http("GET", f"http://127.0.0.1:{port}/readyz")[1])
        assert rz["ready"] and rz["fronts"] == 2 and rz["gpu_processes"] == 1 and rz["gpu_ready"] == 1
        _same(single, _answers(port))
        assert [s for s, _ in single][7:9] == [400, 400]
        url = f"http://127.0.0.1:{port}/indexes/toy/_search"
        want = {i: _hits(_http("POST", url, r)[1]) for i, r in enumerate(REQS[:3])}

        def one(i):                                       # under concurrent load, batches form: the same results
            st, body, h = _http("POST", url, REQS[i % 3])
            return i, st, body, h.get("X-S3-Worker")

        seen = set()
        with ThreadPoolExecutor(32) as ex:
            for i, st, body, w in ex.map(one, range(300)):
                assert st == 200 and _hits(body) == want[i % 3]
                seen.add(w)
        assert seen == {"0", "1"}
        # a write is applied by the GPU process before the front answers: the next search sees it
        first = json.loads(_http("POST", url, {"query": {"match_all": {}}, "sort": [{"price": "desc"}], "size": 1})[1])
        doc = first["hits"]["hits"][-1]["_id"]
        st, body, _ = _http("POST", f"http://127.0.0.1:{port}/indexes/toy/_docs",
                            [{"op": "patch", "_id": doc, "fields": {"price": 0.01}}])
        assert st == 200, body
        low = json.loads(_http("POST", url, {"query": {"match_all": {}}, "sort": [{"price": "asc"}], "size": 1})[1])
        assert low["hits"]["hits"][0]["_id"] == doc
        assert "versions-0.json" in os.listdir(state)
        sock = os.path.join(state, "gpu-0.sock")
        assert stat.S_IMODE(os.stat(sock).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(state).st_mode) == 0o700
    finally:
        assert _stop(p) == 0




def test_two_gpu_processes_are_read_only(built, single):
    port = _free_port()
    p = _start(built, port, ["--fronts", "2", "--gpu-processes", "2"])
    try:
        rz = json.loads(_http("GET", f"http://127.0.0.1:{port}/readyz")[1])
        assert rz["gpu_processes"] == 2 and rz["gpu_ready"] == 2
        _same(single, _answers(port))
        st, body, _ = _http("POST", f"http://127.0.0.1:{port}/indexes/toy/_docs", [])
        assert st == 405 and json.loads(body)["error"]["type"] == "method_not_allowed"
        url = f"http://127.0.0.1:{port}/indexes/toy/_search"           # which process answered, and its index version
        versions = set()
        for j in (0, 1):
            req = urllib.request.Request(url, data=json.dumps(REQS[0]).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "X-S3-Route-Gpu": str(j)})
            with urllib.request.urlopen(req, timeout=30) as r:
                assert r.headers["X-S3-Gpu-Process"] == str(j)
                versions.add(r.headers["X-S3-Index-Version"])
                assert _hits(r.read()) == _hits(single[0][1])
        assert len(versions) == 1 and next(iter(versions)).startswith("seg-")   # both hold the same index state
    finally:
        assert _stop(p) == 0


BAD_BODIES = [b"", b"   ", b"{not json", b'{"query": '] # refused before the engine: 400, as the Python front


def _raw_post(url, body):
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.mark.parametrize("hostpath", ["0", "stub", "stub-prepared"])
def test_binary_protocol_gives_the_same_bodies(built, single, hostpath):
    """S6-22s: on a connection opened with S3B1, a GPU process answers in serving/binproto.py's frames, and the
    bodies decoded the way a native front must encode them equal one process's (but for took_ms); the same with the
    S3_HOSTPATH seam on its Python stub (serving/hostpath.py). A body that is not JSON gets the Python front's 400 on
    the same connection, which stays open."""
    import socket as _socket
    import struct
    import tempfile
    from serving import binproto
    state = tempfile.mkdtemp(prefix="s3b-")
    port = _free_port()
    p = _start(built, port, ["--fronts", "1", "--gpu-processes", "1"],
               {"S3_SERVING_STATE_DIR": state, "S3_HOSTPATH": hostpath})
    sp = _free_port()
    ps = _start(built, sp, [])                              # one process: what the Python front says to a bad body
    try:
        want_bad = [_raw_post(f"http://127.0.0.1:{sp}/indexes/toy/_search", b) for b in BAD_BODIES]
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(os.path.join(state, "gpu-0.sock"))
        s.sendall(binproto.MAGIC)

        def ask(i, req):
            body = req if isinstance(req, bytes) else json.dumps(req).encode()
            name = b"toy"
            s.sendall(binproto.frame(struct.pack("<BQ", 1, i) + struct.pack("<H", len(name)) + name
                                     + struct.pack("<I", len(body)) + body))
            n = struct.unpack("<I", _recv(s, 4))[0]
            return binproto.decode_reply(_recv(s, n))

        kinds = set()
        for i, req in enumerate(REQS):
            id_, st, body, meta = ask(i, req)
            assert id_ == i
            s0, b0 = single[i]
            assert st == s0, (req, body[:200])
            j0, j1 = json.loads(b0), json.loads(body)
            j0.pop("took_ms", None); j1.pop("took_ms", None)
            assert j0 == j1, req
            if st == 200:
                t0, t1 = json.loads(b0)["took_ms"], json.loads(body)["took_ms"]
                assert b0.replace(json.dumps(t0, separators=(",", ":")).encode(), b"") == \
                    body.replace(json.dumps(t1, separators=(",", ":")).encode(), b"")
            kinds.add("pre" in meta)
        assert kinds == {True, False}                     # some columnar (kind 1), some whole (kind 2)
        for k, (bad, (ws, wb)) in enumerate(zip(BAD_BODIES, want_bad)):
            id_, st, body, _ = ask(1000 + k, bad)
            assert (id_, st) == (1000 + k, ws) and json.loads(body) == json.loads(wb), (bad, body, wb)
        id_, st, _, _ = ask(2000, REQS[0])                 # the connection is still open
        assert (id_, st) == (2000, 200)
        mixed = list(REQS) + BAD_BODIES                     # pipelined, so they batch (explain, highlight alone)
        for k, req in enumerate(mixed):
            body = req if isinstance(req, bytes) else json.dumps(req).encode()
            s.sendall(binproto.frame(struct.pack("<BQ", 1, 3000 + k) + struct.pack("<H", 3) + b"toy"
                                     + struct.pack("<I", len(body)) + body))
        got = {}
        for _ in mixed:
            n = struct.unpack("<I", _recv(s, 4))[0]
            id_, st, body, _ = binproto.decode_reply(_recv(s, n))
            got[id_ - 3000] = (st, json.loads(body))
        for k, req in enumerate(mixed):
            want_st, want_body = (single[k] if k < len(REQS) else want_bad[k - len(REQS)])
            j0 = json.loads(want_body)
            j0.pop("took_ms", None); got[k][1].pop("took_ms", None)
            assert got[k] == (want_st, j0), (k, req)
        s.close()
    finally:
        assert _stop(p) == 0
        assert _stop(ps) == 0


def _recv(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        assert chunk, "connection closed"
        buf += chunk
    return buf


def _native_bin():
    here = Path(__file__).resolve().parents[1] / "native_front" / "target" / "release" / "s3-native-front"
    return os.environ.get("S3_NATIVE_FRONT_BIN") or (str(here) if here.exists() else None)


def test_native_front_answers_as_one_process(built, single):
    """S6-22s: the Rust front (serving/native_front) sends the same bodies as one process (skipped unless built)."""
    if not _native_bin():
        pytest.skip("serving/native_front is not built")
    port = _free_port()
    p = _start(built, port, ["--fronts", "2", "--gpu-processes", "1"],
               {"S3_SERVING_FRONT_SERVER": "native", "S3_NATIVE_FRONT_BIN": _native_bin()})
    try:
        _same(single, _answers(port))
        url = f"http://127.0.0.1:{port}/indexes/toy/_search"
        want = _hits(_http("POST", url, REQS[0])[1])
        with ThreadPoolExecutor(16) as ex:
            for st, body, _ in ex.map(lambda _: _http("POST", url, REQS[0]), range(200)):
                assert st == 200 and _hits(body) == want
    finally:
        assert _stop(p) == 0
