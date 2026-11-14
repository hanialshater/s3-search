"""Front-only CPU per request (S6-21s, S6-22s): a fake GPU process answers a canned 2,000-hit columnar response at once;
one front serves HTTP (FRONT=stdlib, granian or native); two aiohttp client processes drive it. Development only.

    FRONT=native S3_NATIVE_FRONT_BIN=... python colab/front_bench.py REPO CONNECTIONS_PER_CLIENT [THREADS]
"""
import asyncio, json, multiprocessing as mp, os, sys, tempfile, time
REPO = sys.argv[1]; sys.path.insert(0, REPO)
import numpy as np

def canned():
    h = 2000; rng = np.random.default_rng(0); sc = np.sort(rng.random(h).astype(np.float32) * 30)[::-1]
    ids = np.array([f"B0{rng.integers(10**8, 10**9)}".encode() for _ in range(h)], dtype="S11")
    return {"hits": {"total": 5000, "columns": {"_id": ids, "_score": sc, "sort": [
        {"values": sc.astype(np.float64), "missing": None}, {"values": rng.integers(0, 10**6, h), "missing": None}]}},
        "took_ms": {"parse": 0.1, "total": 3.0}}

def gpu(sock, state):
    from serving.gpu import GpuServer
    from serving.metrics import Metrics
    r = canned()
    class Ix:
        def search_batch(self, reqs, hits="dicts", ids="str", errors="raise"): return [r] * len(reqs)
    class MI:
        name = "x"; ready = True; index = Ix(); segment_id = "s"; epoch = 0; metrics = Metrics()
    class Mgr:
        def get(self, n): return MI()
        def names(self): return ["x"]
    class App:
        manager = Mgr(); metrics = Metrics()
    GpuServer(App(), sock, 0, state).serve_forever()

def front(sock, port, state, q):
    from serving.app import make_server
    from serving.config import ServingConfig
    from serving.front import FrontApp
    cfg = ServingConfig(port=port, cache_size=0); cfg.worker_state = state; cfg.fronts = 1
    srv = make_server(FrontApp(cfg, [sock]), "127.0.0.1", port)
    q.put(srv.server_address[1]); srv.serve_forever()

def client(port, conns, secs, out):
    import aiohttp
    body = json.dumps({"query": {"match": {"t": "lamp"}}, "size": 2000}).encode()
    async def run():
        n = 0
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
            t_end = time.time() + secs
            async def w():
                nonlocal n
                while time.time() < t_end:
                    async with s.post(f"http://127.0.0.1:{port}/indexes/x/_search", data=body) as r:
                        b = await r.read()
                        if r.status != 200 and n < 3: print("status", r.status, b[:300], flush=True)
                        n += 1
            await asyncio.gather(*[w() for _ in range(conns)])
        out.put(n)
    asyncio.run(run())

if __name__ == "__main__":
    import psutil
    state = tempfile.mkdtemp(); sock = os.path.join(state, "g.sock")
    from serving.gpu import LoadTable; LoadTable(state, create=True)
    g = mp.Process(target=gpu, args=(sock, state), daemon=True); g.start(); time.sleep(0.5)
    if os.environ.get("FRONT") == "native":
        import socket as so, subprocess
        t = so.socket(); t.bind(("127.0.0.1", 0)); port = t.getsockname()[1]; t.close()
        thr = sys.argv[3] if len(sys.argv) > 3 else "1"
        f = subprocess.Popen([os.environ["S3_NATIVE_FRONT_BIN"], "--host", "127.0.0.1", "--port", str(port), "--gpu-sockets", sock, "--threads", thr], start_new_session=True)
        time.sleep(2)
    elif os.environ.get("FRONT") == "granian":
        import socket as so, subprocess
        t = so.socket(); t.bind(("127.0.0.1", 0)); port = t.getsockname()[1]; t.close()
        env = {**os.environ, "S3_SERVING_GPU_SOCKETS": sock, "S3_SERVING_WORKER_STATE": state, "S3_SERVING_CACHE": "0", "PYTHONPATH": REPO}
        f = subprocess.Popen([sys.executable, "-m", "granian", "--interface", "wsgi", "--workers", "1", "--blocking-threads", "32", "--host", "127.0.0.1", "--port", str(port), "--no-ws", "--log-level", "warning", "serving.front_granian:app"], env=env, start_new_session=True)
        time.sleep(3)
    else:
        q = mp.Queue(); f = mp.Process(target=front, args=(sock, 0, state, q), daemon=True); f.start(); port = q.get()
    out = mp.Queue(); secs = 10
    pf = psutil.Process(f.pid)
    tree = lambda: sum(sum(x.cpu_times()[:2]) for x in [pf] + pf.children(recursive=True))
    c0 = tree()
    cs = [mp.Process(target=client, args=(port, int(sys.argv[2]), secs, out)) for _ in range(2)]
    t0 = time.time(); [c.start() for c in cs]; n = sum(out.get() for _ in cs); [c.join() for c in cs]
    c1 = tree()
    import json as _j
    print(_j.dumps({"front": os.environ.get("FRONT", "stdlib"), "req_per_s": round(n / secs), "front_cores": round((c1 - c0) / (time.time() - t0), 2),
                    "front_cpu_ms_per_req": round((c1 - c0) / n * 1e3, 3)}), flush=True)
    if os.environ.get("FRONT") in ("granian", "native"):
        import signal; os.killpg(f.pid, signal.SIGKILL)
    else:
        f.kill()
    g.kill()
