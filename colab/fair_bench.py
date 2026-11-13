#!/usr/bin/env -S colab run --gpu A100 --timeout 1200
"""Matched load test, the engine against OpenSearch on one Colab GPU machine: speed only, development only.

    colab run --gpu A100 --timeout 1200 colab/fair_bench.py [branch-or-commit] [--qps=250,500,1000,2000,4000] [--k=2000]
        [--duration=60] [--engines=engine,opensearch] [--workers=1] [--columns=1] [--mps=0]
        [--fronts=0 --gpu-processes=1] [--gpu-memory-fraction=0] [--log=/content/serving.log]

--columns defaults to 1 since S6-7 (s3-search 67c1cd0): serving asks the engine for columnar hits. Runs before that
commit were the dict form (--columns=0 now). --workers=N runs the engine's server as N worker processes behind the one port (serving/workers.py, S6-7; each holds
its own copy of the index on the GPU); --columns=0 makes serving ask the engine for the dict form of the hits; --mps=1 starts the CUDA MPS daemon before
the server, so the workers' kernels run on the GPU together instead of taking turns.

What is matched, for each engine in turn (never both at once):
- the corpus: every ESCI-US product (1,215,854), through referee/loader.py;
- the queries: the ESCI-US test queries' texts, replayed in the same order;
- the depth: every request asks for the top --k hits (default 2,000), ids and scores only (no _source);
- the interface: HTTP on localhost. The engine behind `python -m serving` (its batching scheduler, query cache off);
  OpenSearch 2.19.1 single node, as the prototype ran it;
- each engine's own tuned lexical request: the engine's tuned BM25F (referee/requests/bm25f-tuned*.json),
  OpenSearch's tuned cross_fields (opensearch-cross-fields-tuned.json, `english` analyzer, as the prototype);
- the load: open loop at each --qps for --duration seconds, climbing until an engine cannot sustain a rate
  (under 90% of it achieved, or over 1% timeouts or errors), the same rule for both. Request i is due at t0 + i/qps whatever happened
  before it, and its latency counts from that due time, so a slow engine queues instead of slowing the clock.
- the CPUs: the load generator (4 processes) is pinned to 4 cores; the engine under test gets the others.
  OpenSearch runs on those CPUs; the engine on the GPU plus those CPUs.

Reports, per engine and rate: achieved QPS, latency p50/p95/p99/max from the due time, errors, timeouts, hits per
response (sampled), GPU utilisation and the server's CPU use (sampled every 0.5 s: which one was the limit),
and the client's send lag (if it falls behind, the client was the limit and the run says so).
Quality at the same depth is the referee's: `referee full --k=2000` for each engine. Writes /content/fair-bench.json.
"""
from __future__ import annotations

import asyncio
import copy
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
OS_VERSION = "2.19.1"
OS_HOME = Path(f"/content/opensearch-{OS_VERSION}")
SERVE_DIR = Path("/content/serve")
CLIENT_PROCS = 4
TIMEOUT_S = 10.0


def opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
REF = ARGS[0] if ARGS else "main"
RATES = [float(x) for x in opt("qps", "250,500,1000,2000,4000").split(",")]
K = int(opt("k", "2000"))
DURATION = float(opt("duration", "60"))
ENGINES = opt("engines", "engine,opensearch").split(",")
WORKERS = int(opt("workers", "1"))
COLUMNS = opt("columns", "1")
MPS = opt("mps", "0") == "1"
FRONTS = int(opt("fronts", "0"))                             # S6-10s: front processes (0: not that layout)
GPUS = int(opt("gpu-processes", "1"))
LOG = opt("log", "/content/serving.log")                     # one server log per configuration
OUT = opt("out", "/content/fair-bench.json")
t_start = time.time()


def say(msg):
    print(f"[fair {time.time() - t_start:7.1f} s] {msg}", flush=True)


def cores():
    n = os.cpu_count() or 8
    client = list(range(min(CLIENT_PROCS, n // 2)))
    server = list(range(len(client), n))
    return client, server


def http_json(method, url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def free_port() -> int:
    """A port no one listens on (Colab's own runtime holds 8080, for example)."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(check, what, seconds=900, proc=None, log=None):
    """Poll ``check`` until it holds; fail at once if ``proc`` (the server) has exited, with its log's tail."""
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc is not None and proc.poll() is not None:
            tail = Path(log).read_text(errors="replace")[-3000:] if log and Path(log).exists() else ""
            raise RuntimeError(f"{what} exited with code {proc.returncode} before it was ready:\n{tail}")
        try:
            if check():
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{what} not ready after {seconds} s")


# ---- the client: open loop, several processes ---------------------------------------------------------------------

def _client_proc(p, n_procs, url, bodies, rate, duration, t0, out_path, cpu):
    try:
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass
    import aiohttp

    async def run():
        n_total = int(rate * duration)
        mine = range(p, n_total, n_procs)
        lat, errors, timeouts, hits, lag, samples, why = [], 0, 0, [], [], [], {}
        conn = aiohttp.TCPConnector(limit=0, force_close=False)
        tout = aiohttp.ClientTimeout(total=TIMEOUT_S)
        async with aiohttp.ClientSession(connector=conn, timeout=tout) as s:
            async def one(i, due):
                nonlocal errors, timeouts
                try:
                    async with s.post(url, data=bodies[i % len(bodies)],
                                      headers={"Content-Type": "application/json"}) as r:
                        raw = await r.read()
                        if r.status != 200:
                            errors += 1
                            why[str(r.status)] = why.get(str(r.status), 0) + 1
                            return
                    lat.append((time.time() - due) * 1e3)
                    if i % 97 == 0:                          # sampled: hits, and a digest to compare unloaded
                        n_hits, digest = result_digest(raw)
                        hits.append(n_hits)
                        samples.append([i, n_hits, digest])
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception as e:
                    errors += 1
                    why[type(e).__name__] = why.get(type(e).__name__, 0) + 1

            tasks = []
            for i in mine:
                due = t0 + i / rate
                delay = due - time.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                lag.append((time.time() - due) * 1e3)
                tasks.append(asyncio.create_task(one(i, due)))
            await asyncio.gather(*tasks)
        json.dump({"lat": lat, "errors": errors, "timeouts": timeouts, "hits": hits, "lag": lag,
                   "sent": len(mine), "samples": samples, "why": why}, open(out_path, "w"))

    asyncio.run(run())


def result_digest(raw: bytes) -> tuple[int, str]:
    """A response's hit count and a digest of everything but took_ms (for the unloaded comparison)."""
    import hashlib
    d = json.loads(raw)
    d.pop("took_ms", None)
    return len(d["hits"]["hits"]), hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()


def compare_unloaded(url, bodies, runs, limit=400):
    """Replay sampled requests one at a time after the ladder and compare with what the loaded runs returned: a
    response changed by load (or emptied) shows as a mismatch; a query with no matches has 0 hits both ways."""
    seen, out = {}, {"checked": 0, "mismatch": [], "zero_hits_loaded": 0, "zero_hits_unloaded_too": 0}
    for r in runs:
        for i, n, dg in r.get("samples", []):
            seen.setdefault(i, []).append((r["target_qps"], n, dg))
    for i in sorted(seen)[:limit]:
        raw = urllib.request.urlopen(urllib.request.Request(url, data=bodies[i % len(bodies)], method="POST",
                                     headers={"Content-Type": "application/json"}), timeout=60).read()
        n0, d0 = result_digest(raw)
        for rate, n, dg in seen[i]:
            out["checked"] += 1
            if n == 0:
                out["zero_hits_loaded"] += 1
                out["zero_hits_unloaded_too"] += n0 == 0
            if dg != d0:
                out["mismatch"].append({"i": i, "rate": rate, "hits_loaded": n, "hits_unloaded": n0})
    return out


def load(url, bodies, rate, duration, client_cpus):
    t0 = time.time() + 2.0
    procs, outs = [], []
    for p in range(len(client_cpus)):
        out = f"/content/fair-client-{p}.json"
        outs.append(out)
        pr = mp.Process(target=_client_proc, args=(p, len(client_cpus), url, bodies, rate, duration, t0, out,
                                                   client_cpus[p]))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join(duration + TIMEOUT_S + 120)
    lat, hits, lag, errors, timeouts, sent, samples, why = [], [], [], 0, 0, 0, [], {}
    for out in outs:
        d = json.load(open(out))
        lat += d["lat"]; hits += d["hits"]; lag += d["lag"]; samples += d["samples"]
        for k, v in d["why"].items():
            why[k] = why.get(k, 0) + v
        errors += d["errors"]; timeouts += d["timeouts"]; sent += d["sent"]
    wall = max(duration, (max(lat) / 1e3) if lat else 0) + 0.0
    import numpy as np

    def pct(x, q):
        return float(np.percentile(x, q)) if x else None

    res = {"target_qps": rate, "sent": sent, "ok": len(lat), "errors": errors, "timeouts": timeouts,
           "achieved_qps": len(lat) / wall,
           "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "max": max(lat) if lat else None},
           "hits_per_response": {"min": min(hits) if hits else None, "median": pct(hits, 50)},
           "client_send_lag_ms_p99": pct(lag, 99)}
    res["errors_by"] = why
    res["samples"] = samples
    res["client_limited"] = bool(res["client_send_lag_ms_p99"] and res["client_send_lag_ms_p99"] > 50)
    return res


# ---- the engines -------------------------------------------------------------------------------------------------

def start_opensearch(server_cpus, port):
    if not (OS_HOME / "bin" / "opensearch").exists():
        tgz = Path(f"/content/opensearch-{OS_VERSION}.tar.gz")
        if not tgz.exists():
            say(f"opensearch: downloading {OS_VERSION}")
            urllib.request.urlretrieve(f"https://artifacts.opensearch.org/releases/bundle/opensearch/{OS_VERSION}/"
                                       f"opensearch-{OS_VERSION}-linux-x64.tar.gz", tgz)
        with tarfile.open(tgz) as t:
            t.extractall("/content")
    subprocess.run(["useradd", "-m", "osuser"], stderr=subprocess.DEVNULL)
    subprocess.run(["chown", "-R", "osuser:osuser", str(OS_HOME)], check=True)
    knn_lib = OS_HOME / "plugins" / "opensearch-knn" / "lib"
    heap = "8g"
    cmd = ["taskset", "-c", ",".join(map(str, server_cpus)), "runuser", "-u", "osuser", "--", "env",
           f"OPENSEARCH_JAVA_OPTS=-Xms{heap} -Xmx{heap}", f"LD_LIBRARY_PATH={knn_lib}",
           str(OS_HOME / "bin" / "opensearch"), "-Ediscovery.type=single-node", "-Eplugins.security.disabled=true",
           f"-Ehttp.port={port}", f"-Enode.processors={len(server_cpus)}"]
    proc = subprocess.Popen(cmd, stdout=open("/content/opensearch.log", "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        wait_until(lambda: http_json("GET", f"http://localhost:{port}")["version"]["number"], "opensearch", 300,
                   proc=proc, log="/content/opensearch.log")
    except Exception:
        stop(proc)
        raise
    say(f"opensearch: up, heap {heap}, {len(server_cpus)} processors")
    return proc


def stop(proc):
    if proc and proc.poll() is None:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def opensearch_target(ds, server_cpus):
    os.environ["S3_SEARCH_OS_ANALYZER"] = "english"         # the prototype indexed OpenSearch with `english`
    port = free_port()
    base = f"http://localhost:{port}"
    os.environ["S3_SEARCH_OPENSEARCH_URL"] = base          # the referee's adapter reads it when imported
    from referee import opensearch as osref
    proc = start_opensearch(server_cpus, port)
    t0 = time.time()
    say("opensearch: indexing 1.2M products (the referee's adapter: same mapping, english analyzer)")
    ix = osref.build_index(ds.catalogue, ds.mapping)
    build_s = time.time() - t0
    http_json("POST", f"{base}/{ix.name}/_forcemerge?max_num_segments=1", timeout=3600)
    say(f"opensearch: index {ix.name} ready in {build_s:.0f} s (force-merged to one segment)")
    tmpl = json.loads(Path("referee/requests/opensearch-cross-fields-tuned.json").read_text())

    def body(q):
        b = copy.deepcopy(tmpl)
        b["query"]["multi_match"]["query"] = q
        return {**b, "size": K, "_source": False, "track_total_hits": False}

    return proc, f"{base}/{ix.name}/_search", body, build_s


def stop_mps():
    subprocess.run(["bash", "-c", "echo quit | nvidia-cuda-mps-control"], check=False)


def engine_target(ds, server_cpus):
    SERVE_DIR.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["bm25"] = json.loads(Path("referee/requests/bm25f-tuned.bm25.json").read_text())
    (SERVE_DIR / "mapping.json").write_text(json.dumps(m))
    src = SERVE_DIR / "products.parquet"
    ds.catalogue.to_parquet(src)
    t0 = time.time()
    if not (SERVE_DIR / "esci").exists():
        say("engine: building the serving index (python -m serving build)")
        subprocess.run([sys.executable, "-m", "serving", "build", "--name", "esci", "--mapping",
                        str(SERVE_DIR / "mapping.json"), "--source", str(src), "--data-dir", str(SERVE_DIR),
                        "--termination-log", str(SERVE_DIR / "build.json")],
                       check=True)
    build_s = time.time() - t0
    env = {**os.environ, "S3_SERVING_CACHE": "0", "S3_SERVING_MAX_BATCH": opt("max-batch", "64"),
           "S3_SERVING_MAX_WAIT_MS": opt("max-wait-ms", "2"), "S3_SEARCH_DEVICE": "cuda",
           "S3_SERVING_WORKERS": str(WORKERS), "S3_SERVING_COLUMNS": COLUMNS,
           "S3_SERVING_FRONTS": str(FRONTS), "S3_SERVING_GPU_PROCESSES": str(GPUS),
           "S3_SERVING_GPU_MEMORY_FRACTION": opt("gpu-memory-fraction", "0")}
    if MPS:                                                  # CUDA MPS: the workers' kernels share the GPU at once
        import atexit
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
        atexit.register(stop_mps)                            # stopped however the run ends
        say("engine: CUDA MPS daemon started")
    port = free_port()
    cmd = ["taskset", "-c", ",".join(map(str, server_cpus)), sys.executable, "-m", "serving", "--data-dir",
           str(SERVE_DIR), "--device", "cuda", "--preload", "esci", "--port", str(port)]
    proc = subprocess.Popen(cmd, env=env, stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        wait_until(lambda: urllib.request.urlopen(f"http://localhost:{port}/readyz", timeout=5).status == 200,
                   "serving", 1800, proc=proc, log=LOG)
    except Exception:
        stop(proc)
        raise
    if WORKERS > 1:                                          # every worker loaded, not only the first (/readyz says)
        wait_until(lambda: http_json("GET", f"http://localhost:{port}/readyz", timeout=5).get("workers_ready", 0)
                   >= WORKERS, f"{WORKERS} workers loaded", 1800, proc=proc, log=LOG)
    say(f"engine: serving on cuda (index built in {build_s:.0f} s; query cache off, max_batch "
        f"{env['S3_SERVING_MAX_BATCH']}, max_wait {env['S3_SERVING_MAX_WAIT_MS']} ms, workers {WORKERS}, "
        f"columns {COLUMNS}, fronts {FRONTS}, gpu processes {GPUS}; log {LOG})")
    tmpl = json.loads(Path("referee/requests/bm25f-tuned.json").read_text())

    def body(q):
        b = copy.deepcopy(tmpl)
        b["query"]["multi_match"]["query"] = q
        return {**b, "size": K}

    return proc, f"http://localhost:{port}/indexes/esci/_search", body, build_s


def _role(p) -> str:
    """What a server process is, from its command line (supervisor, worker i, front i, gpu process j)."""
    try:
        a = p.cmdline()
    except Exception:
        return "?"
    for flag, name in (("--gpu-process", "gpu process"), ("--gpu-sockets", "front"), ("--worker-index", "worker")):
        if flag in a:
            i = a.index(flag)
            j = a[i + 1] if flag != "--gpu-sockets" else a[a.index("--worker-index") + 1] if "--worker-index" in a else "?"
            return f"{name} {j}"
    return "server" if "serving" in " ".join(a) else a[0] if a else "?"     # the supervisor, or a single process


class Sampler:
    """Every 0.5 s during a run: GPU utilisation and memory (nvidia-smi), and the server's CPU use (its whole process
    tree, in cores). Says whether the GPU or the host was the limit."""

    def __init__(self, pid):
        import threading
        self.pid, self.gpu, self.mem, self.cpu = pid, [], [], []
        self.per_cpu, self.per_mem, self.role = {}, {}, {}      # per process: cores and device MiB samples
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil
        root = psutil.Process(self.pid)
        procs = {}
        while not self._stop.wait(0.5):
            try:
                g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5).stdout.split(",")
                self.gpu.append(float(g[0])); self.mem.append(float(g[1]))
            except Exception:
                pass
            try:
                total = 0.0
                for p in [root] + root.children(recursive=True):
                    if p.pid not in procs:
                        procs[p.pid] = p
                        p.cpu_percent(None)                     # the first call only starts the counter
                        self.role[p.pid] = _role(p)
                        continue
                    c = procs[p.pid].cpu_percent(None)
                    total += c
                    self.per_cpu.setdefault(p.pid, []).append(c / 100.0)
                self.cpu.append(total / 100.0)
            except Exception:
                pass
            try:                                                 # device memory per process (what grows, and where)
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                for line in apps:
                    pid, mib = (x.strip() for x in line.split(","))
                    self.per_mem.setdefault(int(pid), []).append(float(mib))
            except Exception:
                pass

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(10)

    def summary(self, n_server_cpus):
        import numpy as np

        def s(x):
            return {"mean": float(np.mean(x)), "p95": float(np.percentile(x, 95)), "max": float(np.max(x))} if x else None
        out = {"gpu_util_pct": s(self.gpu[2:]), "gpu_mem_mib": s(self.mem), "server_cpu_cores": s(self.cpu[2:]),
               "server_cpus_available": n_server_cpus}
        out["per_process"] = sorted(
            ({"pid": pid, "role": self.role.get(pid, "?"), "cpu_cores": s(v[2:] or v),
              "device_mib": s(self.per_mem.get(pid, []))} for pid, v in self.per_cpu.items() if v),
            key=lambda r: -(r["cpu_cores"] or {}).get("p95", 0))[:12]
        busiest = out["per_process"][0] if out["per_process"] else None
        g, c = out["gpu_util_pct"], out["server_cpu_cores"]
        if g and c:
            out["limit"] = ("gpu" if g["mean"] >= 85 else
                            "host cpu (one core: the Python process)" if WORKERS == 1 and not FRONTS and 0.9 <= c["p95"] < 1.6 else
                            "host cpu" if c["p95"] >= 0.85 * n_server_cpus else
                            f"one process at a core: {busiest['role']}" if busiest and busiest["cpu_cores"]
                            and busiest["cpu_cores"]["p95"] >= 0.9 else "neither saturated")
        return out


def main():
    subprocess.run(["bash", "-lc", f"[ -d {REPO} ] || git clone --quiet {REPO_URL} {REPO}"], check=True)
    # a branch or a full sha fetches by name; a short sha does not, so then fetch every branch and resolve it
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"],
                       check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", REF], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.environ.setdefault("S3_SEARCH_DATA", os.path.join(REPO, "data", "esci"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp", "snowballstemmer"], check=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    from referee import loader, runtime
    env = runtime.environment()
    client_cpus, server_cpus = cores()
    say(f"{REF} at {commit}; {env['gpu']}; {os.cpu_count()} CPUs: client {client_cpus}, server {server_cpus}")
    loader.fetch()
    ds = loader.load_dataset("esci-us")
    texts = [str(t) for t in ds.test_queries["query"]]
    say(f"{len(ds.catalogue):,} products, {len(texts):,} query texts, depth {K}, rates {RATES} QPS x {DURATION:.0f} s")
    out = {"kind": "fair-bench", "label": "development only, never a score", "ref": REF, "commit": commit,
           "runtime": env, "cpus": os.cpu_count(), "client_cpus": client_cpus, "server_cpus": server_cpus,
           "n_products": int(len(ds.catalogue)), "k": K, "duration_s": DURATION, "workers": WORKERS,
           "columns": COLUMNS, "mps": MPS, "fronts": FRONTS, "gpu_processes": GPUS, "log": LOG, "engines": {}}
    for name in ENGINES:
        target = engine_target if name == "engine" else opensearch_target
        try:
            proc, url, make_body, build_s = target(ds, server_cpus)
        except Exception as e:                                   # recorded; the other engine still runs
            say(f"{name}: FAILED to start: {e}")
            out["engines"][name] = {"error": str(e)[-3000:]}
            json.dump(out, open(OUT, "w"), indent=1)
            continue
        try:
            bodies = [json.dumps(make_body(t)).encode() for t in texts]
            for b in bodies[:200]:                               # warm-up, one at a time
                urllib.request.urlopen(urllib.request.Request(url, data=b, method="POST",
                                                              headers={"Content-Type": "application/json"}),
                                       timeout=60).read()
            runs = []
            for rate in RATES:
                say(f"{name}: {rate:.0f} QPS for {DURATION:.0f} s")
                with Sampler(proc.pid) as smp:
                    r = load(url, bodies, rate, DURATION, client_cpus)
                r["resources"] = smp.summary(len(server_cpus))
                say(f"{name}: {rate:.0f} QPS resources: GPU {r['resources']['gpu_util_pct']}, server CPU cores "
                    f"{r['resources']['server_cpu_cores']} of {len(server_cpus)}; limit: {r['resources'].get('limit')}")
                say(f"{name}: {rate:.0f} QPS -> achieved {r['achieved_qps']:.0f}, p50 {r['latency_ms']['p50']:.1f} ms, "
                    f"p99 {r['latency_ms']['p99']:.1f} ms, errors {r['errors']}, timeouts {r['timeouts']}, "
                    f"hits {r['hits_per_response']}, client lag p99 {r['client_send_lag_ms_p99']:.1f} ms"
                    + (" (CLIENT-LIMITED)" if r["client_limited"] else ""))
                runs.append(r)
                if r["achieved_qps"] < 0.9 * rate or r["timeouts"] + r["errors"] > 0.01 * r["sent"]:
                    say(f"{name}: cannot sustain {rate:.0f} QPS; no higher rates (the same rule for every engine)")
                    break
                time.sleep(5)
            say(f"{name}: replaying sampled requests unloaded")
            check = compare_unloaded(url, bodies, runs)
            say(f"{name}: unloaded comparison: {check['checked']} checked, {len(check['mismatch'])} differ, "
                f"{check['zero_hits_loaded']} with 0 hits under load ({check['zero_hits_unloaded_too']} also 0 unloaded)")
            out["engines"][name] = {"build_s": build_s, "url": url, "runs": runs, "unloaded_check": check}
        finally:
            stop(proc)
            time.sleep(5)
        json.dump(out, open(OUT, "w"), indent=1)
    say("DONE")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    mp.set_start_method("fork")
    sys.exit(main())
