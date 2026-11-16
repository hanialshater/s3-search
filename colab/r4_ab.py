#!/usr/bin/env -S colab run --gpu L4 --timeout 2400
"""R4's byte A/B on a Colab GPU (hostpath-ab style, S6-arch-206's form): every query text through a real GPU process
with S3_SERVING_R4 off and on. Development only, never a score.

    colab run --gpu L4 --timeout 2400 colab/r4_ab.py [branch-or-commit] [--queries=8956] [--impacts=0,1] [--window=64]

It builds native/s3_hostpath (with R4's Loop) on the VM and ESCI-US's serving index (``python -m serving build``, the
tuned BM25F mapping as colab/fair_bench.py builds it), reading only what members may read (referee/loader.py: the
catalogue and the test queries' texts, never a judgment). Then, per S3_TRITON_FUSED_IMPACTS value and per arm
(S3_SERVING_R4=0, then 1), it starts ``python -m serving --fronts 1 --gpu-processes 1`` on cuda with S3_HOSTPATH=1 and
the fused lexical path on, and on one S3B3 connection to GPU process 0's socket:

- warms up: every deep request once, unrecorded, then waits for the module's HostIndex (built in the background);
- sends every query text as the deep request (tuned BM25F, size 2,000) and as the faceted one (brand and colour
  terms facets, size 10), in pipelined windows of ``--window`` frames, each window's replies read before the next.

Each reply's status and body (took_ms removed) is hashed with sha256, and the two arms are compared request by request:
0 differ is the gate. The R4 arm's ``r4-0.json`` (the Loop's counters and the host path's summary) is kept as evidence
that the Loop and the Rust prepare served. Writes /content/r4-ab.json and prints it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
REF = ARGS[0] if ARGS else "main"


def opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


N_QUERIES = int(opt("queries", "8956"))
IMPACTS = opt("impacts", "0,1").split(",")
WINDOW = int(opt("window", "64"))
SERVE = "/content/r4-serve"
t_start = time.time()
TOOK = re.compile(rb'"took_ms":\{[^{}]*\}')


def say(m):
    print(f"[r4-ab {time.time() - t_start:7.1f} s] {m}", flush=True)


def heartbeat(stop):
    while not stop.wait(60):
        say("still running")


def build_hostpath():
    cargo = os.path.expanduser("~/.cargo/bin")
    if not os.path.exists(os.path.join(cargo, "cargo")):
        subprocess.run("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable",
                       shell=True, check=True)
    os.environ["PATH"] = cargo + os.pathsep + os.environ["PATH"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "maturin>=1.5,<2", "snowballstemmer"], check=True)
    out = "/content/hostpath-wheels"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "maturin", "build", "--release", "-m", "native/s3_hostpath/Cargo.toml",
                    "-o", out], check=True)
    whl = os.path.join(out, sorted(os.listdir(out))[-1])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", whl], check=True)
    subprocess.run([sys.executable, "-c", "import s3_hostpath; s3_hostpath.Loop"], check=True)
    return {"wheel": os.path.basename(whl), "sha256": hashlib.sha256(open(whl, "rb").read()).hexdigest()}


def recv(s, n):
    buf = bytearray()
    while len(buf) < n:
        k = s.recv(n - len(buf))
        if not k:
            raise ConnectionError("the GPU process closed the connection")
        buf += k
    return bytes(buf)


def search_frame(i, body: bytes) -> bytes:
    p = struct.pack("<BQ", 1, i) + struct.pack("<H", 4) + b"esci" + struct.pack("<I", len(body)) + body
    return struct.pack("<I", len(p)) + p


def run_stream(sock_path, bodies, base_id, record=True):
    """Send ``bodies`` in windows; -> {id: sha256(status, body without took_ms)}."""
    from serving import binproto
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sock_path)
    s.sendall(b"S3B3")
    out = {}
    for w in range(0, len(bodies), WINDOW):
        chunk = bodies[w:w + WINDOW]
        s.sendall(b"".join(search_frame(base_id + w + k, b) for k, b in enumerate(chunk)))
        got = 0
        while got < len(chunk):
            n = struct.unpack("<I", recv(s, 4))[0]
            payload = recv(s, n)
            reps = binproto.split_batch4(payload) if payload[0] == 4 else [binproto.decode_reply(payload)]
            for id_, st, body, _ in reps:
                got += 1
                if record:
                    out[id_ - base_id] = hashlib.sha256(struct.pack("<H", st) + TOOK.sub(b"", body)).hexdigest()
    s.close()
    return out


def start_server(r4, impacts, log_path):
    state = tempfile.mkdtemp(prefix="r4ab-")
    port = 18000 + int(time.time()) % 1000
    env = {**os.environ, "S3_SERVING_CACHE": "0", "S3_SEARCH_DEVICE": "cuda", "S3_SERVING_STATE_DIR": state,
           "S3_HOSTPATH": "1", "S3_SERVING_R4": r4, "S3_TRITON_FUSED_LEXICAL": "1",
           "S3_TRITON_FUSED_IMPACTS": impacts, "S3_PLAN_CACHE": "0"}
    log = open(log_path, "w")
    p = subprocess.Popen([sys.executable, "-m", "serving", "--data-dir", SERVE, "--port", str(port), "--preload",
                          "esci", "--fronts", "1", "--gpu-processes", "1"], env=env, stdout=log,
                         stderr=subprocess.STDOUT)
    end = time.time() + 900
    while time.time() < end:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=5).status == 200:
                return p, state
        except OSError:
            pass
        if p.poll() is not None:
            raise RuntimeError(f"the server exited: {open(log_path).read()[-3000:]}")
        time.sleep(1)
    p.kill()
    raise RuntimeError("the server was not ready in 900 s")


def stop_server(p):
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        p.kill()


def main():
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--quiet", REPO_URL, REPO], check=True)
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"],
                       check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", REF], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    wheel = build_hostpath()
    from referee import loader, runtime
    env = runtime.environment()
    say(f"{REF} at {commit}; {env['gpu']}; s3_hostpath {wheel}")
    loader.fetch()
    ds = loader.load_dataset("esci-us")
    tmpl = json.loads(open("referee/requests/bm25f-tuned.json").read())
    fields = tmpl["query"]["multi_match"]["fields"]
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["bm25"] = json.loads(open("referee/requests/bm25f-tuned.bm25.json").read())
    os.makedirs(SERVE, exist_ok=True)
    with open(f"{SERVE}/mapping.json", "w") as f:
        json.dump(m, f)
    ds.catalogue.to_parquet(f"{SERVE}/products.parquet")
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "serving", "build", "--name", "esci", "--mapping", f"{SERVE}/mapping.json",
                    "--source", f"{SERVE}/products.parquet", "--data-dir", SERVE, "--termination-log",
                    f"{SERVE}/build.json"], check=True, capture_output=True)
    say(f"serving index built in {time.time() - t0:.0f} s ({len(ds.catalogue):,} products)")
    texts = [str(t) for t in ds.test_queries["query"].iloc[:N_QUERIES]]
    facets = {"brand": {"terms": {"field": "product_brand.keyword", "size": 10}},
              "color": {"terms": {"field": "product_color.keyword", "size": 10}}}
    kinds = {"deep": [json.dumps({"query": {"multi_match": {"query": q, "fields": fields}}, "size": 2000}).encode()
                      for q in texts],
             "faceted": [json.dumps({"query": {"multi_match": {"query": q, "fields": fields}}, "aggs": facets,
                                     "size": 10}).encode() for q in texts]}
    res = {"kind": "r4-ab", "label": "development only, never a score", "ref": REF, "commit": commit, "runtime": env,
           "hostpath_wheel": wheel, "n_queries": len(texts), "window": WINDOW, "ab": {}, "r4_note": {}}
    for imp in IMPACTS:
        digests = {}
        for r4 in ("0", "1"):
            log_path = f"/content/r4ab-server-imp{imp}-r4{r4}.log"
            p, state = start_server(r4, imp, log_path)
            try:
                sock = os.path.join(state, "gpu-0.sock")
                t0 = time.time()
                run_stream(sock, kinds["deep"], 10_000_000, record=False)     # warm-up; the HostIndex builds meanwhile
                time.sleep(60)
                say(f"impacts {imp} R4 {r4}: warm-up {time.time() - t0:.0f} s")
                for kind, bodies in kinds.items():
                    t0 = time.time()
                    digests[(r4, kind)] = run_stream(sock, bodies, 0 if kind == "deep" else 1_000_000)
                    say(f"impacts {imp} R4 {r4} {kind}: {len(digests[(r4, kind)])} replies in {time.time() - t0:.0f} s")
                time.sleep(6)                                                   # one more 5 s stats note
                note = os.path.join(state, "r4-0.json")
                if os.path.exists(note):
                    res["r4_note"][f"impacts{imp}"] = json.loads(open(note).read())
            finally:
                stop_server(p)
        for kind in kinds:
            a, b = digests[("0", kind)], digests[("1", kind)]
            bad = sorted(i for i in range(len(texts)) if a.get(i) != b.get(i))
            key = f"{kind}_impacts{imp}"
            res["ab"][key] = {"of": len(texts), "replies": [len(a), len(b)], "differ": len(bad), "first_differ": bad[:10],
                              "sha256_off": hashlib.sha256("".join(a[i] for i in sorted(a)).encode()).hexdigest(),
                              "sha256_on": hashlib.sha256("".join(b[i] for i in sorted(b)).encode()).hexdigest()}
            say(f"R4 A/B {key}: {len(bad)} of {len(texts)} replies differ")
    res["passed"] = all(v["differ"] == 0 and v["replies"] == [v["of"], v["of"]] for v in res["ab"].values())
    stop.set()
    json.dump(res, open("/content/r4-ab.json", "w"), indent=1)
    say("DONE " + ("PASSED" if res["passed"] else "FAILED"))
    print(json.dumps(res, indent=1), flush=True)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
