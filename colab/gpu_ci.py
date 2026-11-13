#!/usr/bin/env -S colab run --gpu L4 --timeout 600
"""GPU CI: the referee's fast suite and the engine's tests on a Colab GPU. Development only, never a score.

    colab run --gpu L4 --timeout 600 colab/gpu_ci.py [branch-or-commit]   (a branch, or a full or short sha)

CI runs on CPU, so the cuda paths are exercised only here. A test silent for 120 s gets its stack dumped
(faulthandler), each suite has a hard limit, and a heartbeat line every 60 s keeps the Colab CLI from timing out.
Exit code 0 only when every suite passes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
REF = sys.argv[1] if len(sys.argv) > 1 else "main"
SUITES = [  # (name, pytest targets, hard limit in seconds)
    ("referee fast", None, 900),
    ("engine tests", ["engine/tests"], 1200),
    ("serving tests", ["serving/tests"], 600),       # the platform-engineer's suggestion: serving on cuda too
]


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def heartbeat(stop, label, t0):
    while not stop.wait(60):
        print(f"[gpu-ci] {label}: still running after {int(time.time() - t0)} s", flush=True)


def run_suite(name, targets, limit):
    if targets is None:
        cmd = [sys.executable, "-u", "-m", "referee", "fast", "-o", "faulthandler_timeout=120", "--durations=10"]
    else:
        cmd = [sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "faulthandler_timeout=120",
               "--durations=10", "-rfE", *targets]
    t0, stop = time.time(), threading.Event()
    threading.Thread(target=heartbeat, args=(stop, name, t0), daemon=True).start()
    log = f"/content/gpu-ci-{name.replace(' ', '-')}.log"
    try:
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=limit).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    stop.set()
    took = int(time.time() - t0)
    text = open(log).read()
    print(f"[gpu-ci] {name}: exit {rc} in {took} s (limit {limit} s)", flush=True)
    print(text[-5000:], flush=True)
    return rc == 0, took


def main():
    if not os.path.isdir(REPO):
        sh(["git", "clone", "--quiet", REPO_URL, REPO])
    # A branch or a full sha can be fetched by name; a short sha cannot (git fetch needs a ref or a full id), so
    # fetch every branch and resolve it locally instead.
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        sh(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"])
    else:
        sh(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"])
        sh(["git", "-C", REPO, "checkout", "--quiet", REF])
    os.chdir(REPO)
    missing = [m for m in ("snowballstemmer", "pytest") if __import__("importlib").util.find_spec(m) is None]
    if missing:
        sh([sys.executable, "-m", "pip", "install", "-q", *missing])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    devices = subprocess.run([sys.executable, "-c", "from engine import api; print(api.available_devices())"],
                             capture_output=True, text=True).stdout.strip()
    print(f"[gpu-ci] {REF} at {commit} on {gpu or 'no GPU'}; devices {devices}", flush=True)
    if "cuda" not in devices:
        print("[gpu-ci] FAILED: cuda is not available, so this run would test nothing new", flush=True)
        return 1
    results = [(name, *run_suite(name, targets, limit)) for name, targets, limit in SUITES]
    for name, ok, took in results:
        print(f"[gpu-ci] {'PASSED' if ok else 'FAILED'} {name} ({took} s)", flush=True)
    ok = all(r[1] for r in results)
    print(f"[gpu-ci] {'PASSED' if ok else 'FAILED'}: {REF} at {commit} on {gpu}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
