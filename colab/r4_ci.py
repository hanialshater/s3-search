#!/usr/bin/env -S colab run --timeout 1500
"""R4: build native/s3_hostpath (with R4's Loop) and serving/native_front on a Colab CPU runtime, then hold the Loop and
the GPU process's R4 mode to today's loop. Development only, never a score.

    colab run --timeout 1500 colab/r4_ci.py [branch-or-commit]

Steps (each must pass; the first two fail if any test is skipped):
- "r4 loop": serving/tests/test_r4.py, the Loop's contract on PyLoop and on s3_hostpath.Loop;
- "r4 serving": serving/tests/test_gpu_process.py, every case, S3_SERVING_R4 off, stub and 1, the native front among
  them (test_r4_replies_are_todays is the byte gate on toy);
- "hostpath oracle": the existing Rust tests (prepare against the Python planner, batch4 against binproto.batch4);
- "serving tests", "referee fast", "engine tests": nothing else changed.

It prints the wheel's and both Cargo.locks' sha256.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
REF = next((a for a in sys.argv[1:] if not a.startswith("--")), "main")
t0 = time.time()


def say(m):
    print(f"[r4-ci {time.time() - t0:6.1f} s] {m}", flush=True)


def sh(cmd, check=True, **kw):
    say("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    return subprocess.run(cmd, check=check, shell=isinstance(cmd, str), **kw)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def beat(stop):
    while not stop.wait(60):
        say("still running")


def main():
    stop = threading.Event()
    threading.Thread(target=beat, args=(stop,), daemon=True).start()
    if not os.path.isdir(REPO):
        sh(["git", "clone", "--quiet", REPO_URL, REPO])
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        sh(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"])
    else:
        sh(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"])
        sh(["git", "-C", REPO, "checkout", "--quiet", REF])
    os.chdir(REPO)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"{REF} at {commit}; nproc {os.cpu_count()}")
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    if not os.path.exists(os.path.join(cargo_bin, "cargo")):
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    os.environ["PATH"] = cargo_bin + os.pathsep + os.environ["PATH"]
    sh(["cargo", "--version"])
    sh([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt", "maturin>=1.5,<2"])
    crate = os.path.join(REPO, "native", "s3_hostpath")
    sh([sys.executable, "-m", "maturin", "build", "--release", "-m", os.path.join(crate, "Cargo.toml"),
        "-o", "/content/hostpath-wheels"])
    whl = os.path.join("/content/hostpath-wheels", sorted(os.listdir("/content/hostpath-wheels"))[-1])
    sh([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", whl])
    sh(["cargo", "build", "--release", "-q"], cwd="serving/native_front")
    os.environ["S3_NATIVE_FRONT_BIN"] = os.path.abspath("serving/native_front/target/release/s3-native-front")
    say(f"wheel {os.path.basename(whl)} sha256 {sha(whl)}; s3_hostpath Cargo.lock sha256 "
        f"{sha(os.path.join(crate, 'Cargo.lock'))}; native_front Cargo.lock sha256 "
        f"{sha('serving/native_front/Cargo.lock')}")
    sh([sys.executable, "-c", "import s3_hostpath; print('Loop' in dir(s3_hostpath), s3_hostpath.Loop)"])
    sh([sys.executable, "-c", "from referee import loader; loader.fetch()"])      # esci-sample for the oracle tests
    pyt = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-o", "faulthandler_timeout=120"]
    ok = True
    for name, cmd, no_skips, limit in [
        ("r4 loop", pyt + ["-v", "-rs", "serving/tests/test_r4.py"], True, 300),
        ("r4 serving", pyt + ["-v", "-rs", "--durations=15", "serving/tests/test_gpu_process.py"], True, 1500),
        ("hostpath oracle", pyt + ["-q", "-rs", "engine/tests/test_hostpath.py", "engine/tests/test_hostpath_analysis.py",
                                   "engine/tests/test_hostpath_batch4.py"], True, 700),
        ("serving tests", pyt + ["-q", "-rfE", "serving/tests"], False, 1500),
        ("referee fast", [sys.executable, "-m", "referee", "fast"], False, 300),
        ("engine tests", pyt + ["-q", "engine/tests"], False, 900),
    ]:
        ts = time.time()
        log = f"/content/r4-ci-{name.replace(' ', '-')}.log"
        try:
            with open(log, "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=limit).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        out = open(log, errors="replace").read()
        print(out[-7000:], flush=True)
        if no_skips and "SKIPPED" in out:
            say(f"{name}: a test was skipped")
            rc = "skipped" if rc == 0 else rc
        passed = rc == 0
        say(f"{'PASSED' if passed else 'FAILED'} {name} (exit {rc}) in {time.time() - ts:.0f} s")
        ok = ok and passed
    stop.set()
    say(f"{'PASSED' if ok else 'FAILED'}: {REF} at {commit}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
