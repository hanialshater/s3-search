#!/usr/bin/env -S colab run --timeout 900
"""R1: build native/s3_hostpath on a Colab CPU runtime and run its oracle tests. Development only, never a score.

    colab run --timeout 900 colab/hostpath_ci.py [branch-or-commit]

It installs a minimal Rust toolchain (rustup, stable) and maturin on the VM, builds the wheel (release), installs it,
and runs:
- engine/tests/test_hostpath.py: the Rust prepare against the Python planner, on toy and esci-sample;
- the referee's fast suite and the engine tests, to show nothing else changed.

It prints the wheel's and Cargo.lock's sha256, and copies Cargo.lock to /content/hostpath-Cargo.lock to be committed.
Exit code 0 only when every step passes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time

REPO_URL = "https://github.com/hanialshater/s3-search"
REPO = "/content/s3-search"
REF = sys.argv[1] if len(sys.argv) > 1 else "main"
t0 = time.time()


def say(m):
    print(f"[hostpath-ci {time.time() - t0:6.1f} s] {m}", flush=True)


def sh(cmd, check=True, **kw):
    say("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    return subprocess.run(cmd, check=check, shell=isinstance(cmd, str), **kw)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if not os.path.isdir(REPO):
        sh(["git", "clone", "--quiet", REPO_URL, REPO])
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        sh(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"])
    else:
        sh(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"])
        sh(["git", "-C", REPO, "checkout", "--quiet", REF])
    os.chdir(REPO)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    say(f"{REF} at {commit}")
    cargo = os.path.expanduser("~/.cargo/bin")
    if not os.path.exists(os.path.join(cargo, "cargo")):
        sh("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    os.environ["PATH"] = cargo + os.pathsep + os.environ["PATH"]
    sh(["cargo", "--version"])
    sh([sys.executable, "-m", "pip", "install", "-q", "maturin>=1.5,<2", "snowballstemmer", "pytest"])
    crate = os.path.join(REPO, "native", "s3_hostpath")
    sh([sys.executable, "-m", "maturin", "build", "--release", "-m", os.path.join(crate, "Cargo.toml"),
        "-o", "/content/hostpath-wheels"])
    wheels = sorted(os.listdir("/content/hostpath-wheels"))
    whl = os.path.join("/content/hostpath-wheels", wheels[-1])
    sh([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", whl])
    lock = os.path.join(crate, "Cargo.lock")
    shutil.copy(lock, "/content/hostpath-Cargo.lock")
    say(f"wheel {os.path.basename(whl)} sha256 {sha(whl)}; Cargo.lock sha256 {sha(lock)}")
    sh([sys.executable, "-c", "from referee import loader; loader.fetch()"])     # esci-sample for the oracle test
    ok = True
    for name, cmd in [
        ("hostpath oracle", [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", "-rs",
                             "engine/tests/test_hostpath.py"]),
        ("referee fast", [sys.executable, "-m", "referee", "fast"]),
        ("engine tests", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "engine/tests"]),
    ]:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        out = (r.stdout + r.stderr)[-4000:]
        print(out, flush=True)
        if name == "hostpath oracle" and " SKIPPED" in out:
            say("hostpath oracle: a test was skipped")
            r = subprocess.CompletedProcess(cmd, 1)
        say(f"{'PASSED' if r.returncode == 0 else 'FAILED'} {name}")
        ok = ok and r.returncode == 0
    say(f"{'PASSED' if ok else 'FAILED'}: {REF} at {commit}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
