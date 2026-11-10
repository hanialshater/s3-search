"""Importing the referee pins BLAS and OpenMP threads to the CPU affinity mask, before numpy loads."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(code, env_extra=None, cpus="0"):
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS") and k != "VECLIB_MAXIMUM_THREADS"}
    env.update(env_extra or {})
    cmd = [sys.executable, "-c", code]
    if hasattr(os, "sched_setaffinity"):
        cmd = ["taskset", "-c", cpus] + cmd
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env).stdout.strip()


def test_pins_to_the_affinity_mask_before_numpy():
    out = _run("import referee, os, numpy; print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'],"
               " referee.blas_threads()['numpy_loaded_before_pin'])")
    assert out == "1 1 False"


def test_keeps_what_the_caller_set():
    assert _run("import referee, os; print(os.environ['OPENBLAS_NUM_THREADS'])", {"OPENBLAS_NUM_THREADS": "3"}) == "3"


def test_bench_records_the_threads(tmp_path):
    out = tmp_path / "r.json"
    code = (f"from referee import bench; bench.main(['--dataset=toy', '--engine=reference', '--out={out}']);"
            f"import json; t = json.load(open('{out}'))['host']['threads']; print(t['OPENBLAS_NUM_THREADS'],"
            f" t['cpus_in_affinity_mask'])")
    assert _run(code).splitlines()[-1] == "1 1"
