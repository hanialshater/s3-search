"""The search floor's referee (owner: evaluation-steward; makers do not edit it, P6).

Importing it pins the BLAS and OpenMP thread pools to the process's CPU affinity mask before numpy loads, so a run
under `taskset -c 0,1` uses 2 threads, not one per machine core (timings otherwise vary twofold to sevenfold).
Variables the caller set are kept. Child processes (the judges' serving replicas) inherit the pin.
"""
from __future__ import annotations

import os
import sys

BLAS_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def _cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def pin_blas_threads() -> dict:
    n = str(_cpus())
    for v in BLAS_VARS:
        os.environ.setdefault(v, n)
    return blas_threads()


def blas_threads() -> dict:
    """What a results file records: the affinity mask's size, the pins, and whether numpy loaded before the pin
    (then the pin did not reach it)."""
    return {"cpus_in_affinity_mask": _cpus(), **{v: os.environ.get(v) for v in BLAS_VARS},
            "numpy_loaded_before_pin": _NUMPY_BEFORE}


_NUMPY_BEFORE = "numpy" in sys.modules
pin_blas_threads()
