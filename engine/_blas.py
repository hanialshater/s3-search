"""Keep numpy's OpenBLAS within the process's CPU affinity (S8-3).

OpenBLAS sizes its thread pool from the machine's cores, not the affinity mask, so under ``taskset -c 0,1`` on a
4-core machine it runs 4 threads on 2 cores, and a QR goes from 0.8 s to 7 s. ``limit_to_affinity()`` sets the pool to
the number of CPUs this process may use, through OpenBLAS's own ``*_set_num_threads`` (found in the loaded library;
no new dependency). It is a no-op when no OpenBLAS is loaded or when ``OPENBLAS_NUM_THREADS`` is set by the user.
"""
from __future__ import annotations

import ctypes
import os

_SYMBOLS = ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_", "openblas_set_num_threads",
            "scipy_openblas_set_num_threads")
_done = False


def _libraries():
    try:
        with open("/proc/self/maps") as f:
            return sorted({line.split()[-1] for line in f if "openblas" in line.lower() and line.split()[-1].startswith("/")})
    except OSError:
        return []


def limit_to_affinity() -> int | None:
    """Set OpenBLAS's threads to the affinity mask's size, once per process. Returns the count set, or None."""
    global _done
    if _done or os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS"):
        return None
    _done = True
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    import numpy  # noqa: F401  (loads the library)
    for path in _libraries():
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for sym in _SYMBOLS:
            fn = getattr(lib, sym, None)
            if fn is not None:
                fn(ctypes.c_int(n))
                return n
    return None
