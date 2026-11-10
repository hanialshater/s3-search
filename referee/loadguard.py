"""A quiet machine for timing judges (owner: evaluation-steward, P6).

The load average is machine-wide, so other agents' work shows in it. A timing judge:
1. waits for a quiet start: the 1-minute load average at or below `QUIET_LOAD1` (0.5), for up to `wait_s`; else it
   refuses to start (`Busy`);
2. samples the load average every second while it runs;
3. refuses its timing verdicts when the run's mean 1-minute load exceeds the cores it may use (its affinity mask)
   plus `OVERLOAD_MARGIN` (0.75): its own threads account for up to one unit per core, the rest is someone else's.
Its measurements are kept either way, labelled.
"""
from __future__ import annotations

import os
import threading
import time

QUIET_LOAD1 = 0.5
OVERLOAD_MARGIN = 0.75


class Busy(RuntimeError):
    pass


def cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def load1() -> float:
    return os.getloadavg()[0]


def wait_for_quiet(max_load1: float = QUIET_LOAD1, wait_s: float = 300.0, poll_s: float = 5.0) -> dict:
    t = time.perf_counter()
    while True:
        l1 = load1()
        if l1 <= max_load1:
            return {"load1_at_start": l1, "waited_s": time.perf_counter() - t, "max_load1_to_start": max_load1}
        if time.perf_counter() - t >= wait_s:
            raise Busy(f"the 1-minute load average stayed above {max_load1} for {wait_s:.0f} s (now {l1:.2f}): "
                       "not judging timings on a busy machine; run it in a quiet moment")
        time.sleep(poll_s)


class Sampler:
    """Samples the load average every second in a background thread."""

    def __init__(self, every_s: float = 1.0):
        self.every_s, self.samples, self._stop = every_s, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(load1())
            self._stop.wait(self.every_s)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        s = self.samples or [load1()]
        limit = cpus() + OVERLOAD_MARGIN
        mean = sum(s) / len(s)
        return {"load1_mean": mean, "load1_max": max(s), "samples": len(s), "limit": limit,
                "overloaded": mean > limit}
