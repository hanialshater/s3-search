"""The referee's entry point (owner: evaluation-steward; makers do not edit it, P6).

    python -m referee fast           CPU, in CI: the referee's own tests, then the engine's contract checks on toy
                                     once `engine/api.py` exists (INTERFACES section 5), on every available device
    python -m referee bench ...      quality (fit split), latency, throughput and cost on a named dataset; see
                                     referee/bench.py. CPU numbers are development only
    python -m referee conformance    the DSL conformance suite (B1-5): the engine against the reference or OpenSearch
    python -m referee full           GPU, in Colab, run by the founder: the score of record, written to results.json
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _has_engine() -> bool:
    try:
        return importlib.util.find_spec("engine.api") is not None
    except ModuleNotFoundError:
        return False


class _Tally:
    """Counts outcomes, so the last line of `referee fast` says whether it passed and names what failed."""

    def __init__(self):
        self.counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "error": 0}
        self.failed = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and not report.passed):
            if report.failed:
                key = "failed" if report.when == "call" else "error"
                self.counts[key] += 1
                self.failed.append(report.nodeid)
            elif report.skipped:
                self.counts["xfailed" if hasattr(report, "wasxfail") else "skipped"] += 1
            elif report.passed:
                self.counts["passed"] += 1

    def pytest_collectreport(self, report):
        if report.failed:
            self.counts["error"] += 1
            self.failed.append(f"collecting {report.nodeid}")


def fast(argv=()) -> int:
    import pytest
    targets = [str(HERE / "tests")]
    if _has_engine():
        targets.append(str(HERE / "contract"))
    else:
        print("referee fast: the engine has no API yet (engine/api.py), so only the referee's own tests run")
    t = time.perf_counter()
    tally = _Tally()
    # The judges' full self-tests are marked slow and run with the judges (`python -m pytest -m slow referee/tests`).
    select = [] if any(a == "-m" or a.startswith("-m") for a in argv) else ["-m", "not slow"]
    code = pytest.main(["-q", "-p", "no:cacheprovider", *select, *targets, *argv], plugins=[tally])
    took = time.perf_counter() - t
    c = tally.counts
    counts = ", ".join(f"{n} {k}" for k, n in c.items() if n)
    # The 60 s budget is for the 2-core CPU container (CI and development). It does not apply on Colab, whose
    # contract runs on every device it reports (cpu, torch-cpu, cuda) with GPU warm-ups: the time is shown, not judged.
    import os
    on_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")) or "google.colab" in sys.modules
    over = took > 60 and not on_colab
    budget = "the 60 s budget is for the 2-core CPU container; it does not apply on Colab" if on_colab \
        else "budget 60 s on 2 cores"
    if code == 0 and not over:
        print(f"referee fast: PASSED ({counts}) in {took:.1f} s ({budget})")
        return 0
    shown = "; ".join(tally.failed[:5]) + (f"; and {len(tally.failed) - 5} more" if len(tally.failed) > 5 else "")
    why = f"FAILED ({counts}): {shown}" if tally.failed else f"FAILED (pytest exit {int(code)}; {counts})"
    if over:
        why += f"; over the 60 s budget"
    print(f"referee fast: {why} in {took:.1f} s")
    return int(code) or 1


def bench(argv=()) -> int:
    from referee.bench import main
    return main(list(argv))


def conformance(argv=()) -> int:
    from referee.conformance import main
    return main(list(argv))


def full(argv=()) -> int:
    if not _has_engine() and not any(x in ("--engine=opensearch", "--engine=reference") for x in argv):
        print("referee full: needs engine/api.py (INTERFACES section 5); S1 adds it", file=sys.stderr)
        return 2
    from referee.full import main
    return main(list(argv))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    cmds = {"fast": fast, "bench": bench, "conformance": conformance, "full": full}
    if mode not in cmds:
        print(__doc__)
        sys.exit(2)
    sys.exit(cmds[mode](sys.argv[2:]))
