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


def fast(argv=()) -> int:
    import pytest
    targets = [str(HERE / "tests")]
    if _has_engine():
        targets.append(str(HERE / "contract"))
    else:
        print("referee fast: the engine has no API yet (engine/api.py), so only the referee's own tests run")
    t = time.perf_counter()
    code = pytest.main(["-q", "-p", "no:cacheprovider", *targets, *argv])
    print(f"referee fast: {time.perf_counter() - t:.1f} s (budget 60 s on 2 cores)")
    return code


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
