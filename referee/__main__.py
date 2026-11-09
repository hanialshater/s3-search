"""The referee's entry point (owner: evaluation-steward; makers do not edit it, P6).

    python -m referee fast    CPU, in CI: the referee's own tests, then the engine's contract checks once
                              the engine exposes its API (docs/search-floor/INTERFACES.md §3 in the lab)
    python -m referee full    GPU, in Colab, run by the founder: the score of record, written to results.json
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fast() -> int:
    import pytest
    targets = [str(HERE / "tests")]
    contract = HERE / "contract"
    if importlib.util.find_spec("engine.api") is not None and contract.exists():
        targets.append(str(contract))
    else:
        print("referee fast: the engine has no API yet (engine/api.py), so only the referee's own tests run")
    return pytest.main(["-q", *targets])


def full() -> int:
    if importlib.util.find_spec("engine.api") is None:
        print("referee full: needs engine/api.py, defined in the interfaces (INTERFACES.md §3); S1 adds it",
              file=sys.stderr)
        return 2
    from referee.full import main  # written with the engine's API in S1
    return main()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fast"
    sys.exit({"fast": fast, "full": full}.get(mode, lambda: (print(__doc__), 2)[1])())
