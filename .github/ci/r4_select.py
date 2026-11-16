"""A pytest plugin for the rust workflow (R4, S6-41): keep only the cases that run the real Rust Loop or Rust host path,
so they are run with no skip allowed: the cases parametrized with r4="1" (or "1:inline") or hostpath="1".

    python -m pytest -p r4_select serving/tests/test_gpu_process.py     (with .github/ci on PYTHONPATH)
"""


def pytest_collection_modifyitems(config, items):
    keep, drop = [], []
    for it in items:
        params = getattr(getattr(it, "callspec", None), "params", {})
        rust = str(params.get("r4", "")).partition(":")[0] == "1" or params.get("hostpath") == "1"
        (keep if rust else drop).append(it)
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
