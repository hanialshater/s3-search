"""S6-18: where the Triton kernels may run: Triton imports, its version is the one tested, and the GPU class passed.

Each kernel (``triton_bm25``, ``triton_topk``, ``triton_facets``) gives the torch path's results bit for bit, and its
tests check that in GPU CI. They can only check it on the GPU classes CI ran on, and with the Triton version installed
there. So a kernel runs only where all of these hold, and otherwise the torch path runs, with the same results:

- ``triton`` imports, and the backend is cuda;
- Triton's major.minor is ``TESTED_TRITON`` (``requirements-gpu.txt`` pins it with torch);
- the GPU's compute capability is in ``PASSED_CAPABILITIES``: the classes whose GPU CI ran and passed the kernel
  tests (``colab/gpu_ci.py``). A capability not listed falls back.

``S3_TRITON_CAPABILITIES`` overrides the list for development (for example "9.0" to try an H100 in GPU CI, or "any").
The per-kernel flags still decide whether a kernel is wanted at all: ``S3_TRITON_BM25`` and ``S3_TRITON_TOPK`` are on
unless "0" (S6-18c, the founder's decision); ``S3_TRITON_FACETS`` is off unless "1".
"""
from __future__ import annotations

import functools
import os

try:
    import triton
except Exception:  # noqa: BLE001
    triton = None

TESTED_TRITON = "3.8"
# (major, minor) compute capabilities whose GPU CI passed the kernel tests, with the run that showed it.
# The H100 (9.0) is not listed: Colab refused it on this account (no entitlement), so its GPU CI never ran.
PASSED_CAPABILITIES = {
    (8, 0): "A100-SXM4-40GB, colab/gpu_ci.py at 8f5cb64 (every kernel test run, none skipped)",
    (8, 9): "L4, colab/gpu_ci.py at e098a57 (before the 'kernel ran' checks; the deep top-k test did not run the kernel)",
    (12, 0): "RTX PRO 6000 Blackwell Server Edition (Colab G4), colab/gpu_ci.py at 8f5cb64 (every kernel test run)",
}


def _version_ok() -> bool:
    v = getattr(triton, "__version__", "")
    return v == TESTED_TRITON or v.startswith(TESTED_TRITON + ".")


@functools.lru_cache(maxsize=None)
def _capability(index: int) -> tuple[int, int]:
    import torch
    return tuple(torch.cuda.get_device_capability(index))


def _allowed(cap: tuple[int, int]) -> bool:
    env = os.environ.get("S3_TRITON_CAPABILITIES", "").strip()
    if env == "any":
        return True
    if env:
        return any(cap == tuple(int(x) for x in c.strip().split(".")) for c in env.split(",") if c.strip())
    return cap in PASSED_CAPABILITIES


def supported(b) -> bool:
    """True where a Triton kernel may run on backend ``b`` (the per-kernel flag still decides if it does)."""
    if triton is None or getattr(b, "name", "") != "cuda" or not _version_ok():
        return False
    dev = getattr(b, "device", None)
    index = getattr(dev, "index", None)
    if index is None:                                       # torch.device("cuda"): the current device
        import torch
        index = torch.cuda.current_device()
    return _allowed(_capability(index))


def effective(b) -> dict:
    """Which Triton kernels backend ``b`` would use now, for provenance (a score's record): the kernels on, the Triton
    version, and the GPU's capability (None off cuda)."""
    from . import triton_bm25, triton_facets, triton_topk
    cap = None
    if triton is not None and getattr(b, "name", "") == "cuda":
        dev = getattr(b, "device", None)
        index = getattr(dev, "index", None)
        if index is None:
            import torch
            index = torch.cuda.current_device()
        cap = list(_capability(index))
    return {"bm25": bool(triton_bm25.wanted(b)), "topk": bool(triton_topk.wanted(b, triton_topk.MIN_K)),
            "facets": bool(triton_facets.wanted(b)), "triton": getattr(triton, "__version__", None), "capability": cap}
