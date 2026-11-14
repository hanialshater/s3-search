"""S6-18: where the Triton kernels may run: Triton imports, its version is the one tested, and the GPU class passed.

Each kernel (``triton_bm25``, ``triton_topk``, ``triton_facets``) gives the torch path's results bit for bit, and its
tests check that in GPU CI. They can only check it on the GPU classes CI ran on, and with the Triton version installed
there. So a kernel runs only where all of these hold, and otherwise the torch path runs, with the same results:

- ``triton`` imports, and the backend is cuda;
- Triton's major.minor is ``TESTED_TRITON`` (``requirements-gpu.txt`` pins it with torch);
- the GPU's compute capability is in ``PASSED_CAPABILITIES``: the classes whose GPU CI ran and passed the kernel
  tests (``colab/gpu_ci.py``). A capability not listed falls back.

``S3_TRITON_CAPABILITIES`` overrides the list for development (for example "9.0" to try an H100 in GPU CI, or "any").
The per-kernel flags (``S3_TRITON_BM25`` and the rest) still decide whether a kernel is wanted at all.
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
PASSED_CAPABILITIES = {
    (8, 0): "A100, colab/gpu_ci.py at 06634b1 (BM25F sum); the top-k and facets pending",
    (8, 9): "L4, colab/gpu_ci.py at e098a57",
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
