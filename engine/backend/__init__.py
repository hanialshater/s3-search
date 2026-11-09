"""The array backend every operator calls (docs/INTERFACES.md §0; decision docs/decisions/2026-11-09-gpu-portable-backend.md).

Operators receive a ``Backend`` and use only its methods plus the arrays' own arithmetic, comparison, boolean and
indexing operators. They never import numpy, torch, cupy or CUDA themselves.

Devices: ``"cpu"`` (numpy, the reference, always available), ``"torch-cpu"`` and ``"cuda"`` (torch, imported only when
asked for). Arrays are the device's native type (``numpy.ndarray`` or ``torch.Tensor``). Dtypes are named by string:
``"bool" "uint8" "int32" "int64" "float16" "float32"``.

Determinism rules every implementation keeps:
- scores are float32; reductions are deterministic on a device (sort-based segment sums, never float atomics);
- every ranking op breaks ties by index (doc id) ascending.

The implementations (``engine/backend/numpy_backend.py``, ``engine/backend/torch_backend.py``) land in S1.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, Sequence, runtime_checkable

Array = Any  # numpy.ndarray on "cpu", torch.Tensor on "torch-cpu" and "cuda"

DEVICES = ("cpu", "torch-cpu", "cuda")
ENV_DEVICE = "S3_SEARCH_DEVICE"
ENV_BUDGET = "S3_SEARCH_DEVICE_BUDGET"
ENV_FAST = "S3_SEARCH_FAST"


class DeviceUnavailable(RuntimeError):
    """The requested device cannot be used here (torch missing, no CUDA)."""


class OverBudget(MemoryError):
    """A load or a double-buffered swap would exceed ``Backend.memory_budget()``."""


@runtime_checkable
class Backend(Protocol):
    name: str      # the device string: "cpu", "torch-cpu" or "cuda"

    # ---- creation and transfer ----------------------------------------------------------------------------------
    def asarray(self, x: Any, dtype: str | None = None) -> Array:
        """Host data (numpy array, list) to a device array; no copy when already on the device with that dtype."""

    def to_numpy(self, a: Array) -> "numpy.ndarray":  # noqa: F821
        """A device array to a host numpy array (synchronises)."""

    def zeros(self, shape: int | Sequence[int], dtype: str = "float32") -> Array: ...
    def full(self, shape: int | Sequence[int], value: float, dtype: str = "float32") -> Array: ...
    def arange(self, n: int, dtype: str = "int64") -> Array: ...
    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array: ...
    def astype(self, a: Array, dtype: str) -> Array: ...
    def where(self, cond: Array, a: Array | float, b: Array | float) -> Array: ...

    # ---- sparse CSR (BM25F) -------------------------------------------------------------------------------------
    def csr_expand(self, ptr: Array, rows: Array, owners: Array) -> tuple[Array, Array]:
        """Expand CSR row ranges. ``rows`` are row ids (term ids), ``owners`` one owner (query id) per row.

        Returns ``(owner_of_entry int64[E], entry int64[E])``, where for each i the entries
        ``ptr[rows[i]] .. ptr[rows[i]+1]-1`` appear in order, tagged with ``owners[i]``. E is the total length.
        """

    def segment_sum(self, keys: Array, values: Array, n: int) -> Array:
        """``out[k] = sum(values[keys == k])`` for k in ``range(n)``; float32; deterministic order of summation
        (by entry index) on every device."""

    # ---- dense ----------------------------------------------------------------------------------------------------
    def matmul(self, a: Array, b: Array) -> Array:
        """``a @ b`` accumulated in float32 (inputs may be float16)."""

    def topk(self, scores: Array, k: int, valid: Array | None = None) -> tuple[Array, Array]:
        """Per row of ``scores f32[Q,N]``: the k best indices by score descending, ties by index ascending.

        ``valid bool[Q,N]`` excludes entries. Returns ``(idx int32[Q,k], vals f32[Q,k])``; rows with fewer than k
        valid entries are padded with idx -1 and val -inf.
        """

    # ---- sort and pagination --------------------------------------------------------------------------------------
    def lexsort_rows(self, keys: Sequence[Array], descending: Sequence[bool], valid: Array | None = None) -> Array:
        """Per row, the stable order of indices by ``keys[0]``, then ``keys[1]``, ...; each key ``[Q,N]`` (or ``[N]``,
        broadcast), with its direction; ties after all keys by index ascending. Invalid entries go last.
        Returns ``int64[Q,N]``."""

    # ---- bitmaps (filters) ----------------------------------------------------------------------------------------
    def pack_bits(self, mask: Array) -> Array:
        """``bool[..., N]`` to ``uint32[..., ceil(N/32)]``; bit d is word ``d >> 5``, bit ``d & 31``, LSB first."""

    def unpack_bits(self, words: Array, n: int) -> Array:
        """The inverse of ``pack_bits``: ``uint32[..., W]`` to ``bool[..., n]``."""

    def isin(self, a: Array, values: Array) -> Array:
        """``bool`` mask of ``a``'s elements found in ``values`` (term/terms filters on codes)."""

    def count_nonzero(self, mask: Array, axis: int | None = None) -> Array: ...

    # ---- facets ---------------------------------------------------------------------------------------------------
    def bincount(self, codes: Array, n: int, weights: Array | None = None) -> Array:
        """``int64[n]`` counts of ``codes`` (or float32 sums of ``weights``)."""

    def bincount_rows(self, codes: Array, mask: Array, n: int) -> Array:
        """Per row of ``mask bool[Q,N]``: counts of ``codes int32[N]`` over the masked docs. Returns ``int64[Q,n]``."""

    # ---- memory and sync ------------------------------------------------------------------------------------------
    def nbytes(self, a: Array) -> int: ...

    def memory_budget(self) -> int:
        """Bytes the index may use on this device: ``S3_SEARCH_DEVICE_BUDGET`` if set, else 80% of free CUDA memory
        on "cuda" and 50% of available RAM on CPU devices."""

    def synchronize(self) -> None:
        """Wait for queued device work (a no-op on CPU); timings call it before reading the clock."""


def available_devices() -> list[str]:
    """The devices usable here, ``"cpu"`` first. Imports torch only to probe it."""
    out = ["cpu"]
    try:
        import torch  # noqa: F401  (optional: requirements-gpu.txt)
    except ImportError:
        return out
    out.append("torch-cpu")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def resolve_device(device: str | None = None) -> str:
    """The device string to use: the argument, else ``S3_SEARCH_DEVICE``, else ``"cpu"``. Raises on unknown or
    unavailable devices."""
    device = device or os.environ.get(ENV_DEVICE) or "cpu"
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")
    if device not in available_devices():
        raise DeviceUnavailable(device)
    return device


def get_backend(device: str | None = None) -> Backend:
    """The backend for a device. The implementations land with S1 (S1-2 to S1-4)."""
    device = resolve_device(device)
    raise NotImplementedError(f"backend {device!r}: lands in S1 (engine/backend/numpy_backend.py, torch_backend.py)")
