"""The array backend every operator calls (docs/INTERFACES.md §0; decision docs/decisions/2026-11-09-gpu-portable-backend.md).

Operators receive a ``Backend`` and use only its methods plus the arrays' own arithmetic, comparison, boolean and
indexing operators. They never import numpy, torch, cupy or CUDA themselves.

Devices: ``"cpu"`` (numpy, the reference, always available), ``"torch-cpu"`` and ``"cuda"`` (torch, imported only when
asked for). Arrays are the device's native type (``numpy.ndarray`` or ``torch.Tensor``). Dtypes are named by string:
``"bool" "uint8" "int32" "int64" "float16" "float32"``.

Determinism rules every implementation keeps:
- scores are float32; reductions are deterministic on a device (sort-based segment sums, never float atomics);
- every ranking op breaks ties by index (doc id) ascending.

The implementations: ``engine/backend/numpy_backend.py`` and ``engine/backend/torch_backend.py``. On torch devices a
bitmap lives as a bool mask; ``pack_bits``/``unpack_bits`` give the packed uint32 form for storage.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, Sequence, runtime_checkable

Array = Any  # numpy.ndarray on "cpu", torch.Tensor on "torch-cpu" and "cuda"

DEVICES = ("cpu", "torch-cpu", "cuda")
ENV_DEVICE = "S3_SEARCH_DEVICE"
ENV_BUDGET = "S3_SEARCH_DEVICE_BUDGET"
ENV_FAST = "S3_SEARCH_FAST"


from ..errors import DeviceUnavailable, OverBudget  # noqa: E402  (one home for the named errors)


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
    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array: ...
    def copy(self, a: Array) -> Array: ...

    # ---- elementwise, reductions, indexing (names differ between numpy and torch, so they are here)
    def log1p(self, a: Array) -> Array: ...
    def isnan(self, a: Array) -> Array: ...
    def sum(self, a: Array, axis: int | None = None) -> Array: ...
    def nonzero(self, mask: Array) -> tuple[Array, ...]:
        """Indices of true entries, one int64 array per axis, in row-major order."""
    def take_along(self, a: Array, idx: Array, axis: int = -1) -> Array: ...
    def searchsorted(self, sorted_a: Array, v: Array, right: bool = False) -> Array: ...
    def unique(self, a: Array) -> Array:
        """Sorted distinct values, on the device."""
    def argsort(self, a: Array, descending: bool = False) -> Array:
        """Stable argsort along the last axis (equal keys keep index order)."""

    # ---- sparse CSR (BM25F) -------------------------------------------------------------------------------------
    def csr_expand(self, ptr: Array, rows: Array, owners: Array) -> tuple[Array, Array]:
        """Expand CSR row ranges. ``rows`` are row ids (term ids), ``owners`` one owner (query id) per row.

        Returns ``(owner_of_entry int64[E], entry int64[E])``, where for each i the entries
        ``ptr[rows[i]] .. ptr[rows[i]+1]-1`` appear in order, tagged with ``owners[i]``. E is the total length.
        """

    def segment_sum(self, keys: Array, values: Array, n: int) -> Array:
        """``out[k] = sum(values[keys == k])`` for k in ``range(n)``; float32; deterministic order of summation
        (by entry index) on every device."""

    def segment_count(self, keys: Array, n: int) -> Array:
        """``int32[n]`` counts of ``keys``."""

    # ---- dense ----------------------------------------------------------------------------------------------------
    def matmul(self, a: Array, b: Array) -> Array:
        """``a @ b`` as float32, accumulated in float64 (so a row's result does not depend on the batch it is in);
        chunked over b's columns. A float32 GEMM is an S6 fast path."""

    def topk(self, scores: Array, k: int, valid: Array | None = None) -> tuple[Array, Array]:
        """Per row of ``scores f32[Q,N]``: the k best indices by score descending, ties by index ascending.

        ``valid bool[Q,N]`` excludes entries. Returns ``(idx int32[Q,k], vals f32[Q,k])``; rows with fewer than k
        valid entries are padded with idx -1 and val -inf.
        """

    # Multi-key sorts (pagination) are composed by the operators from stable ``argsort`` passes, last key first.

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


_CACHE: dict = {}
ENV_ACCUM = "S3_SEARCH_ACCUM"


def get_backend(device: str | None = None, accumulate: str | None = None) -> Backend:
    """The backend for a device (one instance per device and accumulation).

    ``accumulate``: ``"float64"`` (the reference default: BM25F sums and the dense matmul accumulate in float64 and
    round to float32, so results do not depend on the batch) or ``"float32"`` (an option for GPUs with slow FP64:
    BM25F stays deterministic and batch-invariant; the dense float32 GEMM may not be batch-invariant). ``None`` reads
    ``S3_SEARCH_ACCUM``, else float64."""
    device = resolve_device(device)
    accumulate = accumulate or os.environ.get(ENV_ACCUM) or "float64"
    key = (device, accumulate)
    if key not in _CACHE:
        if device == "cpu":
            from .numpy_backend import NumpyBackend
            _CACHE[key] = NumpyBackend(accumulate)
        else:
            from .torch_backend import TorchBackend
            _CACHE[key] = TorchBackend(device, accumulate)
    return _CACHE[key]
