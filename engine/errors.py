"""The engine's named errors, all in one module (docs/INTERFACES.md §5). Re-exported by ``engine.api``; also by
``engine.backend`` (DeviceUnavailable, OverBudget) and ``engine.segment`` (CorruptSegment), which define none of their
own. Importing this module imports nothing else."""
from __future__ import annotations


class UnsupportedQuery(ValueError):
    """A request outside the supported DSL subset. ``path`` is the JSON path of the offending key."""

    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path, self.reason = path, reason


class InvalidRequest(UnsupportedQuery):
    """A malformed request inside the subset (bad types, from + size over 10,000, unknown fields)."""


class InvalidMapping(ValueError):
    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path, self.reason = path, reason


class NotHot(ValueError):
    """A patch to a field that is not declared hot (``_meta.engine.hot``)."""


class NoSuchKey(KeyError):
    """``Index.patch`` or ``Index.delete`` named an ``_id`` the segment never held. A ``KeyError``."""


class DeviceUnavailable(RuntimeError):
    """The requested device cannot be used here (torch missing, no CUDA)."""


class OverBudget(MemoryError):
    """A load or a double-buffered swap would exceed ``Backend.memory_budget()``."""


class CorruptSegment(IOError):
    """A segment file is missing, or its sha256 or the manifest checksum does not match."""


__all__ = ["UnsupportedQuery", "InvalidRequest", "InvalidMapping", "NotHot", "NoSuchKey", "DeviceUnavailable",
           "OverBudget", "CorruptSegment"]
