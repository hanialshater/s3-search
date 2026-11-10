"""The data plane's three calls into the engine (docs/INTERFACES.md §3, §4), behind one seam.

- ``build_segment(catalogue, mapping) -> SegmentArrays`` (``engine.build``)
- ``write_segment(seg, path) -> Path`` (``engine.segment``)
- ``index_from_segment(path, device) -> Index`` (``engine.api.Index.from_segment``; the updater only)

The engine is imported lazily, on first use. ``use(...)`` installs other hooks with the same signatures (for example
to instrument a call in a test).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class EngineHooks:
    build_segment: Callable[[Any, dict], Any]
    write_segment: Callable[[Any, Any], Any]
    index_from_segment: Callable[[Any, str], Any]


_override: EngineHooks | None = None


def hooks() -> EngineHooks:
    if _override is not None:
        return _override
    from engine.build import build_segment
    from engine.segment import write_segment

    def index_from_segment(path, device="cpu"):
        from engine.api import Index
        return Index.from_segment(path, device=device)

    return EngineHooks(build_segment, write_segment, index_from_segment)


@contextlib.contextmanager
def use(h: EngineHooks) -> Iterator[EngineHooks]:
    global _override
    prev, _override = _override, h
    try:
        yield h
    finally:
        _override = prev
