"""s3_hostpath: R1's host path for the GPU process (the architect's side of serving/hostpath.py's seam).

- ``HostIndex`` (Rust, ``_native``) parses and plans a batch of request bodies for the fused lexical path
  (engine/hostpath.py builds it per index state; engine/tests/test_hostpath.py holds it to the Python planner).
- ``prepare(payloads, index, out)`` and ``reply(msgs)`` are the seam's contract (serving/hostpath.py, PR #32):
  - ``prepare`` takes the raw binproto search payloads of one batch for one index. It returns a
    ``serving.hostpath.Prepared`` whose ``layout`` is the HostIndex output (``engine.api.Index.search_prepared``
    takes it) and whose ``plans`` are its chunks.
  - Each eligible position has ``requests[i] = None``. ``fallback`` lists the rest (Python parses them, and
    classifies empty or non-JSON bodies itself).
  - The HostIndex for an index state is built in a background thread the first time it is asked for. Until it is
    ready, every position falls back.
  - ``reply`` writes the kind 2 and 3 frames with the Python encoder.
- ``batch4(msgs)`` (R3c) writes an S3B3 connection's kind 4 frame for one batch's replies in Rust: the same bytes as
  ``serving.binproto.batch4``, held to the platform's golden4 frames. ``S3_HOSTPATH_BATCH4=0`` uses the Python writer.
"""
from __future__ import annotations

import os
import threading

from ._native import HostIndex, analyze, stem, token, words_of
from ._native import batch4 as _batch4

__all__ = ["HostIndex", "analyze", "batch4", "prepare", "reply", "stats", "stem", "token", "words_of"]

_lock = threading.Lock()
_built: dict = {}            # id(IndexState) -> (state, HostIndex), the state kept so its id is not reused
_building: set = set()
stats = {"batches": 0, "served": 0, "fallback": 0, "not_ready": 0}


def _host_index(index):
    st = index._st
    key = id(st)
    with _lock:
        got = _built.get(key)
        if got is not None and got[0] is st:
            return got[1]
        if key in _building:
            return None
        _building.add(key)

    def build():
        from engine import hostpath
        try:
            hi = hostpath.host_index(st)
            with _lock:
                for k in [k for k, (s, _) in _built.items() if s is not st]:
                    del _built[k]                         # an older state of this process's indexes
                _built[key] = (st, hi)
        finally:
            with _lock:
                _building.discard(key)
    threading.Thread(target=build, name="s3_hostpath-build", daemon=True).start()
    return None


def prepare(payloads, index, out=None):
    from serving import binproto
    from serving.hostpath import Prepared
    n = len(payloads)
    stats["batches"] += 1
    hi = _host_index(index) if index is not None else None
    if hi is None:
        stats["not_ready"] += 1
        stats["fallback"] += n
        return Prepared([None] * n, (), list(range(n)))
    bodies = [bytes(binproto.search_body(p)) for p in payloads]
    raw = hi.prepare(bodies)
    stats["served"] += n - len(raw["fallback"])
    stats["fallback"] += len(raw["fallback"])
    return Prepared([None] * n, (), raw["fallback"], layout=raw, plans=raw["chunks"])


def reply(msgs) -> bytes:
    from serving.hostpath import reply_frame
    return b"".join(reply_frame(m) for m in msgs)


_CTX: list = []


def _ctx():
    if not _CTX:
        import numpy as np
        from serving.encode import encode
        _CTX.append((np.ndarray, np.dtype(np.float32), np.dtype(np.float64), np.asarray, np.dtype(np.int64), encode, round))
    return _CTX[0]


def batch4(msgs) -> bytes:
    """``serving.binproto.batch4(msgs)``'s bytes, written in Rust (``S3_HOSTPATH_BATCH4=0``: the Python writer)."""
    if os.environ.get("S3_HOSTPATH_BATCH4", "1") == "0":
        from serving import binproto
        return binproto.batch4(msgs)
    return _batch4(list(msgs), _ctx())
