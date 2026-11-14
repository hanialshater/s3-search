"""The GPU process's host path on native-front connections (S6-22s): where ``s3_hostpath`` plugs in.

``s3_hostpath`` is the architect's pyo3 module (Rust inside the GPU process). It is meant to parse and plan a batch of
search frames straight from their binproto payloads, and to write the batch's reply frames. This module is the seam:

    S3_HOSTPATH unset or 0   the default: the reader thread parses each frame (binproto.parse_request), as before
    S3_HOSTPATH=stub         PyStub below: the same contract in Python, so the plumbing is tested before the Rust lands
    S3_HOSTPATH=stub-prepared  PyStub with a layout: plain requests come back eligible (``requests[i]`` None), and the
                             GPU process answers them through ``search_prepared`` (the index's, or search_prepared_stub)
    S3_HOSTPATH=1            ``import s3_hostpath``; it must provide ``prepare`` and ``reply`` as PyStub does

The contract (agreed through the coordinator, 2026-11-14):

- ``prepare(payloads, index, out) -> Prepared``: the raw type-1 payloads of one batch (``u8 1, u64 id, u16 len, name,
  u32 len, request JSON``, ``serving/binproto.py``); ``index`` the engine index the batch reads (the module keeps its
  own ``s3_hostpath.Index`` per index version, built once, and falls back on a version it does not hold); ``out`` the
  caller's reusable pinned host tensor (one per GPU process; Rust never links CUDA), or None. ``Prepared.layout``
  describes what the module wrote into ``out`` (offsets and lengths of the row arrays, the msm, boost and facet
  tables, the eligible positions) and ``Prepared.plans`` per eligible request its from, size, aggs (name, field, size)
  and fused-group id; both None from the stub. ``Prepared.requests[i]`` is the request dict for payload i (the stub
  fills every one; a module may leave eligible ones None when ``plans`` covers them), or None when ``i`` is in
  ``errors`` or ``fallback``. ``errors`` holds ``(i, status, body_dict)`` for a request refused before the
  engine, with the same status and body ``serving.errors.classify`` gives (a body that is not JSON: 400 invalid_request,
  "body is not JSON: ..."; an empty body: "empty body"). ``fallback`` lists positions the module will not handle; the
  GPU process parses those in Python, unchanged.
- ``reply(msgs) -> bytes``: the reply frames for a list of the GPU process's reply messages, concatenated so the
  writer makes one ``sendall``. Each frame is byte-identical to ``binproto.columnar(...)`` or ``binproto.whole(...)``
  for the same message, took_ms masked; ``serving/tests/golden/binproto_frames.json`` holds fixed cases a module must
  reproduce. (The architect's engine side may add a page-level reply, from the page buffer and the facet buckets; the
  frames it writes are held to the same golden cases.)
- Eligible positions (the architect's 4c, 2026-11-14): when ``Prepared.layout`` is not None, ``requests[i]`` is None
  for each position in ``layout["eligible"]``, and the GPU process calls ``index.search_prepared(prep, hits=...,
  ids=...)``, which returns the responses for those positions, in that order and in today's response form; the other
  positions go through ``search_batch`` as before, and the answers are matched back by position. An index without
  ``search_prepared`` gets ``search_prepared_stub`` (the stub's own bodies through ``search_batch``). Empty and non-JSON
  bodies come back as ``fallback`` from the module (the Python path keeps the error text).
- Any exception from ``prepare`` falls that batch back to Python parsing, and from ``reply`` to the Python encoder;
  both are counted (``fallbacks``). No request is lost.

Development only; the default is off.
"""
from __future__ import annotations

import json
import os

from . import binproto
from .errors import BadJSON, classify


class Prepared:
    __slots__ = ("requests", "errors", "fallback", "layout", "plans")

    def __init__(self, requests, errors=(), fallback=(), layout=None, plans=None):
        self.requests, self.errors, self.fallback = list(requests), list(errors), list(fallback)
        self.layout, self.plans = layout, plans


def parse_body(body: bytes):
    """The request dict, or raises BadJSON with the Python front's messages (``serving/front.py`` ``_parse``)."""
    if not body or not body.strip():
        raise BadJSON("empty body")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise BadJSON(f"body is not JSON: {exc}") from None


def reply_frame(msg) -> bytes:
    """One reply message as binproto frames (what ``_BinConn`` wrote before the seam)."""
    from .encode import encode
    if msg[0] == "multi":
        return b"".join(reply_frame(m) for m in msg[1])
    if msg[0] == "search" and msg[2] == "ok":
        _, id_, _, resp, version, queue_ms, *rest = msg
        meta = {"version": f"{version[0]}:{version[1]}", "queue_ms": round(float(queue_ms), 3),
                "batch": rest[0] if rest else 1}
        return binproto.columnar(id_, resp, meta) or binproto.whole(id_, 200, encode(resp), meta)
    if msg[0] == "search":                                  # ("search", id, "error", status, body)
        return binproto.whole(msg[1], msg[3], json.dumps(msg[4], separators=(",", ":")).encode())
    _, id_, status, body, ctype, headers = msg               # ("call", ...)
    return binproto.whole(id_, status, body, {"type": ctype})


_OWN = ("explain", "highlight", "_source", "profile")          # never eligible in the stub


class PyStub:
    """The contract in Python: what ``s3_hostpath`` replaces. ``prepared=True`` also returns a layout, with every plain
    request eligible, so the GPU process's search_prepared path runs before the Rust lands."""
    name = "python-stub"

    def __init__(self, prepared: bool = False):
        self.prepared = prepared

    def prepare(self, payloads, index=None, out=None) -> Prepared:
        requests, errors = [], []
        for i, p in enumerate(payloads):
            try:
                requests.append(parse_body(binproto.search_body(p)))
            except BadJSON as exc:
                requests.append(None)
                status, body = classify(exc)
                errors.append((i, status, body))
        if not self.prepared:
            return Prepared(requests, errors)
        refused = {i for i, _, _ in errors}
        eligible = [i for i, r in enumerate(requests) if i not in refused and isinstance(r, dict)
                    and not any(r.get(k) for k in _OWN)]
        layout = {"eligible": eligible, "stub_bodies": [requests[i] for i in eligible]}
        for i in eligible:
            requests[i] = None
        return Prepared(requests, errors, layout=layout)

    def reply(self, msgs) -> bytes:
        return b"".join(reply_frame(m) for m in msgs)


class _Guarded:
    """Wraps a module: its exceptions fall back to Python and are counted."""

    def __init__(self, impl, name):
        self.impl, self.name, self.fallbacks, self._py = impl, name, 0, PyStub()

    def prepare(self, payloads, index=None, out=None) -> Prepared:
        try:
            return self.impl.prepare(payloads, index, out)
        except Exception:  # noqa: BLE001 - never lose a batch to the module
            self.fallbacks += 1
            return self._py.prepare(payloads, index, out)

    def reply(self, msgs) -> bytes:
        try:
            return self.impl.reply(msgs)
        except Exception:  # noqa: BLE001
            self.fallbacks += 1
            return self._py.reply(msgs)


def search_prepared_stub(index, prep: Prepared, **kw) -> list:
    """An index without ``search_prepared``: the stub layout's own request dicts through ``search_batch``."""
    out = index.search_batch(list(prep.layout["stub_bodies"]), **kw)
    return list(out)


def load():
    """None (the default), or the host path S3_HOSTPATH names, guarded."""
    mode = os.environ.get("S3_HOSTPATH", "0")
    if mode in ("", "0"):
        return None
    if mode == "stub":
        return _Guarded(PyStub(), "python-stub")
    if mode == "stub-prepared":
        return _Guarded(PyStub(prepared=True), "python-stub-prepared")
    if mode == "1":
        import s3_hostpath                                   # the architect's pyo3 module
        return _Guarded(s3_hostpath, getattr(s3_hostpath, "__name__", "s3_hostpath"))
    raise ValueError(f"S3_HOSTPATH={mode!r}: 0, stub, stub-prepared or 1")
