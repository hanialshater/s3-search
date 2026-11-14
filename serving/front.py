"""A front process (S6-10s, docs/INTERFACES.md §7 "One GPU process"): HTTP, parsing and encoding, no ``Index``.

``FrontApp.handle(method, path, body, headers) -> Response`` is ``App.handle``'s interface, so ``make_server`` and the
RPC server put it on the wire unchanged. A search is parsed here, sent to the GPU process with the fewest requests in
flight (``serving/gpu.py``), and its columns are encoded here (``serving/encode.py``): the same bytes a single process
sends. Every other route is forwarded to GPU process 0 and answered with what it returns, after it has been applied.
With more than one GPU process the set is read-only, as N workers are (405 for writes and admin calls).

The cache (``cache_size`` entries) is keyed by the index's ``(segment_id, epoch)`` as GPU process 0 publishes it in the
state directory (``versions-0.json``, rewritten within 20 ms of any swap, patch or delete), so a write forwarded by
any front leaves no stale entry reachable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from urllib.parse import parse_qsl

from . import encode as encode_mod
from . import metrics as metrics_mod
from .app import _READ_ACTIONS, Response, _json, _search_json
from .cache import LRUCache, request_key
from .config import ServingConfig
from .errors import BadJSON, InvalidRequestBody, MethodNotAllowed, NotReady, ServingError, classify
from .gpu import GpuClient

log = logging.getLogger("serving.front")
_INDEX_ROUTE = re.compile(r"^/indexes/([^/]+)(?:/([^/]+))?$")


class FrontApp:
    def __init__(self, cfg: ServingConfig, sockets: list[str]):
        self.cfg = cfg
        self.metrics = metrics_mod.standard()
        self.gpus = [GpuClient(p) for p in sockets]
        self.load = None
        if cfg.worker_state and len(sockets) > 1:
            from .gpu import LoadTable
            try:
                self.load = LoadTable(cfg.worker_state, create=False)
            except OSError:
                self.load = None
        self.cache = LRUCache(cfg.cache_size)
        self.draining = False
        self._versions: dict = {}
        self._versions_mtime = None
        self._vlock = threading.Lock()

    def preload(self) -> None:                                  # the GPU processes load; a front holds nothing
        return None

    def close(self) -> None:
        return None

    # -- versions for the cache
    def _version(self, name: str):
        if not self.cfg.worker_state:
            return None
        path = os.path.join(self.cfg.worker_state, "versions-0.json")
        try:
            m = os.stat(path).st_mtime_ns
        except OSError:
            return None
        with self._vlock:
            if m != self._versions_mtime:
                try:
                    with open(path) as f:
                        self._versions = json.load(f)
                    self._versions_mtime = m
                except (OSError, ValueError):
                    return None
            v = self._versions.get(name)
        return tuple(v) if v else None

    # -- dispatch
    def handle(self, method: str, path: str, body=b"", headers: dict | None = None) -> Response:
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        rid = headers.get("x-request-id") or uuid.uuid4().hex[:16]
        route = "unknown"
        try:
            bare, _, qs = path.partition("?")
            if callable(body):
                reader = body
                body = b"".join(iter(lambda: reader(1 << 20), b""))
            route, resp = self._route(method.upper(), bare, path, body, headers)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a named error body
            status, obj = classify(exc)
            if status >= 500 and not isinstance(exc, ServingError):
                log.exception("request %s failed", rid)
            self.metrics.inc("s3_errors_total", type=obj["error"]["type"])
            resp = _json(status, obj)
        resp.headers["X-Request-Id"] = rid
        if self.cfg.worker_index is not None:
            resp.headers["X-S3-Worker"] = str(self.cfg.worker_index)
        self.metrics.inc("s3_http_requests_total", route=route, status=str(resp.status))
        return resp

    def _route(self, method: str, bare: str, full: str, body: bytes, headers: dict) -> tuple[str, Response]:
        if bare == "/healthz":
            if method != "GET":
                raise MethodNotAllowed(f"{bare} takes GET")
            return "healthz", _json(200, {"status": "ok"})
        if bare == "/metrics" and method == "GET":
            return "metrics", Response(200, self.metrics.render().encode(), "text/plain; version=0.0.4")
        if bare == "/readyz" and method == "GET":
            return "readyz", self._readyz()
        m = _INDEX_ROUTE.match(bare)
        action = m.group(2) if m else None
        if m and action in ("_search", "_msearch") and method == "POST":
            return action.lstrip("_"), (self._search(m.group(1), body, headers) if action == "_search"
                                        else self._msearch(m.group(1), body))
        if len(self.gpus) > 1 and m and not (method == "GET" or action in _READ_ACTIONS):
            raise MethodNotAllowed(f"this server runs {len(self.gpus)} GPU processes and serves reads only; writes "
                                   "and admin calls go to a server with one GPU process (S3_SERVING_GPU_PROCESSES=1)")
        return "forward", self._forward(method, full, body, headers)

    def _wait(self, fut, what: str):
        try:
            return fut.result(timeout=self.cfg.request_timeout_s)
        except FutureTimeout:
            raise NotReady(f"{what} timed out in the GPU process") from None

    def _forward(self, method: str, path: str, body: bytes, headers: dict) -> Response:
        msg = self._wait(self.gpus[0].call(method, path, body, headers), "the call")
        _, _, status, rbody, ctype, rheaders = msg
        return Response(status, rbody, ctype, {k: v for k, v in rheaders.items() if k not in ("X-Request-Id", "X-S3-Worker")})

    def _readyz(self) -> Response:
        ready, indexes = [], []
        for g in self.gpus:
            try:
                msg = self._wait(g.call("GET", "/readyz", b"", {}), "readyz")
                ready.append(msg[2] == 200)
                indexes = indexes or json.loads(msg[3]).get("indexes", [])
            except Exception:  # noqa: BLE001
                ready.append(False)
        ok = all(ready) and not self.draining
        return _json(200 if ok else 503, {"ready": ok, "draining": self.draining, "indexes": indexes,
                                          "fronts": self.cfg.fronts, "gpu_processes": len(self.gpus),
                                          "gpu_ready": sum(ready), "worker": self.cfg.worker_index})

    # -- search
    @staticmethod
    def _parse(body: bytes):
        if not body or not body.strip():
            raise BadJSON("empty body")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise BadJSON(f"body is not JSON: {exc}") from None

    def _pick(self) -> GpuClient:
        """The least loaded GPU process by the shared load table (what every front sees), this front's own in-flight
        count breaking ties; without the table, the own count alone."""
        if self.load is None:
            return min(self.gpus, key=lambda g: g.inflight)
        return min(enumerate(self.gpus), key=lambda jg: (self.load.get(jg[0]), jg[1].inflight))[1]

    def _submit(self, name: str, request: dict, route: int | None = None):
        """Send one search to a GPU process: the least loaded, or ``route`` (the ``X-S3-Route-Gpu`` request header, for
        diagnosis). The future's result carries which process answered (``fut.gpu``)."""
        with self._vlock:
            g = self.gpus[route] if route is not None and 0 <= route < len(self.gpus) else self._pick()
            g.inflight += 1

        def done(_f):
            with self._vlock:
                g.inflight -= 1
        try:
            fut = g.search(name, request)
        except Exception:
            done(None)
            raise
        fut.add_done_callback(done)
        fut.gpu = self.gpus.index(g)
        return fut

    def _one(self, name: str, request: dict, fut=None, route: int | None = None):
        """``(response, version, info)`` or raises the named error. ``info`` names the GPU process that answered."""
        t0 = time.perf_counter()
        fut = fut if fut is not None else self._submit(name, request, route)
        msg = self._wait(fut, "search")
        if msg[2] == "error":
            raise _Answered(msg[3], msg[4])
        resp, version, queue_ms = msg[3], msg[4], msg[5]
        return resp, version, {"cache": "miss", "queue_ms": queue_ms, "serve_ms": (time.perf_counter() - t0) * 1000.0,
                               "gpu": getattr(fut, "gpu", 0), "batch": msg[6] if len(msg) > 6 else 1,
                               "batch_ms": msg[7] if len(msg) > 7 else None}

    def _search(self, name: str, body: bytes, headers: dict | None = None) -> Response:
        request = self._parse(body)
        if not isinstance(request, dict):
            raise InvalidRequestBody("a search request is a JSON object", path="")
        t0 = time.perf_counter()
        cacheable = self.cache.capacity and request.get("_source") in (None, False)
        v = self._version(name) if cacheable else None
        if v is not None:
            hit = self.cache.get((name, v, request_key(request)))
            if hit is not None:
                return _search_json(hit, {"X-S3-Cache": "hit", "X-S3-Queue-Ms": "0.000",
                                          "X-S3-Serve-Ms": f"{(time.perf_counter() - t0) * 1000.0:.3f}"})
        route = None
        if headers and headers.get("x-s3-route-gpu", "").strip().isdigit():
            route = int(headers["x-s3-route-gpu"])
        try:
            resp, version, info = self._one(name, request, route=route)
        except _Answered as a:
            return _json(a.status, a.body)
        if cacheable:
            self.cache.put((name, tuple(version), request_key(request)), resp)
        hdr = {"X-S3-Cache": info["cache"], "X-S3-Queue-Ms": f"{info['queue_ms']:.3f}",
               "X-S3-Serve-Ms": f"{info['serve_ms']:.3f}", "X-S3-Gpu-Process": str(info["gpu"]),
               "X-S3-Index-Version": f"{version[0]}:{version[1]}", "X-S3-Batch-Size": str(info["batch"])}
        if info.get("batch_ms") is not None:                    # S6-22s: the GPU process's search_batch wall time
            hdr["X-S3-Batch-Ms"] = f"{info['batch_ms']:.3f}"
        return _search_json(resp, hdr)

    def _msearch(self, name: str, body: bytes) -> Response:
        requests = self._parse(body)
        if not isinstance(requests, list) or not all(isinstance(r, dict) for r in requests):
            raise InvalidRequestBody("_msearch takes a JSON list of search requests", path="")
        futs = [self._submit(name, r) for r in requests]          # they reach the GPU together: one batch
        out = []
        for r, f in zip(requests, futs):
            try:
                out.append(self._one(name, r, f)[0])
            except _Answered as a:                            # as a single process: the request's error fails it
                return _json(a.status, a.body)
        return _search_json(out)


class _Answered(Exception):
    def __init__(self, status: int, body: dict):
        super().__init__(status)
        self.status, self.body = status, body
