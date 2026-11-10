"""The HTTP front (docs/INTERFACES.md §7), on the standard library only.

``create_app(config, ports) -> App``. ``App.handle(method, path, body, headers) -> Response`` is the whole API and is
what the tests call; ``serve(app, host, port)`` puts it behind a ``ThreadingHTTPServer`` (one thread per connection,
HTTP/1.1 keep-alive). Concurrent connections are what the batching scheduler groups.

Routes::

    POST /indexes/{name}/_search     §5 request -> §5 response   (``_source`` in the request adds the stored doc)
    POST /indexes/{name}/_msearch    [request, ...] -> [response, ...]        (one Index.search_batch)
    PUT  /indexes/{name}             {"mapping", "source": {"path"}} -> {"name", "version", "segment_id"}
    POST /indexes/{name}/_docs       [{"op", "_id", "doc"|"fields"}, ...] -> {"first_offset", "last_offset", ...}
    POST /indexes/{name}/_reindex    {} -> {"segment_id", "version"}
    GET  /indexes/{name}/versions    -> [{"segment_id", "version", "created", "live"}]
    POST /indexes/{name}/_rankers    {"name", "path"} -> {"name", "rankers"}    (a ranker JSON under the data dir, for rescore)
    GET  /indexes/{name}/_mapping    {name: {"mappings": ..., "s3_fields": {field: capabilities}}}
    POST /indexes/{name}/_rollback   {"segment_id"} -> {"segment_id", "version"}
    GET  /indexes, /indexes/{name}   what is loaded
    GET  /healthz /readyz /metrics   liveness, readiness, Prometheus text

Response headers on a search: ``X-Request-Id`` (echoed or generated), ``X-S3-Cache`` (hit|miss), ``X-S3-Queue-Ms``
(time in the scheduler) and ``X-S3-Serve-Ms`` (serving time). The engine's ``took_ms`` is passed through unchanged.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import metrics as metrics_mod
from .config import ServingConfig
from .errors import (BadJSON, InvalidRequestBody, MethodNotAllowed, NotFound, NotReady, ServingError, classify)
from .manager import IndexManager
from .ports import Ports, default_ports

log = logging.getLogger("serving")

_INDEX_ROUTE = re.compile(r"^/indexes/([^/]+)(?:/([^/]+))?$")
_ACTIONS = {"_search": "POST", "_msearch": "POST", "_docs": "POST", "_reindex": "POST", "_rollback": "POST", "_rankers": "POST", "_mapping": "GET",
            "versions": "GET"}


@dataclass
class Response:
    status: int
    body: bytes
    content_type: str = "application/json"
    headers: dict = field(default_factory=dict)

    def json(self):
        return json.loads(self.body)


def _json(status: int, obj, headers: dict | None = None) -> Response:
    return Response(status, json.dumps(obj, separators=(",", ":")).encode(), headers=headers or {})


class App:
    def __init__(self, cfg: ServingConfig, ports: Ports | None = None):
        self.cfg = cfg
        self.metrics = metrics_mod.standard()
        self.manager = IndexManager(cfg, ports or default_ports(), self.metrics)
        self.draining = False

    # -- lifecycle --------------------------------------------------------------------------------------------------
    def preload(self) -> None:
        for name, seg in self.cfg.preload:
            self.manager.open_existing(name, seg)

    def close(self) -> None:
        self.manager.close()

    # -- dispatch ---------------------------------------------------------------------------------------------------
    def handle(self, method: str, path: str, body: bytes = b"", headers: dict | None = None) -> Response:
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        rid = headers.get("x-request-id") or uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        route = "unknown"
        try:
            path = path.split("?", 1)[0]
            route, resp = self._route(method.upper(), path, body)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a named error body
            status, obj = classify(exc)
            if status >= 500 and not isinstance(exc, ServingError):
                log.exception("request %s failed", rid)
            self.metrics.inc("s3_errors_total", type=obj["error"]["type"])
            resp = _json(status, obj)
        resp.headers["X-Request-Id"] = rid
        self.metrics.inc("s3_http_requests_total", route=route, status=str(resp.status))
        log.debug("req=%s %s %s -> %s %.1fms", rid, method, path, resp.status, (time.perf_counter() - t0) * 1000)
        return resp

    def _route(self, method: str, path: str, body: bytes) -> tuple[str, Response]:
        if path in ("/healthz", "/readyz", "/metrics", "/indexes"):
            if method != "GET":
                raise MethodNotAllowed(f"{path} takes GET")
            if path == "/healthz":
                return "healthz", _json(200, {"status": "ok"})
            if path == "/readyz":
                ok = self.manager.ready() and not self.draining
                return "readyz", _json(200 if ok else 503, {"ready": ok, "draining": self.draining,
                                                           "indexes": self.manager.names()})
            if path == "/metrics":
                return "metrics", Response(200, self.metrics.render().encode(), "text/plain; version=0.0.4")
            return "indexes", _json(200, [self.manager.get(n).info() for n in self.manager.names()])
        m = _INDEX_ROUTE.match(path)
        if not m:
            raise NotFound(f"no route {path}", path=path)
        name, action = m.group(1), m.group(2)
        if action is None:
            if method == "PUT":
                return "create", self._create(name, body)
            if method == "GET":
                return "index_info", _json(200, self.manager.get(name).info())
            raise MethodNotAllowed(f"/indexes/{{name}} takes PUT or GET")
        if action not in _ACTIONS:
            raise NotFound(f"no route {path}", path=path)
        if method != _ACTIONS[action]:
            raise MethodNotAllowed(f"{action} takes {_ACTIONS[action]}")
        mi = self.manager.get(name)
        return action.lstrip("_"), getattr(self, "_" + action.lstrip("_"))(mi, body)

    # -- handlers ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _parse(body: bytes, default=None):
        if not body or not body.strip():
            if default is not None:
                return default
            raise BadJSON("empty body")
        try:
            return json.loads(body)
        except ValueError as exc:
            raise BadJSON(f"body is not JSON: {exc}") from None

    def _search(self, mi, body: bytes) -> Response:
        request = self._parse(body)
        if not isinstance(request, dict):
            raise InvalidRequestBody("a search request is a JSON object", path="")
        src = request.get("_source")
        try:
            resp, info = mi.search(request)
        except FutureTimeout:
            raise NotReady("search timed out in the scheduler") from None
        except Exception:
            mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="error")
            raise
        mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="ok")
        if src not in (None, False):
            mi.attach_source(resp, src)
        return _json(200, resp, {"X-S3-Cache": info["cache"], "X-S3-Queue-Ms": f"{info['queue_ms']:.3f}",
                                 "X-S3-Serve-Ms": f"{info['serve_ms']:.3f}"})

    def _msearch(self, mi, body: bytes) -> Response:
        requests = self._parse(body)
        if not isinstance(requests, list) or not all(isinstance(r, dict) for r in requests):
            raise InvalidRequestBody("_msearch takes a JSON list of search requests", path="")
        try:
            out = mi.msearch(requests) if requests else []
        except Exception:
            mi.metrics.inc("s3_search_requests_total", len(requests), index=mi.name, outcome="error")
            raise
        mi.metrics.inc("s3_search_requests_total", len(requests), index=mi.name, outcome="ok")
        for r, resp in zip(requests, out):
            if r.get("_source") not in (None, False):
                mi.attach_source(resp, r["_source"])
        return _json(200, out)

    def _create(self, name: str, body: bytes) -> Response:
        spec = self._parse(body)
        if not isinstance(spec, dict) or "mapping" not in spec:
            raise InvalidRequestBody('body is {"mapping": {...}, "source": {"path": "..."}}', path="mapping")
        return _json(200, self.manager.create(name, spec["mapping"], spec.get("source")))

    def _docs(self, mi, body: bytes) -> Response:
        return _json(200, mi.add_docs(self._parse(body)))

    def _reindex(self, mi, body: bytes) -> Response:
        return _json(200, mi.reindex())

    def _rankers(self, mi, body: bytes) -> Response:
        spec = self._parse(body)
        if not isinstance(spec, dict):
            raise InvalidRequestBody('body is {"name": "...", "path": "..."}', path="")
        return _json(200, mi.add_ranker(spec.get("name"), spec.get("path")))

    def _mapping(self, mi, body: bytes) -> Response:
        return _json(200, {mi.name: {"mappings": mi.mapping, "s3_fields": mi.field_capabilities()}})

    def _versions(self, mi, body: bytes) -> Response:
        return _json(200, mi.versions())

    def _rollback(self, mi, body: bytes) -> Response:
        spec = self._parse(body)
        sid = spec.get("segment_id") if isinstance(spec, dict) else None
        if not isinstance(sid, str):
            raise InvalidRequestBody('body is {"segment_id": "..."}', path="segment_id")
        return _json(200, mi.rollback(sid))


def create_app(config: ServingConfig | None = None, ports: Ports | None = None) -> App:
    return App(config or ServingConfig.from_env(), ports)


# ---- the standard-library server --------------------------------------------------------------------------------------
def make_server(app: App, host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "s3-serving"

        def _do(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            if n > app.cfg.max_body_bytes:
                resp = _json(413, {"error": {"type": "invalid_request", "path": "", "reason": "body too large"}})
            else:
                resp = app.handle(self.command, self.path, self.rfile.read(n) if n else b"", dict(self.headers))
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(resp.body)))
            for k, v in resp.headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.body)

        do_GET = do_POST = do_PUT = do_DELETE = _do

        def log_message(self, fmt, *args) -> None:  # request logging goes through ``App.handle``
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 256

    return Server((host if host is not None else app.cfg.host, app.cfg.port if port is None else port), Handler)


def serve_forever(app: App, server: ThreadingHTTPServer) -> None:
    """Run until SIGTERM/SIGINT; on the signal, fail readiness, wait ``drain_seconds`` for the endpoint to be removed
    from the Service, then stop. This is what keeps a rolling update from dropping in-flight queries."""
    import signal

    def stop(signum, frame):  # noqa: ARG001
        def _shutdown():
            app.draining = True
            time.sleep(app.cfg.drain_seconds)
            server.shutdown()
        threading.Thread(target=_shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server.serve_forever()
    server.server_close()
    app.close()
