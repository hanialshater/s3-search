"""RPC serving next to HTTP: length-prefixed JSON frames over TCP, standard library only.

Decision record: docs/decisions/2026-11-10-rpc-and-operator-watches.md. The RPC is the HTTP API without HTTP: a frame is
a 4-byte big-endian length and a UTF-8 JSON object, and a request names a method and a path exactly as the HTTP route does,
so **the same handler (``App.handle``) answers both** and the responses are the same by construction.

    request   {"id": 7, "method": "POST", "path": "/indexes/shop/_search", "body": {...}, "headers": {...}}
    response  {"id": 7, "status": 200, "body": {...}, "headers": {"X-S3-Cache": "miss", ...}}

``body`` is the parsed JSON (a string for ``/metrics``). A connection serves requests one after another; open several
connections (``RpcPool``) for concurrency, which is what the batching scheduler groups. ``RpcClient`` is the Python client.
"""
from __future__ import annotations

import json
import socket
import socketserver
import struct
import threading
from typing import Any

from .app import App

MAX_FRAME = 64 * 1024 * 1024
_LEN = struct.Struct(">I")


class RpcError(Exception):
    """A non-2xx RPC response: ``status`` and the named error body."""

    def __init__(self, status: int, body: Any):
        err = body.get("error", {}) if isinstance(body, dict) else {}
        super().__init__(f"{status} {err.get('type', '')}: {err.get('reason', body)}")
        self.status, self.body = status, body
        self.type, self.path, self.reason = err.get("type"), err.get("path"), err.get("reason")


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else bytes(buf)  # a short read mid-frame is returned short and fails to parse
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket, max_frame: int = MAX_FRAME) -> dict | None:
    """The next frame as a dict, or None on a clean close. Raises ValueError for a bad frame."""
    head = _read_exact(sock, 4)
    if head is None:
        return None
    if len(head) < 4:
        raise ValueError("connection closed inside a frame header")
    (n,) = _LEN.unpack(head)
    if n > max_frame:
        raise ValueError(f"frame of {n} bytes is over the {max_frame} limit")
    data = _read_exact(sock, n)
    if data is None or len(data) < n:
        raise ValueError("connection closed inside a frame")
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("a frame is a JSON object")
    return obj


def write_frame(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(_LEN.pack(len(data)) + data)


def _error(rid, status: int, type_: str, reason: str) -> dict:
    return {"id": rid, "status": status, "body": {"error": {"type": type_, "path": "", "reason": reason}}, "headers": {}}


def handle_frame(app: App, req: dict) -> dict:
    rid = req.get("id")
    method, path = req.get("method", "POST"), req.get("path")
    if not isinstance(method, str) or not isinstance(path, str):
        return _error(rid, 400, "invalid_request", 'a request needs a string "path" (and "method", default POST)')
    body = req.get("body")
    raw = b"" if body is None else json.dumps(body).encode()
    headers = dict(req.get("headers") or {})
    if rid is not None:
        headers.setdefault("X-Request-Id", f"rpc-{rid}")
    resp = app.handle(method, path, raw, headers)
    out = resp.json() if resp.content_type == "application/json" else resp.body.decode()
    return {"id": rid, "status": resp.status, "body": out, "headers": resp.headers}


def make_rpc_server(app: App, host: str = "0.0.0.0", port: int = 0) -> socketserver.ThreadingTCPServer:
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            sock = self.request
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    req = read_frame(sock)
                except (ValueError, OSError) as exc:
                    try:
                        write_frame(sock, _error(None, 400, "invalid_request", f"bad frame: {exc}"))
                    except OSError:
                        pass
                    return
                if req is None:
                    return
                try:
                    write_frame(sock, handle_frame(app, req))
                except OSError:
                    return

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True
        request_queue_size = 256

    return Server((host, port), Handler)


class RpcClient:
    """One connection, safe to share between threads (calls take turns). Reconnects once if the connection was dropped."""

    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.addr, self.timeout = (host, port), timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._next = 0

    def _connect(self) -> socket.socket:
        s = socket.create_connection(self.addr, timeout=self.timeout)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def call(self, method: str, path: str, body: Any = None, headers: dict | None = None) -> tuple[int, Any, dict]:
        """``(status, body, headers)``; never raises for an HTTP-style error status."""
        with self._lock:
            self._next += 1
            frame = {"id": self._next, "method": method, "path": path, "body": body, "headers": headers or {}}
            for attempt in (0, 1):
                try:
                    if self._sock is None:
                        self._sock = self._connect()
                    write_frame(self._sock, frame)
                    resp = read_frame(self._sock)
                    if resp is None:
                        raise ConnectionError("server closed the connection")
                    return resp["status"], resp["body"], resp.get("headers", {})
                except (OSError, ValueError):
                    self.close_locked()
                    if attempt:
                        raise
            raise AssertionError("unreachable")

    def close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def close(self) -> None:
        with self._lock:
            self.close_locked()

    def __enter__(self) -> "RpcClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- the API, as methods; each raises RpcError on a non-200 -------------------------------------------------------
    def _ok(self, method: str, path: str, body: Any = None) -> Any:
        status, out, _ = self.call(method, path, body)
        if status != 200:
            raise RpcError(status, out)
        return out

    def search(self, index: str, request: dict) -> dict:
        return self._ok("POST", f"/indexes/{index}/_search", request)

    def msearch(self, index: str, requests: list[dict]) -> list[dict]:
        return self._ok("POST", f"/indexes/{index}/_msearch", requests)

    def mapping(self, index: str) -> dict:
        return self._ok("GET", f"/indexes/{index}/_mapping")

    def versions(self, index: str) -> list[dict]:
        return self._ok("GET", f"/indexes/{index}/versions")

    def docs(self, index: str, changes: list[dict]) -> dict:
        return self._ok("POST", f"/indexes/{index}/_docs", changes)

    def reindex(self, index: str) -> dict:
        return self._ok("POST", f"/indexes/{index}/_reindex", {})

    def rollback(self, index: str, segment_id: str) -> dict:
        return self._ok("POST", f"/indexes/{index}/_rollback", {"segment_id": segment_id})

    def ready(self) -> bool:
        return self.call("GET", "/readyz")[0] == 200
