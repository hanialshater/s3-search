"""A front on granian (S6-21s): the same ``FrontApp`` behind granian's Rust HTTP server, through WSGI.

    S3_SERVING_FRONT_SERVER=granian python -m serving --fronts F --gpu-processes K ...

``serving/front.py``'s ``FrontApp`` does the work exactly as before: parsing the body, the cache, sending the search to
a GPU process and encoding the columns. So the response bodies are the same bytes. What changes is the HTTP layer.
Instead of Python's ``BaseHTTPRequestHandler``, with one thread per connection, granian (BSD-3-Clause, a Rust server
built on hyper) parses HTTP and runs the app on its blocking threads; only ``FrontApp.handle`` holds the GIL.

- **Processes:** granian runs F worker processes on the port (SO_REUSEPORT), in place of serving's own F fronts on a
  shared socket. Each worker builds one ``FrontApp`` on first use.
- **TCP_NODELAY:** granian sets it on its listening socket (granian v2.8.8 ``src/net.rs:89``), and Linux passes it to
  every accepted connection; the Nagle check covers it.
- **Headers:** granian writes its own ``server`` and ``date`` headers; serving's ``X-S3-*`` headers are passed through.
- **Fallback:** ``S3_SERVING_FRONT_SERVER=stdlib`` (the default) keeps the stdlib front.
"""
from __future__ import annotations

import os
import threading
from http import HTTPStatus

_APP = None
_LOCK = threading.Lock()


def _front():
    global _APP
    if _APP is None:
        with _LOCK:
            if _APP is None:
                from .config import ServingConfig
                from .front import FrontApp
                cfg = ServingConfig.from_env()
                cfg.worker_index = int(os.environ.get("S3_SERVING_FRONT_INDEX", "0"))
                _APP = FrontApp(cfg, os.environ["S3_SERVING_GPU_SOCKETS"].split(","))
    return _APP


def app(environ, start_response):
    """WSGI: one request through ``FrontApp.handle``."""
    front = _front()
    n = int(environ.get("CONTENT_LENGTH") or 0)
    body = environ["wsgi.input"].read(n) if n else b""
    path = environ.get("PATH_INFO", "/")
    if environ.get("QUERY_STRING"):
        path += "?" + environ["QUERY_STRING"]
    headers = {k[5:].replace("_", "-").lower(): v for k, v in environ.items() if k.startswith("HTTP_")}
    if environ.get("CONTENT_TYPE"):
        headers["content-type"] = environ["CONTENT_TYPE"]
    r = front.handle(environ["REQUEST_METHOD"], path, body, headers)
    try:
        reason = HTTPStatus(r.status).phrase
    except ValueError:
        reason = ""
    out = [("Content-Type", r.content_type), ("Content-Length", str(len(r.body)))]
    out += [(k, str(v)) for k, v in r.headers.items()]
    start_response(f"{r.status} {reason}", out)
    return [r.body]
