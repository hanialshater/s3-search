"""Serve the demo page locally, next to a serving API. Local only: it binds 127.0.0.1 and nothing here deploys anything.

    python -m tools.demo --api http://127.0.0.1:8080            # page on :8000, /indexes/* proxied to the API
    python -m tools.demo --serving-cmd "python -m serving ..."   # start the platform engineer's server first
    python -m tools.demo --real [--products N] [--fields]        # start a real serving replica over ESCI-sample (tools/stack.py)

The page calls ``/indexes/{name}/_search`` on its own origin; this server forwards those calls to the API, so the page
needs no CORS. Open ``http://127.0.0.1:8000/?index=NAME``.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def make_server(api_url: str, port: int = 8000, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    api_url = api_url.rstrip("/")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _static(self):
            name = self.path.split("?")[0].lstrip("/") or "index.html"
            f = (DEMO_DIR / name).resolve()
            if DEMO_DIR not in f.parents or not f.is_file():
                self.send_error(404)
                return
            raw = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TYPES.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _proxy(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0))) if self.command == "POST" else None
            req = urllib.request.Request(api_url + self.path, data=body, method=self.command,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    code, raw = r.status, r.read()
            except urllib.error.HTTPError as e:
                code, raw = e.code, e.read()
            except OSError as e:
                code, raw = 502, ('{"error": {"type": "bad_gateway", "reason": "%s"}}' % str(e).replace('"', "'")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self._proxy() if self.path.startswith(("/indexes/", "/healthz", "/readyz")) else self._static()

        def do_POST(self):
            self._proxy() if self.path.startswith("/indexes/") else self.send_error(404)

    return ThreadingHTTPServer((host, port), H)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", default="http://127.0.0.1:8080", help="the serving API (INTERFACES §7)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--real", action="store_true", help="start a real serving replica over ESCI-sample instead of --api")
    ap.add_argument("--products", type=int, default=None, help="with --real: the first N products (default all)")
    ap.add_argument("--fields", action="store_true", help="with --real: add synthetic price, stock and category")
    ap.add_argument("--geo", action="store_true", help="with --real: add synthetic locations (geo_point) and nested offers")
    ap.add_argument("--serving-cmd", help="a command that starts the serving API; stopped when this exits")
    args = ap.parse_args(argv)

    api, proc, stack = args.api, None, None
    if args.real:
        from tools.stack import Stack
        stack = Stack(args.products, args.fields, args.geo)
        api = stack.base
        print(f"real serving replica at {api}")
    elif args.serving_cmd:
        proc = subprocess.Popen(shlex.split(args.serving_cmd))
    srv = make_server(api, args.port)
    print(f"demo at http://127.0.0.1:{args.port}/  (API {api}); Ctrl-C to stop. Local only: nothing is deployed.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        if proc:
            proc.terminate()
        if stack:
            stack.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
