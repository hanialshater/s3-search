"""zvec 0.8.1 behind a minimal HTTP server, for the matched load test (S6-18s). Development only.

zvec is an in-process library, so it enters an HTTP load test only through a server someone writes. This one reuses
the engine's own serving stack, so both sides pay the same HTTP cost (the steward's ruling,
lab/benchmarks/fairness-rulings.md):

- the same HTTP server (``serving.app.make_server``: stdlib, one thread per connection, HTTP/1.1 keep-alive);
- the same encoder (``serving.encode``: orjson for the numbers, one bytes template), with the same response shape as
  the engine's searches (``{"hits": {"total": n, "hits": [{"_id", "_score", "sort": []}]}}``, depth as asked);
- worker processes on one shared listening socket (``serving.workers``). Each opens the collection read-only, which
  zvec supports, and zvec runs its own query threads (``zvec.init(query_threads=...)``).

A request is the engine's own ``multi_match`` request: ``{"query": {"multi_match": {"query": text, "fields":
["f^w", ...]}}, "size": k}``. zvec scores one full-text field per query route with BM25. Several fields mean one route
per field, fused by zvec's own re-ranker:

- ``ZVEC_FUSION=weighted`` (the default): ``WeightedReRanker`` with the request's field weights; zvec normalises each
  route's scores itself;
- ``ZVEC_FUSION=rrf``: ``RrfReRanker()`` (rank constant 60, zvec's default).

The steward chooses which fusion counts. The wrapper holds no work zvec does not need: per hit, it only reads ``id``
and ``score`` from the documents zvec returns (``output_fields=[]``, so no stored fields are fetched).

    python colab/competitors/zvec_server.py --collection DIR --port 8080 --workers N [--query-threads T]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np  # noqa: E402

from serving.app import Response, make_server  # noqa: E402
from serving.encode import encode  # noqa: E402

log = logging.getLogger("zvec-server")
_ROUTE = re.compile(r"^/indexes/([^/]+)/_search$")
_FIELD = re.compile(r"^([^\^]+)(?:\^([0-9.eE+-]+))?$")


class Cfg:
    host = "0.0.0.0"
    port = 8080
    max_body_bytes = 64 * 1024 * 1024
    drain_seconds = 0.5
    worker_index = None


class ZvecApp:
    """``handle(method, path, body, headers) -> Response``, the interface serving's HTTP server calls."""

    def __init__(self, collection_dir: str, fusion: str):
        import zvec
        self.zvec = zvec
        self.col = zvec.open(collection_dir, option=zvec.CollectionOption(read_only=True))
        self.fusion = fusion
        self.cfg = Cfg()
        self.draining = False

    def close(self) -> None:
        pass

    def _json(self, status: int, obj) -> Response:
        return Response(status, json.dumps(obj, separators=(",", ":")).encode())

    def handle(self, method: str, path: str, body=b"", headers=None) -> Response:
        path = path.partition("?")[0]
        if path in ("/healthz", "/readyz"):
            return self._json(200, {"status": "ok", "ready": True})
        if method != "POST" or not _ROUTE.match(path):
            return self._json(404, {"error": {"type": "not_found", "path": path, "reason": "only _search"}})
        try:
            req = json.loads(body)
            mm = req["query"]["multi_match"]
            text, size = str(mm["query"]), int(req.get("size", 10))
            fields = []
            for f in mm.get("fields") or []:
                m = _FIELD.match(f)
                fields.append((m.group(1), float(m.group(2) or 1.0)))
        except (ValueError, KeyError, TypeError) as exc:
            return self._json(400, {"error": {"type": "invalid_request", "path": "", "reason": str(exc)}})
        z = self.zvec
        routes = [z.Query(field_name=f, fts=z.Fts(match_string=text)) for f, _ in fields]
        if len(routes) == 1:
            docs = self.col.query(queries=routes[0], topk=size, output_fields=[])
        else:
            rr = z.RrfReRanker() if self.fusion == "rrf" else z.WeightedReRanker([w for _, w in fields])
            docs = self.col.query(queries=routes, topk=size, output_fields=[], reranker=rr)
        ids = [d.id for d in docs]
        scores = np.fromiter((d.score for d in docs), dtype=np.float32, count=len(ids))
        resp = {"hits": {"total": len(ids), "columns": {"_id": ids, "_score": scores, "sort": []}}}
        return Response(200, encode(resp))


def serve(args, sock=None) -> None:
    import zvec
    zvec.init(query_threads=args.query_threads) if args.query_threads else zvec.init()
    app = ZvecApp(args.collection, args.fusion)
    srv = make_server(app, "0.0.0.0", args.port, sock=sock)
    signal.signal(signal.SIGTERM, lambda *a: threading.Thread(target=srv.shutdown, daemon=True).start())
    log.info("zvec server on port %s (fusion %s, query threads %s)", srv.server_address[1], args.fusion,
             args.query_threads or "zvec's default")
    srv.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--query-threads", type=int, default=0, dest="query_threads")
    ap.add_argument("--fusion", default=os.environ.get("ZVEC_FUSION", "weighted"), choices=("weighted", "rrf"))
    ap.add_argument("--listen-fd", type=int, dest="listen_fd", help=argparse.SUPPRESS)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.listen_fd is not None:                              # a worker
        from serving.workers import inherited
        serve(args, inherited(args.listen_fd))
        return 0
    if args.workers <= 1:
        serve(args)
        return 0
    import subprocess
    from serving.workers import listen
    sock = listen("0.0.0.0", args.port)
    cmd = [sys.executable, __file__, "--collection", args.collection, "--fusion", args.fusion,
           "--query-threads", str(args.query_threads), "--listen-fd", str(sock.fileno())]
    procs = [subprocess.Popen(cmd, pass_fds=(sock.fileno(),)) for _ in range(args.workers)]
    signal.signal(signal.SIGTERM, lambda *a: [p.terminate() for p in procs])
    signal.signal(signal.SIGINT, lambda *a: [p.terminate() for p in procs])
    log.info("zvec server: %d workers on port %d", args.workers, sock.getsockname()[1])
    return max(p.wait() for p in procs)


if __name__ == "__main__":
    sys.exit(main())
