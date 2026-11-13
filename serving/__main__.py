"""``python -m serving`` serves; ``python -m serving build`` is the indexer Job the Kubernetes operator runs.

    python -m serving [--data-dir D] [--device cpu|torch-cpu|cuda] [--port 8080] [--preload name[:segment_id]]
                      [--workers N]       (N > 1: N read-only worker processes behind the one port, serving/workers.py)
    python -m serving build --name N --mapping mapping.json --source products.parquet [--data-dir D]
                            [--termination-log /dev/termination-log]

``build`` ingests the source into the store at <data_dir>/<name>/store.sqlite, builds a segment into
<data_dir>/<name>/segments/ and writes ``{"segment_id", "version", "n_docs"}`` to the termination log, which is how
the operator learns what the Job built without any API access from the pod.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

from .app import create_app, make_server, serve_forever
from .config import ServingConfig, parse_preload
from .manager import ManagedIndex
from .metrics import standard


def _serve(args: argparse.Namespace) -> int:
    cfg = ServingConfig.from_env()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.device:
        cfg.device = args.device
    if args.port is not None:
        cfg.port = args.port
    if args.rpc_port is not None:
        cfg.rpc_port = args.rpc_port
    if args.preload:
        cfg.preload = parse_preload(args.preload)
    if args.workers:
        cfg.workers = max(1, args.workers)
    if cfg.workers > 1 and args.listen_fd is None:              # the supervisor: no index, no engine import
        from .workers import supervise
        if not cfg.preload:
            logging.warning("--workers %d without --preload: the workers are read-only and will hold no index", cfg.workers)
        passed = [a for a in (argv_of(args)) if a]
        return supervise(cfg.workers, cfg.host, cfg.port, cfg.rpc_port, passed)
    sock = rpc_sock = None
    if args.listen_fd is not None:                              # a worker
        from .workers import inherited
        sock, rpc_sock = inherited(args.listen_fd), inherited(args.rpc_listen_fd)
        cfg.worker_index = args.worker_index
        cfg.read_only = cfg.workers > 1
    app = create_app(cfg)
    if sock is not None and args.worker_index:                  # workers after the first accept once loaded
        app.preload()
        logging.info("worker %s preloaded %s", args.worker_index, [n for n, _ in cfg.preload])
        from .workers import mark_ready
        mark_ready(cfg.worker_state, args.worker_index)
    server = make_server(app, sock=sock)
    extra = ()
    if cfg.rpc_port or rpc_sock is not None:
        from .rpc import make_rpc_server
        rpc = make_rpc_server(app, cfg.host, cfg.rpc_port, sock=rpc_sock)
        threading.Thread(target=rpc.serve_forever, name="rpc", daemon=True).start()
        extra = (rpc,)
        logging.info("rpc on %s:%s", cfg.host, cfg.rpc_port)  # listen first: /healthz answers while a big segment loads, /readyz says no
    def load() -> None:
        try:
            app.preload()
            logging.info("preloaded %s", [n for n, _ in cfg.preload])
            if cfg.worker_index is not None:
                from .workers import mark_ready
                mark_ready(cfg.worker_state, cfg.worker_index)
        except Exception:  # noqa: BLE001
            logging.exception("preload failed; staying not-ready")

    if not (sock is not None and args.worker_index):
        threading.Thread(target=load, name="preload", daemon=True).start()
    logging.info("serving on %s:%s device=%s data_dir=%s%s", cfg.host, server.server_address[1], cfg.resolved_device(),
                 cfg.data_dir, "" if cfg.worker_index is None else f" worker={cfg.worker_index}/{cfg.workers}")
    serve_forever(app, server, extra)
    return 0


def argv_of(args: argparse.Namespace) -> list[str]:
    """The serve flags as given, for the workers' command line (the supervisor adds the worker flags)."""
    out: list[str] = []
    for flag, v in (("--data-dir", args.data_dir), ("--device", args.device), ("--port", args.port),
                    ("--rpc-port", args.rpc_port), ("--preload", args.preload)):
        if v is not None:
            out += [flag, str(v)]
    return out


def _build(args: argparse.Namespace) -> int:
    from .ports import default_ports
    cfg = ServingConfig.from_env()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    mapping = json.loads(Path(args.mapping).read_text())
    mi = ManagedIndex(args.name, Path(cfg.data_dir) / args.name, mapping, cfg, default_ports(), standard())
    try:
        result = mi.build_only({"path": args.source})
    finally:
        mi.close()
    print(json.dumps(result))
    if args.termination_log:
        try:
            Path(args.termination_log).write_text(json.dumps(result))
        except OSError as exc:
            print(f"cannot write {args.termination_log}: {exc}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m serving")
    ap.add_argument("--data-dir")
    ap.add_argument("--device")
    ap.add_argument("--port", type=int)
    ap.add_argument("--rpc-port", type=int, dest="rpc_port")
    ap.add_argument("--preload", help="name[:segment_id][,...]")
    ap.add_argument("--workers", type=int, help="worker processes behind the one port (default 1; more is read-only)")
    ap.add_argument("--listen-fd", type=int, dest="listen_fd", help=argparse.SUPPRESS)
    ap.add_argument("--rpc-listen-fd", type=int, dest="rpc_listen_fd", help=argparse.SUPPRESS)
    ap.add_argument("--worker-index", type=int, dest="worker_index", default=0, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build", help="the indexer Job: ingest a source and build a segment")
    b.add_argument("--data-dir", dest="data_dir")
    b.add_argument("--name", required=True)
    b.add_argument("--mapping", required=True)
    b.add_argument("--source", required=True)
    b.add_argument("--termination-log", default="/dev/termination-log")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return _build(args) if args.cmd == "build" else _serve(args)


if __name__ == "__main__":
    sys.exit(main())
