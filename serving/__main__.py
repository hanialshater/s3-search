"""``python -m serving`` serves; ``python -m serving build`` is the indexer Job the Kubernetes operator runs.

    python -m serving [--data-dir D] [--device cpu|torch-cpu|cuda] [--port 8080] [--preload name[:segment_id]]
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
    if args.port:
        cfg.port = args.port
    if args.rpc_port is not None:
        cfg.rpc_port = args.rpc_port
    if args.preload:
        cfg.preload = parse_preload(args.preload)
    app = create_app(cfg)
    server = make_server(app)
    extra = ()
    if cfg.rpc_port:
        from .rpc import make_rpc_server
        rpc = make_rpc_server(app, cfg.host, cfg.rpc_port)
        threading.Thread(target=rpc.serve_forever, name="rpc", daemon=True).start()
        extra = (rpc,)
        logging.info("rpc on %s:%s", cfg.host, cfg.rpc_port)  # listen first: /healthz answers while a big segment loads, /readyz says no
    def load() -> None:
        try:
            app.preload()
            logging.info("preloaded %s", [n for n, _ in cfg.preload])
        except Exception:  # noqa: BLE001
            logging.exception("preload failed; staying not-ready")

    threading.Thread(target=load, name="preload", daemon=True).start()
    logging.info("serving on %s:%s device=%s data_dir=%s", cfg.host, cfg.port, cfg.resolved_device(), cfg.data_dir)
    serve_forever(app, server, extra)
    return 0


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
