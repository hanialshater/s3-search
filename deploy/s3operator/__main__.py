"""``python -m deploy.s3operator``: run the reconcile loop.

    --interval 3         seconds between passes (the loop is level-triggered: each pass lists every SearchIndex)
    --namespace NS       only this namespace (default: WATCH_NAMESPACE, else all)
    --once               one pass, then exit (smoke tests)
    --server/--token/--ca/--insecure    an out-of-cluster API server (envtest); default is the in-cluster service account

One replica of the operator is enough: reconciling is idempotent, so a second one would only repeat work.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

from .kube import Kube
from .reconcile import Reconciler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deploy.s3operator")
    ap.add_argument("--interval", type=float, default=float(os.environ.get("S3_OPERATOR_INTERVAL", "3")))
    ap.add_argument("--namespace", default=os.environ.get("WATCH_NAMESPACE") or None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--server")
    ap.add_argument("--token")
    ap.add_argument("--ca")
    ap.add_argument("--insecure", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    kube = Kube(args.server, args.token, args.ca, args.insecure) if args.server else Kube()
    rec = Reconciler(kube)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    logging.info("s3 operator up: namespace=%s interval=%ss", args.namespace or "(all)", args.interval)
    while not stop.is_set():
        try:
            rec.reconcile_all(args.namespace)
        except Exception:  # noqa: BLE001 - the API server may be briefly away
            logging.exception("reconcile pass failed")
        if args.once:
            break
        stop.wait(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
