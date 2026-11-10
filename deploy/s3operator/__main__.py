"""``python -m deploy.s3operator``: watch SearchIndexes and reconcile them; one replica at a time, chosen by a Lease.

    --resync 30            seconds between full resyncs (the backstop behind the watches)
    --namespace NS         only this namespace (default: WATCH_NAMESPACE, else all)
    --once                 one pass over every SearchIndex, then exit (smoke tests); no watches, no election
    --no-leader-elect      run without a Lease (a single replica, or tests)
    --lease-name/--lease-namespace/--identity    the Lease (default s3-operator in POD_NAMESPACE or the service
                           account's namespace; identity POD_NAME or hostname + a random suffix)
    --lease-duration 15 --renew-deadline 10 --retry-period 2     seconds, as in client-go
    --server/--token/--ca/--insecure    an out-of-cluster API server (envtest); default is the in-cluster service account
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import uuid

from .controller import Controller
from .kube import Kube
from .leader import LeaderElector
from .reconcile import Reconciler


def _default_namespace() -> str:
    if os.environ.get("POD_NAMESPACE"):
        return os.environ["POD_NAMESPACE"]
    try:
        return open(f"{Kube.SA}/namespace").read().strip()
    except OSError:
        return "s3-search-system"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m deploy.s3operator")
    ap.add_argument("--resync", type=float, default=float(os.environ.get("S3_OPERATOR_RESYNC", "30")))
    ap.add_argument("--namespace", default=os.environ.get("WATCH_NAMESPACE") or None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-leader-elect", action="store_true")
    ap.add_argument("--lease-name", default="s3-operator")
    ap.add_argument("--lease-namespace", default=None)
    ap.add_argument("--identity", default=None)
    ap.add_argument("--lease-duration", type=float, default=15.0)
    ap.add_argument("--renew-deadline", type=float, default=10.0)
    ap.add_argument("--retry-period", type=float, default=2.0)
    ap.add_argument("--server")
    ap.add_argument("--token")
    ap.add_argument("--ca")
    ap.add_argument("--insecure", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    kube = Kube(args.server, args.token, args.ca, args.insecure) if args.server else Kube()
    rec = Reconciler(kube)
    if args.once:
        rec.reconcile_all(args.namespace)
        return 0

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    current: list[Controller] = []

    def start() -> None:
        c = Controller(kube, rec, args.namespace, args.resync)
        current.append(c)
        c.start()

    def halt() -> None:
        while current:
            current.pop().stop()

    if args.no_leader_elect:
        logging.info("s3 operator up (no leader election): namespace=%s resync=%ss", args.namespace or "(all)", args.resync)
        start()
        stop.wait()
        halt()
        return 0
    identity = args.identity or os.environ.get("POD_NAME") or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    elector = LeaderElector(kube, args.lease_namespace or _default_namespace(), args.lease_name, identity,
                            args.lease_duration, args.renew_deadline, args.retry_period)
    logging.info("s3 operator up: identity=%s lease=%s/%s namespace=%s resync=%ss", identity, elector.ns, args.lease_name,
                 args.namespace or "(all)", args.resync)
    elector.run(start, halt, stop)
    halt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
