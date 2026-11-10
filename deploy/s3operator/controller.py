"""The event-driven loop: watches feed a de-duplicating work queue, one worker reconciles, a resync is the backstop.

Watches: ``SearchIndex`` (every change), and the Deployments and Jobs the operator owns (selected by the
``managed-by`` label), each mapped to its SearchIndex by the ``search.s3.dev/index`` label, so a replica becoming ready or
a Job finishing reconciles its index within milliseconds. Each watcher lists, enqueues what it finds, then watches from
the list's resourceVersion; on any error or a 410 it backs off and lists again. Every ``resync`` seconds every
SearchIndex is enqueued anyway (a lost event, or a ConfigMap edit, which is not watched, is caught by the next resync).
Reconcile is idempotent and writes nothing when nothing changed, so the status write it makes does not loop.
"""
from __future__ import annotations

import logging
import threading
import time

from . import render
from .kube import DEPLOYMENT, JOB, SEARCHINDEX, ApiError, Kube
from .reconcile import Reconciler

log = logging.getLogger("s3operator")
OWNED = f"app.kubernetes.io/managed-by={render.MANAGER}"


class Controller:
    def __init__(self, kube: Kube, reconciler: Reconciler, namespace: str | None = None, resync: float = 30.0,
                 watch_timeout: int = 60):
        self.kube, self.rec, self.ns, self.resync, self.watch_timeout = kube, reconciler, namespace, resync, watch_timeout
        self._cv = threading.Condition()
        self._queue: list[tuple[str, str]] = []
        self._queued: set[tuple[str, str]] = set()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- queue ------------------------------------------------------------------------------------------------------
    def enqueue(self, ns: str, name: str) -> None:
        key = (ns, name)
        with self._cv:
            if key not in self._queued:
                self._queued.add(key)
                self._queue.append(key)
                self._cv.notify()

    def _next(self) -> tuple[str, str] | None:
        with self._cv:
            while not self._queue and not self._stop.is_set():
                self._cv.wait(0.5)
            if self._stop.is_set():
                return None
            key = self._queue.pop(0)
            self._queued.discard(key)   # an event arriving while we reconcile queues it again
            return key

    def _worker(self) -> None:
        while (key := self._next()) is not None:
            ns, name = key
            try:
                si = self.kube.get(SEARCHINDEX, ns, name)
                if si is None or si["metadata"].get("deletionTimestamp"):
                    continue
                log.info("reconcile %s/%s", ns, name)
                self.rec.reconcile(si)
            except Exception:  # noqa: BLE001 - retry soon, do not die
                log.exception("reconcile %s/%s failed; retrying", ns, name)
                threading.Timer(2.0, self.enqueue, (ns, name)).start()

    # -- watches ----------------------------------------------------------------------------------------------------
    def _owner_key(self, obj: dict) -> tuple[str, str] | None:
        idx = (obj["metadata"].get("labels") or {}).get(render.LABEL_INDEX)
        return (obj["metadata"]["namespace"], idx) if idx else None

    def _watcher(self, res, selector: str | None, to_key) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                items, rv = self.kube.list_with_version(res, self.ns, selector)
                for it in items:
                    if (k := to_key(it)):
                        self.enqueue(*k)
                backoff = 0.5
                while not self._stop.is_set():       # re-watch from the last seen version until an error
                    for ev in self.kube.watch(res, self.ns, rv, selector, self.watch_timeout):
                        obj = ev["object"]
                        rv = obj["metadata"].get("resourceVersion", rv)
                        if ev["type"] != "BOOKMARK" and (k := to_key(obj)):
                            self.enqueue(*k)
                        if self._stop.is_set():
                            return
            except ApiError as e:
                if e.status != 410:
                    log.warning("watch %s: %s", res.plural, e)
            except Exception as e:  # noqa: BLE001
                if not self._stop.is_set():
                    log.warning("watch %s: %s", res.plural, e)
            self._stop.wait(backoff)
            backoff = min(backoff * 2, 10)

    def _resyncer(self) -> None:
        while not self._stop.wait(self.resync):
            try:
                for si in self.kube.list(SEARCHINDEX, self.ns):
                    self.enqueue(si["metadata"]["namespace"], si["metadata"]["name"])
            except Exception as e:  # noqa: BLE001
                log.warning("resync: %s", e)

    # -- lifecycle --------------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        specs = [(SEARCHINDEX, None, lambda o: (o["metadata"]["namespace"], o["metadata"]["name"])),
                 (DEPLOYMENT, OWNED, self._owner_key), (JOB, OWNED, self._owner_key)]
        self._threads = [threading.Thread(target=self._worker, name="worker", daemon=True),
                         threading.Thread(target=self._resyncer, name="resync", daemon=True)]
        self._threads += [threading.Thread(target=self._watcher, args=a, name=f"watch-{a[0].plural}", daemon=True)
                          for a in specs]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        # watcher threads end when their current stream closes (at most watch_timeout seconds); they are daemons and
        # only enqueue, so a stopped controller never reconciles again.
        self._queue.clear()
        self._queued.clear()
        self._threads = []
