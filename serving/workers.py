"""Worker processes behind one port (S6-7): ``python -m serving --workers N`` (``S3_SERVING_WORKERS``).

One Python process serves on one core: at depth 2,000 the matched load test of 2026-11-13 saw the server at one core
of 8 and the GPU at 7%. With ``workers`` > 1 a supervisor binds the HTTP (and RPC) listening socket once and starts N
copies of ``python -m serving`` that inherit it; each worker loads its own copy of the preloaded indexes (on the GPU when
the device is cuda: N copies of the segment in device memory) and accepts connections from the shared socket, so the
kernel's accept queue spreads connections over the workers and a connection stays with the worker that took it.

- **Reads only.** Each worker holds its own index, so a write to one would not reach the others. With ``workers`` > 1
  every worker serves ``GET`` routes, ``_search``, ``_msearch``, ``_mapping`` and ``versions``, and refuses the rest
  with 405 ``method_not_allowed``; writes and admin calls go to a single-worker server (the replica Deployment of the
  operator is read-only already: its pods preload a segment the indexer Job built).
- **Start-up.** Worker 0 accepts at once, so ``/healthz`` answers while the segment loads and ``/readyz`` says 503 until
  it is loaded. The other workers accept only once their own preload is done, so a request never reaches a worker that
  is still loading once worker 0 is ready.
- **Per worker:** the query cache, the batching scheduler and ``/metrics`` (a scrape reads the worker that took the
  connection). Every response carries ``X-S3-Worker`` (the worker's number).
- **Threads.** Each worker's OpenMP/BLAS pools default to its share of the CPUs (``OMP_NUM_THREADS`` and the like,
  unless set).
- **Signals.** SIGTERM or SIGINT to the supervisor is passed to every worker, which drains as a single server does
  (readiness fails, ``drain_seconds``, then stop). A worker that exits on its own is started again after a pause (2 s, doubling to 60 s while it keeps dying
  within a minute of its start).
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time

log = logging.getLogger("serving.workers")


def _cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                                      # macOS
        return os.cpu_count() or 1


def listen(host: str, port: int, backlog: int = 1024) -> socket.socket:
    """A listening TCP socket the workers share (non-blocking: an idle worker that loses the race to accept gets
    EAGAIN, which socketserver ignores, instead of blocking)."""
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    sock.setblocking(False)
    sock.set_inheritable(True)
    return sock


def adopt(server, sock: socket.socket):
    """Make a socketserver server (built with ``bind_and_activate=False``) accept from an inherited listening socket.
    An accepted connection is made blocking (BSD accept copies the listener's non-blocking flag; Linux does not)."""
    try:
        server.socket.close()
    except OSError:
        pass
    server.socket = sock
    server.server_address = sock.getsockname()[:2]

    def get_request():
        conn, addr = sock.accept()
        conn.setblocking(True)
        return conn, addr

    server.get_request = get_request
    return server


def inherited(fd: int | None) -> socket.socket | None:
    if fd is None:
        return None
    s = socket.socket(fileno=fd)
    s.setblocking(False)
    return s


class Supervisor:
    """Starts ``n`` workers on shared sockets, restarts one that dies, passes SIGTERM/SIGINT on."""

    def __init__(self, n: int, host: str, port: int, rpc_port: int, argv: list[str], restart_pause_s: float = 1.0):
        self.n, self.argv, self.restart_pause_s = n, list(argv), restart_pause_s
        self.http = listen(host, port)
        self.rpc = listen(host, rpc_port) if rpc_port else None
        self.port = self.http.getsockname()[1]
        self.procs: list[subprocess.Popen | None] = [None] * n
        self.started = [0.0] * n
        self.stopping = threading.Event()

    def _cmd(self, i: int) -> list[str]:
        cmd = [sys.executable, "-m", "serving", *self.argv, "--worker-index", str(i),
               "--listen-fd", str(self.http.fileno())]
        if self.rpc is not None:
            cmd += ["--rpc-listen-fd", str(self.rpc.fileno())]
        return cmd

    def _start(self, i: int) -> None:
        fds = (self.http.fileno(),) + ((self.rpc.fileno(),) if self.rpc is not None else ())
        env = {**os.environ, "S3_SERVING_WORKERS": str(self.n)}
        share = str(max(1, _cores() // self.n))                  # N workers x all-core thread pools oversubscribe
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env.setdefault(var, share)
        self.procs[i] = subprocess.Popen(self._cmd(i), pass_fds=fds, env=env)
        self.started[i] = time.monotonic()
        log.info("worker %d started (pid %d) on port %d", i, self.procs[i].pid, self.port)

    def start(self) -> "Supervisor":
        for i in range(self.n):
            self._start(i)
        return self

    def stop(self, sig: int = signal.SIGTERM) -> None:
        self.stopping.set()
        for p in self.procs:
            if p is not None and p.poll() is None:
                try:
                    p.send_signal(sig)
                except OSError:
                    pass

    def wait(self, timeout: float | None = None) -> int:
        """Supervise until every worker has exited after a stop; returns the highest worker exit code."""
        t_end = None if timeout is None else time.monotonic() + timeout
        due: dict[int, float] = {}                              # worker -> when to start it again
        pause = [self.restart_pause_s] * self.n                  # doubles while a worker keeps dying young (to 60 s)
        while True:
            now = time.monotonic()
            for i, p in enumerate(self.procs):
                if self.stopping.is_set() or p is None or p.poll() is None:
                    continue
                if i not in due:
                    young = now - self.started[i] < 60
                    pause[i] = min(60.0, pause[i] * 2) if young else self.restart_pause_s
                    log.error("worker %d (pid %d) exited with %s; starting it again in %.0f s", i, p.pid,
                              p.returncode, pause[i])
                    due[i] = now + pause[i]
                elif now >= due[i]:
                    del due[i]
                    self._start(i)
            if self.stopping.is_set() and all(p is None or p.poll() is not None for p in self.procs):
                break
            if t_end is not None and time.monotonic() > t_end:
                break
            time.sleep(0.2)
        self.http.close()
        if self.rpc is not None:
            self.rpc.close()
        stopped = (-signal.SIGTERM, -signal.SIGINT)              # a worker still loading dies of the signal itself
        return max((abs(p.returncode) for p in self.procs
                    if p is not None and p.returncode is not None and p.returncode not in stopped), default=0)


def supervise(n: int, host: str, port: int, rpc_port: int, argv: list[str]) -> int:
    sup = Supervisor(n, host, port, rpc_port, argv).start()

    def on_signal(signum, frame):  # noqa: ARG001
        sup.stop(signum)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log.info("supervising %d workers on %s:%d", n, host, sup.port)
    return sup.wait()
