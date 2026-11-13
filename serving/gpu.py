"""One GPU process (S6-10s, docs/INTERFACES.md §7 "One GPU process"): the process that holds the ``Index``.

    python -m serving --fronts F --gpu-processes K ...

The supervisor (``serving/workers.py``) starts K GPU processes and F front processes (``serving/front.py``). A GPU
process holds the only copy of each preloaded index it serves (one per process), a full ``App`` for every route that is
not a search, and a batch loop that makes **one** ``search_batch(requests, hits="columns", errors="return")`` call at
a time with at most ``MAX_BATCH`` (64) requests: whatever the fronts have queued when the previous call returns. There is
no batch window. Requests with ``explain`` or ``highlight`` go in batches of their own (§5: per-hit work that would
slow the whole batch); everything else shares a batch.

**Transport.** A Unix-domain socket per GPU process, in the supervisor's state directory (mode 0700), the socket itself
mode 0600. A frame is an 8-byte big-endian length and a pickle. Pickle is code execution on load, so the socket must
stay inside one trust boundary: the service user's own processes. Never expose it on TCP.

    front -> gpu   ("search", id, index, request)          ("call", id, method, path, body, headers)
    gpu -> front   ("search", id, "ok", response, version, queue_ms)
                   ("search", id, "error", status, body)
                   ("call", id, status, body, content_type, headers)

``version`` is ``(segment_id, epoch)`` of the managed index the batch read: what a front's cache keys on.

**Errors.** An ``{"error": ...}`` entry (one request refused) maps the engine's class name to §7's named error and
status (``serving/errors.py``). When the call itself raises (a failure that is not one request's: OverBudget,
DeviceUnavailable, CUDA), the requests are tried one at a time, but after ``RETRY_CAP`` such failures in a row the rest
of the batch fails with that error instead of retrying each.

**Writes.** Calls (writes, admin, GET routes) run on one call thread, one at a time, next to the batch loop: an
``Index`` write is copy-on-write and ``search_batch`` reads its state once (§5), so a batch that starts after a call
returns sees it. The front answers the write only after this process has applied it.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import socket
import struct
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .errors import NotReady, Overloaded, classify, classify_entry

log = logging.getLogger("serving.gpu")
MAX_BATCH = 64
RETRY_CAP = 2
_LEN = struct.Struct(">Q")
_OWN_BATCH = ("explain", "highlight")


# ---- frames ------------------------------------------------------------------------------------------------------------
def send_msg(sock: socket.socket, obj: Any, lock: threading.Lock) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    with lock:
        sock.sendall(_LEN.pack(len(data)) + data)


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Any | None:
    head = _read_exact(sock, 8)
    if head is None:
        return None
    data = _read_exact(sock, _LEN.unpack(head)[0])
    return None if data is None else pickle.loads(data)


def unix_listen(path: str) -> socket.socket:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)                                       # created 0600: the service user only
    try:
        s.bind(path)
    finally:
        os.umask(old)
    os.chmod(path, 0o600)
    s.listen(256)
    return s


def version_of(mi) -> tuple:
    return (mi.segment_id, mi.epoch)


# ---- the GPU process ------------------------------------------------------------------------------------------------------
class _Conn:
    """One front's connection: a writer thread sends what the batch loop and the call thread queue, so neither ever
    waits on a slow reader. Replies of one batch to one front travel as one ("multi", [msg, ...]) frame."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.out: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._writer, daemon=True).start()

    def put(self, msg) -> None:
        self.out.put(msg)

    def _writer(self) -> None:
        while True:
            msg = self.out.get()
            if msg is None:
                return
            try:
                send_msg(self.sock, msg, self.lock)
            except OSError:
                return                                          # the front went away; its callers time out

    def close(self) -> None:
        self.out.put(None)


class _Item:
    __slots__ = ("conn", "id", "name", "request", "t")

    def __init__(self, conn, id_, name, request):
        self.conn, self.id, self.name, self.request = conn, id_, name, request
        self.t = time.perf_counter()


def _search_batch(index, requests: list) -> tuple[list, bool]:
    """``(responses, per_request_errors)``: errors="return" when the index takes it."""
    from .scheduler import takes_param
    kw = {}
    if takes_param(index.search_batch, "hits"):
        kw["hits"] = "columns"
        if takes_param(index.search_batch, "ids"):              # S6-12: _id as one bytes array (pickles ~7x faster)
            kw["ids"] = "bytes"
    per_request = takes_param(index.search_batch, "errors")
    if per_request:
        kw["errors"] = "return"
    return index.search_batch(requests, **kw), per_request


class GpuServer:
    def __init__(self, app, sock_path: str, index: int = 0, state_dir: str | None = None):
        self.app, self.sock_path, self.index, self.state_dir = app, sock_path, index, state_dir
        self.q: queue.Queue[_Item] = queue.Queue()
        self.calls = ThreadPoolExecutor(1, thread_name_prefix="gpu-call")
        self.listener = unix_listen(sock_path)
        self._stop = threading.Event()
        self.n_batches = self.n_requests = 0
        self.busy_s = 0.0                                       # time inside search_batch (the batch loop's work)
        self._out: dict[_Conn, list] | None = None              # replies of the batch in hand, per front
        self.in_batch = 0
        self.load = LoadTable(state_dir, create=False) if state_dir else None

    # -- connections
    def serve_forever(self) -> None:
        threading.Thread(target=self._batch_loop, name="gpu-batches", daemon=True).start()
        threading.Thread(target=self._publish_versions, name="gpu-versions", daemon=True).start()
        threading.Thread(target=self._stats, name="gpu-stats", daemon=True).start()
        self.listener.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(None)
            threading.Thread(target=self._reader, args=(conn,), daemon=True).start()

    def shutdown(self) -> None:
        self._stop.set()

    def _reader(self, sock: socket.socket) -> None:
        conn = _Conn(sock)
        try:
            while True:
                msg = recv_msg(sock)
                if msg is None:
                    return
                if msg[0] == "search":
                    self.q.put(_Item(conn, msg[1], msg[2], msg[3]))
                    self._publish_load()
                elif msg[0] == "call":
                    self.calls.submit(self._call, conn, msg)
        except (OSError, EOFError, pickle.UnpicklingError):
            return
        finally:
            conn.close()
            try:
                sock.close()
            except OSError:
                pass

    def _publish_load(self) -> None:
        if self.load is not None:
            self.load.set(self.index, self.q.qsize() + self.in_batch)

    def _reply(self, it: _Item, msg: tuple) -> None:
        if self._out is not None:                               # inside a batch: one frame per front at its end
            self._out.setdefault(it.conn, []).append(msg)
        else:
            it.conn.put(msg)

    def _flush(self) -> None:
        out, self._out = self._out or {}, None
        for conn, msgs in out.items():
            conn.put(("multi", msgs) if len(msgs) > 1 else msgs[0])

    def _call(self, conn, msg) -> None:
        _, id_, method, path, body, headers = msg
        r = self.app.handle(method, path, body, headers)
        conn.put(("call", id_, r.status, r.body, r.content_type, r.headers))

    # -- the batch loop
    def _collect(self) -> list[_Item]:
        try:
            first = self.q.get(timeout=0.2)
        except queue.Empty:
            return []
        batch = [first]
        while len(batch) < MAX_BATCH:
            try:
                batch.append(self.q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _batch_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            self.in_batch = len(batch)
            self._publish_load()
            self._out = {}
            groups: dict[tuple, list[_Item]] = {}
            for it in batch:                                    # one search_batch per index; explain/highlight alone
                own = isinstance(it.request, dict) and any(it.request.get(k) for k in _OWN_BATCH)
                groups.setdefault((it.name, id(it) if own else None), []).append(it)
            for (name, _), items in groups.items():
                try:
                    self._run(name, items)
                except Exception as exc:  # noqa: BLE001 - never lose the loop
                    log.exception("batch for %s failed", name)
                    status, body = classify(exc)
                    for it in items:
                        self._reply(it, ("search", it.id, "error", status, body))
            self._flush()
            self.in_batch = 0
            self._publish_load()

    def _fail(self, items: list[_Item], exc: BaseException) -> None:
        status, body = classify(exc)
        for it in items:
            self._reply(it, ("search", it.id, "error", status, body))

    def _run(self, name: str, items: list[_Item]) -> None:
        try:
            mi = self.app.manager.get(name)
        except Exception as exc:  # noqa: BLE001 - UnknownIndex
            return self._fail(items, exc)
        if not mi.ready:
            return self._fail(items, NotReady(f"index {name!r} is not loaded"))
        index, version = mi.index, version_of(mi)
        t0 = time.perf_counter()
        waits = [(t0 - it.t) * 1000.0 for it in items]
        m = self.app.metrics
        m.inc("s3_batches_total", index=name)
        m.observe("s3_batch_size", len(items), index=name)
        bad = [i for i, it in enumerate(items) if not isinstance(it.request, dict)]
        for i in bad:
            self._fail([items[i]], TypeError("a search request is a JSON object"))
        items = [it for i, it in enumerate(items) if i not in set(bad)]
        if not items:
            return
        try:
            tb = time.perf_counter()
            out, per_request = _search_batch(index, [it.request for it in items])
            self.busy_s += time.perf_counter() - tb
            self.n_batches += 1
            self.n_requests += len(items)
            if len(out) != len(items):
                raise RuntimeError(f"search_batch returned {len(out)} responses for {len(items)} requests")
        except Exception as exc:  # noqa: BLE001 - not one request's failure: one at a time, capped
            return self._one_by_one(index, version, mi, items, waits, exc)
        for it, resp, w in zip(items, out, waits):
            self._answer(it, resp, version, w, mi, len(items))

    def _one_by_one(self, index, version, mi, items, waits, exc) -> None:
        failures = 0
        for k, (it, w) in enumerate(zip(items, waits)):
            if failures >= RETRY_CAP:                           # it keeps failing: fail the rest, do not retry each
                return self._fail(items[k:], exc)
            try:
                resp = _search_batch(index, [it.request])[0][0]
            except Exception as e:  # noqa: BLE001
                self._fail([it], e)
                if classify(e)[0] >= 500 or type(e).__name__ in ("OverBudget", "DeviceUnavailable"):
                    failures, exc = failures + 1, e             # not this request's own error
                continue
            failures = 0
            self._answer(it, resp, version, w, mi)

    def _answer(self, it: _Item, resp: dict, version: tuple, wait_ms: float, mi, size: int = 1) -> None:
        if isinstance(resp, dict) and set(resp) == {"error"}:
            status, body = classify_entry(resp["error"])
            mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="error")
            return self._reply(it, ("search", it.id, "error", status, body))
        src = it.request.get("_source")
        if src not in (None, False):
            from .encode import expand_columns
            try:
                resp = expand_columns(resp)
                mi.attach_source(resp, src)
            except Exception as exc:  # noqa: BLE001
                return self._fail([it], exc)
        mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="ok")
        self._reply(it, ("search", it.id, "ok", resp, version, wait_ms, size))

    # -- what the process is doing, every 5 s in its log (and gauges): batches, queue, device memory
    def _stats(self, every: float = 5.0) -> None:
        last = (0, 0, 0.0, time.perf_counter())
        while not self._stop.wait(every):
            nb, nr, busy, t = self.n_batches, self.n_requests, self.busy_s, time.perf_counter()
            dt = t - last[3]
            mem = device_memory()
            m = self.app.metrics
            m.set("s3_gpu_queue_depth", self.q.qsize())
            for k, v in mem.items():
                m.set("s3_device_memory_bytes", v, kind=k)
            if nb != last[0] or self.q.qsize():
                log.info("gpu process %d: %.0f req/s in %.1f batches/s (mean %.1f), search_batch busy %.0f%%, "
                         "queue %d, device MiB %s", self.index, (nr - last[1]) / dt, (nb - last[0]) / dt,
                         (nr - last[1]) / max(1, nb - last[0]), 100 * (busy - last[2]) / dt, self.q.qsize(),
                         {k: round(v / 2**20) for k, v in mem.items()})
            last = (nb, nr, busy, t)

    # -- versions for the fronts' caches
    def _publish_versions(self) -> None:
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, f"versions-{self.index}.json")
        last = None
        while not self._stop.is_set():
            try:
                now = {n: list(version_of(self.app.manager.get(n))) for n in self.app.manager.names()}
            except Exception:  # noqa: BLE001
                now = last
            if now != last:
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(now, f)
                os.replace(tmp, path)
                last = now
            self._stop.wait(0.02)


class LoadTable:
    """Each GPU process's load (requests queued plus in its batch in hand), one int64 slot per process in a file in the
    state directory, mmapped by every front: one writer per slot (that GPU process), so it never drifts, and every front
    picks the least loaded process from the same numbers (S6-12s; it was each front's own in-flight count)."""

    SLOTS = 64

    def __init__(self, state_dir: str, create: bool):
        import mmap
        path = os.path.join(state_dir, "load.bin")
        if create or not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(b"\0" * 8 * self.SLOTS)
        self._f = open(path, "r+b")
        self._m = mmap.mmap(self._f.fileno(), 8 * self.SLOTS)

    def set(self, j: int, v: int) -> None:
        struct.pack_into("<q", self._m, 8 * j, int(v))

    def get(self, j: int) -> int:
        return struct.unpack_from("<q", self._m, 8 * j)[0]


def device_memory() -> dict:
    """torch's caching allocator on cuda: allocated (live tensors), reserved (held by the cache), peak allocated."""
    try:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return {}
        return {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()}
    except Exception:  # noqa: BLE001
        return {}


# ---- the front's client ---------------------------------------------------------------------------------------------------
class GpuClient:
    """One connection from a front to one GPU process; calls from many threads; reconnects after a drop."""

    def __init__(self, path: str):
        self.path = path
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()                           # sends
        self._conn_lock = threading.Lock()
        self._pending: dict[int, Future] = {}
        self._ids = iter(range(1, 1 << 62))
        self.inflight = 0

    def _connect(self) -> socket.socket:
        with self._conn_lock:
            if self._sock is None:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.path)
                self._sock = s
                threading.Thread(target=self._reader, args=(s,), daemon=True).start()
            return self._sock

    def _reader(self, s: socket.socket) -> None:
        try:
            while True:
                msg = recv_msg(s)
                if msg is None:
                    break
                for m in (msg[1] if msg[0] == "multi" else (msg,)):
                    fut = self._pending.pop(m[1], None)
                    if fut is not None:
                        fut.set_result(m)
        except (OSError, EOFError, pickle.UnpicklingError):
            pass
        with self._conn_lock:
            if self._sock is s:
                self._sock = None
        for k in list(self._pending):                           # fail what this connection carried
            fut = self._pending.pop(k, None)
            if fut is not None and not fut.done():
                fut.set_exception(NotReady("the GPU process connection was lost"))

    def _send(self, make) -> Future:
        fut: Future = Future()
        id_ = next(self._ids)
        self._pending[id_] = fut
        try:
            s = self._connect()
            send_msg(s, make(id_), self._lock)
        except OSError:
            self._pending.pop(id_, None)
            with self._conn_lock:
                self._sock = None
            raise NotReady("the GPU process is not reachable") from None
        return fut

    def search(self, name: str, request: dict) -> Future:
        return self._send(lambda i: ("search", i, name, request))

    def call(self, method: str, path: str, body: bytes, headers: dict) -> Future:
        return self._send(lambda i: ("call", i, method, path, body, headers))


def gpu_main(cfg, sock_path: str, index: int) -> int:
    """``python -m serving --gpu-process J --gpu-socket PATH``: load, then serve the fronts until SIGTERM."""
    import signal
    from .app import create_app
    from .workers import mark_ready
    frac = float(os.environ.get("S3_SERVING_GPU_MEMORY_FRACTION", "0") or 0)
    if frac > 0:                                                # cap torch's cache so K processes fit predictably
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(frac)
                log.info("gpu process %d: torch memory capped at %.2f of the device", index, frac)
        except Exception:  # noqa: BLE001
            log.exception("could not cap the device memory")
    app = create_app(cfg)
    srv = GpuServer(app, sock_path, index, cfg.worker_state)

    def load() -> None:
        try:
            app.preload()
            log.info("gpu process %d preloaded %s", index, [n for n, _ in cfg.preload])
            mark_ready(cfg.worker_state, f"gpu-{index}")
        except Exception:  # noqa: BLE001
            log.exception("preload failed; staying not-ready")

    threading.Thread(target=load, name="preload", daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *a: srv.shutdown())
    signal.signal(signal.SIGINT, lambda *a: srv.shutdown())
    log.info("gpu process %d on %s device=%s", index, sock_path, cfg.resolved_device())
    srv.serve_forever()
    app.close()
    return 0


__all__ = ["GpuServer", "GpuClient", "gpu_main", "MAX_BATCH", "Overloaded"]
