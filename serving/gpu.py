"""One GPU process (S6-10s, docs/INTERFACES.md §7 "One GPU process"): the process that holds the ``Index``.

    python -m serving --fronts F --gpu-processes K ...

The supervisor (``serving/workers.py``) starts K GPU processes and F front processes (``serving/front.py``). A GPU
process holds the only copy of each preloaded index it serves (one per process), a full ``App`` for every route that is
not a search, and a batch loop that makes **one** ``search_batch(requests, hits="columns", errors="return")`` call at
a time with at most ``MAX_BATCH`` (64) requests: whatever the fronts have queued when the previous call returns. There is
no batch window by default; ``S3_SERVING_GPU_BATCH_WAIT_MS`` (default 0) opens one, waiting up to that long after a
batch's first request for more, and ``S3_SERVING_GPU_MAX_BATCH`` (default 64) caps it (S6-22s: plat-m20 found the
scheduler's ``S3_SERVING_MAX_WAIT_MS`` and ``S3_SERVING_MAX_BATCH`` do not reach this loop). Requests with ``explain`` or ``highlight`` go in batches of their own (§5: per-hit work that would
slow the whole batch); everything else shares a batch.

**Transport.** A Unix-domain socket per GPU process, in the supervisor's state directory (mode 0700), the socket itself
mode 0600. A frame is a 12-byte header (pickle length, a buffer count of 0) and a pickle (protocol 5). Pickle is code execution on load, so the socket must
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

**R4** (``S3_SERVING_R4=1``, opt-in; ``stub`` for its Python stand-in; serving/r4.py): a native front's connection is
handed, after its 4-byte magic, to a ``Loop`` (``s3_hostpath.Loop``, Rust) whose threads read its frames, queue the
searches, hold the calls and write the replies, waiting without the GIL. The batch loop takes its batches from the
Loop (``_collect``) and runs them exactly as below; each front's replies are encoded at the batch's end as that
protocol's writer encoded them (``_R4Conn``) and handed to the Loop's writer. A pickle connection keeps its Python
reader and enters the same queue as a token. The default (0) leaves this loop untouched.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import queue
import socket
import struct
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .errors import NotReady, Overloaded, classify, classify_entry
from .encode import flush_response

log = logging.getLogger("serving.gpu")
MAX_BATCH = 64


def _batch_settings() -> tuple[int, float]:
    """(max batch, wait in seconds) for the batch loop, from S3_SERVING_GPU_MAX_BATCH and S3_SERVING_GPU_BATCH_WAIT_MS."""
    mb = int(os.environ.get("S3_SERVING_GPU_MAX_BATCH", str(MAX_BATCH)))
    wait_ms = float(os.environ.get("S3_SERVING_GPU_BATCH_WAIT_MS", "0"))
    if mb < 1 or wait_ms < 0:
        raise ValueError(f"S3_SERVING_GPU_MAX_BATCH={mb} must be >= 1 and S3_SERVING_GPU_BATCH_WAIT_MS={wait_ms} >= 0")
    return mb, wait_ms / 1000.0
RETRY_CAP = 2
_LEN = struct.Struct(">Q")
_HEAD = struct.Struct(">QI")                                    # pickle length, number of out-of-band buffers
_OWN_BATCH = ("explain", "highlight")
THREAD_CPU: dict = {}                                           # thread name -> its CPU seconds so far (S6-19s)


def _account(kind: str) -> None:
    """Record this thread's CPU time under its kind (reader, writer, batches, ...); the 5 s log sums them."""
    THREAD_CPU[(kind, threading.get_ident())] = time.thread_time()


ACCOUNT_EVERY = 32          # R4a: a reader records its CPU time every 32 frames (the 5 s log lags by at most that)


class _BatchQueue:
    """R4a: the requests waiting for the batch loop, a deque under one condition variable. The batch loop takes a
    whole batch in one lock hold; a queue.Queue took one lock and one condition round per item. ``put`` and ``qsize``
    are queue.Queue's; ``take`` is ``_collect``'s rule: the first item (waiting up to ``first_timeout``), then whatever
    is queued up to ``mb``, then, with a ``wait``, arrivals until ``wait`` after the first item or the batch is full."""

    def __init__(self):
        self._d: deque = deque()
        self._c = threading.Condition(threading.Lock())

    def put(self, item) -> None:
        with self._c:
            self._d.append(item)
            self._c.notify()

    def qsize(self) -> int:
        return len(self._d)

    def take(self, mb: int, wait: float, first_timeout: float = 0.2) -> list:
        d, c = self._d, self._c
        with c:
            if not d:
                end = time.perf_counter() + first_timeout
                while not d:
                    left = end - time.perf_counter()
                    if left <= 0 or not c.wait(left) and not d:
                        return []
            deadline = time.perf_counter() + wait
            batch = [d.popleft() for _ in range(min(mb, len(d)))]
            while len(batch) < mb and wait > 0:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                if not d:
                    c.wait(left)
                while d and len(batch) < mb:
                    batch.append(d.popleft())
            return batch


# ---- frames ------------------------------------------------------------------------------------------------------------
def send_msg(sock: socket.socket, obj: Any, lock: threading.Lock) -> None:
    data = pickle.dumps(obj, protocol=5)
    with lock:
        sock.sendall(_HEAD.pack(len(data), 0) + data)


def _read_into(sock: socket.socket, n: int) -> bytearray | None:
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            return None
        got += k
    return buf


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    b = _read_into(sock, n)
    return None if b is None else bytes(b)


def recv_msg(sock: socket.socket) -> Any | None:
    head = _read_into(sock, _HEAD.size)
    if head is None:
        return None
    return _finish_msg(sock, bytes(head))


def _finish_msg(sock: socket.socket, head: bytes) -> Any | None:
    n, k = _HEAD.unpack(head)
    sizes = _read_into(sock, 8 * k) if k else b""
    if sizes is None:
        return None
    data = _read_into(sock, n)
    if data is None:
        return None
    bufs = []
    for j in range(k):
        b = _read_into(sock, _LEN.unpack_from(sizes, 8 * j)[0])
        if b is None:
            return None
        bufs.append(b)
    return pickle.loads(data, buffers=bufs)


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
            _account("writer")

    def close(self) -> None:
        self.out.put(None)


class _BinConn(_Conn):
    """A native front's connection (S6-22s, ``serving/binproto.py``): replies as binary frames, the hits as columns
    the front encodes itself, anything else encoded here. ``proto`` 2 (an S3B2 connection, opt-in): the hits go as
    kind 3, unchecked, the front escaping the ids and writing the floats as Python does; and each reply of a batch is
    sent as soon as it is encoded, not with the whole batch (plat-m25: every reply waited for its batch's encode)."""

    def __init__(self, sock, proto: int = 1):
        self.proto = proto
        super().__init__(sock)

    def _bytes(self, msg) -> bytes:
        from . import hostpath
        hp = _hostpath()
        if hp is not None and self.proto == 1:                  # S3_HOSTPATH: the module writes the frames
            return hp.reply(msg[1] if msg[0] == "multi" else [msg])
        return hostpath.reply_frame(msg, self.proto)

    def _writer(self) -> None:
        while True:
            msg = self.out.get()
            if msg is None:
                return
            try:
                if self.proto == 3:                             # S3B3: the batch's replies as one kind 4 frame
                    from . import binproto
                    msgs = msg[1] if msg[0] == "multi" else [msg]
                    hp = _hostpath()                            # R3c: s3_hostpath's Rust writer, guarded
                    data = hp.batch4(msgs) if hp is not None else binproto.batch4(msgs)
                    with self.lock:
                        self.sock.sendall(data)
                    _account("writer")
                    continue
                parts = msg[1] if self.proto == 2 and msg[0] == "multi" else [msg]
                for m in parts:                                 # S3B2: one send per reply, as it is ready
                    data = self._bytes(m)
                    with self.lock:
                        self.sock.sendall(data)
            except OSError:
                return
            _account("writer")


def _bin_bytes(msg, proto: int) -> bytes:
    """The bytes a native front's writer (``_BinConn._writer``) sends for one queued message, all of them at once."""
    from . import hostpath
    hp = _hostpath()
    if proto == 3:                                              # S3B3: the batch's replies as one kind 4 frame
        from . import binproto
        msgs = msg[1] if msg[0] == "multi" else [msg]
        return hp.batch4(msgs) if hp is not None else binproto.batch4(msgs)
    if proto == 2:                                              # S3B2: one frame per reply (sent back to back)
        parts = msg[1] if msg[0] == "multi" else [msg]
        return b"".join(hostpath.reply_frame(m, 2) for m in parts)
    if hp is not None:                                          # S3B1 under S3_HOSTPATH: the module writes the frames
        return hp.reply(msg[1] if msg[0] == "multi" else [msg])
    return hostpath.reply_frame(msg, 1)


class _R4Conn:
    """R4: a native front's connection owned by the Loop. ``put`` encodes the message where it is called (the batch
    loop at a batch's end, or the call thread) as ``_BinConn``'s writer did, and queues the bytes on the Loop's writer
    thread, which sends them without the GIL. An encoding error closes the connection (a writer thread's exception
    ended its sends before) and is counted; it never stops the batch loop."""
    __slots__ = ("loop", "cid", "proto", "counts")

    def __init__(self, loop, cid: int, proto: int, counts: dict | None = None):
        self.loop, self.cid, self.proto = loop, cid, proto
        self.counts = counts if counts is not None else {"encode_errors": 0}   # the server's, in r4-<j>.json

    def put(self, msg) -> None:
        try:
            data = _bin_bytes(msg, self.proto)
        except Exception:  # noqa: BLE001 - never lose the batch loop to one front's reply
            self.counts["encode_errors"] += 1
            log.exception("R4: encoding a reply for connection %d failed; closing it", self.cid)
            self.loop.close_conn(self.cid)
            return
        self.loop.send(self.cid, data)

    def close(self) -> None:
        self.loop.close_conn(self.cid)


class _R4Queue:
    """R4: ``GpuServer.q`` over the Loop. A pickle connection's reader puts its ``_Item`` here: it waits in a dict
    under a token, and the token joins the Loop's queue, so the frames and the Python front's requests share one
    queue in arrival order."""

    def __init__(self, loop):
        self.loop = loop
        self.items: dict = {}
        self._tokens = iter(range(1, 1 << 62))
        self._lock = threading.Lock()

    def put(self, item) -> None:
        with self._lock:
            tok = next(self._tokens)
        self.items[tok] = item
        self.loop.push_py(tok)

    def qsize(self) -> int:
        return self.loop.qsize()


def _r4_loop_class():
    from .r4 import load
    return load()


_HOSTPATH: list = []


def _hostpath():
    """serving/hostpath.py's load(), once per process (None by default)."""
    if not _HOSTPATH:
        from .hostpath import load
        _HOSTPATH.append(load())
    return _HOSTPATH[0]


def binproto_search_body(raw):
    from .binproto import search_body
    return search_body(raw)


class _Item:
    __slots__ = ("conn", "id", "name", "request", "t", "raw", "prep", "parse_ms")

    def __init__(self, conn, id_, name, request, raw=None):
        self.conn, self.id, self.name, self.request, self.raw = conn, id_, name, request, raw
        self.prep = None                                        # (Prepared, position) when s3_hostpath planned it
        self.parse_ms = None                                    # the reader's parse of the frame (S6-22s timers)
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
        self.q = _BatchQueue()                                  # R4a: a deque and one condition variable
        self._ok: dict = {}                                     # R4a: index name -> [mi, ok answers this batch]
        self.calls = ThreadPoolExecutor(1, thread_name_prefix="gpu-call")
        self.listener = unix_listen(sock_path)
        self._stop = threading.Event()
        self.n_batches = self.n_requests = 0
        self.busy_s = 0.0                                       # time inside search_batch (the batch loop's work)
        self._out: dict[_Conn, list] | None = None              # replies of the batch in hand, per front
        self.in_batch = 0
        self.load = LoadTable(state_dir, create=False) if state_dir else None
        self.batch_settings = _batch_settings()                # S3_SERVING_GPU_MAX_BATCH, S3_SERVING_GPU_BATCH_WAIT_MS
        self.r4 = None                                          # R4 (S3_SERVING_R4): the Loop that owns the fronts
        cls = _r4_loop_class()
        if cls is not None:
            self.r4 = cls(time.perf_counter(), os.path.join(state_dir, "load.bin") if state_dir else None, index)
            self.q = _R4Queue(self.r4)
            self._r4_conns: dict[int, _R4Conn] = {}
            self._r4_ids = iter(range(1, 1 << 31))
            # the server's own R4 counts: replies it could not encode (that connection closed), batch items and calls
            # of connections already gone (their fronts left), connections the Loop refused
            self._r4_counts = {"encode_errors": 0, "gone_items": 0, "gone_calls": 0, "refused_conns": 0}
            log.warning("gpu process %d: R4 serving loop on (%s)", index, getattr(self.r4, "name", type(self.r4).__name__))
            if _hostpath() is None:
                log.warning("gpu process %d: R4 without S3_HOSTPATH: each native frame's JSON is parsed in the batch "
                            "loop's thread, serially with the GPU work (set S3_HOSTPATH=1)", index)
            self._r4_note()

    # -- connections
    def serve_forever(self) -> None:
        threading.Thread(target=self._batch_loop, name="gpu-batches", daemon=True).start()
        threading.Thread(target=self._publish_versions, name="gpu-versions", daemon=True).start()
        threading.Thread(target=self._stats, name="gpu-stats", daemon=True).start()
        if getattr(self, "r4", None) is not None:
            threading.Thread(target=self._r4_calls, name="gpu-r4-calls", daemon=True).start()
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
        if getattr(self, "r4", None) is not None:              # R4: the Loop's threads end, its sockets close
            left = self.r4.close(2.0)                           # (each writer sends what it has queued first)
            if left:
                log.warning("gpu process %d: R4 closed with %d connection(s) still sending", self.index, left)

    def _reader(self, sock: socket.socket) -> None:
        first = _read_into(sock, 4)
        if first is None:
            return sock.close()
        if bytes(first) in (b"S3B1", b"S3B2", b"S3B3"):         # a native front (serving/binproto.py)
            proto = {b"S3B1": 1, b"S3B2": 2, b"S3B3": 3}[bytes(first)]
            if getattr(self, "r4", None) is not None:            # R4: the Loop reads and writes it from here on
                return self._r4_add(sock, proto)
            return self._bin_reader(sock, proto)
        conn = _Conn(sock)
        pending = bytes(first)
        nread = 0
        try:
            while True:
                if pending:                                     # the first frame's header began with those 4 bytes
                    rest = _read_into(sock, _HEAD.size - 4)
                    if rest is None:
                        return
                    msg = _finish_msg(sock, pending + bytes(rest))
                    pending = b""
                else:
                    msg = recv_msg(sock)
                if msg is None:
                    return
                if msg[0] == "search":
                    self.q.put(_Item(conn, msg[1], msg[2], msg[3]))
                    nread += 1
                    if nread % ACCOUNT_EVERY == 1:
                        _account("reader")
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

    def _bin_reader(self, sock: socket.socket, proto: int = 1) -> None:
        from . import binproto
        conn = _BinConn(sock, proto)
        nread = 0
        try:
            while True:
                head = _read_into(sock, 4)
                if head is None:
                    return
                payload = _read_into(sock, struct.unpack("<I", head)[0])
                if payload is None:
                    return
                hp = _hostpath()
                if hp is not None and payload[0] == 1:          # S3_HOSTPATH: the batch loop prepares the raw frame
                    id_, name = binproto.search_head(payload)
                    self.q.put(_Item(conn, id_, name, None, raw=bytes(payload)))
                    nread += 1
                    if nread % ACCOUNT_EVERY == 1:
                        _account("reader")
                    self._publish_load()
                    continue
                tp = time.perf_counter()
                msg = binproto.parse_request(bytes(payload))
                parse_ms = (time.perf_counter() - tp) * 1000.0
                if msg[0] == "search" and isinstance(msg[3], Exception):   # not JSON: 400, as the Python front
                    status, body = classify(msg[3])
                    conn.put(("search", msg[1], "error", status, body))
                elif msg[0] == "search":
                    item = _Item(conn, msg[1], msg[2], msg[3])
                    item.parse_ms = parse_ms
                    self.q.put(item)
                    nread += 1
                    if nread % ACCOUNT_EVERY == 1:
                        _account("reader")
                    self._publish_load()
                else:
                    _, id_, method, path, body = msg
                    self.calls.submit(self._call, conn, ("call", id_, method, path, body, {}))
        except (OSError, ValueError):
            return
        finally:
            conn.close()
            try:
                sock.close()
            except OSError:
                pass

    def _publish_load(self) -> None:
        if getattr(self, "r4", None) is not None:              # R4: the Loop writes the slot with its queue length
            return self.r4.set_in_batch(self.in_batch)
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
        """The next batch: the first request, then whatever is queued, up to the max batch; with a batch wait, keep
        taking arrivals until the wait after the first request has passed or the batch is full (_BatchQueue.take)."""
        mb, wait = getattr(self, "batch_settings", None) or _batch_settings()
        if getattr(self, "r4", None) is not None:
            return self._r4_items(self.r4.next_batch(mb, wait, 0.2))
        return self.q.take(mb, wait)

    # -- R4: the Loop's side (serving/r4.py)
    def _r4_add(self, sock: socket.socket, proto: int) -> None:
        cid = next(self._r4_ids)
        self._r4_conns[cid] = _R4Conn(self.r4, cid, proto, self._r4_counts)   # before the Loop can queue a frame
        fd = sock.detach()
        try:
            self.r4.add_conn(cid, fd, proto)
        except ValueError:                                      # refused: the fd is still ours (serving/r4.py)
            self._r4_conns.pop(cid, None)
            self._r4_counts["refused_conns"] += 1
            os.close(fd)
            log.warning("R4: the Loop refused connection %d", cid, exc_info=True)
        except Exception:  # noqa: BLE001 - its threads did not start: the Loop has shut the fd
            self._r4_conns.pop(cid, None)
            self._r4_counts["refused_conns"] += 1
            log.exception("R4: the Loop could not start connection %d", cid)

    def _r4_forget(self) -> None:
        """Drop the connections the Loop reports closed (both its threads ended), so ``_r4_conns`` does not grow."""
        for cid in self.r4.closed_conns():
            self._r4_conns.pop(cid, None)

    def _r4_items(self, entries: list) -> list[_Item]:
        """The Loop's batch as the batch loop's items. A frame becomes what ``_bin_reader`` made of it: the raw
        payload under S3_HOSTPATH (prepared in ``_prepare_raw``), else the request parsed here, a body that is not
        JSON answered at once with the Python front's 400 (its own frame, as the reader sent it)."""
        hp, conns, py, out = _hostpath(), self._r4_conns, self.q.items, []
        for cid, id_, name, raw, t in entries:
            if cid < 0:                                         # a pickle connection's request, by its token
                it = py.pop(id_, None)
                if it is not None:
                    out.append(it)
                continue
            conn = conns.get(cid)
            if conn is None:                                    # its front has gone and the Loop closed it
                self._r4_counts["gone_items"] += 1
                continue
            if hp is not None:
                it = _Item(conn, id_, name, None, raw=raw)
            else:
                from . import binproto
                tp = time.perf_counter()
                try:
                    msg = binproto.parse_request(raw)
                except Exception:  # noqa: BLE001 - a frame its reader would have died on: the connection closes
                    self.r4.close_conn(cid)
                    continue
                parse_ms = (time.perf_counter() - tp) * 1000.0
                if isinstance(msg[3], Exception):               # not JSON: 400, as the Python front
                    status, body = classify(msg[3])
                    conn.put(("search", msg[1], "error", status, body))
                    continue
                it = _Item(conn, msg[1], msg[2], msg[3])
                it.parse_ms = parse_ms
            it.t = t                                            # queued when the Loop read it
            out.append(it)
        return out

    def _r4_calls(self) -> None:
        """The Loop's call frames onto the call thread, as ``_bin_reader`` submitted them."""
        from . import binproto
        while not self._stop.is_set():
            got = self.r4.next_call(0.2)
            self._r4_forget()
            if got is None:
                continue
            cid, payload = got
            conn = self._r4_conns.get(cid)
            try:
                _, id_, method, path, body = binproto.parse_request(payload)
            except Exception:  # noqa: BLE001 - its reader would have died on it: the connection closes
                self.r4.close_conn(cid)
                continue
            if conn is None:                                    # its connection closed before the call ran
                self._r4_counts["gone_calls"] += 1
                log.warning("R4: a call (%s %s, id %d) for connection %d, which has closed; not run", method, path,
                            id_, cid)
                continue
            self.calls.submit(self._call, conn, ("call", id_, method, path, body, {}))

    def _r4_note(self, stats: dict | None = None) -> None:
        """``r4-<index>.json`` in the state directory: the Loop's name and, every 5 s, its counters (tests read it)."""
        if not self.state_dir:
            return
        path = os.path.join(self.state_dir, f"r4-{self.index}.json")
        hp = _hostpath() if stats is not None else None         # the host path's coverage too (S3_HOSTPATH)
        try:
            with open(path + ".tmp", "w") as f:
                json.dump({"loop": getattr(self.r4, "name", type(self.r4).__name__), "stats": stats,
                           "server": {**self._r4_counts, "conns": len(self._r4_conns)},
                           "hostpath": hp.summary() if hasattr(hp, "summary") else None}, f, default=str)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    def _batch_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if not batch:
                continue
            self.in_batch = len(batch)
            self._publish_load()
            self._out = {}
            batch = self._prepare_raw(batch)                    # S3_HOSTPATH: the raw frames, before the grouping
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
            self._count_ok()
            self._flush()
            self.in_batch = 0
            _account("batches")
            self._publish_load()

    def _prepare_raw(self, batch: list[_Item]) -> list[_Item]:
        """S3_HOSTPATH (serving/hostpath.py): the batch's raw search frames through ``prepare``, per index, so every
        item has its request before the grouping (explain and highlight still run alone). Refused frames are answered
        here; a fallback position, or an index the manager cannot give, is parsed in Python as before."""
        raw = [it for it in batch if it.raw is not None and it.request is None]
        if not raw:
            return batch
        from .hostpath import parse_body
        hp, refused = _hostpath(), {}
        by_name: dict[str, list[_Item]] = {}
        for it in raw:
            by_name.setdefault(it.name, []).append(it)
        for name, its in by_name.items():
            try:
                index = self.app.manager.get(name).index
            except Exception:  # noqa: BLE001 - UnknownIndex: _run answers it, from the parsed request
                index = None
            prep = hp.prepare([it.raw for it in its], index, None) if index is not None else None
            errs = {i: (st, body) for i, st, body in (prep.errors if prep else [])}
            eligible = set(prep.layout.get("eligible", ())) if prep is not None and prep.layout is not None else set()
            for i, it in enumerate(its):
                if i in errs:
                    refused[id(it)] = (it, errs[i])
                elif i in eligible and i not in prep.fallback:      # planned by the module: search_prepared in _run
                    it.prep = (prep, i)
                elif prep is not None and i not in prep.fallback and prep.requests[i] is not None:
                    it.request = prep.requests[i]
                else:
                    try:
                        it.request = parse_body(binproto_search_body(it.raw))
                    except Exception as exc:  # noqa: BLE001
                        refused[id(it)] = (it, classify(exc))
        for it, (st, body) in refused.values():
            self._reply(it, ("search", it.id, "error", st, body))
        self.hp_batches = getattr(self, "hp_batches", 0) + 1
        if self.hp_batches % 2000 == 1 and hasattr(hp, "summary"):  # the serving log shows the Rust line's coverage
            log.warning("hostpath stats after %d batches: %s", self.hp_batches, json.dumps(hp.summary()))
        return [it for it in batch if id(it) not in refused]

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
        planned = [it for it in items if it.prep is not None]
        if planned:                                             # s3_hostpath: the eligible positions at once
            by_prep: dict[int, list[_Item]] = {}
            for it in planned:
                by_prep.setdefault(id(it.prep[0]), []).append(it)
            rest = [it for it in items if it.prep is None]     # before _run_prepared clears prep on a fallback
            back = [x for group in by_prep.values() for x in self._run_prepared(index, version, mi, group, len(items))]
            items = rest + back
        bad = [i for i, it in enumerate(items) if not isinstance(it.request, dict)]
        for i in bad:
            self._fail([items[i]], TypeError("a search request is a JSON object"))
        items = [it for i, it in enumerate(items) if i not in set(bad)]
        if not items:
            return
        try:
            tb = time.perf_counter()
            out, per_request = _search_batch(index, [it.request for it in items])
            batch_ms = (time.perf_counter() - tb) * 1000.0          # X-S3-Batch-Ms: the call's wall time
            self.busy_s += batch_ms / 1000.0
            self.n_batches += 1
            self.n_requests += len(items)
            if len(out) != len(items):
                raise RuntimeError(f"search_batch returned {len(out)} responses for {len(items)} requests")
        except Exception as exc:  # noqa: BLE001 - not one request's failure: one at a time, capped
            return self._one_by_one(index, version, mi, items, waits, exc)
        for it, resp, w in zip(items, out, waits):
            self._answer(it, resp, version, w, mi, len(items), batch_ms)

    def _run_prepared(self, index, version, mi, planned: list[_Item], size: int) -> list[_Item]:
        """``index.search_prepared`` (or hostpath.search_prepared_stub) for the planned items, answered here and matched
        back by position; returns the items to run through search_batch instead (their bodies parsed in Python) when
        the call fails, counted as a host-path fallback."""
        from . import hostpath
        prep = planned[0].prep[0]
        order = list(prep.layout["eligible"])
        kw = {}
        from .scheduler import takes_param
        fn = getattr(index, "search_prepared", None)
        if fn is not None:
            if takes_param(fn, "hits"):
                kw["hits"] = "columns"
            if takes_param(fn, "ids"):
                kw["ids"] = "bytes"
        else:
            kw = {k: v for k, v in (("hits", "columns"), ("ids", "bytes")) if takes_param(index.search_batch, k)}
        t0 = time.perf_counter()
        try:
            out = fn(prep, **kw) if fn is not None else hostpath.search_prepared_stub(index, prep, **kw)
            if len(out) != len(order):
                raise RuntimeError(f"search_prepared returned {len(out)} responses for {len(order)} positions")
        except Exception:  # noqa: BLE001 - never lose a batch to the module
            log.exception("search_prepared failed; the batch falls back to search_batch")
            hp = _hostpath()
            if hp is not None:
                hp.fallbacks += 1
            for it in planned:
                it.prep = None
                it.request = hostpath.parse_body(binproto_search_body(it.raw))
            return planned
        batch_ms = (time.perf_counter() - t0) * 1000.0
        self.busy_s += batch_ms / 1000.0
        self.n_batches += 1
        self.n_requests += len(planned)
        by_pos = dict(zip(order, out))
        for it in planned:
            self._answer(it, by_pos[it.prep[1]], version, (t0 - it.t) * 1000.0, mi, size, batch_ms)
        return []

    def _one_by_one(self, index, version, mi, items, waits, exc) -> None:
        failures = 0
        for k, (it, w) in enumerate(zip(items, waits)):
            if failures >= RETRY_CAP:                           # it keeps failing: fail the rest, do not retry each
                return self._fail(items[k:], exc)
            try:
                tb = time.perf_counter()
                resp = _search_batch(index, [it.request])[0][0]
                one_ms = (time.perf_counter() - tb) * 1000.0
            except Exception as e:  # noqa: BLE001
                self._fail([it], e)
                if classify(e)[0] >= 500 or type(e).__name__ in ("OverBudget", "DeviceUnavailable"):
                    failures, exc = failures + 1, e             # not this request's own error
                continue
            failures = 0
            self._answer(it, resp, version, w, mi, 1, one_ms)

    def _answer(self, it: _Item, resp: dict, version: tuple, wait_ms: float, mi, size: int = 1,
                batch_ms: float | None = None) -> None:
        if isinstance(resp, dict) and set(resp) == {"error"}:
            status, body = classify_entry(resp["error"])
            mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="error")
            return self._reply(it, ("search", it.id, "error", status, body))
        resp = flush_response(resp)        # the one flush of subnormal floats, before _source and any writer (lab f134183)
        src = it.request.get("_source") if isinstance(it.request, dict) else None   # planned: no _source
        if src not in (None, False):
            from .encode import expand_columns
            try:
                resp = expand_columns(resp)
                mi.attach_source(resp, src)
            except Exception as exc:  # noqa: BLE001
                return self._fail([it], exc)
        if getattr(self, "_out", None) is None:                 # outside a batch: counted now, as before
            mi.metrics.inc("s3_search_requests_total", index=mi.name, outcome="ok")
        else:                                                   # R4a: inside a batch, once per batch (_count_ok)
            okd = self.__dict__.setdefault("_ok", {})
            ok = okd.get(mi.name)
            if ok is None or ok[0] is not mi:
                self._count_ok()                                # a different index object: count what we have first
                okd = self.__dict__.setdefault("_ok", {})
                okd[mi.name] = ok = [mi, 0]
            ok[1] += 1
        msg = ("search", it.id, "ok", resp, version, wait_ms, size)
        # [7] batch_ms (X-S3-Batch-Ms), [8] the answer's perf_counter (the writer's wait from it: post_ms), [9] the
        # reader's parse_ms: the front's share of a request split into its parts (S6-22s timers)
        self._reply(it, msg + (batch_ms, time.perf_counter(), it.parse_ms))

    def _count_ok(self) -> None:
        """The batch's ok answers into s3_search_requests_total, one inc per index (R4a; it was one per answer)."""
        ok, self._ok = getattr(self, "_ok", {}), {}
        for name, (mi, n) in ok.items():
            if n:
                mi.metrics.inc("s3_search_requests_total", n, index=name, outcome="ok")

    # -- what the process is doing, every 5 s in its log (and gauges): batches, queue, device memory
    def _stats(self, every: float = 5.0) -> None:
        last = (0, 0, 0.0, time.perf_counter())
        last_cpu: dict = {}
        last_loop: dict = {}                                    # R4: the Loop's thread CPU at the last log
        while not self._stop.wait(every):
            cpu = {}
            for (kind, _), v in list(THREAD_CPU.items()):
                cpu[kind] = cpu.get(kind, 0.0) + v
            nb, nr, busy, t = self.n_batches, self.n_requests, self.busy_s, time.perf_counter()
            dt = t - last[3]
            mem = device_memory()
            m = self.app.metrics
            m.set("s3_gpu_queue_depth", self.q.qsize())
            for k, v in mem.items():
                m.set("s3_device_memory_bytes", v, kind=k)
            if nb != last[0] or self.q.qsize():
                share = {k: round(100 * (v - last_cpu.get(k, 0.0)) / dt) for k, v in sorted(cpu.items())}
                log.info("gpu process %d: %.0f req/s in %.1f batches/s (mean %.1f), search_batch busy %.0f%%, "
                         "queue %d, thread CPU %% of a core %s, device MiB %s", self.index, (nr - last[1]) / dt,
                         (nb - last[0]) / dt, (nr - last[1]) / max(1, nb - last[0]), 100 * (busy - last[2]) / dt,
                         self.q.qsize(), share, {k: round(v / 2**20) for k, v in mem.items()})
            if getattr(self, "r4", None) is not None:
                st = dict(self.r4.stats())
                lcpu = st.get("cpu_s") or {}                    # the Loop's threads: the host time that left Python's
                if nb != last[0] or self.q.qsize():
                    share = {k: round(100 * (v - last_loop.get(k, 0.0)) / dt) for k, v in sorted(lcpu.items())}
                    log.info("gpu process %d: R4 loop CPU %% of a core %s; %s; server %s", self.index, share,
                             json.dumps(st), json.dumps(self._r4_counts))
                last_loop = dict(lcpu)
                self._r4_note(st)
            last = (nb, nr, busy, t)
            last_cpu = cpu

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
    sw = os.environ.get("S3_SERVING_GPU_SWITCH_INTERVAL_MS")    # S6-22s: the GIL switch interval (default Python's 5 ms)
    if sw:                                                      # so the reader and writer threads wait less behind
        sys.setswitchinterval(float(sw) / 1000.0)               # the batch thread
        log.info("gpu process %d: GIL switch interval %.3f ms", index, sys.getswitchinterval() * 1000.0)
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
