"""R4 (docs/proposals/2026-11-15-r4-rust-serving-loop.md): the GPU process's front connections owned by a ``Loop``.

    S3_SERVING_R4 unset or 0   the default: today's loop (serving/gpu.py), untouched
    S3_SERVING_R4=stub         PyLoop below: the same contract in Python threads, so the GPU process's R4 mode is
                               tested on any machine before the Rust is built
    S3_SERVING_R4=1            ``s3_hostpath.Loop`` (Rust, native/s3_hostpath/src/serve.rs): readers, framing, the batch
                               queue and writers on Rust threads, the GIL released while they wait

What a Loop does (the contract both implementations follow; serving/tests/test_r4.py holds them to it):

- ``Loop(perf_now, load_path=None, load_slot=0)``: ``perf_now`` is ``time.perf_counter()`` when it is made, so the
  enqueue times it returns are on Python's clock. ``load_path`` is the state directory's ``load.bin``; the Loop then
  writes this process's load (queued plus in the batch in hand) into slot ``load_slot`` as ``LoadTable.set`` does.
- ``add_conn(cid, fd, proto)``: the Loop takes the connected socket ``fd`` (a native front's, after its 4-byte magic;
  ``proto`` 1, 2 or 3 for S3B1, S3B2, S3B3) as connection ``cid``, which the caller picks. A reader thread reads its
  frames (u32 length, payload). A search frame (type 1) goes on the batch queue with its id and index name read from
  the head and its enqueue time. Any other frame goes on the calls queue. A frame whose head cannot be read (shorter
  than 11 bytes, or a name that is not UTF-8) closes the connection, as a Python reader's exception did. A writer
  thread sends what ``send`` queues, in order, so the batch loop never waits on a slow front.
- ``push_py(token)``: a request from a pickle connection (the Python front), which keeps its Python reader; the token
  comes back from ``next_batch`` in arrival order with the frames.
- ``next_batch(max, wait_s, first_timeout_s)``: ``_BatchQueue.take``'s rule (the first item, waiting up to
  ``first_timeout_s``; then what is queued, up to ``max``; with a wait, arrivals until ``wait_s`` after the first or
  full), waiting without the GIL. Returns a list of ``(cid, id, name, payload, t)`` for frames and
  ``(-1, token, None, None, t)`` for pushed tokens; ``t`` is the enqueue time on ``time.perf_counter``'s clock.
- ``set_in_batch(n)``: the batch in hand's size, for the load slot (written with the queue length, under its lock).
- ``next_call(timeout_s)``: ``(cid, payload)`` of the next call frame, or None.
- ``send(cid, data) -> bool``: queue bytes for connection ``cid``'s writer; False when it is gone (counted).
  On a ValueError (the Loop closed, ``cid`` in use) ``fd`` is still the caller's to close; on a RuntimeError (its
  threads could not start) the Loop took it and has shut it.
- ``close_conn(cid)``: shut the connection down (its threads then end).
- ``closed_conns() -> list``: the connections both of whose threads have ended since the last call, each once, so
  the caller can forget them.
- ``close(timeout_s=2.0) -> int``: shutdown. No new connection or frame is taken; each connection's reader ends and
  its writer sends what is already queued, then closes the socket. Waits up to ``timeout_s`` for every thread to end
  (so ``closed_conns`` then lists every connection; a front that
  reads nothing can hold its writer) and returns how many connections are still open.
- ``qsize()``, ``stats() -> dict``: the queue length; the Loop's counters, with ``cpu_s`` (``{"reader", "writer"}``:
  the CPU seconds of the Loop's reader and writer threads, so the 5 s log shows where the host time went once it has
  left Python's threads).

The Loop holds no Python object while it waits (the proposal's GIL rule): frames are bytes it owns until
``next_batch`` hands them over. Development only; the default is off.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
from collections import deque

_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")


def search_head(p: bytes):
    """``(id, name)`` of a type-1 payload as serving/binproto.py's ``search_head`` reads it (a name cut short by the
    payload's end is cut, as a slice is), or None where that raises (a head under 11 bytes, a name not UTF-8)."""
    if len(p) < 11:
        return None
    id_ = struct.unpack_from("<Q", p, 1)[0]
    n = struct.unpack_from("<H", p, 9)[0]
    try:
        return id_, bytes(p[11:11 + n]).decode()
    except UnicodeDecodeError:
        return None


class _PyConn:
    def __init__(self, loop, cid, sock, proto):
        self.loop, self.cid, self.sock, self.proto = loop, cid, sock, proto
        self.out: deque = deque()
        self.c = threading.Condition(threading.Lock())
        self.reader_done = self.dead = False
        self.ended = 0


class PyLoop:
    """The Loop contract in Python (S3_SERVING_R4=stub): what s3_hostpath.Loop replaces."""
    name = "python-stub"

    def __init__(self, perf_now: float, load_path: str | None = None, load_slot: int = 0):
        self._d: deque = deque()
        self._c = threading.Condition(threading.Lock())
        self._in_batch = 0
        self._calls: deque = deque()
        self._cc = threading.Condition(threading.Lock())
        self._conns: dict = {}
        self._lock = threading.Lock()
        self._load = os.open(load_path, os.O_RDWR) if load_path else None
        self._slot = int(load_slot)
        self._closing = False
        self._closed: list = []
        self._cpu = {"reader": 0.0, "writer": 0.0}            # ended threads' CPU seconds
        self._live: dict = {}                                    # thread ident -> (kind, its thread_time so far)
        self._threads = 0                                        # reader and writer threads running
        self.st = {k: 0 for k in ("conns_opened", "conns_closed", "frames", "searches", "calls", "bad_heads",
                                  "pushed", "batches", "batch_items", "sends", "send_bytes", "send_dropped",
                                  "send_errors", "max_out_depth")}

    # -- the load slot
    def _publish(self) -> None:                                 # under self._c
        if self._load is not None:
            os.pwrite(self._load, _I64.pack(len(self._d) + self._in_batch), 8 * self._slot)

    def set_in_batch(self, n: int) -> None:
        with self._c:
            self._in_batch = int(n)
            self._publish()

    def qsize(self) -> int:
        return len(self._d)

    # -- connections
    def add_conn(self, cid: int, fd: int, proto: int) -> None:
        with self._lock:                                        # checked before the fd is wrapped: on a refusal it
            if self._closing:                                   # stays the caller's
                raise ValueError("Loop: closed")
            if cid in self._conns:
                raise ValueError(f"connection {cid} exists")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
            sock.setblocking(True)
            c = _PyConn(self, cid, sock, proto)
            self._conns[cid] = c
            self._threads += 2
            self.st["conns_opened"] += 1
        threading.Thread(target=self._reader, args=(c,), name=f"r4-read-{cid}", daemon=True).start()
        threading.Thread(target=self._writer, args=(c,), name=f"r4-write-{cid}", daemon=True).start()

    def _read(self, sock, n: int):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = sock.recv_into(view[got:], n - got)
            if not k:
                return None
            got += k
        return bytes(buf)

    def _tick(self, kind: str) -> None:
        self._live[threading.get_ident()] = (kind, time.thread_time())

    def _end(self, c: _PyConn, kind: str) -> None:
        self._live.pop(threading.get_ident(), None)
        with self._lock:
            self._cpu[kind] += time.thread_time()
            c.ended += 1
            if c.ended == 2:
                self._closed.append(c.cid)
            self._threads -= 1

    def _reader(self, c: _PyConn) -> None:
        try:
            self._reader_loop(c)
        finally:
            self._end(c, "reader")

    def _reader_loop(self, c: _PyConn) -> None:
        try:
            while True:
                head = self._read(c.sock, 4)
                if head is None:
                    return
                p = self._read(c.sock, _U32.unpack(head)[0])
                if not p or self._closing:                      # EOF, an empty frame (a Python reader's IndexError)
                    return                                      # or closing
                self.st["frames"] += 1
                self._tick("reader")
                if p[0] == 1:
                    h = search_head(p)
                    if h is None:
                        self.st["bad_heads"] += 1
                        return
                    with self._c:
                        self._d.append((c.cid, h[0], h[1], p, time.perf_counter()))
                        self.st["searches"] += 1
                        self._publish()
                        self._c.notify()
                else:
                    with self._cc:
                        self._calls.append((c.cid, p))
                        self.st["calls"] += 1
                        self._cc.notify()
        except OSError:
            return
        finally:
            with c.c:
                c.reader_done = True
                c.c.notify()

    def _writer(self, c: _PyConn) -> None:
        try:
            self._writer_loop(c)
        finally:
            self._end(c, "writer")

    def _writer_loop(self, c: _PyConn) -> None:
        try:
            while True:
                with c.c:
                    while not c.out and not c.reader_done and not c.dead:
                        c.c.wait()
                    if c.dead or not c.out:
                        return
                    bufs = list(c.out)
                    c.out.clear()
                try:
                    for b in bufs:
                        c.sock.sendall(b)
                    self._tick("writer")
                except OSError:
                    with c.c:
                        c.dead = True
                    self.st["send_errors"] += 1
                    return
        finally:
            with self._lock:
                self._conns.pop(c.cid, None)
                self.st["conns_closed"] += 1
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.sock.close()

    def push_py(self, token: int) -> None:
        with self._c:
            self._d.append((-1, int(token), None, None, time.perf_counter()))
            self.st["pushed"] += 1
            self._publish()
            self._c.notify()

    # -- the batch queue (gpu._BatchQueue.take's rule)
    def next_batch(self, mb: int, wait: float, first_timeout: float = 0.2) -> list:
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
            self.st["batches"] += 1
            self.st["batch_items"] += len(batch)
            return batch

    def next_call(self, timeout: float):
        with self._cc:
            if not self._calls:
                self._cc.wait(timeout)
            return self._calls.popleft() if self._calls else None

    # -- replies
    def send(self, cid: int, data: bytes) -> bool:
        c = self._conns.get(cid)
        if c is None:
            self.st["send_dropped"] += 1
            return False
        with c.c:
            if c.dead:
                self.st["send_dropped"] += 1
                return False
            c.out.append(bytes(data))
            self.st["sends"] += 1
            self.st["send_bytes"] += len(data)
            self.st["max_out_depth"] = max(self.st["max_out_depth"], len(c.out))
            c.c.notify()
        return True

    def close_conn(self, cid: int) -> None:
        c = self._conns.get(cid)
        if c is not None:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def closed_conns(self) -> list:
        with self._lock:
            out, self._closed = self._closed, []
        return out

    def close(self, timeout: float = 2.0) -> int:
        with self._lock:
            self._closing = True
            conns = list(self._conns.values())
        for c in conns:
            try:
                c.sock.shutdown(socket.SHUT_RD)
            except OSError:
                pass
            with c.c:
                c.reader_done = True
                c.c.notify()
        end = time.perf_counter() + timeout
        while (self._conns or self._threads) and time.perf_counter() < end:
            time.sleep(0.005)
        return len(self._conns)

    def stats(self) -> dict:
        cpu = dict(self._cpu)
        for kind, v in list(self._live.values()):
            cpu[kind] += v
        return {**self.st, "queued": len(self._d), "conns": len(self._conns), "cpu_s": cpu}


def load():
    """None (the default), or the Loop class S3_SERVING_R4 names."""
    mode = os.environ.get("S3_SERVING_R4", "0").strip()
    if mode in ("", "0"):
        return None
    if mode == "stub":
        return PyLoop
    if mode == "1":
        import s3_hostpath                                      # the architect's pyo3 module, with R4's Loop
        cls = getattr(s3_hostpath, "Loop", None)
        if cls is None:
            raise RuntimeError("S3_SERVING_R4=1 but this s3_hostpath has no Loop (rebuild native/s3_hostpath)")
        return cls
    raise ValueError(f"S3_SERVING_R4={mode!r}: 0, stub or 1")
