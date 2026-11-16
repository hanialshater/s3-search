"""R4 (serving/r4.py): the Loop's contract, held on its Python stand-in (PyLoop) and on s3_hostpath.Loop (Rust; skipped
unless built): framing, searches and calls apart, the batch rule, the load slot, the writers, closing, a slow front."""
import os
import socket
import struct
import threading
import time

import pytest

from serving import r4


def _rust():
    try:
        import s3_hostpath
        return getattr(s3_hostpath, "Loop", None)
    except ImportError:
        return None


IMPLS = ["stub", "rust"]


@pytest.fixture(params=IMPLS)
def make(request, tmp_path):
    cls = r4.PyLoop if request.param == "stub" else _rust()
    if cls is None:
        pytest.skip("s3_hostpath.Loop is not built")
    path = tmp_path / "load.bin"
    path.write_bytes(b"\0" * 8 * 64)
    loops = []

    def mk(slot=3):
        lp = cls(time.perf_counter(), str(path), slot)
        loops.append(lp)
        return lp
    mk.path = path
    yield mk


def _search(i, name=b"toy", body=b"{}"):
    return struct.pack("<BQ", 1, i) + struct.pack("<H", len(name)) + name + struct.pack("<I", len(body)) + body


def _call(i):
    m, p = b"GET", b"/indexes/toy/_mapping"
    return struct.pack("<BQ", 2, i) + struct.pack("<H", len(m)) + m + struct.pack("<H", len(p)) + p + struct.pack("<I", 0)


def _frame(p):
    return struct.pack("<I", len(p)) + p


def _conn(lp, cid, proto=3):
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    lp.add_conn(cid, a.detach(), proto)
    return b


def _wait(pred, t=5.0):
    end = time.time() + t
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def _load(path, slot=3):
    return struct.unpack_from("<q", path.read_bytes(), 8 * slot)[0]


def test_frames_become_searches_and_calls_in_order(make):
    lp = make()
    front = _conn(lp, 7)
    t0 = time.perf_counter()
    front.sendall(b"".join(_frame(_search(i, body=b'{"size": %d}' % i)) for i in range(5)) + _frame(_call(99)))
    assert _wait(lambda: lp.qsize() == 5)
    got = lp.next_batch(64, 0.0, 0.2)
    assert [(c, i, n) for c, i, n, _, _ in got] == [(7, i, "toy") for i in range(5)]
    assert [p for _, _, _, p, _ in got] == [_search(i, body=b'{"size": %d}' % i) for i in range(5)]
    assert all(t0 <= t <= time.perf_counter() for *_, t in got)      # enqueue times on perf_counter's clock
    assert lp.next_call(2.0) == (7, _call(99))
    assert lp.next_call(0.05) is None
    st = lp.stats()
    assert (st["frames"], st["searches"], st["calls"]) == (6, 5, 1)
    front.close()


def test_the_batch_rule(make):
    lp = make()
    front = _conn(lp, 1)
    front.sendall(b"".join(_frame(_search(i)) for i in range(6)))
    assert _wait(lambda: lp.qsize() == 6)
    assert [x[1] for x in lp.next_batch(4, 0.0, 0.2)] == [0, 1, 2, 3]
    assert [x[1] for x in lp.next_batch(4, 0.0, 0.2)] == [4, 5]
    t = time.perf_counter()
    assert lp.next_batch(4, 0.0, 0.1) == [] and 0.08 <= time.perf_counter() - t < 2.0
    front.sendall(_frame(_search(10)))

    def later():
        for i in (11, 12):
            time.sleep(0.03)
            front.sendall(_frame(_search(i)))
    threading.Thread(target=later).start()
    t = time.perf_counter()
    got = lp.next_batch(64, 0.2, 1.0)                           # a wait gathers late arrivals until its deadline
    assert [x[1] for x in got] == [10, 11, 12] and 0.15 <= time.perf_counter() - t < 2.0
    front.sendall(b"".join(_frame(_search(i)) for i in range(20, 23)))
    assert _wait(lambda: lp.qsize() == 3)
    t = time.perf_counter()
    assert [x[1] for x in lp.next_batch(3, 5.0, 0.2)] == [20, 21, 22] and time.perf_counter() - t < 1.0   # full
    front.close()


def test_pushed_tokens_share_the_queue_in_arrival_order(make):
    lp = make()
    front = _conn(lp, 2)
    lp.push_py(100)
    front.sendall(_frame(_search(1)))
    assert _wait(lambda: lp.qsize() == 2)
    lp.push_py(101)
    got = lp.next_batch(64, 0.0, 0.2)
    assert [(c, i, n, p) for c, i, n, p, _ in got] == [(-1, 100, None, None), (2, 1, "toy", _search(1)),
                                                       (-1, 101, None, None)]
    front.close()


def test_the_load_slot(make):
    lp = make(slot=3)
    front = _conn(lp, 1)
    front.sendall(b"".join(_frame(_search(i)) for i in range(4)))
    assert _wait(lambda: _load(make.path) == 4)
    b = lp.next_batch(64, 0.0, 0.2)
    lp.set_in_batch(len(b))
    assert _load(make.path) == 4
    lp.set_in_batch(0)
    assert _load(make.path) == 0
    assert struct.unpack_from("<q", make.path.read_bytes(), 0)[0] == 0      # the other slots untouched
    front.close()


def test_replies_in_order_and_a_gone_front(make):
    lp = make()
    front = _conn(lp, 5)
    parts = [bytes([i]) * (1000 + i) for i in range(50)]
    for p in parts:
        assert lp.send(5, p)
    want = b"".join(parts)
    got = b""
    front.settimeout(5)
    while len(got) < len(want):
        got += front.recv(65536)
    assert got == want
    assert lp.send(99, b"x") is False                            # no such connection
    front.close()                                                 # the front goes away: its connection ends
    assert _wait(lambda: lp.stats()["conns"] == 0)
    assert lp.send(5, b"late") is False
    st = lp.stats()
    assert st["conns_opened"] == st["conns_closed"] == 1 and st["send_dropped"] == 2


def test_a_bad_head_closes_only_its_connection(make):
    lp = make()
    bad, good = _conn(lp, 1), _conn(lp, 2)
    bad.sendall(_frame(b"\x01\x00\x00"))                         # a search frame shorter than its head
    bad.settimeout(5)
    assert bad.recv(10) == b""                                   # closed by the Loop
    bad2 = _conn(lp, 3)
    bad2.sendall(_frame(_search(1, name=b"\xff\xfe")))          # a name that is not UTF-8
    bad2.settimeout(5)
    assert bad2.recv(10) == b""
    good.sendall(_frame(_search(8)))
    assert _wait(lambda: lp.qsize() == 1)
    assert [x[1] for x in lp.next_batch(64, 0.0, 0.2)] == [8]
    assert lp.stats()["bad_heads"] == 2
    good.close()


def test_close_conn(make):
    lp = make()
    front = _conn(lp, 4)
    lp.close_conn(4)
    front.settimeout(5)
    assert front.recv(10) == b""
    assert _wait(lambda: lp.stats()["conns"] == 0)


def test_a_slow_front_never_stalls_send(make):
    """A front that reads nothing: send never blocks (its writer's queue grows and is counted), and another front's
    replies still arrive."""
    lp = make()
    slow, fast = _conn(lp, 1), _conn(lp, 2)
    blob = b"z" * (1 << 20)
    t = time.perf_counter()
    for _ in range(32):                                           # 32 MiB, far past the socket buffers
        assert lp.send(1, blob)
    assert time.perf_counter() - t < 2.0
    assert lp.send(2, b"hello")
    fast.settimeout(5)
    assert fast.recv(16) == b"hello"
    assert lp.stats()["max_out_depth"] >= 2
    slow.close()
    fast.close()


def test_many_fronts_concurrently(make):
    """Four fronts sending 500 frames each while batches are taken: every frame once, each front's in its order."""
    lp = make()
    fronts = [_conn(lp, c) for c in range(1, 5)]

    def go(c, s):
        s.sendall(b"".join(_frame(_search(i)) for i in range(500)))
    ts = [threading.Thread(target=go, args=(c, s)) for c, s in zip(range(1, 5), fronts)]
    for t in ts:
        t.start()
    got = []
    end = time.time() + 20
    while len(got) < 2000 and time.time() < end:
        b = lp.next_batch(64, 0.0, 0.2)
        assert len(b) <= 64
        got += [(c, i) for c, i, *_ in b]
    for t in ts:
        t.join()
    assert sorted(got) == sorted((c, i) for c in range(1, 5) for i in range(500))
    for c in range(1, 5):
        assert [i for cc, i in got if cc == c] == list(range(500))
    for s in fronts:
        s.close()


def test_load_names_the_switch(monkeypatch):
    monkeypatch.delenv("S3_SERVING_R4", raising=False)
    assert r4.load() is None
    monkeypatch.setenv("S3_SERVING_R4", "0")
    assert r4.load() is None
    monkeypatch.setenv("S3_SERVING_R4", "stub")
    assert r4.load() is r4.PyLoop
    monkeypatch.setenv("S3_SERVING_R4", "yes")
    with pytest.raises(ValueError):
        r4.load()


def test_close_sends_what_is_queued_then_closes(make):
    """Shutdown: the writers send what is queued, every connection closes, nothing new is taken."""
    lp = make()
    a, b = _conn(lp, 1), _conn(lp, 2)
    assert lp.send(1, b"bye-1") and lp.send(2, b"bye-2")
    assert lp.close(5.0) == 0
    for s, want in ((a, b"bye-1"), (b, b"bye-2")):
        s.settimeout(5)
        got = b""
        while True:
            k = s.recv(64)
            if not k:
                break
            got += k
        assert got == want                                       # its queued reply, then EOF
    assert sorted(lp.closed_conns()) == [1, 2] and lp.closed_conns() == []
    x, y = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(ValueError):                              # closed: refused, and the fd is still ours
        lp.add_conn(3, x.fileno(), 3)
    os.fstat(x.fileno())
    x.close()
    y.close()


def test_closed_conns_reports_each_once_and_a_refused_fd_stays_the_callers(make):
    lp = make()
    front = _conn(lp, 9)
    x, y = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with pytest.raises(ValueError):                              # cid in use
        lp.add_conn(9, x.fileno(), 3)
    os.fstat(x.fileno())                                         # not closed by the refusal
    x.sendall(b"ok")
    assert y.recv(2) == b"ok"
    x.close()
    y.close()
    assert lp.closed_conns() == []
    front.close()
    assert _wait(lambda: lp.stats()["conns"] == 0)
    assert _wait(lambda: lp.closed_conns() == [9])
    assert lp.closed_conns() == []


def test_stats_carry_the_loops_cpu(make):
    lp = make()
    front = _conn(lp, 1)
    front.sendall(b"".join(_frame(_search(i)) for i in range(200)))
    assert _wait(lambda: lp.qsize() == 200)
    cpu = lp.stats()["cpu_s"]
    assert set(cpu) == {"reader", "writer"} and all(v >= 0 for v in cpu.values())
    front.close()


class _RefusingLoop:
    def __init__(self, exc):
        self.exc = exc

    def add_conn(self, cid, fd, proto):
        raise self.exc


def _r4_server(loop):
    from serving import gpu
    srv = object.__new__(gpu.GpuServer)
    srv.r4, srv._r4_conns, srv._r4_ids, srv.index = loop, {}, iter(range(1, 100)), 0
    srv._r4_counts = {"encode_errors": 0, "gone_items": 0, "gone_calls": 0, "refused_conns": 0}
    return srv


def test_a_refused_connection_is_closed_and_forgotten():
    srv = _r4_server(_RefusingLoop(ValueError("Loop: closed")))
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    srv._r4_add(a, 3)
    b.settimeout(5)
    assert b.recv(4) == b""                                      # the GPU process closed its end
    assert srv._r4_conns == {} and srv._r4_counts["refused_conns"] == 1
    b.close()


def test_closed_connections_are_forgotten_and_their_items_and_calls_counted(make, caplog, monkeypatch):
    from serving import gpu
    lp = make()
    srv = _r4_server(lp)
    srv.q = gpu._R4Queue(lp)
    srv._stop = threading.Event()
    monkeypatch.setattr(gpu, "_hostpath", lambda: object())     # frames stay raw: no parse needed here
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    srv._r4_add(a, 3)
    b.sendall(_frame(_search(1)) + _frame(_call(2)))
    assert _wait(lambda: lp.qsize() == 1)
    b.close()
    assert _wait(lambda: lp.stats()["conns"] == 0)
    srv._r4_forget()
    assert srv._r4_conns == {}
    assert srv._r4_items(lp.next_batch(64, 0.0, 0.2)) == [] and srv._r4_counts["gone_items"] == 1
    t = threading.Thread(target=srv._r4_calls, daemon=True)
    with caplog.at_level("WARNING", logger="serving.gpu"):
        t.start()
        assert _wait(lambda: srv._r4_counts["gone_calls"] == 1)
        srv._stop.set()
        t.join(5)
    assert any("has closed; not run" in r.getMessage() for r in caplog.records)


def test_shutdown_closes_the_loop(make):
    lp = make()
    srv = _r4_server(lp)
    srv._stop = threading.Event()
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    srv._r4_add(a, 3)
    assert lp.send(1, b"last")
    srv.shutdown()
    b.settimeout(5)
    assert _recv_all(b) == b"last" and lp.stats()["conns"] == 0
    b.close()


def _recv_all(s):
    got = b""
    while True:
        k = s.recv(64)
        if not k:
            return got
        got += k
