"""The GPU process's batch window (S6-22s, after plat-m20): S3_SERVING_GPU_BATCH_WAIT_MS and S3_SERVING_GPU_MAX_BATCH.
The default (wait 0, max 64) takes exactly what is queued, as before."""
import threading
import time

import pytest

from serving import gpu


def _server(mb, wait_s):
    srv = object.__new__(gpu.GpuServer)
    srv.q = gpu._BatchQueue()
    srv.batch_settings = (mb, wait_s)
    return srv


def test_settings_from_the_environment(monkeypatch):
    monkeypatch.delenv("S3_SERVING_GPU_MAX_BATCH", raising=False)
    monkeypatch.delenv("S3_SERVING_GPU_BATCH_WAIT_MS", raising=False)
    assert gpu._batch_settings() == (64, 0.0)
    monkeypatch.setenv("S3_SERVING_GPU_MAX_BATCH", "32")
    monkeypatch.setenv("S3_SERVING_GPU_BATCH_WAIT_MS", "10")
    assert gpu._batch_settings() == (32, 0.01)
    monkeypatch.setenv("S3_SERVING_GPU_MAX_BATCH", "0")
    with pytest.raises(ValueError):
        gpu._batch_settings()


def test_no_wait_takes_what_is_queued_up_to_the_cap():
    srv = _server(4, 0.0)
    for i in range(6):
        srv.q.put(i)
    assert srv._collect() == [0, 1, 2, 3]
    assert srv._collect() == [4, 5]
    srv.q.put(9)
    t = time.perf_counter()
    assert srv._collect() == [9] and time.perf_counter() - t < 0.12     # no waiting for more (the 0.15 s case below waits)


def test_a_wait_gathers_late_arrivals_until_the_deadline():
    srv = _server(64, 0.15)
    srv.q.put(0)

    def later():
        for i in (1, 2, 3):
            time.sleep(0.02)
            srv.q.put(i)
    threading.Thread(target=later).start()
    t = time.perf_counter()
    got = srv._collect()
    took = time.perf_counter() - t
    assert got == [0, 1, 2, 3] and 0.12 <= took < 2.0


def test_a_wait_stops_when_the_batch_is_full():
    srv = _server(3, 5.0)
    for i in range(3):
        srv.q.put(i)
    t = time.perf_counter()
    assert srv._collect() == [0, 1, 2] and time.perf_counter() - t < 1.0      # far under the 5 s wait


def test_an_empty_queue_returns_nothing():
    assert _server(64, 0.0)._collect() == []


def test_the_batch_queue_under_concurrent_readers():
    """R4a: four readers putting 2,000 items each while the batch loop takes batches: every item exactly once, each
    reader's items in its own order, no batch over the cap."""
    srv = _server(64, 0.0)
    done = threading.Event()

    def reader(r):
        for i in range(2000):
            srv.q.put((r, i))
    ts = [threading.Thread(target=reader, args=(r,)) for r in range(4)]
    for t in ts:
        t.start()
    got = []
    while len(got) < 8000:
        b = srv._collect()
        assert len(b) <= 64
        got += b
    for t in ts:
        t.join()
    done.set()
    assert sorted(got) == sorted((r, i) for r in range(4) for i in range(2000))
    for r in range(4):
        assert [i for rr, i in got if rr == r] == list(range(2000))
    assert srv._collect() == []
