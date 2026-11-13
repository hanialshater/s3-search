"""The batching scheduler: concurrent requests are grouped into one ``Index.search_batch`` call.

A worker thread takes the first waiting request, then collects more until ``max_batch`` are in hand or ``max_wait_ms``
has passed since the first one arrived, and sends them together. ``max_wait_ms`` is the latency the scheduler may add
to a lone request, so the referee's latency gate sets it (S4 predictions); ``max_wait_ms = 0`` sends whatever is
already queued without waiting.

The index is resolved when a batch is dispatched (``get_index()``), so a swap takes effect between batches and a
batch in flight finishes on the index it started on.

If a batch call raises, the requests are retried one by one, so one bad request fails alone with its own named error.

``columns=True`` asks an index whose ``search_batch`` takes ``hits`` for columnar hits (docs/INTERFACES.md §5,
"Columnar hits"; serving encodes them, ``serving/encode.py``). An index without the parameter answers in the dict form.
"""
from __future__ import annotations

import functools
import inspect
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable

from .errors import Overloaded
from .metrics import Metrics


def takes_hits(fn) -> bool:
    """Whether ``search``/``search_batch`` accepts ``hits=`` (the engine's columnar hits, S6-7)."""
    return takes_param(fn, "hits")


def takes_param(fn, name: str) -> bool:
    return _takes(getattr(fn, "__func__", fn), name)


@functools.lru_cache(maxsize=128)
def _takes(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def search_batch(index, requests: list, columns: bool) -> list:
    if columns and takes_hits(index.search_batch):
        return index.search_batch(requests, hits="columns")
    return index.search_batch(requests)


def search_one(index, request: dict, columns: bool) -> dict:
    if columns and takes_hits(index.search):
        return index.search(request, hits="columns")
    return index.search(request)


class _Item:
    __slots__ = ("request", "future", "enqueued")

    def __init__(self, request: dict):
        self.request = request
        self.future: Future = Future()
        self.enqueued = time.perf_counter()


class BatchScheduler:
    def __init__(self, get_index: Callable[[], Any], *, name: str = "", max_batch: int = 32, max_wait_ms: float = 2.0,
                 max_queue: int = 4096, metrics: Metrics | None = None, columns: bool = False):
        self.get_index = get_index
        self.name = name
        self.max_batch = max(1, int(max_batch))
        self.max_wait = max(0.0, float(max_wait_ms)) / 1000.0
        self.metrics = metrics or Metrics()
        self.columns = bool(columns)
        self._q: queue.Queue[_Item] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"batcher-{name}", daemon=True)
        self._thread.start()

    # -- client side ------------------------------------------------------------------------------------------------
    def submit(self, request: dict) -> Future:
        """Queue one request; the Future resolves to ``(response, queue_wait_ms)`` or raises its named error."""
        if self._stop.is_set():
            raise Overloaded("scheduler is stopped")
        item = _Item(request)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            raise Overloaded(f"scheduler queue is full ({self._q.maxsize})") from None
        self.metrics.set("s3_queue_depth", self._q.qsize(), index=self.name)
        return item.future

    def close(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        while True:  # fail anything left rather than hang its caller
            try:
                self._q.get_nowait().future.set_exception(Overloaded("scheduler stopped"))
            except queue.Empty:
                break

    # -- worker -----------------------------------------------------------------------------------------------------
    def _collect(self) -> list[_Item]:
        try:
            first = self._q.get(timeout=0.05)
        except queue.Empty:
            return []
        batch = [first]
        deadline = first.enqueued + self.max_wait
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            try:
                batch.append(self._q.get_nowait() if remaining <= 0 else self._q.get(timeout=remaining))
            except queue.Empty:
                break
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self._collect()
            if batch:
                self._dispatch(batch)

    def _dispatch(self, batch: list[_Item]) -> None:
        now = time.perf_counter()
        waits = [(now - it.enqueued) * 1000.0 for it in batch]
        for w in waits:
            self.metrics.observe("s3_queue_wait_ms", w, index=self.name)
        self.metrics.inc("s3_batches_total", index=self.name)
        self.metrics.observe("s3_batch_size", len(batch), index=self.name)
        self.metrics.set("s3_queue_depth", self._q.qsize(), index=self.name)
        try:
            index = self.get_index()
            responses = search_batch(index, [it.request for it in batch], self.columns)
            if len(responses) != len(batch):
                raise RuntimeError(f"search_batch returned {len(responses)} responses for {len(batch)} requests")
        except Exception:
            self._one_by_one(batch, waits)
            return
        for it, resp, w in zip(batch, responses, waits):
            it.future.set_result((resp, w))

    def _one_by_one(self, batch: list[_Item], waits: list[float]) -> None:
        for it, w in zip(batch, waits):
            try:
                it.future.set_result((search_one(self.get_index(), it.request, self.columns), w))
            except Exception as exc:  # noqa: BLE001 - delivered to the caller, which names it
                it.future.set_exception(exc)
