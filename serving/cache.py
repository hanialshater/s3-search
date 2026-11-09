"""A small thread-safe LRU for query results.

The key carries the index's identity (segment id, version, patch epoch), so a swap or a patch makes every older entry
unreachable at once; ``clear()`` also frees them. Callers get a deep copy, so nobody mutates a cached response.
"""
from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from typing import Any, Hashable


def request_key(request: Any) -> str:
    """Canonical JSON: key order does not matter."""
    return json.dumps(request, sort_keys=True, separators=(",", ":"), default=str)


class LRUCache:
    def __init__(self, capacity: int = 1024):
        self.capacity = max(0, int(capacity))
        self._d: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        if not self.capacity:
            return None
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._d[key])
            self.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        if not self.capacity:
            return
        value = copy.deepcopy(value)
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)
