"""A small fake engine and data plane that follow docs/INTERFACES.md §3 to §5, for testing serving alone.

``FakeIndex`` answers ``match`` and ``match_all`` over title text, honours ``from``/``size``, patches hot fields and
deletes, and records every ``search_batch`` size. ``fake_ports()`` gives an in-memory store with an ordered change log,
a "builder" that writes a real directory with ``manifest.json`` plus ``docs.json``, and a loader for it.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from serving.ports import Ports


class UnsupportedQuery(ValueError):
    def __init__(self, path, reason):
        super().__init__(f"{path}: {reason}")
        self.path, self.reason = path, reason


class InvalidRequest(UnsupportedQuery):
    pass


class OverBudget(MemoryError):
    pass


class NotHot(ValueError):
    pass


class FakeIndex:
    HOT = ("price", "stock")

    def __init__(self, docs: list[dict], version: int = 0, device: str = "cpu", delay: float = 0.0):
        self.docs = {d["_id"]: dict(d) for d in docs}
        self.order = [d["_id"] for d in docs]
        self.live = {i: True for i in self.order}
        self.version, self.device, self.delay = version, device, delay
        self.n_docs = len(docs)
        self.mapping: dict = {}
        self.device_nbytes = 0
        self.batch_sizes: list[int] = []
        self.search_calls = 0
        self._lock = threading.Lock()

    def search(self, request: dict) -> dict:
        self.search_calls += 1
        return self._one(request)

    def search_batch(self, requests: list[dict]) -> list[dict]:
        with self._lock:
            self.batch_sizes.append(len(requests))
        return [self._one(r) for r in requests]

    def _one(self, request: dict) -> dict:
        t0 = time.perf_counter()
        if self.delay:
            time.sleep(self.delay)
        q = request.get("query", {"match_all": {}})
        extra = set(request) - {"query", "from", "size", "_source", "sort", "explain"}
        if extra:
            raise UnsupportedQuery(sorted(extra)[0], "unknown key")
        size, frm = request.get("size", 10), request.get("from", 0)
        if frm + size > 10_000:
            raise InvalidRequest("from+size", "over 10000")
        if "match" in q:
            (field, text), = q["match"].items()
            words = str(text).lower().split()
            ids = [i for i in self.order if self.live[i] and
                   any(w in str(self.docs[i].get(field, "")).lower() for w in words)]
        elif "match_all" in q:
            ids = [i for i in self.order if self.live[i]]
        else:
            raise UnsupportedQuery("query." + next(iter(q)), "not in the subset")
        page = ids[frm:frm + size]
        took = (time.perf_counter() - t0) * 1000
        return {"hits": {"total": len(ids), "hits": [{"_id": i, "_score": 1.0, "sort": [1.0, self.order.index(i)]}
                                                      for i in page]},
                "took_ms": {"parse": 0.01, "retrieve": took, "page": 0.01, "total": took}}

    def patch(self, field: str, ids, values) -> None:
        if field not in self.HOT:
            raise NotHot(field)
        for i, v in zip(ids, values):
            if i in self.docs:
                self.docs[i][field] = v

    def delete(self, ids) -> None:
        for i in ids:
            if i in self.live:
                self.live[i] = False

    def set_version(self, version: int) -> None:
        self.version = version


class FakeStore:
    """In-memory: ``changes`` is the ordered log, ``docs`` the live key-value view."""

    def __init__(self):
        self.log: list[dict] = []
        self.docs: dict[str, dict] = {}
        self.order: list[str] = []

    def apply(self, changes):
        first = len(self.log) + 1
        for ch in changes:
            rec = {**ch, "offset": len(self.log) + 1, "ts": datetime.now(timezone.utc).isoformat()}
            if rec["op"] == "patch" and rec["_id"] not in self.docs:
                raise KeyError(rec["_id"])
            self.log.append(rec)
            if rec["op"] == "upsert":
                if rec["_id"] not in self.docs:
                    self.order.append(rec["_id"])
                self.docs[rec["_id"]] = dict(rec["doc"], _id=rec["_id"])
            elif rec["op"] == "patch":
                self.docs[rec["_id"]].update(rec["fields"])
            elif rec["op"] == "delete" and rec["_id"] in self.docs:
                del self.docs[rec["_id"]]
                self.order.remove(rec["_id"])
        return first, len(self.log)

    def head(self) -> int:
        return len(self.log)

    def get(self, ids):
        return [dict(self.docs[i]) if i in self.docs else None for i in ids]

    def changes(self, after: int, limit: int = 10_000):
        return self.log[after:after + limit]


def fake_ports(delay: float = 0.0, over_budget_on: set | None = None) -> Ports:
    stores: dict[str, FakeStore] = {}
    counter = {"n": 0}
    built: list[Path] = []

    def open_store(path: Path, mapping: dict):
        return stores.setdefault(str(path), FakeStore())

    def ingest(source: str, mapping: dict, store: FakeStore, rejected: Path):
        docs = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
        store.apply([{"op": "upsert", "_id": d["_id"], "doc": {k: v for k, v in d.items() if k != "_id"}}
                     for d in docs])
        return {"accepted": len(docs), "rejected": 0}

    def prepare(changes, mapping):
        out, rej = [], []
        for ch in changes:
            if ch["op"] == "upsert" and "title" not in ch["doc"]:
                rej.append({"_id": ch["_id"], "reason": "bad_type", "field": "title", "detail": "missing title"})
            else:
                out.append(ch)
        return out, rej

    def build_segment(store: FakeStore, mapping: dict, out_dir: Path) -> Path:
        counter["n"] += 1
        head = store.head()
        seg_id = f"seg-{head}-{counter['n']:08x}"
        d = Path(out_dir) / seg_id
        d.mkdir(parents=True)
        docs = [dict(store.docs[i], _id=i) for i in store.order]
        (d / "docs.json").write_text(json.dumps(docs))
        (d / "manifest.json").write_text(json.dumps({
            "segment_id": seg_id, "version": head, "n_docs": len(docs),
            "created": f"2026-11-09T12:00:{counter['n']:02d}Z"}))
        built.append(d)
        return d

    def load_index(path: Path, device: str):
        m = json.loads((Path(path) / "manifest.json").read_text())
        if over_budget_on and m["segment_id"] in over_budget_on:
            raise OverBudget("2 * device_nbytes + work_nbytes > budget")
        return FakeIndex(json.loads((Path(path) / "docs.json").read_text()), version=m["version"], device=device,
                         delay=delay)

    p = Ports(open_store, ingest, prepare, build_segment, load_index)
    p.stores = stores  # type: ignore[attr-defined]
    return p


def write_source(path: Path, n: int = 20) -> str:
    rows = [{"_id": f"P{i:03d}", "title": f"{'red' if i % 2 else 'blue'} widget {i}", "price": float(i), "stock": i}
            for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return str(path)


MAPPING = {"_meta": {"engine": {"id_field": "_id", "hot": ["price", "stock"]}},
           "properties": {"title": {"type": "text"}, "price": {"type": "float"}, "stock": {"type": "integer"}}}
