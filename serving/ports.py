"""What serving needs from the engine and the data plane, as five functions.

The defaults call the real packages (``engine.api``, ``dataplane.kv``, ``dataplane.intake``, ``dataplane.indexer``)
exactly as docs/INTERFACES.md §2 to §5 define them, importing lazily so that serving imports and its unit tests run
before those packages exist. Tests pass a ``Ports`` whose functions are fakes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class Ports:
    # open_store(path, mapping) -> Store: apply(changes) -> (first, last), head(), get(ids), changes(after, limit)
    open_store: Callable[[Path, dict], Any]
    # ingest(source_path, mapping, store, rejected_path) -> report with .accepted/.rejected
    ingest: Callable[[str, dict, Any, Path], Any]
    # prepare(changes, mapping) -> (changes ready for store.apply, rejected list of dicts)
    prepare: Callable[[list[dict], dict], tuple[list[dict], list[dict]]]
    # build_segment(store, mapping, out_dir) -> segment directory (version = store head)
    build_segment: Callable[[Any, dict, Path], Path]
    # load_index(segment_dir, device) -> Index (engine.api.Index.from_segment)
    load_index: Callable[[Path, str], Any]
    # load_ranker(path) -> model with predict(X) (for Index.add_ranker); a JSON file written by GBDT.to_dict()
    load_ranker: Callable[[Path], Any] | None = None


def default_ports() -> Ports:
    def open_store(path: Path, mapping: dict):
        from dataplane.kv import Store
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return Store(path, mapping)

    def ingest(source: str, mapping: dict, store, rejected: Path):
        from dataplane.intake import ingest as _ingest
        return _ingest(source, mapping, store, rejected)

    def prepare(changes: list[dict], mapping: dict):
        from dataplane.intake import enrich, validate
        out: list[dict] = []
        rejected: list[dict] = []
        for ch in changes:
            if ch.get("op") != "upsert":
                out.append(ch)
                continue
            res = validate(ch["doc"], mapping)
            if type(res).__name__ == "Rejection":
                rejected.append({"_id": getattr(res, "_id", None) or ch.get("_id"), "reason": res.reason,
                                 "field": res.field, "detail": res.detail})
            else:
                out.append({**ch, "doc": res})
        ups = [c for c in out if c.get("op") == "upsert"]
        if ups:
            for c, doc in zip(ups, enrich([c["doc"] for c in ups], mapping, None)):
                c["doc"] = doc
        return out, rejected

    def build_segment(store, mapping: dict, out_dir: Path) -> Path:
        from dataplane.indexer import build
        return Path(build(store, mapping, out_dir))

    def load_index(path: Path, device: str):
        from engine.api import Index
        return Index.from_segment(path, device=device)

    def load_ranker(path: Path):
        # engine.api has no ranker save/load; the model class has to_dict/from_dict, so a ranker file is its JSON.
        from engine.rank.gbdt import GBDT
        return GBDT.from_dict(json.loads(Path(path).read_text()))

    return Ports(open_store, ingest, prepare, build_segment, load_index, load_ranker)


def read_manifest(segment_dir: Path) -> dict:
    return json.loads((Path(segment_dir) / "manifest.json").read_text())
