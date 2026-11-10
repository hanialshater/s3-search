"""The S8-7 write-stream judge, self-tested on toy before S8-4 lands (no verdicts here).

`referee fast` runs the quick ones; the full runs are marked `slow` and run with the judge:
`python -m pytest -m slow referee/tests/test_stream.py` (or `python -m referee.stream --self-test`).

- The real updater as it stands (no delta segments yet) makes the judge stop with UpdaterAbsent.
- A test-only replica with S8-4's documented behaviour (upserts searchable after a poll) runs the whole stream clean.
- Planted errors are caught: a replica that never applies deletes, and one that answers with a partial result.
"""
import tempfile
import threading
import time
from pathlib import Path

import pytest

api = pytest.importorskip("engine.api")
pytest.importorskip("dataplane.kv")

from referee import bench, loader, stream  # noqa: E402


def _toy_store(tmp):
    from dataplane import indexer, intake
    from dataplane.kv import Store
    ds = loader.load_dataset("toy")
    base, pool = ds.catalogue.iloc[:150], ds.catalogue.iloc[150:]
    src = Path(tmp) / "base.parquet"
    base.to_parquet(src, index=False)
    st = Store(Path(tmp) / "s.sqlite", ds.mapping)
    intake.ingest(src, ds.mapping, st, Path(tmp) / "rej.jsonl")
    seg = indexer.build(st, ds.mapping, Path(tmp) / "seg")
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    reqs = [{**bench.fill(bench.DEFAULT_TEMPLATE, t), "size": 100} for t in q["query"]]
    return ds, st, seg, list(base["product_id"]), pool.to_dict("records"), ds.catalogue.to_dict("records"), reqs


class RebuildReplica:
    """S8-4's documented surface with the simplest correct behaviour: each poll that sees changes rebuilds."""

    def __init__(self, store, mapping, drop_deletes=False, partial_every=0):
        self.store, self.mapping = store, mapping
        self.drop_deletes, self.partial_every, self.n = drop_deletes, partial_every, 0
        self._lock, self._stop, self._thread = threading.Lock(), threading.Event(), None
        self.index = self._build()

    def _build(self):
        snap = self.store.snapshot()
        version = snap.attrs.get("version", self.store.head())
        if self.drop_deletes:  # planted: documents deleted in the stream stay searchable
            import pandas as pd
            if not hasattr(self, "_kept"):
                self._kept = {}
            self._kept.update({r["product_id"]: r for r in snap.to_dict("records")})
            live = set(snap["product_id"])
            snap = pd.concat([snap, pd.DataFrame([r for p, r in self._kept.items() if p not in live])],
                             ignore_index=True) if len(self._kept) > len(live) else snap
        idx = api.build_index(snap.reset_index(drop=True), self.mapping)
        idx.set_version(version)
        return idx

    @property
    def version(self):
        return self.index.version

    def poll(self):
        if self.store.head() != self.index.version:
            with self._lock:
                self.index = self._build()

    def search(self, r):
        out = self.index.search(r)
        self.n += 1
        if self.partial_every and self.n % self.partial_every == 0:
            out = {**out, "hits": {**out["hits"], "total": max(0, out["hits"]["total"] - 1)}}
        return out

    def search_batch(self, rs):
        return [self.search(r) for r in rs]

    def stats(self):
        return {}

    def start(self, interval=0.05):
        def run():
            while not self._stop.is_set():
                self.poll()
                self._stop.wait(interval)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()


def test_the_current_updater_is_reported_absent(tmp_path):
    from dataplane.updater import Replica
    ds, st, seg, base_ids, pool, texts, reqs = _toy_store(tmp_path)
    rep = Replica(st, seg, "cpu").start(interval=0.02)
    try:
        if hasattr(rep, "refresh") or "delta" in (rep.stats() or {}):
            pytest.skip("the updater looks like S8-4's: this test is for its absence")
        with pytest.raises(stream.UpdaterAbsent, match="S8-4"):
            stream.require_s8_updater(st, rep, stream._doc(texts[0]), wait_s=0.5)
    finally:
        rep.stop()


def _run(tmp, duration_s=1.5, idle_s=0.4, rate=100, **fake):
    ds, st, seg, base_ids, pool, texts, reqs = _toy_store(tmp)
    rep = RebuildReplica(st, ds.mapping, **fake).start(interval=0.02)
    try:
        stream.require_s8_updater(st, rep, stream._doc(texts[0]), wait_s=2.0)
        return stream.run_stream(st, rep, ds.mapping, base_ids, st.head(), pool, texts, reqs, rate=rate,
                                 duration_s=duration_s, idle_s=idle_s, settle_timeout_s=3.0, out_dir=Path(tmp) / "rb")
    finally:
        rep.stop()


@pytest.mark.slow
def test_a_correct_updater_runs_clean(tmp_path):
    r = _run(tmp_path)
    w = r["writes"]
    assert w["changes"] > 100 and sum(w["never_visible"].values()) == 0
    assert all(n > 0 for n in w["by_kind_visible"].values())
    assert r["visibility_ms"]["p99"] is not None and r["queries"]["failed"] == 0 and r["queries"]["n_wrong_totals"] == 0
    assert r["queries"]["witnesses"] > 0 and r["queries"]["idle"]["n"] > 0 and r["queries"]["stream"]["n"] > 0
    q = r["quiesced"]
    assert q["same_version"] and q["differ"] == 0 and q["n"] == 20


@pytest.mark.slow
def test_planted_lost_deletes_are_caught(tmp_path):
    r = _run(tmp_path, drop_deletes=True)
    assert r["writes"]["never_visible"]["delete"] > 0 or r["quiesced"]["differ"] > 0


def test_planted_partial_results_are_caught(tmp_path):
    """The quick one fast keeps: a short stream, and a replica that answers some queries with a partial result."""
    r = _run(tmp_path, duration_s=0.4, idle_s=0.1, rate=50, partial_every=3)
    assert r["queries"]["n_wrong_totals"] > 0


def test_marker_words_survive_the_analyzer():
    from referee.analysis import analyze
    for n in (0, 1, 18, 19, 500, 12345):
        m = stream.marker(n)
        assert analyze(f"{m} lamp") == [m, "lamp"]
    assert len({stream.marker(n) for n in range(5000)}) == 5000


@pytest.mark.slow
def test_intake_timings_and_bulk_on_toy(tmp_path):
    ds = loader.load_dataset("toy")
    cat = ds.catalogue.drop(columns=["embedding"])
    m = {**ds.mapping, "properties": {k: v for k, v in ds.mapping["properties"].items() if k != "embedding"}}
    t = stream.intake_timings(cat, m, tmp_path)
    e = t["encoder_intake"]
    assert t["lexical_intake"]["docs_per_s"] > 0 and t["build_from_store_s"] > 0
    assert 0 < e["engine_s"] <= e["total_s"] and abs(e["engine_s"] + e["dataplane_s"] - e["total_s"]) < 1e-9
    b = stream.bulk_http(cat, m, tmp_path)
    assert b["all_products"]["all_accounted"], b["all_products"]
    assert b["mixed"]["all_accounted"], b["mixed"]
    assert b["mixed"]["by_op_status"].get("create 409") == 10 and b["mixed"]["by_op_status"].get("update 404") == 10
