"""S8-7, the write-stream judge (owner: evaluation-steward, P6). Built before S8-4 lands; it judges nothing until the
coordinator asks. Every number is CPU, development only.

    taskset -c 0,1 python -m referee.stream [--duration=60] [--rate=200] [--out=results/judge-s8.json]

What it measures, for S8's pre-registered predictions (the lab's `company/milestones.json`, S8):

1. **The stream** (prediction 1). ESCI-sample's first `BASE` products are ingested and built into a base segment; a
   `Replica(store, segment, "cpu").start(interval)` runs the data plane's own poller. A writer **process** applies
   `rate` changes a second to the store (`store.apply`, batches every 50 ms): new products (the rest of
   ESCI-sample), full-text edits (another product's text, with a unique marker word in the title) and deletes of
   base products (each used at most once), in the `MIX` proportions.
   - **Visibility** is timed from `store.apply` returning to the moment a probe query sees the change: a new id is
     found by `terms` on `product_id`, an edit by its marker in `product_title`, a delete by its id's absence. The
     probe runs every 10 ms, so the resolution is about 10 ms plus a probe's time. p50, p99 and max over every change.
   - **Quiesced equality**: after the writer stops and the replica has caught up (its version equals the store's
     head, and every change is visible), `dataplane.indexer.build(store, ...)` at the same version and
     `Index.from_segment` answer every fit query (size 100) like the replica: totals and ids in order exactly, scores
     within 1e-4.
2. **Query latency under the stream** (prediction 3). An **open loop**: fit queries at batch 1 with Poisson
   arrivals at `qps` (100/s), each timed from its scheduled arrival; first for `IDLE_S` seconds with no writes
   (idle), then during the stream. p50, p99 and the rise of p99 over idle. Why this design (fixed before any judged
   run, with what the steward had seen) is written above `_State` below.
   - **No failed or partial query across merge swaps.** A monitor samples `len(index.segments)` every 20 ms; a drop is
     a merge swap. Every tenth query is a `match_all` (size 0) whose total must equal the live count at the replica's
     version (known from the writer's own changes) whenever the version did not move during the call; a merge keeps
     the version, so a partial result shows as a wrong total. Errors and wrong totals are counted overall and inside
     merge-swap windows. If no merge swap is observed, the run says so: that part of prediction 3 is then not judged.
3. **`_bulk` over HTTP** (prediction 4): a real `python -m serving`; all 48,639 products as `index` actions in one
   NDJSON body, timed; every action must have an item, applied (2xx) or rejected with a named `s3_reason`. A second,
   mixed request (`create` of existing ids, `update` of missing ids, `delete`, `index` without an id) checks the
   accounting of each op.
4. **Intake and build speed** (prediction 2): lexical intake docs/s; build from the store; intake that fits the
   tfidf-svd-128 encoder, timed as the total, the engine's part (time inside `engine.encode.TfidfSvdEncoder`) and the
   data plane's part (the rest).

**Load.** The judge waits (up to 5 minutes) for a quiet machine, a 1-minute load average at or below 0.5, and
refuses to start otherwise; it samples the load during the run and marks the timings not judged if the mean exceeds
the cores it may use plus 0.75 (`referee/loadguard.py`).

**If S8-4 is absent** (the updater does not make an upserted document searchable), the stream part stops with
`UpdaterAbsent` and a message saying what it waited for, instead of measuring something else.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from referee import bench, checks, loader

MIX = {"new": 0.40, "edit": 0.45, "delete": 0.15}
BASE = 40_000
IDLE_S = 15.0
SEED = 20261111
_MARK_LETTERS = "bcdfghjklmnpqrtvwxz"   # no vowels, no s or y: the Porter stemmer leaves the word alone


class UpdaterAbsent(RuntimeError):
    """The data plane's updater does not do what S8-4 documents (upserts searchable through delta segments)."""


def marker(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n, len(_MARK_LETTERS))
        s += _MARK_LETTERS[r]
    return "zq" + s


def _doc(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}


def _pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q)) if len(xs) else None


# ---- the capability probe --------------------------------------------------------------------------------------------

def require_s8_updater(store, replica, sample_doc: dict, wait_s: float = 5.0):
    """Upsert one new document and wait for it to become searchable through the replica; raise UpdaterAbsent if not."""
    for name in ("start", "stop", "poll", "search", "search_batch", "stats", "index", "version"):
        if not hasattr(replica, name):
            raise UpdaterAbsent(f"Replica has no `{name}` (INTERFACES section 3): the updater is not the documented one")
    if not hasattr(replica.index, "segments"):
        raise UpdaterAbsent("the replica's Index has no `segments` (S8-2): not a multi-segment index")
    doc = dict(sample_doc, product_id="S8PROBE-0001")
    store.apply([{"op": "upsert", "_id": doc["product_id"], "doc": doc}])
    t = time.perf_counter()
    while time.perf_counter() - t < wait_s:
        r = replica.search({"query": {"bool": {"filter": [{"terms": {"product_id": [doc["product_id"]]}}]}},
                            "size": 1})
        if r["hits"]["total"] == 1:
            store.apply([{"op": "delete", "_id": doc["product_id"]}])
            return time.perf_counter() - t
        time.sleep(0.05)
    store.apply([{"op": "delete", "_id": doc["product_id"]}])
    raise UpdaterAbsent(f"an upserted document was not searchable through the replica within {wait_s} s with its "
                        "poller running: the updater does not build delta segments yet (S8-4, INTERFACES section 3: "
                        "`poll` counts upserts as pending until a full build)")


# ---- the stream ------------------------------------------------------------------------------------------------------
#
# The query design (fixed 2026-11-10, before any judged S8 run). Prediction 3 says "query p99 at batch 1" without
# saying how queries are driven; its intent is a serving replica under realistic traffic, so:
# - the writer is a separate process (`_writer_proc`), as in a deployment, where writes reach the store through
#   intake or `_bulk` while the replica's process only polls the store. It applies `rate` changes a second in
#   batches every 50 ms and reports each batch (offsets, `time.monotonic()` after `store.apply`) over a queue;
#   CLOCK_MONOTONIC is system-wide on Linux, so visibility is timed across the two processes;
# - queries are an open loop: Poisson arrivals at `qps` (default 100/s), each sent at batch 1 at its scheduled time
#   (at once if the client is behind), and timed from the scheduled time, so a query that queues behind a slow one
#   counts its wait. A closed back-to-back loop measures something else: one client saturating the replica (and,
#   in one process, starving the writer of Python's lock), not traffic.
# - edit and delete targets are base products, each used at most once, so the writer needs nothing back from probes.
# Disclosed: before writing this, the steward saw the dataplane-engineer's development numbers (refresh 22-29 ms;
# visibility p99 534-721 ms; query p99 4-6x idle, mostly lock waits) and the note that the loop design moves p99.

class _State:
    def __init__(self, base_ids, base_head):
        self.lock = threading.Lock()
        self.pending_new, self.pending_edit, self.pending_del = {}, {}, {}   # id -> t_applied (edit: (marker, t))
        self.visible_ms = {"new": [], "edit": [], "delete": []}
        self.live_at = [(base_head, len(base_ids))]   # (offset, live count after it), ascending
        self.errors = []
        self.writer_done = threading.Event()
        self.write_s = None

    def live_count(self, version):
        i = np.searchsorted([o for o, _ in self.live_at], version, side="right") - 1
        return self.live_at[max(i, 0)][1]


def _plan_changes(pool, texts, targets, rate, duration_s, seed):
    """Every batch the writer will apply: [[(kind, change, marker)], ...], one batch per 50 ms tick."""
    rng = np.random.default_rng(seed)
    kinds, probs = list(MIX), np.array(list(MIX.values()))
    per, carry, n_new, n_mark, t_i = rate * 0.05, 0.0, 0, 0, 0
    batches = []
    for _ in range(int(round(duration_s / 0.05))):
        carry += per
        k = int(carry)
        carry -= k
        batch = []
        for kind in rng.choice(kinds, size=k, p=probs / probs.sum()):
            if kind == "new" and n_new < len(pool):
                doc = _doc(pool[n_new])
                n_new += 1
                batch.append(("new", {"op": "upsert", "_id": doc["product_id"], "doc": doc}, None))
            elif kind in ("edit", "delete") and t_i < len(targets):
                pid = targets[t_i]
                t_i += 1
                if kind == "edit":
                    src = _doc(texts[int(rng.integers(len(texts)))])
                    mk = marker(n_mark)
                    n_mark += 1
                    doc = dict(src, product_id=pid, product_title=f"{mk} {src.get('product_title') or ''}")
                    batch.append(("edit", {"op": "upsert", "_id": pid, "doc": doc}, mk))
                else:
                    batch.append(("delete", {"op": "delete", "_id": pid}, None))
        batches.append(batch)
    return batches


def _writer_proc(store_path, mapping, batches, queue):
    """The writer process: applies each batch on its 50 ms tick, reports (first offset, t_applied, meta)."""
    from dataplane.kv import Store
    store = Store(store_path, mapping)
    t_start = time.monotonic()
    for i, batch in enumerate(batches):
        due = t_start + 0.05 * i
        delay = due - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if batch:
            first, _ = store.apply([c for _, c, _ in batch])
            queue.put((first, time.monotonic(), [(k, c["_id"], m) for k, c, m in batch]))
    queue.put(("done", time.monotonic() - t_start, None))


def _writer_thread(store, batches, queue):
    t_start = time.monotonic()
    for i, batch in enumerate(batches):
        delay = t_start + 0.05 * i - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if batch:
            first, _ = store.apply([c for _, c, _ in batch])
            queue.put((first, time.monotonic(), [(k, c["_id"], m) for k, c, m in batch]))
    queue.put(("done", time.monotonic() - t_start, None))


def _drain(queue, st):
    """Moves the writer's reports into the state: pending changes, and the live count after each offset."""
    while True:
        first, t, meta = queue.get()
        if first == "done":
            st.write_s = t
            st.writer_done.set()
            return
        with st.lock:
            count = st.live_at[-1][1]
            for i, (kind, pid, mk) in enumerate(meta):
                if kind == "new":
                    st.pending_new[pid] = t
                    count += 1
                elif kind == "edit":
                    st.pending_edit[pid] = (mk, t)
                else:
                    st.pending_del[pid] = t
                    count -= 1
                st.live_at.append((first + i, count))


def _probe_once(replica, st):
    with st.lock:
        new, edit, dele = list(st.pending_new), dict(st.pending_edit), list(st.pending_del)
    seen = []
    for i in range(0, len(new), 500):
        ids = new[i:i + 500]
        r = replica.search({"query": {"bool": {"filter": [{"terms": {"product_id": ids}}]}}, "size": len(ids)})
        seen += [("new", h["_id"]) for h in r["hits"]["hits"]]
    items = list(edit.items())
    for i in range(0, len(items), 200):
        chunk = items[i:i + 200]
        r = replica.search({"query": {"match": {"product_title": " ".join(m for _, (m, _) in chunk)}},
                            "size": 2 * len(chunk)})
        want = {pid for pid, _ in chunk}
        seen += [("edit", h["_id"]) for h in r["hits"]["hits"] if h["_id"] in want]
    for i in range(0, len(dele), 500):
        ids = dele[i:i + 500]
        r = replica.search({"query": {"bool": {"filter": [{"terms": {"product_id": ids}}]}}, "size": len(ids)})
        still = {h["_id"] for h in r["hits"]["hits"]}
        seen += [("delete", pid) for pid in ids if pid not in still]
    now = time.monotonic()
    with st.lock:
        for kind, pid in seen:
            if kind == "new" and pid in st.pending_new:
                st.visible_ms["new"].append((now - st.pending_new.pop(pid)) * 1e3)
            elif kind == "edit" and pid in st.pending_edit:
                st.visible_ms["edit"].append((now - st.pending_edit.pop(pid)[1]) * 1e3)
            elif kind == "delete" and pid in st.pending_del:
                st.visible_ms["delete"].append((now - st.pending_del.pop(pid)) * 1e3)


def _prober(replica, st, stop):
    while not stop.is_set():
        try:
            _probe_once(replica, st)
        except Exception as e:  # noqa: BLE001  (a probe failure is a failed query; recorded, not hidden)
            st.errors.append(f"probe: {type(e).__name__}: {str(e)[:200]}")
        time.sleep(0.01)


def _querier(replica, st, reqs, stop, out, qps=100.0, seed=SEED, witness_every=10):
    """The open loop: Poisson arrivals at qps, batch 1, latency from the scheduled arrival."""
    rng = np.random.default_rng(seed)
    i, due = 0, time.monotonic()
    while not stop.is_set():
        due += float(rng.exponential(1.0 / qps))
        delay = due - time.monotonic()
        if delay > 0:
            if stop.wait(delay):
                break
        witness = i % witness_every == witness_every - 1
        req = {"query": {"match_all": {}}, "size": 0} if witness else reqs[i % len(reqs)]
        v0 = replica.version
        try:
            r = replica.search(req)
            err = None
        except Exception as e:  # noqa: BLE001
            r, err = None, f"{type(e).__name__}: {str(e)[:200]}"
        done = time.monotonic()
        v1 = replica.version
        row = {"t0": due, "ms": (done - due) * 1e3, "witness": witness, "error": err, "wrong_total": None}
        if witness and r is not None and v0 == v1:
            with st.lock:
                exp = st.live_count(v0)
            if r["hits"]["total"] != exp:
                row["wrong_total"] = (r["hits"]["total"], exp, v0)
        out.append(row)
        i += 1


def _monitor(replica, stop, out):
    while not stop.is_set():
        try:
            out.append((time.monotonic(), len(replica.index.segments), replica.version))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.02)


def run_stream(store, replica, mapping, base_ids, base_head, pool, texts, fit_reqs, rate=200, duration_s=60.0,
               idle_s=IDLE_S, settle_timeout_s=60.0, seed=SEED, build=None, out_dir=None, qps=100.0,
               writer="process"):
    """The stream, the idle and stream latencies, merge-swap witnesses, and quiesced equality with a rebuild.
    writer: "process" (the judged design) or "thread" (in-process, for self-tests and comparison only)."""
    import multiprocessing as mp
    import queue as _queue

    import engine.api as api
    from dataplane import indexer
    st = _State(base_ids, base_head)
    page1 = [{**r, "size": 10} for r in fit_reqs]
    targets = list(np.random.default_rng(seed + 1).permutation(np.asarray(base_ids, dtype=object)))
    batches = _plan_changes(pool, texts, targets, rate, duration_s, seed)
    res = {"rate": rate, "duration_s": duration_s, "mix": MIX, "qps": qps, "writer": writer,
           "query_design": "open loop, Poisson arrivals, batch 1, latency from the scheduled arrival"}

    stop_q = threading.Event()
    idle_rows = []
    th = threading.Thread(target=_querier, args=(replica, st, page1, stop_q, idle_rows, qps, seed), daemon=True)
    th.start()
    time.sleep(idle_s)
    stop_q.set()
    th.join()

    stop_all = threading.Event()
    rows, segs = [], []
    threads = [threading.Thread(target=_querier, args=(replica, st, page1, stop_all, rows, qps, seed + 2), daemon=True),
               threading.Thread(target=_prober, args=(replica, st, stop_all), daemon=True),
               threading.Thread(target=_monitor, args=(replica, stop_all, segs), daemon=True)]
    for t in threads:
        t.start()
    if writer == "process":
        q = mp.get_context("spawn").Queue()
        proc = mp.get_context("spawn").Process(target=_writer_proc, args=(str(store.path), mapping, batches, q))
        proc.start()
    else:
        q = _queue.Queue()
        proc = threading.Thread(target=_writer_thread, args=(store, batches, q), daemon=True)
        proc.start()
    drain = threading.Thread(target=_drain, args=(q, st), daemon=True)
    drain.start()
    st.writer_done.wait(duration_s * 3 + 60)
    proc.join()
    stop_writes_at = time.monotonic()
    # Settle: the replica reaches the store's head and every change is visible.
    while time.monotonic() - stop_writes_at < settle_timeout_s:
        with st.lock:
            pending = len(st.pending_new) + len(st.pending_edit) + len(st.pending_del)
        if not pending and replica.version == store.head():
            break
        time.sleep(0.05)
    settle_s = time.monotonic() - stop_writes_at
    stop_all.set()
    for t in threads:
        t.join()
    with st.lock:
        never = {"new": len(st.pending_new), "edit": len(st.pending_edit), "delete": len(st.pending_del)}
    vis_all = [x for v in st.visible_ms.values() for x in v]
    n_changes = len(st.live_at) - 1
    write_s = st.write_s
    res["writes"] = {"changes": n_changes, "applied_per_s": n_changes / write_s if write_s else None,
                     "by_kind_visible": {k: len(v) for k, v in st.visible_ms.items()}, "never_visible": never,
                     "settle_s": settle_s}
    res["visibility_ms"] = {"p50": _pct(vis_all, 50), "p99": _pct(vis_all, 99), "max": max(vis_all) if vis_all else None,
                            "by_kind_p99": {k: _pct(v, 99) for k, v in st.visible_ms.items()},
                            "resolution": "probe every 10 ms"}

    # Latency, failures, witnesses, merge swaps.
    drops = [(segs[i - 1][0], segs[i][0]) for i in range(1, len(segs)) if segs[i][1] < segs[i - 1][1]]  # monotonic
    in_swap = lambda r: any(a <= r["t0"] + r["ms"] / 1e3 and r["t0"] <= b for a, b in drops)  # noqa: E731
    lat = lambda rs: [r["ms"] for r in rs if not r["witness"] and not r["error"]]  # noqa: E731
    idle_p99, stream_p99 = _pct(lat(idle_rows), 99), _pct(lat(rows), 99)
    res["queries"] = {
        "idle": {"n": len(lat(idle_rows)), "p50": _pct(lat(idle_rows), 50), "p99": idle_p99},
        "stream": {"n": len(lat(rows)), "p50": _pct(lat(rows), 50), "p99": stream_p99},
        "p99_rise": (stream_p99 / idle_p99 - 1.0) if idle_p99 and stream_p99 else None,
        "failed": sum(1 for r in rows + idle_rows if r["error"]) + len(st.errors),
        "failed_examples": ([r["error"] for r in rows if r["error"]] + st.errors)[:5],
        "witnesses": sum(r["witness"] for r in rows), "wrong_totals": [r["wrong_total"] for r in rows if r["wrong_total"]][:5],
        "n_wrong_totals": sum(1 for r in rows if r["wrong_total"]),
        "merge_swaps_observed": len(drops),
        "queries_in_merge_swaps": sum(1 for r in rows if in_swap(r)),
        "failed_or_partial_in_merge_swaps": sum(1 for r in rows if in_swap(r) and (r["error"] or r["wrong_total"])),
        "segments": {"max": max((s for _, s, _ in segs), default=None), "last": segs[-1][1] if segs else None},
    }
    if not drops:
        res["queries"]["note"] = "no merge swap observed in this run: the merge part of prediction 3 is not judged"

    # Quiesced equality with a rebuild from the store at the same version.
    head = store.head()
    seg = (build or indexer.build)(store, mapping, Path(out_dir or tempfile.mkdtemp(prefix="s8-rebuild-")))
    man = json.loads((Path(seg) / "manifest.json").read_text())
    fresh = api.Index.from_segment(seg)
    cmp_ = compare(replica, fresh, fit_reqs)
    res["quiesced"] = {"store_head": head, "replica_version": replica.version, "rebuild_version": man.get("version"),
                       "same_version": head == replica.version == man.get("version"), **cmp_}
    return res


def compare(a, b, reqs, batch=64):
    """Totals and ids in order exactly; scores within 1e-4 (relative to max(1, |score|))."""
    differ, examples, worst = 0, [], 0.0
    for i in range(0, len(reqs), batch):
        ra, rb = a.search_batch(reqs[i:i + batch]), b.search_batch(reqs[i:i + batch])
        for j, (x, y) in enumerate(zip(ra, rb)):
            xi, yi = [h["_id"] for h in x["hits"]["hits"]], [h["_id"] for h in y["hits"]["hits"]]
            xs, ys = [h["_score"] for h in x["hits"]["hits"]], [h["_score"] for h in y["hits"]["hits"]]
            if xs and len(xs) == len(ys):
                worst = max(worst, max(abs(p - q) / max(1.0, abs(q)) for p, q in zip(xs, ys)))
            ok = x["hits"]["total"] == y["hits"]["total"] and xi == yi and all(checks._close(p, q) for p, q in zip(xs, ys))
            if not ok:
                differ += 1
                if len(examples) < 3:
                    examples.append({"query": reqs[i + j]["query"], "totals": [x["hits"]["total"], y["hits"]["total"]],
                                     "first_ids": [xi[:3], yi[:3]]})
    return {"n": len(reqs), "differ": differ, "examples": examples, "max_rel_score_diff": worst}


# ---- intake, build and encoder timings --------------------------------------------------------------------------------

class _EngineEncoderClock:
    """Time spent inside engine.encode.TfidfSvdEncoder's public methods (outermost calls only), in this process."""

    METHODS = ("fit", "fit_transform", "encode")

    def __init__(self):
        self.s, self._depth, self._saved = 0.0, 0, {}

    def __enter__(self):
        from engine import encode
        cls = encode.TfidfSvdEncoder
        for m in self.METHODS:
            if hasattr(cls, m):
                orig = getattr(cls, m)
                self._saved[m] = orig

                def wrapped(*a, _orig=orig, **k):
                    self._depth += 1
                    t = time.perf_counter()
                    try:
                        return _orig(*a, **k)
                    finally:
                        self._depth -= 1
                        if self._depth == 0:
                            self.s += time.perf_counter() - t
                setattr(cls, m, wrapped)
        return self

    def __exit__(self, *exc):
        from engine import encode
        for m, orig in self._saved.items():
            setattr(encode.TfidfSvdEncoder, m, orig)


def encoder_mapping(mapping):
    import copy
    m = copy.deepcopy(mapping)
    m["properties"]["embedding"] = {"type": "knn_vector", "dimension": 128,
                                    "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
    m["_meta"]["engine"]["embeddings"] = {"embedding": {"model": "tfidf-svd-128",
                                                        "inputs": ["product_title", "product_brand", "product_color"]}}
    return m


def intake_timings(catalogue, mapping, work: Path) -> dict:
    from dataplane import indexer, intake
    from dataplane.kv import Store
    src = work / "catalogue.parquet"
    catalogue.to_parquet(src, index=False)
    out = {"n_docs": len(catalogue)}
    st = Store(work / "lex.sqlite", mapping)
    t = time.perf_counter()
    intake.ingest(src, mapping, st, work / "rej-lex.jsonl")
    s = time.perf_counter() - t
    out["lexical_intake"] = {"s": s, "docs_per_s": len(catalogue) / s}
    t = time.perf_counter()
    indexer.build(st, mapping, work / "seg-lex")
    out["build_from_store_s"] = time.perf_counter() - t
    em = encoder_mapping(mapping)
    st2 = Store(work / "enc.sqlite", em)
    with _EngineEncoderClock() as clock:
        t = time.perf_counter()
        intake.ingest(src, em, st2, work / "rej-enc.jsonl")
        total = time.perf_counter() - t
    out["encoder_intake"] = {"total_s": total, "engine_s": clock.s, "dataplane_s": total - clock.s,
                             "note": "engine_s = time inside engine.encode.TfidfSvdEncoder (fit, fit_transform, encode)"}
    return out


# ---- _bulk over HTTP ----------------------------------------------------------------------------------------------------

def _post_ndjson(base, path, lines, timeout=900):
    import urllib.request
    body = ("\n".join(json.dumps(x) for x in lines) + "\n").encode()
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": "application/x-ndjson"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return resp, time.perf_counter() - t


def _account(resp, n_actions, want_ops=None):
    items = resp.get("items", [])
    bad, by = [], {}
    for i, it in enumerate(items):
        if not isinstance(it, dict) or len(it) != 1:
            bad.append(f"item {i}: not one op")
            continue
        (op, body), = it.items()
        st = int(body.get("status", 0))
        key = (op, st)
        by[f"{op} {st}"] = by.get(f"{op} {st}", 0) + 1
        if want_ops is not None and op != want_ops[i]:
            bad.append(f"item {i}: op {op}, sent {want_ops[i]}")
        if st >= 300 and not (body.get("error") or {}).get("type"):
            bad.append(f"item {i}: status {st} without a named error")
        del key
    return {"actions": n_actions, "items": len(items), "all_accounted": len(items) == n_actions and not bad,
            "by_op_status": by, "problems": bad[:5], "errors_flag": resp.get("errors")}


def bulk_http(catalogue, mapping, work: Path) -> dict:
    from referee.judge import Server
    one = work / "one.parquet"
    catalogue.iloc[:1].to_parquet(one, index=False)
    srv = Server(work / "serving")
    try:
        srv.call("PUT", "/indexes/esci", {"mapping": mapping, "source": {"path": str(one)}})
        lines, ops = [], []
        for rec in catalogue.to_dict("records"):
            d = _doc(rec)
            lines += [{"index": {"_id": d["product_id"]}}, d]
            ops.append("index")
        resp, s = _post_ndjson(srv.base, "/indexes/esci/_bulk", lines)
        out = {"all_products": {"s": s, "docs_per_s": len(ops) / s, **_account(resp, len(ops), ops)}}
        ids = list(catalogue["product_id"].iloc[:40])
        mixed, mops = [], []
        for pid in ids[:10]:
            mixed += [{"create": {"_id": pid}}, {"product_id": pid, "product_title": "dup"}]
            mops.append("create")
        for k in range(10):
            mixed += [{"update": {"_id": f"S8-MISSING-{k}"}}, {"doc": {"product_title": "x"}}]
            mops.append("update")
        for pid in ids[10:20]:
            mixed += [{"delete": {"_id": pid}}]
            mops.append("delete")
        for k in range(5):
            mixed += [{"index": {}}, {"product_title": f"generated id {k}"}]
            mops.append("index")
        for pid in ids[20:25]:
            mixed += [{"update": {"_id": pid}}, {"doc": {"product_title": "edited"}}]
            mops.append("update")
        resp2, s2 = _post_ndjson(srv.base, "/indexes/esci/_bulk", mixed)
        out["mixed"] = {"s": s2, **_account(resp2, len(mops), mops)}
    finally:
        srv.close()
    return out


# ---- the judge -----------------------------------------------------------------------------------------------------------

def setup_esci_sample(work: Path, base_n=BASE):
    from dataplane import indexer, intake
    from dataplane.kv import Store
    ds = loader.load_dataset("esci-sample")
    cat = ds.catalogue
    base, pool = cat.iloc[:base_n], cat.iloc[base_n:]
    src = work / "base.parquet"
    base.to_parquet(src, index=False)
    store = Store(work / "store.sqlite", ds.mapping)
    intake.ingest(src, ds.mapping, store, work / "rej.jsonl")
    seg = indexer.build(store, ds.mapping, work / "segments")
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    fit_reqs = [{**bench.fill(bench.DEFAULT_TEMPLATE, t), "size": 100} for t in q["query"]]
    return ds, store, seg, list(base["product_id"]), pool.to_dict("records"), cat.to_dict("records"), fit_reqs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m referee.stream")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--interval", type=float, default=0.05, help="the replica's poll interval (Replica.start)")
    ap.add_argument("--qps", type=float, default=100.0, help="the open loop's query rate (Poisson arrivals)")
    ap.add_argument("--writer", default="process", choices=("process", "thread"),
                    help="process: the judged design; thread: in-process, for comparison only")
    ap.add_argument("--parts", default="stream,bulk,intake")
    ap.add_argument("--out", default="results/judge-s8.json")
    ap.add_argument("--self-test", action="store_true", help="run the judge's full self-tests on toy, then stop")
    ap.add_argument("--ignore-load", action="store_true",
                    help="measure even on a busy machine (the results are then labelled, never verdicts)")
    a = ap.parse_args(argv)
    if a.self_test:
        import pytest
        return int(pytest.main(["-q", "-p", "no:cacheprovider", str(Path(__file__).parent / "tests" / "test_stream.py")]))
    from dataplane.updater import Replica
    work = Path(tempfile.mkdtemp(prefix="judge-s8-"))
    res = {"kind": "judge-s8", "label": "CPU, development only", "dataset": "esci-sample",
           "threads": __import__("referee").blas_threads()}
    parts = set(a.parts.split(","))
    from referee import loadguard
    try:
        res["load_at_start"] = loadguard.wait_for_quiet() if not a.ignore_load else {"ignored": True,
                                                                                     "load1": loadguard.load1()}
    except loadguard.Busy as e:
        res["not_judged"] = str(e)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(res["not_judged"])
        return 3
    sampler = loadguard.Sampler().__enter__()
    ds = None
    if "stream" in parts:
        ds, store, seg, base_ids, pool, texts, fit_reqs = setup_esci_sample(work)
        replica = Replica(store, seg, "cpu").start(interval=a.interval)
        try:
            res["capability_probe_s"] = require_s8_updater(store, replica, _doc(texts[0]))
            res["stream"] = run_stream(store, replica, ds.mapping, base_ids, store.head(), pool, texts, fit_reqs,
                                       rate=a.rate, duration_s=a.duration, out_dir=work / "rebuild", qps=a.qps,
                                       writer=a.writer)
        except UpdaterAbsent as e:
            res["stream"] = {"not_judged": f"UpdaterAbsent: {e}"}
        finally:
            replica.stop()
    ds = ds or loader.load_dataset("esci-sample")
    if "bulk" in parts:
        res["bulk"] = bulk_http(ds.catalogue, ds.mapping, work)
    if "intake" in parts:
        res["intake"] = intake_timings(ds.catalogue, ds.mapping, work)
    sampler.__exit__(None, None, None)
    res["load_during_run"] = sampler.summary()
    if res["load_during_run"]["overloaded"] or a.ignore_load:
        res["timings_not_judged"] = ("the machine was busy during the run (mean 1-minute load "
                                     f"{res['load_during_run']['load1_mean']:.2f} > {res['load_during_run']['limit']:.2f})"
                                     if not a.ignore_load else "--ignore-load: measured, not judged")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: (v if k != "stream" else {kk: vv for kk, vv in v.items() if kk in
                                                    ("not_judged", "visibility_ms", "queries", "quiesced")})
                      for k, v in res.items()}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
