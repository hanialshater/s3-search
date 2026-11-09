"""Serving against a fake engine (serving/tests/fakes.py). No engine or dataplane needed."""
import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from serving.app import create_app, make_server
from serving.cache import LRUCache
from serving.config import ServingConfig
from serving.errors import classify
from serving.metrics import Metrics
from serving.scheduler import BatchScheduler

from .fakes import MAPPING, FakeIndex, fake_ports, write_source


@pytest.fixture
def make(tmp_path):
    made = []

    def _make(**kw):
        ports = kw.pop("ports", None) or fake_ports()
        cfg = ServingConfig(data_dir=str(tmp_path / "data"), max_wait_ms=kw.pop("max_wait_ms", 20.0), **kw)
        app = create_app(cfg, ports)
        made.append(app)
        return app, ports

    yield _make
    for a in made:
        a.close()


def call(app, method, path, obj=None, raw=None, headers=None):
    body = raw if raw is not None else (b"" if obj is None else json.dumps(obj).encode())
    r = app.handle(method, path, body, headers)
    return r.status, (r.json() if r.content_type == "application/json" else r.body.decode()), r.headers


def create(app, tmp_path, name="shop", n=20):
    src = write_source(tmp_path / f"{name}.jsonl", n)
    st, body, _ = call(app, "PUT", f"/indexes/{name}", {"mapping": MAPPING, "source": {"path": src}})
    assert st == 200, body


@pytest.fixture
def idx(make, tmp_path):
    app, ports = make()
    create(app, tmp_path)
    return app, ports


# ---- create, search, errors -------------------------------------------------------------------------------------
def test_create_then_search(idx):
    app, _ = idx
    st, body, h = call(app, "POST", "/indexes/shop/_search", {"query": {"match": {"title": "red"}}, "size": 3})
    assert st == 200 and body["hits"]["total"] == 10 and len(body["hits"]["hits"]) == 3
    assert h["X-S3-Cache"] == "miss" and "X-Request-Id" in h and "X-S3-Queue-Ms" in h


def test_create_conflict_and_bad_name(idx):
    app, _ = idx
    st, body, _ = call(app, "PUT", "/indexes/shop", {"mapping": MAPPING, "source": {"path": "x"}})
    assert st == 409 and body["error"]["type"] == "index_exists"
    st, body, _ = call(app, "PUT", "/indexes/Bad%20Name", {"mapping": MAPPING, "source": {"path": "x"}})
    assert st == 400 and body["error"]["type"] == "invalid_request"
    st, body, _ = call(app, "PUT", "/indexes/nosrc", {"mapping": MAPPING})
    assert st == 400 and body["error"]["path"] == "source.path"


def test_failed_create_leaves_nothing(make, tmp_path):
    app, _ = make()
    st, _, _ = call(app, "PUT", "/indexes/broken", {"mapping": MAPPING, "source": {"path": str(tmp_path / "missing")}})
    assert st == 500
    assert not (tmp_path / "data" / "broken").exists()
    assert call(app, "GET", "/indexes/broken")[0] == 404


def test_named_errors(idx):
    app, _ = idx
    st, body, _ = call(app, "POST", "/indexes/shop/_search", {"query": {"fuzzy": {}}})
    assert st == 400 and body["error"] == {"type": "unsupported_query", "path": "query.fuzzy",
                                          "reason": "not in the subset"}
    st, body, _ = call(app, "POST", "/indexes/shop/_search", {"from": 9999, "size": 10})
    assert st == 400 and body["error"]["type"] == "invalid_request"   # the subclass is named before its parent
    st, body, _ = call(app, "POST", "/indexes/shop/_search", raw=b"{nope")
    assert st == 400 and body["error"]["type"] == "invalid_request"
    assert call(app, "POST", "/indexes/ghost/_search", {})[0] == 404
    assert call(app, "GET", "/indexes/shop/_search")[0] == 405
    assert call(app, "POST", "/nowhere", {})[0] == 404
    assert classify(MemoryError("x"))[0] == 500
    assert "s3_errors_total" in app.metrics.render()


def test_bad_request_does_not_fail_its_batch_neighbours(idx):
    app, _ = idx
    reqs = [{"query": {"match": {"title": "red"}}}, {"query": {"fuzzy": {}}}, {"query": {"match_all": {}}}]
    with ThreadPoolExecutor(3) as ex:
        out = list(ex.map(lambda r: call(app, "POST", "/indexes/shop/_search", r)[0], reqs))
    assert out == [200, 400, 200]


# ---- batching ---------------------------------------------------------------------------------------------------
def test_scheduler_groups_concurrent_requests():
    index = FakeIndex([{"_id": "a", "title": "x"}])
    s = BatchScheduler(lambda: index, max_batch=8, max_wait_ms=100, metrics=Metrics())
    futs = [s.submit({"query": {"match_all": {}}}) for _ in range(20)]
    [f.result(5) for f in futs]
    s.close()
    assert sum(index.batch_sizes) == 20 and max(index.batch_sizes) == 8      # capped by max_batch
    assert len(index.batch_sizes) <= 4


def test_scheduler_max_wait_bounds_added_latency():
    index = FakeIndex([{"_id": "a", "title": "x"}])
    s = BatchScheduler(lambda: index, max_batch=64, max_wait_ms=30, metrics=Metrics())
    t0 = time.perf_counter()
    _, wait = s.submit({}).result(5)
    took = time.perf_counter() - t0
    s.close()
    assert 0.02 < took < 0.5 and wait >= 25                                   # a lone request waits ~max_wait
    s2 = BatchScheduler(lambda: index, max_batch=64, max_wait_ms=0, metrics=Metrics())
    _, wait0 = s2.submit({}).result(5)
    s2.close()
    assert wait0 < 25


def test_http_requests_are_batched(make, tmp_path):
    app, _ = make(max_wait_ms=60, cache_size=0)
    create(app, tmp_path)
    mi = app.manager.get("shop")
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda i: call(app, "POST", "/indexes/shop/_search",
                                         {"query": {"match": {"title": "widget"}}, "size": i + 1})[0], range(12)))
    assert res == [200] * 12
    assert max(mi.index.batch_sizes) > 1 and sum(mi.index.batch_sizes) == 12
    assert app.metrics.value("s3_batches_total", index="shop") == len(mi.index.batch_sizes)


def test_msearch_is_one_search_batch(idx):
    app, _ = idx
    mi = app.manager.get("shop")
    reqs = [{"query": {"match": {"title": "red"}}, "size": n} for n in (1, 2, 3)]
    st, body, _ = call(app, "POST", "/indexes/shop/_msearch", reqs)
    assert st == 200 and [len(b["hits"]["hits"]) for b in body] == [1, 2, 3]
    assert mi.index.batch_sizes == [3]
    assert call(app, "POST", "/indexes/shop/_msearch", {"not": "a list"})[0] == 400


# ---- cache ------------------------------------------------------------------------------------------------------
def test_lru_evicts_oldest_and_returns_copies():
    c = LRUCache(2)
    c.put("a", {"v": 1})
    c.put("b", {"v": 2})
    c.get("a")
    c.put("c", {"v": 3})
    assert c.get("b") is None and c.get("a") == {"v": 1} and len(c) == 2
    got = c.get("a")
    got["v"] = 99
    assert c.get("a") == {"v": 1}
    assert LRUCache(0).get("x") is None


def test_cache_hit_then_invalidated_by_patch_and_version_change(idx):
    app, _ = idx
    req = {"query": {"match": {"title": "red"}}, "size": 2}

    def state():
        return call(app, "POST", "/indexes/shop/_search", req)[2]["X-S3-Cache"]

    assert [state(), state()] == ["miss", "hit"]
    st, body, _ = call(app, "POST", "/indexes/shop/_docs", [{"op": "patch", "_id": "P001", "fields": {"price": 1.5}}])
    assert st == 200 and body["applied"]["patched"] == 1
    assert state() == "miss"                              # a patch moves the epoch
    assert state() == "hit"
    call(app, "POST", "/indexes/shop/_reindex", {})
    assert state() == "miss"                              # a new segment: another key
    r = app.handle("POST", "/indexes/shop/_search",
                   json.dumps({"size": 2, "query": {"match": {"title": "red"}}}).encode())
    assert r.headers["X-S3-Cache"] == "hit"               # reordered keys are the same query
    assert app.metrics.value("s3_cache_hits_total", index="shop") >= 3


# ---- docs, versions, reindex, rollback --------------------------------------------------------------------------
def test_docs_patch_delete_visible_at_once_upsert_after_reindex(idx):
    app, _ = idx
    mi = app.manager.get("shop")
    st, body, _ = call(app, "POST", "/indexes/shop/_docs", [
        {"op": "patch", "_id": "P003", "fields": {"price": 0.5, "stock": 0}},
        {"op": "delete", "_id": "P001"},
        {"op": "upsert", "_id": "NEW", "doc": {"title": "red gadget", "price": 3.0, "stock": 1}},
        {"op": "upsert", "_id": "BAD", "doc": {"price": 3.0}}])
    assert st == 200 and body["first_offset"] == 21 and body["last_offset"] == 23
    assert body["rejected"][0]["_id"] == "BAD"
    assert body["applied"] == {"patched": 2, "deleted": 1, "pending_upserts": 1}   # patched counts field values
    assert mi.index.docs["P003"]["price"] == 0.5 and not mi.index.live["P001"]
    assert mi.index.version == 23

    def total(word="red"):
        return call(app, "POST", "/indexes/shop/_search", {"query": {"match": {"title": word}}})[1]["hits"]["total"]

    assert total() == 9 and total("gadget") == 0          # P001 deleted; NEW not visible before the rebuild
    call(app, "POST", "/indexes/shop/_reindex", {})
    assert total() == 10 and total("gadget") == 1         # rebuilt from the store: P001 gone, NEW present


def test_docs_validation(idx):
    app, _ = idx
    assert call(app, "POST", "/indexes/shop/_docs", [])[0] == 400
    assert call(app, "POST", "/indexes/shop/_docs", [{"op": "zap", "_id": "a"}])[0] == 400
    assert call(app, "POST", "/indexes/shop/_docs", [{"op": "patch", "_id": "a"}])[1]["error"]["path"] == "[0].fields"


def test_versions_reindex_rollback(idx):
    app, _ = idx
    mi = app.manager.get("shop")
    v1 = call(app, "GET", "/indexes/shop/versions")[1]
    assert len(v1) == 1 and v1[0]["live"] and v1[0]["version"] == 20
    call(app, "POST", "/indexes/shop/_docs", [{"op": "delete", "_id": "P002"}])
    st, r, _ = call(app, "POST", "/indexes/shop/_reindex", {})
    assert st == 200 and r["version"] == 21
    v2 = call(app, "GET", "/indexes/shop/versions")[1]
    assert [v["live"] for v in v2] == [False, True]
    st, r, _ = call(app, "POST", "/indexes/shop/_rollback", {"segment_id": v1[0]["segment_id"]})
    assert st == 200 and r["segment_id"] == v1[0]["segment_id"]
    # rolled back to the older segment, then the log after it is replayed: the delete still holds
    assert not mi.index.live["P002"] and mi.index.version == 21
    err = call(app, "POST", "/indexes/shop/_rollback", {"segment_id": "seg-nope"})[1]["error"]
    assert err["type"] == "segment_not_found"
    assert call(app, "POST", "/indexes/shop/_rollback", {})[0] == 400


def test_old_segments_are_pruned_but_live_one_kept(make, tmp_path):
    app, _ = make(keep_segments=2)
    create(app, tmp_path)
    for _ in range(4):
        call(app, "POST", "/indexes/shop/_reindex", {})
    v = call(app, "GET", "/indexes/shop/versions")[1]
    assert len(v) == 2 and v[-1]["live"]


def test_over_budget_swap_is_refused_and_old_index_keeps_serving(make, tmp_path):
    app, _ = make(ports=fake_ports(over_budget_on={"seg-20-00000002"}))
    create(app, tmp_path)
    st, body, _ = call(app, "POST", "/indexes/shop/_reindex", {})
    assert st == 409 and body["error"]["type"] == "over_budget"
    assert call(app, "POST", "/indexes/shop/_search", {"size": 1})[0] == 200
    assert call(app, "GET", "/indexes/shop")[1]["segment_id"] == "seg-20-00000001"


def test_swap_under_load_never_fails_a_query(make, tmp_path):
    app, _ = make(max_wait_ms=2)
    create(app, tmp_path)
    stop, bad = threading.Event(), []

    def hammer():
        i = 0
        while not stop.is_set():
            st, body, _ = call(app, "POST", "/indexes/shop/_search",
                               {"query": {"match": {"title": "red"}}, "size": 1 + i % 5})
            if st != 200 or body["hits"]["total"] != 10:
                bad.append((st, body))
            i += 1

    ts = [threading.Thread(target=hammer) for _ in range(4)]
    for t in ts:
        t.start()
    for _ in range(5):
        call(app, "POST", "/indexes/shop/_reindex", {})
    stop.set()
    for t in ts:
        t.join()
    assert bad == []


def test_source_from_the_store(idx):
    app, _ = idx
    st, body, _ = call(app, "POST", "/indexes/shop/_search",
                       {"query": {"match": {"title": "red"}}, "size": 2, "_source": ["title"]})
    assert st == 200 and all(set(h["_source"]) == {"title"} for h in body["hits"]["hits"])
    body = call(app, "POST", "/indexes/shop/_search", {"size": 1, "_source": True})[1]
    assert set(body["hits"]["hits"][0]["_source"]) >= {"title", "price", "stock"}
    assert "_source" not in call(app, "POST", "/indexes/shop/_search", {"size": 1})[1]["hits"]["hits"][0]


# ---- health, metrics, preload -----------------------------------------------------------------------------------
def test_health_ready_metrics(make, tmp_path):
    app, _ = make()
    assert call(app, "GET", "/healthz")[0] == 200
    assert call(app, "GET", "/readyz")[0] == 503                 # no index yet
    create(app, tmp_path)
    assert call(app, "GET", "/readyz")[0] == 200
    call(app, "POST", "/indexes/shop/_search", {"query": {"match": {"title": "red"}}})
    app.draining = True
    assert call(app, "GET", "/readyz")[0] == 503 and call(app, "GET", "/healthz")[0] == 200
    st, text, _ = call(app, "GET", "/metrics")
    assert st == 200
    for needle in ('s3_http_requests_total{route="search",status="200"}',
                   's3_search_requests_total{index="shop",outcome="ok"} 1',
                   's3_stage_ms_bucket{index="shop",stage="retrieve",le="+Inf"} 1',
                   's3_batch_size_count{index="shop"} 1', 's3_index_version{index="shop"} 20',
                   's3_swaps_total{index="shop",kind="load"} 1', "# TYPE s3_queue_wait_ms histogram"):
        assert needle in text, needle


def test_preload_loads_a_pinned_segment_and_replica_has_no_store(make, tmp_path):
    app, ports = make()
    create(app, tmp_path)
    first = call(app, "GET", "/indexes/shop/versions")[1][0]["segment_id"]
    call(app, "POST", "/indexes/shop/_docs", [{"op": "delete", "_id": "P000"}])
    call(app, "POST", "/indexes/shop/_reindex", {})
    # a replica pod: same data dir, no store file (the fake store is in memory, so none exists on disk)
    replica = create_app(ServingConfig(data_dir=str(tmp_path / "data"), preload=[("shop", first)]), ports)
    try:
        replica.preload()
        assert replica.handle("GET", "/readyz").status == 200
        assert replica.manager.get("shop").segment_id == first
        r = replica.handle("POST", "/indexes/shop/_search", b'{"size": 1, "_source": true}')
        assert r.status == 400 and r.json()["error"]["path"] == "_source"
    finally:
        replica.close()


# ---- the real server over a socket ------------------------------------------------------------------------------
def test_http_server_keepalive_and_headers(make, tmp_path):
    app, _ = make()
    create(app, tmp_path)
    server = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        req = urllib.request.Request(url + "/indexes/shop/_search", data=b'{"size": 2}', method="POST",
                                     headers={"X-Request-Id": "abc"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 200 and r.headers["X-Request-Id"] == "abc" and len(json.load(r)["hits"]["hits"]) == 2
        with urllib.request.urlopen(url + "/metrics") as r:
            assert r.headers["Content-Type"].startswith("text/plain")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(url + "/indexes/ghost")
        assert e.value.code == 404 and json.load(e.value)["error"]["type"] == "index_not_found"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(importlib.util.find_spec("engine.api") is None, reason="engine.api not importable yet (S1-6)")
def test_real_engine_smoke(make, tmp_path):
    """With the real engine and dataplane (default ports): create from the toy mapping and search."""
    pytest.importorskip("dataplane.kv")
    from serving.ports import default_ports
    app, _ = make(ports=default_ports())
    toy = Path(__file__).resolve().parents[2] / "deploy" / "samples" / "toy-products.jsonl"
    mapping = json.loads((toy.parent / "toy-mapping.json").read_text())
    st, body, _ = call(app, "PUT", "/indexes/toy", {"mapping": mapping, "source": {"path": str(toy)}})
    assert st == 200, body
    st, body, _ = call(app, "POST", "/indexes/toy/_search", {"query": {"match": {"product_title": "shirt"}}})
    assert st == 200 and body["hits"]["total"] > 0 and "total" in body["took_ms"]
