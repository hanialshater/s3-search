"""_bulk (OpenSearch NDJSON), refresh modes, refresh_interval and the lag metrics, against the fake engine."""
import http.client
import json
import threading
import time

import pytest

from serving.app import create_app, make_server
from serving.bulk import BulkFormatError, iter_lines, parse_actions
from serving.config import ServingConfig

from .fakes import MAPPING, fake_ports, write_source


def nd(*objs):
    return ("\n".join(o if isinstance(o, str) else json.dumps(o) for o in objs) + "\n").encode()


@pytest.fixture
def app(tmp_path):
    made = []

    def make(**kw):
        a = create_app(ServingConfig(data_dir=str(tmp_path / f"d{len(made)}"), max_wait_ms=2, **{"refresh_interval": "-1", **kw}),
                       fake_ports())
        made.append(a)
        src = write_source(tmp_path / "s.jsonl", 10)
        assert a.handle("PUT", "/indexes/shop", json.dumps({"mapping": MAPPING, "source": {"path": src}}).encode()).status == 200
        return a

    yield make
    for a in made:
        a.close()


def bulk(app, body, query=""):
    r = app.handle("POST", f"/indexes/shop/_bulk{query}", body)
    return r.status, r.json()


def total(app, word="widget"):
    r = app.handle("POST", "/indexes/shop/_search", json.dumps({"query": {"match": {"title": word}}}).encode())
    return r.json()["hits"]["total"]


def item(out, i):
    (op, body), = out["items"][i].items()
    return op, body


def test_every_action_is_accounted_for(app):
    a = app()
    body = nd({"index": {"_id": "N1"}}, {"title": "new widget", "price": 1.0, "stock": 3},       # created
              {"index": {"_id": "P001"}}, {"title": "replaced widget", "price": 2.0, "stock": 1},  # updated
              {"create": {"_id": "P002"}}, {"title": "dup", "price": 1.0, "stock": 1},            # 409
              {"create": {"_id": "N2"}}, {"title": "created widget", "price": 1.0, "stock": 1},   # created
              {"update": {"_id": "P003"}}, {"doc": {"price": 9.5}},                               # patch
              {"update": {"_id": "P004"}}, {"doc": {"title": "merged widget"}},                   # merge, non-hot
              {"update": {"_id": "NOPE"}}, {"doc": {"price": 1}},                                 # 404
              {"update": {"_id": "UP1"}}, {"doc": {"title": "upserted"}, "doc_as_upsert": True},  # created
              {"delete": {"_id": "P005"}},                                                        # deleted
              {"delete": {"_id": "GHOST"}},                                                       # not_found
              {"index": {"_id": "BAD"}}, {"title": 7},                                            # rejected by the data plane
              {"update": {"_id": "P006"}}, {"doc": {"price": 1}, "script": "x"},                  # unsupported
              {"index": {}}, {"title": "auto id"},
              {"update": {"_id": "P007"}}, {"doc": {}},                                           # noop
              {"index": {"_id": "X", "_index": "other"}}, {"title": "wrong index"})
    st, out = bulk(a, body)
    assert st == 200 and out["errors"] is True and len(out["items"]) == 15
    got = [(op, b["status"], b.get("result"), (b.get("error") or {}).get("type")) for op, b in map(lambda i: item(out, i), range(15))]
    assert got == [("index", 201, "created", None), ("index", 200, "updated", None),
                   ("create", 409, None, "version_conflict_engine_exception"), ("create", 201, "created", None),
                   ("update", 200, "updated", None), ("update", 200, "updated", None),
                   ("update", 404, None, "document_missing_exception"), ("update", 201, "created", None),
                   ("delete", 200, "deleted", None), ("delete", 404, "not_found", None),
                   ("index", 400, None, "mapper_parsing_exception"), ("update", 400, None, "illegal_argument_exception"),
                   ("index", 201, "created", None), ("update", 200, "noop", None),
                   ("index", 404, None, "index_not_found_exception")]
    assert item(out, 10)[1]["error"]["s3_reason"] == "bad_type" and item(out, 10)[1]["error"]["field"] == "title"
    seqs = [item(out, i)[1].get("_seq_no") for i in range(15)]
    applied = [s for s in seqs if s is not None]
    assert applied == list(range(11, 11 + len(applied)))           # consecutive store offsets after the 10 created docs
    assert len(item(out, 12)[1]["_id"]) == 32                       # generated id
    mi = a.manager.get("shop")
    store = a.manager.ports.stores[str(mi.store_path)]
    assert store.docs["P003"]["price"] == 9.5 and store.docs["P004"]["title"] == "merged widget"
    assert "P005" not in store.docs and store.docs["P001"]["title"] == "replaced widget"


def test_metrics_count_items_rejections_rate_and_lag(app):
    a = app()
    bulk(a, nd({"index": {"_id": "B1"}}, {"title": "x"}, {"index": {"_id": "B2"}}, {"title": 5}, {"delete": {"_id": "ZZ"}}))
    text = a.handle("GET", "/metrics").body.decode()
    for needle in ('s3_bulk_requests_total{index="shop",refresh="false"} 1', 's3_bulk_items_total{index="shop",op="index",outcome="ok"} 1',
                   's3_bulk_items_total{index="shop",op="index",outcome="error"} 1', 's3_bulk_rejected_total{index="shop",reason="bad_type"} 1',
                   's3_bulk_items_total{index="shop",op="delete",outcome="ok"} 1', "s3_bulk_items_per_second{", 's3_index_lag{index="shop"} 1',
                   's3_index_pending_upserts{index="shop"} 1'):
        assert needle in text, needle


def test_streaming_applies_chunk_by_chunk_before_the_body_ends(app):
    a = app(bulk_chunk=4)
    mi = a.manager.get("shop")
    store = a.manager.ports.stores[str(mi.store_path)]
    lines = [x for i in range(12) for x in ({"index": {"_id": f"S{i}"}}, {"title": f"stream {i}"})]
    pieces = [nd(*lines[i:i + 8]) for i in range(0, len(lines), 8)]        # 4 actions per piece
    seen = []

    def read(n):
        if pieces:
            seen.append(store.head())                                      # what is stored when the next piece is asked for
            return pieces.pop(0)
        return b""

    st, out = a.handle("POST", "/indexes/shop/_bulk", read).status, None
    assert st == 200 and seen[0] == 10 and seen[1] == 14 and seen[2] == 18   # applied as it streams in, not at the end
    assert store.head() == 22


def test_chunk_boundaries_do_not_change_semantics(app):
    a = app(bulk_chunk=2)
    st, out = bulk(a, nd({"index": {"_id": "C1"}}, {"title": "a"}, {"create": {"_id": "C1"}}, {"title": "b"},
                         {"update": {"_id": "C1"}}, {"doc": {"price": 3}}, {"delete": {"_id": "C1"}},
                         {"create": {"_id": "C1"}}, {"title": "again"}))
    assert [item(out, i)[1]["status"] for i in range(5)] == [201, 409, 200, 200, 201]


def test_bad_action_line_ends_the_request_with_what_was_applied(app):
    a = app()
    st, out = bulk(a, nd({"index": {"_id": "G1"}}, {"title": "good"}, "{not json", {"index": {"_id": "G2"}}, {"title": "never"}))
    assert st == 400 and out["error"]["line"] == 3 and len(out["items"]) == 1 and out["errors"] is True
    st, out = bulk(a, nd({"explode": {}}))
    assert st == 400 and "action line" in out["error"]["reason"]
    st, out = bulk(a, nd({"index": {"_id": "H"}}, "[1]", {"index": {"_id": "H2"}}))
    assert st == 200 and item(out, 0)[1]["status"] == 400 and item(out, 1)[1]["status"] == 400   # source missing for the last


def test_refresh_modes_and_visibility(app):
    a = app()
    doc = nd({"index": {"_id": "V1"}}, {"title": "visible gadget"})
    bulk(a, doc)
    assert total(a, "gadget") == 0                                          # stored, not yet visible (refresh=false)
    st, out = bulk(a, nd({"index": {"_id": "V2"}}, {"title": "visible gadget two"}), "?refresh=true")
    assert out["forced_refresh"] is True and total(a, "gadget") == 2        # a refresh makes everything visible
    mi = a.manager.get("shop")
    assert mi.lag() == 0 and mi.pending_upserts == 0
    bulk(a, nd({"index": {"_id": "V3"}}, {"title": "visible gadget three"}), "?refresh")   # a bare ?refresh is true
    assert total(a, "gadget") == 3
    assert bulk(a, doc, "?refresh=maybe")[0] == 400


def test_wait_for_returns_after_the_interval_refresh_and_forces_one_when_off(app):
    a = app(refresh_interval="200ms")
    a.manager.get("shop")
    t0 = time.perf_counter()
    st, out = bulk(a, nd({"index": {"_id": "W1"}}, {"title": "waited gadget"}), "?refresh=wait_for")
    assert st == 200 and out["forced_refresh"] is False and total(a, "gadget") == 1
    assert time.perf_counter() - t0 < 5
    b = app()
    b2 = b.manager.get("shop")
    st, out = bulk(b, nd({"index": {"_id": "W2"}}, {"title": "waited gadget"}), "?refresh=wait_for")   # interval -1: forced
    assert out["forced_refresh"] is True and total(b, "gadget") == 1 and b2.lag() == 0


def test_background_refresh_follows_refresh_interval_setting(app):
    a = app()                                                               # interval -1: nothing refreshes by itself
    bulk(a, nd({"index": {"_id": "T1"}}, {"title": "timed gadget"}))
    time.sleep(0.4)
    assert total(a, "gadget") == 0
    r = a.handle("PUT", "/indexes/shop/_settings", json.dumps({"index": {"refresh_interval": "100ms"}}).encode())
    assert r.status == 200 and r.json()["index"]["refresh_interval"] == "100ms"
    for _ in range(50):
        if total(a, "gadget") == 1:
            break
        time.sleep(0.1)
    assert total(a, "gadget") == 1
    assert a.handle("GET", "/indexes/shop/_settings").json()["shop"]["settings"]["index"]["refresh_interval"] == "100ms"
    for bad in ({"index": {"refresh_interval": "soon"}}, {"index": {"number_of_shards": 3}}, [1]):
        assert a.handle("PUT", "/indexes/shop/_settings", json.dumps(bad).encode()).status == 400
    assert a.handle("POST", "/indexes/shop/_refresh", b"").status == 200


def test_update_of_a_pending_doc_does_not_break_the_live_index(app):
    a = app()
    st, out = bulk(a, nd({"index": {"_id": "Q1"}}, {"title": "pending gadget", "price": 1.0},
                         {"update": {"_id": "Q1"}}, {"doc": {"price": 2.0}}, {"delete": {"_id": "Q1"}}))
    assert st == 200 and not out["errors"]
    bulk(a, nd({"index": {"_id": "Q2"}}, {"title": "pending gadget"}), "?refresh=true")
    assert total(a, "gadget") == 1


def test_parsers():
    assert list(iter_lines(iter([b"a\nb", b"c\n\nd"]).__next__ if False else _reader([b"a\nb", b"c\n\nd"]))) == [b"a", b"bc", b"", b"d"]
    with pytest.raises(BulkFormatError):
        list(parse_actions(iter([b'{"index": {}, "delete": {}}'])))


def _reader(pieces):
    pieces = list(pieces)
    return lambda n: pieces.pop(0) if pieces else b""


def test_http_content_length_and_chunked_bodies(app):
    a = app(bulk_chunk=3)
    server = make_server(a, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        c = http.client.HTTPConnection("127.0.0.1", port)
        body = nd(*[x for i in range(7) for x in ({"index": {"_id": f"H{i}"}}, {"title": f"http {i}"})])
        c.request("POST", "/indexes/shop/_bulk?refresh=true", body, {"Content-Type": "application/x-ndjson"})
        r = c.getresponse()
        out = json.loads(r.read())
        assert r.status == 200 and len(out["items"]) == 7 and not out["errors"]
        chunks = [body[i:i + 50] for i in range(0, len(body), 50)]           # Transfer-Encoding: chunked, no length
        c.request("POST", "/indexes/shop/_bulk", iter(chunks), {"Content-Type": "application/x-ndjson"}, encode_chunked=True)
        r = c.getresponse()
        out = json.loads(r.read())
        assert r.status == 200 and [item(out, i)[1]["result"] for i in range(7)] == ["updated"] * 7
    finally:
        server.shutdown()
        server.server_close()
