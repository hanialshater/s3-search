"""RPC serving: the same handlers as HTTP, so the same responses."""
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from serving.app import create_app
from serving.config import ServingConfig
from serving.rpc import RpcClient, RpcError, make_rpc_server, read_frame, write_frame

from .fakes import MAPPING, fake_ports, write_source


@pytest.fixture
def rpc(tmp_path):
    app = create_app(ServingConfig(data_dir=str(tmp_path / "d"), max_wait_ms=5), fake_ports())
    src = write_source(tmp_path / "s.jsonl")
    assert app.handle("PUT", "/indexes/shop", json.dumps({"mapping": MAPPING, "source": {"path": src}}).encode()).status == 200
    server = make_rpc_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = RpcClient("127.0.0.1", server.server_address[1])
    yield app, server, client
    client.close()
    server.shutdown()
    server.server_close()
    app.close()


def strip(r):
    r = json.loads(json.dumps(r))
    r.pop("took_ms", None)
    return r


def http(app, method, path, obj=None):
    r = app.handle(method, path, b"" if obj is None else json.dumps(obj).encode())
    return r.status, r.json()


def test_rpc_equals_http_for_every_route(rpc):
    app, _, c = rpc
    cases = [("POST", "/indexes/shop/_search", {"query": {"match": {"title": "red"}}, "size": 3}),
             ("POST", "/indexes/shop/_search", {"query": {"fuzzy": {}}}),                      # a named error
             ("POST", "/indexes/shop/_msearch", [{"size": 1}, {"query": {"match": {"title": "blue"}}}]),
             ("GET", "/indexes/shop/_mapping", None), ("GET", "/indexes/shop/versions", None),
             ("GET", "/indexes/ghost/_mapping", None), ("POST", "/indexes/shop/_search", {"size": 2, "_source": ["title"]})]
    for method, path, body in cases:
        hs, hb = http(app, method, path, body)
        rs, rb, _ = c.call(method, path, body)
        assert rs == hs, path
        assert [strip(x) for x in rb] == [strip(x) for x in hb] if isinstance(hb, list) else strip(rb) == strip(hb)


def test_client_methods_and_errors(rpc):
    _, _, c = rpc
    assert c.ready() and c.search("shop", {"size": 2})["hits"]["total"] == 20
    assert [len(r["hits"]["hits"]) for r in c.msearch("shop", [{"size": 1}, {"size": 2}])] == [1, 2]
    assert c.docs("shop", [{"op": "delete", "_id": "P001"}])["applied"]["deleted"] == 1
    v = c.versions("shop")
    assert c.reindex("shop")["version"] == 21 and c.rollback("shop", v[0]["segment_id"])["segment_id"] == v[0]["segment_id"]
    with pytest.raises(RpcError) as e:
        c.search("shop", {"query": {"fuzzy": {}}})
    assert e.value.status == 400 and e.value.type == "unsupported_query" and e.value.path == "query.fuzzy"
    status, text, _ = c.call("GET", "/metrics")
    assert status == 200 and "s3_http_requests_total" in text


def test_concurrent_connections_are_batched_and_one_connection_is_shared_safely(rpc):
    app, server, c = rpc
    mi = app.manager.get("shop")
    mi.cache.capacity = 0
    port = server.server_address[1]
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda i: RpcClient("127.0.0.1", port).search("shop", {"size": 1 + i}), range(12)))
    assert [len(r["hits"]["hits"]) for r in res] == [min(1 + i, 20) for i in range(12)]
    assert max(mi.index.batch_sizes) > 1
    with ThreadPoolExecutor(8) as ex:  # one client, many threads: calls take turns
        assert all(r["hits"]["total"] == 20 for r in ex.map(lambda _: c.search("shop", {"size": 1}), range(40)))


def test_bad_frames_get_a_named_error_and_reconnect_works(rpc):
    _, server, c = rpc
    s = socket.create_connection(("127.0.0.1", server.server_address[1]))
    s.sendall(b"\x00\x00\x00\x05nope!")
    assert read_frame(s)["body"]["error"]["type"] == "invalid_request"
    s.close()
    s = socket.create_connection(("127.0.0.1", server.server_address[1]))
    write_frame(s, {"id": 1, "body": {}})                                   # no path
    assert read_frame(s)["status"] == 400
    s.close()
    c.search("shop", {"size": 1})
    c._sock.close()                                                         # the server side drops: the client redials
    assert c.search("shop", {"size": 1})["hits"]["total"] == 20


def test_rpc_equals_http_on_the_200_conformance_requests(tmp_path):
    """The real engine on `toy`: every request of the referee's conformance suite, over HTTP handlers and over RPC."""
    pytest.importorskip("engine.api")
    from referee import conformance, loader
    from engine.api import build_index
    from serving.manager import ManagedIndex
    ds = loader.load_dataset("toy")
    cfg = ServingConfig(data_dir=str(tmp_path / "d"), max_wait_ms=1, cache_size=0)
    app = create_app(cfg, fake_ports())
    mi = ManagedIndex("toy", tmp_path / "d" / "toy", ds.mapping, cfg, app.manager.ports, app.metrics)
    mi.index, mi.segment_id = build_index(ds.catalogue, ds.mapping), "seg-0-00000000"
    app.manager._indexes["toy"] = mi
    server = make_rpc_server(app, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    suite = conformance.build_suite(ds)
    assert len(suite) == 200
    try:
        with RpcClient("127.0.0.1", server.server_address[1]) as c:
            for s in suite:
                hs, hb = http(app, "POST", "/indexes/toy/_search", s["request"])
                rs, rb, _ = c.call("POST", "/indexes/toy/_search", s["request"])
                assert (rs, strip(rb)) == (hs, strip(hb)), s["id"]
    finally:
        server.shutdown()
        server.server_close()
        app.close()
