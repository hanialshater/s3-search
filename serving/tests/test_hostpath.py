"""serving/hostpath.py, the S3_HOSTPATH seam (S6-22s): the golden binproto reply frames (the oracle s3_hostpath must
reproduce, took_ms masked), the Python stub's prepare and reply, and the guard's fallback."""
import json
import re
import struct
from pathlib import Path

import pytest

from serving import binproto, hostpath
from serving.errors import BadJSON, classify
from serving.tests.golden.make_binproto_frames import build

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden" / "binproto_frames.json").read_text())["cases"]
_TOOK = re.compile(rb'"took_ms":\{[^{}]*\}')


def frames(buf: bytes) -> list:
    out, o = [], 0
    while o < len(buf):
        (n,) = struct.unpack_from("<I", buf, o)
        out.append(buf[o + 4:o + 4 + n])
        o += 4 + n
    assert o == len(buf)
    return out


def masked(payload: bytes) -> bytes:
    """A reply payload with took_ms blanked inside its meta JSON (the length prefix rewritten to match)."""
    kind, id_ = struct.unpack_from("<BQ", payload, 0)
    (n,) = struct.unpack_from("<I", payload, 9)
    meta = _TOOK.sub(b'"took_ms":0', payload[13:13 + n])
    meta = re.sub(rb'\\"took_ms\\":\{[^{}]*\}', rb'\\"took_ms\\":0', meta)   # the shell inside "pre"
    rest = payload[13 + n:]
    if kind == 2:                                       # a whole response: took_ms in its body
        st, m = struct.unpack_from("<HI", rest, 0)
        body = _TOOK.sub(b'"took_ms":0', rest[6:6 + m])
        rest = struct.pack("<HI", st, len(body)) + body + rest[6 + m:]
    return struct.pack("<BQI", kind, id_, len(meta)) + meta + rest


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_frames_are_what_the_python_encoder_writes(case):
    got = hostpath.reply_frame(build(case["spec"]))
    assert got == bytes.fromhex(case["frame_hex"])


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_the_stub_reply_matches_the_golden_frames_with_took_ms_masked(case):
    msg = build(case["spec"])
    got = hostpath.PyStub().reply(msg[1] if msg[0] == "multi" else [msg])
    want = bytes.fromhex(case["frame_hex"])
    assert [masked(f) for f in frames(got)] == [masked(f) for f in frames(want)]


def test_masking_ignores_took_ms_only():
    spec = next(c["spec"] for c in GOLDEN if c["spec"]["kind"] == "ok" and c["frame_hex"][8:10] == "01")
    a = hostpath.reply_frame(build({**spec, "took_ms": {"parse": 9.9}}))
    b = hostpath.reply_frame(build(spec))
    assert a != b and [masked(f) for f in frames(a)] == [masked(f) for f in frames(b)]
    c = hostpath.reply_frame(build({**spec, "total": spec["total"] + 1}))
    assert [masked(f) for f in frames(c)] != [masked(f) for f in frames(b)]


def _payload(id_, body: bytes, name=b"toy"):
    return struct.pack("<BQ", 1, id_) + struct.pack("<H", len(name)) + name + struct.pack("<I", len(body)) + body


def test_search_head_and_body():
    p = _payload(42, b'{"size": 3}', b"esci")
    assert binproto.search_head(p) == (42, "esci") and binproto.search_body(p) == b'{"size": 3}'
    assert binproto.search_head(memoryview(p)) == (42, "esci")


def test_stub_prepare_parses_and_refuses_like_the_python_front():
    bodies = [b'{"query": {"match_all": {}}}', b"", b"  ", b"{nope", b'[1, 2]']
    prep = hostpath.PyStub().prepare([_payload(i, b) for i, b in enumerate(bodies)], index=None, out=None)
    assert prep.requests[0] == {"query": {"match_all": {}}} and prep.requests[4] == [1, 2]
    assert prep.fallback == [] and prep.layout is None and prep.plans is None
    errs = {i: (st, body) for i, st, body in prep.errors}
    assert sorted(errs) == [1, 2, 3]
    for i in (1, 2, 3):
        try:
            hostpath.parse_body(bodies[i])
        except BadJSON as exc:
            assert errs[i] == classify(exc)
    assert errs[1][1]["error"]["reason"] == "empty body"
    assert errs[3] == (400, {"error": {"type": "invalid_request", "path": "",
                                       "reason": "body is not JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"}})


def test_parse_request_returns_bad_json_instead_of_raising():
    msg = binproto.parse_request(_payload(5, b"{x"))
    assert msg[:3] == ("search", 5, "toy") and isinstance(msg[3], BadJSON)


def test_the_guard_falls_back_and_counts():
    class Broken:
        def prepare(self, payloads, index, out):
            raise RuntimeError("boom")

        def reply(self, msgs):
            raise RuntimeError("boom")
    g = hostpath._Guarded(Broken(), "broken")
    prep = g.prepare([_payload(1, b'{"size": 1}')])
    assert prep.requests == [{"size": 1}]
    msg = build(GOLDEN[0]["spec"])
    assert g.reply([msg]) == hostpath.reply_frame(msg) and g.fallbacks == 2


def test_load(monkeypatch):
    monkeypatch.delenv("S3_HOSTPATH", raising=False)
    assert hostpath.load() is None
    monkeypatch.setenv("S3_HOSTPATH", "0")
    assert hostpath.load() is None
    monkeypatch.setenv("S3_HOSTPATH", "stub")
    assert hostpath.load().name == "python-stub"
    monkeypatch.setenv("S3_HOSTPATH", "maybe")
    with pytest.raises(ValueError):
        hostpath.load()


def test_stub_prepared_marks_plain_requests_eligible():
    bodies = [b'{"query": {"match_all": {}}, "size": 2}', b'{"query": {"match_all": {}}, "explain": true}', b"",
              b'{"size": 1, "_source": true}', b'{"size": 3}']
    prep = hostpath.PyStub(prepared=True).prepare([_payload(i, b) for i, b in enumerate(bodies)])
    assert prep.layout["eligible"] == [0, 4]
    assert prep.requests[0] is None and prep.requests[4] is None
    assert prep.requests[1]["explain"] is True and prep.requests[3]["_source"] is True
    assert [i for i, _, _ in prep.errors] == [2]
    assert prep.layout["stub_bodies"] == [{"query": {"match_all": {}}, "size": 2}, {"size": 3}]


class _FakeIndex:
    """search_batch answers {"n": size}; search_prepared (optional) answers the layout's bodies the same way."""

    def __init__(self, prepared=True, broken=False):
        self.calls, self.broken = [], broken
        if prepared:
            self.search_prepared = self._search_prepared

    def search_batch(self, requests, hits=None, ids=None, errors=None):
        self.calls.append(("batch", len(requests)))
        return [{"n": r.get("size")} for r in requests]

    def _search_prepared(self, prep, hits=None, ids=None):
        self.calls.append(("prepared", len(prep.layout["eligible"])))
        if self.broken:
            raise RuntimeError("boom")
        return [{"n": r.get("size"), "via": "prepared"} for r in prep.layout["stub_bodies"]]


class _FakeMI:
    name, ready = "toy", True

    def __init__(self, index):
        self.index = index
        import types
        self.metrics = types.SimpleNamespace(inc=lambda *a, **k: None, observe=lambda *a, **k: None)

    def attach_source(self, resp, src):
        pass


def _server(index, monkeypatch):
    import types
    from serving import gpu
    monkeypatch.setattr(gpu, "_HOSTPATH", [hostpath._Guarded(hostpath.PyStub(prepared=True), "python-stub-prepared")])
    monkeypatch.setattr(gpu, "version_of", lambda mi: ("seg", 1))
    mi = _FakeMI(index)
    srv = object.__new__(gpu.GpuServer)
    srv.app = types.SimpleNamespace(manager=types.SimpleNamespace(get=lambda name: mi),
                                    metrics=types.SimpleNamespace(inc=lambda *a, **k: None, observe=lambda *a, **k: None))
    srv.busy_s, srv.n_batches, srv.n_requests, srv._out = 0.0, 0, 0, None
    sent = []
    srv._reply = lambda it, msg: sent.append(msg)
    return gpu, srv, sent


@pytest.mark.parametrize("prepared,broken", [(True, False), (False, False), (True, True)])
def test_run_answers_planned_positions_by_search_prepared_and_the_rest_by_search_batch(monkeypatch, prepared, broken):
    index = _FakeIndex(prepared=prepared, broken=broken)
    gpu, srv, sent = _server(index, monkeypatch)
    bodies = [b'{"size": 1}', b'{"size": 2, "explain": true}', b"{bad", b'{"size": 4}']
    items = [gpu._Item(None, 100 + i, "toy", None, raw=_payload(100 + i, b)) for i, b in enumerate(bodies)]
    batch = srv._prepare_raw(items)
    assert [it.id for it in batch] == [100, 101, 103]                # the non-JSON body answered at once
    assert sent[0][:4] == ("search", 102, "error", 400)
    srv._run("toy", batch)
    got = {m[1]: m[3] for m in sent[1:]}
    assert set(got) == {100, 101, 103} and got[101] == {"n": 2}
    if prepared and not broken:
        assert got[100] == {"n": 1, "via": "prepared"} and got[103] == {"n": 4, "via": "prepared"}
        assert index.calls == [("prepared", 2), ("batch", 1)]
    elif not prepared:                                                # an index without search_prepared: the stub
        assert got[100] == {"n": 1} and got[103] == {"n": 4} and ("batch", 2) in index.calls
    else:                                                             # search_prepared raised: search_batch, counted
        assert got[100] == {"n": 1} and got[103] == {"n": 4}
        assert index.calls == [("prepared", 2), ("batch", 3)] and gpu._HOSTPATH[0].fallbacks == 1
