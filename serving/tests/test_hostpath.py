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
