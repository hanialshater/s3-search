"""serving/binproto.py's kind-1 checks (S6-22s): the single-pass _plain_ids and _safe decide exactly as the numpy forms
they replaced (kept here as the reference), on random and edge-case inputs; and a columnar reply still decodes to the
bytes serving.encode writes."""
import json
import numpy as np
import pytest

from serving import binproto


def _plain_ids_ref(ids) -> bool:                       # the numpy form before plat-m19's profile
    if not (isinstance(ids, np.ndarray) and ids.dtype.kind == "S"):
        return False
    b = np.frombuffer(ids.tobytes(), np.uint8)
    return not ((b >= 0x7F) | ((b < 0x20) & (b != 0)) | (b == 0x22) | (b == 0x5C)).any()


def _safe_ref(a) -> bool:
    m = np.abs(a)
    return bool(np.isfinite(a).all() and ((m == 0) | ((m >= 1e-4) & (m < 1e16))).all())


def test_plain_ids_matches_the_reference_on_every_single_byte():
    for c in range(256):
        for ids in (np.array([bytes([c])], "S1"), np.array([b"B07" + bytes([c]) + b"X", b"ok"], "S5")):
            assert binproto._plain_ids(ids) == _plain_ids_ref(ids), (c, ids)


def test_plain_ids_matches_the_reference_on_random_ids():
    rng = np.random.default_rng(22)
    for trial in range(400):
        w = int(rng.integers(1, 16))
        alphabet = rng.choice([0x41, 0x30, 0x20, 0x00, 0x22, 0x5C, 0x7F, 0x80, 0x0A, 0x7E], size=3)
        raw = rng.choice(np.concatenate([np.arange(0x41, 0x5B), alphabet]), size=(int(rng.integers(0, 50)), w))
        ids = np.frombuffer(raw.astype(np.uint8).tobytes(), f"S{w}")
        assert binproto._plain_ids(ids) == _plain_ids_ref(ids)
    assert binproto._plain_ids([b"a"]) is False and binproto._plain_ids(np.array(["a"])) is False


@pytest.mark.parametrize("vals", [
    [], [0.0], [-0.0], [1e-4], [-1e-4], [9.9e-5], [5e-324], [1e16], [-1e16], [9.99e15], [np.inf], [-np.inf], [np.nan],
    [1.0, np.nan, 2.0], [0.0, 1e-5], [0.0, 3.5, 1e15], [2.5] * 2000, [1e-4, 0.0, -0.0, 7.25],
])
def test_safe_matches_the_reference_on_edges(vals):
    a = np.array(vals, np.float64)
    assert binproto._safe(a) == _safe_ref(a), vals


def test_safe_matches_the_reference_on_random_values():
    rng = np.random.default_rng(23)
    picks = np.array([0.0, -0.0, 1e-4, 9e-5, 1e16, 9.9e15, np.inf, -np.inf, np.nan, 3.0, -7.5, 1e-300])
    for trial in range(400):
        a = rng.normal(0, 10, size=int(rng.integers(0, 60))) * 10.0 ** rng.integers(-8, 18)
        if a.size and trial % 2:
            a[rng.integers(0, a.size)] = rng.choice(picks)
        assert binproto._safe(a) == _safe_ref(a), a
        f32 = a.astype(np.float32).astype(np.float64)
        assert binproto._safe(f32) == _safe_ref(f32)


def test_a_columnar_reply_still_decodes_to_the_encoded_bytes():
    from serving.encode import encode
    ids = np.array([b"B000A", b"B000B", b"B000C"], "S5")
    sc = np.array([9.5, 3.25, 1.0], np.float32)
    resp = {"took_ms": {"parse": 0.1}, "hits": {"total": 3, "columns": {
        "_id": ids, "_score": sc, "sort": [{"values": sc.astype(np.float64), "missing": None},
                                           {"values": np.array([7, 3, 1], np.int64), "missing": None}]}}}
    frame = binproto.columnar(5, resp, {"version": "seg:1", "queue_ms": 0.0, "batch": 1})
    assert frame is not None
    id_, st, body, _ = binproto.decode_reply(frame[4:])
    assert (id_, st) == (5, 200) and body == encode(resp)


def test_kind3_frames_carry_any_ids_and_floats_and_decode_to_the_encoded_bytes():
    """S3B2: columnar3 sends ids and floats kind 1 refuses (quote, non-ASCII, NaN, inf, exponent floats) as kind 3,
    and decode_reply (Python's encoder on the decoded columns) gives serving.encode's bytes; the native front's
    render is held to the same bytes by its own tests (tests/golden3.txt)."""
    from serving.encode import encode
    ids = np.array(['B0"X'.encode(), "é中😀".encode(), b"x\\y", b"ok"], "S12")
    sc = np.array([np.nan, np.inf, 1e-7, 3.0e17], np.float32)
    sort = [{"values": sc.astype(np.float64), "missing": None},
            {"values": np.array([-np.inf, 5e-324, -0.0, 1e16]), "missing": None},
            {"values": np.array([1, -2, 3, 2**62], np.int64), "missing": None}]
    resp = {"took_ms": {"parse": 0.1}, "hits": {"total": 4, "columns": {"_id": ids, "_score": sc, "sort": sort}}}
    assert binproto.columnar(1, resp, {"version": "s:1", "queue_ms": 0.0, "batch": 1}) is None   # kind 1 refuses
    f = binproto.columnar3(1, resp, {"version": "s:1", "queue_ms": 0.0, "batch": 1})
    assert f is not None and f[4] == 3
    id_, st, body, _ = binproto.decode_reply(f[4:])
    assert (id_, st) == (1, 200) and body == encode(resp)


def test_kind3_still_sends_whole_what_it_cannot_carry():
    sc = np.array([1.0, 2.0], np.float32)
    ids = np.array([b"a", b"b"], "S1")
    for sort in ([{"values": np.array([1.0, 2.0]), "missing": [False, True]}],
                 [{"values": np.array([1.0, 2.0], np.float32), "missing": None}]):
        resp = {"took_ms": 1, "hits": {"total": 2, "columns": {"_id": ids, "_score": sc, "sort": sort}}}
        assert binproto.columnar3(1, resp, {}) is None
    assert binproto.columnar3(1, {"took_ms": 1, "hits": {"total": 0, "hits": []}}, {}) is None


def _resp(ids, sc, sort, total=None, aggs=False):
    r = {"took_ms": {"parse": 0.1}, "hits": {"total": total or len(ids), "columns": {"_id": ids, "_score": sc, "sort": sort}}}
    if aggs:
        r["aggregations"] = {"c": {"buckets": [{"key": "a\"b", "doc_count": 3}]}}
    return r


def test_batch4_records_decode_to_the_encoded_bytes_of_each_reply():
    """S3B3 (kind 4): one frame for a batch's replies (columnar, field-sorted with missing values, ids S3B1 refuses,
    a whole response, an error, a call); each record split out encodes to serving.encode's bytes for its reply."""
    from serving.encode import encode
    rng = np.random.default_rng(4)
    a_ids = np.array([b"B001", b"B002", b"B003"], "S4")
    a_sc = np.array([3.5, 2.0, 1.0], np.float32)
    msgs = [
        ("search", 11, "ok", _resp(a_ids, a_sc, [{"values": a_sc.astype(np.float64), "missing": None}]), ("seg", 1), 0.5, 4, 2.0),
        ("search", 12, "ok", _resp(np.array(['x"y'.encode(), "é中".encode()], "S6"), np.array([np.nan, 1e-7], np.float32),
                                   [{"values": np.array([1.5, -0.0]), "missing": [False, True]},
                                    {"values": np.array([7, 2**62], np.int64), "missing": None}], aggs=True), ("seg", 1), 0.4, 4),
        ("search", 13, "ok", _resp(np.array([b"q"], "S1"), np.array([9.0], np.float32),
                                   [{"values": np.array([3], np.int64), "missing": None},
                                    {"values": np.array([2.25]), "missing": None}]), ("seg", 1), 0.3, 4),
        ("search", 14, "ok", _resp(np.array([b"a\x00b", b"c"], "S3"), np.array([1.0, 2.0], np.float32), []), ("seg", 1), 0.2, 4),
        ("search", 15, "error", 400, {"error": {"type": "invalid_request", "path": "", "reason": "no"}}),
        ("call", 16, 200, b'{"status":"ok"}', "application/json", {}),
    ]
    f = binproto.batch4(msgs)
    assert f[4] == 4
    recs = binproto.split_batch4(f[4:])
    assert [r[0] for r in recs] == [11, 12, 13, 14, 15, 16]
    for (id_, st, body, meta), msg in zip(recs, msgs):
        if msg[0] == "search" and msg[2] == "ok":
            assert st == 200 and body == encode(msg[3]), id_
        elif msg[0] == "search":
            assert st == 400 and json.loads(body) == msg[4]
        else:
            assert st == 200 and body == b'{"status":"ok"}' and meta == {"type": "application/json"}
    assert "pre" in recs[0][3] and "pre" not in recs[3][3]          # an id with an interior NUL goes whole (rkind 2)


def test_batch4_aligns_its_numeric_arrays():
    import struct
    msgs = [("search", 1, "ok", _resp(np.array([b"abc"], "S3"), np.array([1.0], np.float32),
                                      [{"values": np.array([2.0]), "missing": None}]), ("s", 1), 0.0, 1)]
    p = binproto.batch4(msgs)[4:]
    n, w, ht = struct.unpack_from("<IHI", p, 9)
    o = 19 + ht * w
    o += -o % 8
    assert o % 8 == 0 and p[19 + ht * w:o] == b"\x00" * (o - 19 - ht * w)
    sc = np.frombuffer(p, "<f4", ht, o)
    assert sc.tolist() == [1.0]
