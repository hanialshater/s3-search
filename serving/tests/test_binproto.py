"""serving/binproto.py's kind-1 checks (S6-22s): the single-pass _plain_ids and _safe decide exactly as the numpy forms
they replaced (kept here as the reference), on random and edge-case inputs; and a columnar reply still decodes to the
bytes serving.encode writes."""
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
