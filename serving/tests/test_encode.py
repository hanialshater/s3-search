"""S6-7: serving's encoder gives the dict form's bytes for columnar hits (serving/encode.py, INTERFACES §5 and §7)."""
import json

import numpy as np
import pytest

from serving import encode as enc


def _dumps(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":")).encode()


def _columnar(h=5, sort=None, extra=None):
    cols = {"_id": [f"B{i:04d}" for i in range(h)], "_score": np.linspace(9.1, 1.3, h).astype(np.float32),
            "sort": sort if sort is not None else []}
    return {"took_ms": {"total": 1.5}, "hits": {"total": 42, "columns": cols}, **(extra or {})}


def test_columns_without_the_engine():
    h = 6
    sort = [{"values": np.array([1, 0, 1, 1, 0, 0], np.int64), "missing": None},
            {"values": np.array([1.5, np.nan, np.inf, -np.inf, 2.0, 0.1]), "missing": np.array([0, 0, 0, 0, 1, 0], bool)},
            {"values": ["red", "bl\"ue", "é", "", "x", "y"], "missing": np.array([0, 0, 0, 1, 0, 0], bool)},
            {"values": np.array([True, False, True, True, False, True]), "missing": None},
            {"values": np.arange(h, dtype=np.int64) * 10**12, "missing": None}]
    r = _columnar(h, sort, {"aggregations": {"brand": {"buckets": [{"key": "a", "doc_count": 3}]}}})
    assert enc.encode(r) == _dumps(enc.expand(r))
    d = enc.expand(r)
    assert list(d) == ["hits", "took_ms", "aggregations"] and list(d["hits"]) == ["total", "hits"]
    assert d["hits"]["hits"][4]["sort"][1] is None and d["hits"]["hits"][3]["sort"][2] is None
    assert d["hits"]["hits"][0]["_score"] == float(np.float32(9.1))
    for n in (0, 1, 2):                                      # the specialised joins
        rr = _columnar(4, sort=[{"values": np.arange(4, dtype=np.int64), "missing": None}] * n)
        assert enc.encode(rr) == _dumps(enc.expand(rr))
    empty = _columnar(0)
    assert enc.encode(empty) == _dumps(enc.expand(empty)) and b'"hits":[]' in enc.encode(empty)


def test_dict_form_and_lists_pass_through():
    d = {"hits": {"total": 1, "hits": [{"_id": "a", "_score": 1.0, "sort": [1.0]}]}}
    assert enc.encode(d) == _dumps(d) and enc.expand(d) is d
    both = [d, _columnar(3)]
    assert enc.encode(both) == _dumps(enc.expand(both))
    assert enc.encode({"x": [1, 2]}) == _dumps({"x": [1, 2]})


api = pytest.importorskip("engine.api")


def test_bytes_equal_the_engine_dict_form():
    """Over the engine's own columnar test requests (engine/tests/test_columnar_hits.py), on every device: serving's
    bytes are json.dumps of expand_hits, which equals the dict form."""
    from engine.tests.test_columnar_hits import _requests, _without_took
    from referee import loader
    ds = loader.load_dataset("toy")
    for device in api.available_devices():
        ix = api.build_index(ds.catalogue, ds.mapping, device=device)
        reqs = _requests(ds)
        n = 0
        for r, d, c in zip(reqs, ix.search_batch(reqs), ix.search_batch(reqs, hits="columns")):
            n += "columns" in c["hits"]
            assert enc.encode(_without_took(c)) == _dumps(_without_took(d)), r
            assert enc.encode(c) == _dumps(api.expand_hits(c)), r
            assert enc.expand_columns(c) == api.expand_hits(c), r
        assert n >= len(reqs) // 2


def test_ids_as_a_bytes_array_encode_the_same():
    """S6-12: ``_id`` as one numpy S<w> array (UTF-8, NUL-padded) gives the same bytes as the list[str] form."""
    for ids in (["B0001", "B02", "X"], ["a\"b", "c\\d", "tab\tx", "é", "日本"], ["plain", "naïve"]):
        r = _columnar(len(ids))
        as_str = {**r, "hits": {**r["hits"], "columns": {**r["hits"]["columns"], "_id": ids}}}
        arr = np.array([i.encode("utf-8") for i in ids], dtype=f"S{max(len(i.encode()) for i in ids)}")
        as_bytes = {**r, "hits": {**r["hits"], "columns": {**r["hits"]["columns"], "_id": arr}}}
        assert enc.encode(as_bytes) == enc.encode(as_str) == _dumps(enc.expand(as_str))
        assert enc.expand(as_bytes) == enc.expand(as_str)
