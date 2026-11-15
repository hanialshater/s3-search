"""plat-m35: the default path's one new host cost per reply was serving's subnormal flush (flush_response, 8 to 24 us
a reply after #46). The fused route now flushes its page columns once per batch, and marks them FlushedColumns,
which flush_response returns as they are. The bytes must be exactly what the per-reply flush gave."""
import numpy as np
import pytest

from engine import api
from engine.plan.execute import FlushedColumns, flush_subnormals
from referee import checks, loader
from serving import encode as enc


def _edge(dtype):
    info = np.finfo(dtype)
    vals = [0.0, -0.0, 1.0, -2.5, np.inf, -np.inf, np.nan, info.tiny, -info.tiny, info.tiny / 2, -info.tiny / 3,
            info.smallest_subnormal, -info.smallest_subnormal, info.max]
    rng = np.random.default_rng(5)
    a = np.array(vals + list(rng.normal(size=200) * 10.0 ** rng.integers(-320, 30, 200)), dtype)
    return a.reshape(2, -1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_the_engines_flush_is_the_serving_flush_bit_for_bit(dtype):
    a = _edge(dtype)
    ut = np.uint32 if dtype == np.float32 else np.uint64
    got, want = flush_subnormals(a), enc.flush_subnormals(a)
    assert np.array_equal(np.asarray(got).view(ut), np.asarray(want).view(ut))
    clean = np.array([1.0, 2.0], dtype)
    assert flush_subnormals(clean) is clean
    assert flush_subnormals(np.array([1, 2])).dtype.kind == "i"


def test_flush_response_returns_flushed_columns_as_they_are():
    c = FlushedColumns({"_id": np.array([b"a"], "S1"), "_score": np.array([1e-40], np.float32), "sort": []})
    r = {"hits": {"total": 1, "columns": c}}
    assert enc.flush_response(r) is r                   # trusted: the engine flushed them


def test_fused_route_bytes_are_the_per_reply_flushs(monkeypatch):
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "reference")
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    reqs = [{**r, "size": 50} for r in checks.lexical_requests(ds)[:24]]
    out = ix.search_batch(reqs, hits="columns", ids="bytes")
    assert all(isinstance(r["hits"]["columns"], FlushedColumns) for r in out)
    for r in out:
        plain = {**r, "hits": {**r["hits"], "columns": dict(r["hits"]["columns"])}}
        assert enc.encode(enc.flush_response(r)) == enc.encode(enc.flush_response(plain))


@pytest.mark.parametrize("mutate", [
    lambda c: c.__setitem__("_score", c["_score"]), lambda c: c.__delitem__("sort"), lambda c: c.update(x=1),
    lambda c: c.setdefault("y", 2), lambda c: c.pop("sort"), lambda c: c.popitem(), lambda c: c.clear(),
    lambda c: c.__ior__({"z": 3})])
def test_any_mutation_drops_the_marker(mutate):
    c = FlushedColumns({"_id": np.array([b"a"], "S1"), "_score": np.array([1e-40], np.float32), "sort": []})
    assert c.s3_flushed is True
    mutate(c)
    assert c.s3_flushed is False
    r = {"hits": {"total": 1, "columns": c}}
    if "_score" in c:                                    # flushed per reply again
        out = enc.flush_response(r)
        assert out is not r and out["hits"]["columns"]["_score"].view(np.uint32)[0] == 0
    assert FlushedColumns({}).s3_flushed is True         # the class default is untouched


def test_a_subnormal_in_the_fused_page_is_flushed_once_with_the_per_reply_bytes(monkeypatch):
    """Scores made subnormal (boost 1e-40): the fused route's batch flush gives the bytes that today's (unfused) path
    plus the per-reply flush gives."""
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    texts = [str(t) for t in ds.test_queries["query"]][:8]
    reqs = [{"query": {"match": {"product_title": {"query": t, "boost": 1e-40}}}, "size": 30} for t in texts]
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "0")
    want = [enc.encode({k: v for k, v in enc.flush_response(r).items() if k != "took_ms"})
            for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    raw = ix.search_batch(reqs, hits="columns", ids="bytes")
    sub = [r for r in raw if np.any((r["hits"]["columns"]["_score"] != 0)
                                    & (np.abs(r["hits"]["columns"]["_score"]) < np.finfo(np.float32).tiny))]
    assert sub, "the unfused path gave no subnormal score: the case is not exercised"
    monkeypatch.setenv("S3_TRITON_FUSED_LEXICAL", "reference")
    got = ix.search_batch(reqs, hits="columns", ids="bytes")
    assert all(isinstance(r["hits"]["columns"], FlushedColumns) for r in got)
    for r in got:
        sc = r["hits"]["columns"]["_score"].view(np.uint32)
        assert not np.any(((sc & 0x7FFFFFFF) - 1) < 0x7FFFFF)               # no subnormal left
    assert [enc.encode({k: v for k, v in enc.flush_response(r).items() if k != "took_ms"}) for r in got] == want
