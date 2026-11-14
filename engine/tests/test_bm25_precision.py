"""S6-21, the precision experiment: BM25F's term weights rounded to float16 (``S3_BM25_ADDEND``) and its sum in float32
or float64 (``S3_BM25_SUM``). The default is unchanged. Every variant is deterministic, batch-invariant and the same
bits on every device, and ``S3_BM25_SUM=float32`` equals the existing float32 accumulation option for BM25F."""
import numpy as np
import pytest

from engine import api
from engine.backend import available_devices, bm25_precision, get_backend
from engine.build import build_segment
from engine.ops import lexical as L
from referee import checks, loader

VARIANTS = [("float32", "float64"), ("float32", "float32"), ("float16", "float64"), ("float16", "float32")]


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.fixture(scope="module")
def seg(ds):
    return build_segment(ds.catalogue, ds.mapping, version=1)


def _set(monkeypatch, addend, acc):
    monkeypatch.setenv("S3_BM25_ADDEND", addend)
    monkeypatch.setenv("S3_BM25_SUM", acc)


def _strip(r):
    return {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}


def _reqs(ds):
    return [{**r, "size": 50} for r in checks.lexical_requests(ds)[:30]]


def test_the_default_is_float32_weights_summed_in_the_backends_dtype(monkeypatch):
    monkeypatch.delenv("S3_BM25_ADDEND", raising=False)
    monkeypatch.delenv("S3_BM25_SUM", raising=False)
    assert bm25_precision(get_backend("cpu")) == ("float32", "float64")
    assert bm25_precision(get_backend("cpu", "float32")) == ("float32", "float32")
    monkeypatch.setenv("S3_BM25_ADDEND", "bfloat16")
    with pytest.raises(ValueError):
        bm25_precision(get_backend("cpu"))
    monkeypatch.setenv("S3_BM25_ADDEND", "float32")
    monkeypatch.setenv("S3_BM25_SUM", "float16")                 # a float16 sum is not offered
    with pytest.raises(ValueError):
        bm25_precision(get_backend("cpu"))


def test_unset_equals_the_explicit_default(ds, seg, monkeypatch):
    ix = api.Index(seg, "cpu")
    monkeypatch.delenv("S3_BM25_ADDEND", raising=False)
    monkeypatch.delenv("S3_BM25_SUM", raising=False)
    want = [_strip(r) for r in ix.search_batch(_reqs(ds))]
    _set(monkeypatch, "float32", "float64")
    assert [_strip(r) for r in ix.search_batch(_reqs(ds))] == want


def test_sum_float32_equals_the_float32_accumulation_option(ds, seg, monkeypatch):
    for dev in [d for d in available_devices() if d in ("cpu", "torch-cpu", "cuda")]:
        monkeypatch.delenv("S3_BM25_ADDEND", raising=False)
        monkeypatch.delenv("S3_BM25_SUM", raising=False)
        want = [_strip(r) for r in api.Index(seg, dev, accumulate="float32").search_batch(_reqs(ds))]
        _set(monkeypatch, "float32", "float32")
        got = [_strip(r) for r in api.Index(seg, dev).search_batch(_reqs(ds))]
        assert got == want, dev


@pytest.mark.parametrize("addend,acc", VARIANTS)
def test_each_variant_is_the_same_on_every_device_alone_and_batched(ds, seg, addend, acc, monkeypatch):
    _set(monkeypatch, addend, acc)
    reqs = _reqs(ds)
    ref = None
    for dev in [d for d in available_devices() if d in ("cpu", "torch-cpu", "cuda")]:
        ix = api.Index(seg, dev)
        batched = [_strip(r) for r in ix.search_batch(reqs)]
        assert [_strip(ix.search(r)) for r in reqs[:8]] == batched[:8], dev   # batch-invariant
        assert [_strip(r) for r in ix.search_batch(reqs)] == batched, dev       # run to run
        if ref is None:
            ref = batched
        assert batched == ref, dev                                              # the same bits on every device


def test_float16_weights_change_scores_by_about_float16s_precision_only(ds, seg, monkeypatch):
    b = get_backend("cpu")
    ix = api.Index(seg, "cpu")
    st = ix._st
    part = st.parts[0]
    from engine.analysis import analyze
    texts = list(dict.fromkeys(list(ds.fit["query"]) + list(ds.test_queries["query"])))[:64]
    terms = [sorted({part.vocab[t] for t in analyze(x) if t in part.vocab}) for x in texts]
    F = len(st.text_fields)
    w = np.linspace(0.7, 2.3, F).astype(np.float32)
    bb = np.full(F, 0.75, np.float32)
    out = {}
    for addend, acc in VARIANTS:
        _set(monkeypatch, addend, acc)
        s, n = L.bm25f(b, part, terms, w, bb, 1.2, part.idf)
        out[(addend, acc)] = (np.asarray(s), np.asarray(n))
    base_s, base_n = out[("float32", "float64")]
    for k, (s, n) in out.items():
        assert np.array_equal(n, base_n), k                                    # match counts never change
        assert np.allclose(s, base_s, rtol=2e-3, atol=1e-3), k                 # fp16: about 3 digits per weight
    assert not np.array_equal(out[("float16", "float64")][0], base_s)          # the switch takes effect
