"""S9-2 and S9-3: the GPU encoder fit and GPU trees are optional. Here: what runs without CUDA (the CPU result is
unchanged, the GPU option is refused by name, the XGBoost export equals the engine's GBDT when XGBoost is installed,
round trips); with CUDA (Colab): the GPU fit and the GPU trees against the reference."""
import copy

import numpy as np
import pytest

from engine.api import DeviceUnavailable, Index, build_index
from engine.encode import TfidfSvdEncoder, doc_texts
from engine.rank.gbdt import GBDT

INPUTS = ["product_title", "product_brand", "product_color"]


def has_cuda():
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def has_xgboost():
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


cuda = pytest.mark.skipif(not has_cuda(), reason="needs CuPy and a CUDA GPU (Colab)")
xgb = pytest.mark.skipif(not has_xgboost(), reason="needs xgboost (requirements-gpu.txt)")


def test_cpu_fit_is_unchanged_by_the_option(catalogue):
    texts = doc_texts(catalogue, INPUTS)
    ref = TfidfSvdEncoder(16).fit_transform(texts)
    auto = TfidfSvdEncoder(16, fit="auto")
    got = auto.fit_transform(texts)
    if not has_cuda():
        assert auto.fitted_with == "numpy" and np.array_equal(got, ref)


@pytest.mark.skipif(has_cuda(), reason="the refusal is for machines without CUDA")
def test_gpu_fit_refused_without_cuda(catalogue, mapping):
    with pytest.raises(DeviceUnavailable):
        TfidfSvdEncoder(16, fit="gpu").fit(doc_texts(catalogue, INPUTS))
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["embeddings"]["embedding"]["fit"] = "cuml"
    with pytest.raises(DeviceUnavailable):
        build_index(catalogue, m)


def test_fitted_encoder_round_trip(catalogue):
    e = TfidfSvdEncoder(16).fit(doc_texts(catalogue, INPUTS))
    back = TfidfSvdEncoder.from_arrays(e.to_arrays())
    q = ["black running shoes", "wool socks"]
    assert np.array_equal(back.encode(q), e.encode(q))


def gbdt_and_rows(seed=0, n=10_000, f=18):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    X[:50, 2] = 0.25                                       # values that sit on split edges
    y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 3]) - (np.nan_to_num(X[:, 5]) > 0)
    return GBDT(n_rounds=60, max_depth=5, min_leaf=10).fit(X[:5000], y[:5000]), X


@xgb
def test_xgboost_export_equals_the_engine():
    from engine.rank.gpu_trees import GpuTrees
    m, X = gbdt_and_rows()
    g = GpuTrees(m, X.shape[1], "xgboost", "cpu")
    assert np.abs(g.predict(X) - m.predict(X)).max() <= 1e-5


@xgb
def test_rescore_with_xgboost_runtime(segment, catalogue):
    ix = Index(copy.deepcopy(segment))
    rng = np.random.default_rng(1)
    Xtr = rng.normal(size=(500, len(ix.feature_names())))
    model = GBDT(n_rounds=20, max_depth=3, min_leaf=5).fit(Xtr, Xtr[:, 0] - Xtr[:, 3])
    req = {"query": {"hybrid": {"queries": [{"match": {"product_title": "black shoes"}},
                                            {"neural": {"embedding": {"query_text": "black shoes", "k": 20}}}]}},
           "rescore": {"window_size": 30, "s3_ranker": {"model": "m"}}, "size": 40}
    ix.add_ranker("m", model)
    a = ix.search(req)["hits"]["hits"]
    ix.add_ranker("m", model, runtime="xgboost")
    b = ix.search(req)["hits"]["hits"]
    assert [h["_id"] for h in a] == [h["_id"] for h in b]
    assert np.allclose([h["_score"] for h in a], [h["_score"] for h in b], atol=1e-5)


@cuda
def test_gpu_fit_close_to_numpy(catalogue):
    texts = doc_texts(catalogue, INPUTS)
    a = TfidfSvdEncoder(16).fit_transform(texts)
    g = TfidfSvdEncoder(16, fit="gpu")
    b = g.fit_transform(texts)
    assert g.fitted_with == "gpu"
    assert np.median(np.sum(a * b, axis=1)) > 0.999


@cuda
@xgb
def test_xgboost_on_cuda_equals_the_engine():
    from engine.rank.gpu_trees import GpuTrees
    m, X = gbdt_and_rows()
    g = GpuTrees(m, X.shape[1], "xgboost", "cuda")
    assert np.abs(g.predict(X) - m.predict(X)).max() <= 1e-5
