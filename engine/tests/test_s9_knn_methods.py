"""S9-1: knn ``method`` (exact | cagra | ivf_pq) and the response's ``knn.<field> = {"requested", "ran"}``.

Without CUDA and cuVS the approximate methods run exact (and say so); the segment-merging path the cuVS methods use
is checked here with a stand-in segment searcher that returns each segment's exact neighbours, so it must equal
exact knn over segments, deletes and filters. With CUDA and cuVS (Colab): recall against exact, the prefilter, and
the per-segment index rebuilt on merge."""
import copy

import numpy as np
import pytest

from engine import ops
from engine.api import Index, InvalidRequest
from engine.build import build_segment, merge_segments
from engine import ann
from engine.tests.test_s8_segments import incremental, mapping, scenario, vec


def has_cuvs():
    try:
        import cupy as cp
        from cuvs.neighbors import cagra  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


cuvs = pytest.mark.skipif(not has_cuvs(), reason="needs cuVS, CuPy and a CUDA GPU (Colab)")


def knn(v, k=10, method=None, flt=None, params=None):
    spec = {"vector": list(map(float, v)), "k": k}
    if method:
        spec["method"] = method
    if flt:
        spec["filter"] = flt
    if params:
        spec["method_parameters"] = params
    return {"query": {"knn": {"embedding": spec}}, "size": k}


def ids(r):
    return [h["_id"] for h in r["hits"]["hits"]]


@pytest.mark.parametrize("method", ["exact", "cagra", "ivf_pq"])
def test_method_named_and_fallback_equals_exact(index, method):
    if has_cuvs():
        pytest.skip("the fallback is for machines without cuVS")
    v = np.linspace(-1, 1, 16)
    a = index.search(knn(v, 12))
    b = index.search(knn(v, 12, method))
    assert b["knn"] == {"embedding": {"requested": method, "ran": "exact"}}
    assert a["knn"] == {"embedding": {"requested": "exact", "ran": "exact"}}
    assert ids(a) == ids(b) and [h["_score"] for h in a["hits"]["hits"]] == [h["_score"] for h in b["hits"]["hits"]]


def test_no_knn_no_field(index):
    assert "knn" not in index.search({"query": {"match": {"product_title": "shoes"}}})


def test_several_clauses_on_one_field_are_a_list(index):
    v = np.linspace(-1, 1, 16).tolist()
    r = index.search({"query": {"hybrid": {"queries": [
        {"knn": {"embedding": {"vector": v, "k": 5, "method": "cagra"}}},
        {"neural": {"embedding": {"query_text": "black shoes", "k": 5}}}]}}})
    ran = "cagra" if has_cuvs() else "exact"
    assert r["knn"] == {"embedding": [{"requested": "cagra", "ran": ran}, {"requested": "exact", "ran": "exact"}]}


@pytest.mark.parametrize("spec,where", [
    ({"method": "hnsw"}, "method"),
    ({"method": "ivf_pq", "method_parameters": {"itopk_size": 64}}, "method_parameters.itopk_size"),
    ({"method": "cagra", "method_parameters": {"itopk_size": 0}}, "method_parameters.itopk_size"),
    ({"method": "exact", "method_parameters": {"n_probes": 4}}, "method_parameters.n_probes"),
    ({"method": "cagra", "method_parameters": []}, "method_parameters"),
])
def test_bad_method_is_invalid(index, spec, where):
    body = {"vector": [0.1] * 16, "k": 3, **spec}
    with pytest.raises(InvalidRequest) as e:
        index.search({"query": {"knn": {"embedding": body}}})
    assert e.value.path.endswith(where)


def exact_segment_searcher(b, part, field, method, params, q, k, valid_local):
    """A stand-in for cuVS: each segment's exact top-k among its valid docs."""
    sims = ops.dense_scores(b, part.vectors[field], q)
    idx, _ = b.topk(sims, k, valid_local.reshape(1, -1))
    return b.astype(idx, "int64")


@pytest.mark.parametrize("seed", [0, 1])
def test_segment_merge_path_equals_exact(monkeypatch, device, seed):
    base, deltas, final = scenario(seed)
    ix = incremental(base, deltas, device)               # three segments, edits and deletes
    reqs = [(vec("P0007"), None), (vec("P0042"), {"range": {"price": {"gte": 20, "lte": 80}}}),
            (vec("P0101"), {"term": {"category": "socks"}})]
    want = [ix.search(knn(v, 15, None, f)) for v, f in reqs]
    monkeypatch.setattr(ann, "available", lambda b: True)
    monkeypatch.setattr(ann, "search_part", exact_segment_searcher)
    for (v, f), w in zip(reqs, want):
        for m in ("cagra", "ivf_pq"):
            got = ix.search(knn(v, 15, m, f))
            assert got["knn"] == {"embedding": {"requested": m, "ran": m}}
            assert ids(got) == ids(w) and np.allclose([h["_score"] for h in got["hits"]["hits"]],
                                                      [h["_score"] for h in w["hits"]["hits"]], atol=1e-6)


def test_segment_searcher_sees_only_valid_docs(monkeypatch):
    base, deltas, final = scenario(0)
    ix = incremental(base, deltas)
    seen = []

    def spy(b, part, field, method, params, q, k, valid_local):
        seen.append((part.n, int(np.count_nonzero(b.to_numpy(valid_local))), k, dict(params)))
        return exact_segment_searcher(b, part, field, method, params, q, k, valid_local)

    monkeypatch.setattr(ann, "available", lambda b: True)
    monkeypatch.setattr(ann, "search_part", spy)
    ix.search(knn(vec("P0007"), 15, "ivf_pq", {"term": {"category": "socks"}}, {"n_probes": 7}))
    assert len(seen) == len(ix.segments)
    st = ix._st
    for (n, nvalid, k, params), part in zip(seen, st.parts):
        assert nvalid <= len(part.live_local) and k == min(15, nvalid) and params == {"n_probes": 7}


# ---- on the GPU (Colab)

@cuvs
@pytest.mark.parametrize("method", ["cagra", "ivf_pq"])
def test_cuvs_recall_and_filter(segment, method):
    ix = Index(copy.deepcopy(segment), "cuda")
    rng = np.random.default_rng(0)
    rec = []
    for _ in range(20):
        v = rng.normal(size=16)
        a, b = ix.search(knn(v, 10)), ix.search(knn(v, 10, method))
        assert b["knn"]["embedding"]["ran"] == method
        rec.append(len(set(ids(a)) & set(ids(b))) / 10)
        f = {"term": {"category": "socks"}}
        allowed = set(ids(ix.search({"query": f, "size": 1000})))
        assert set(ids(ix.search(knn(v, 10, method, f)))) <= allowed
    assert np.mean(rec) >= 0.9


@cuvs
def test_cuvs_index_rebuilt_on_merge():
    base, deltas, final = scenario(0)
    ix = incremental(base, deltas, "cuda")
    ix.search(knn(vec("P0007"), 10, "cagra"))
    assert all("_ann" in p.__dict__ for p in ix._st.parts)
    segs, v = ix.snapshot_segments()
    ix.replace_segments([s["segment_id"] for s in ix.segments], merge_segments(segs, version=v))
    assert all("_ann" not in p.__dict__ for p in ix._st.parts)
    r = ix.search(knn(vec("P0007"), 10, "cagra"))
    assert r["knn"]["embedding"]["ran"] == "cagra" and all("_ann" in p.__dict__ for p in ix._st.parts)
