"""BM25F against an independent plain-Python implementation (the prototype's check, on every device)."""
import math

import numpy as np
import pytest

from engine.analysis import analyze

FIELDS = ["product_title", "product_brand", "product_color", "product_bullet_point", "product_description"]


def reference_bm25f(docs_tokens, query_tokens, w, b, k1, one=None):
    N, F = len(docs_tokens), len(w)
    lens = np.array([[len(d[f]) for f in range(F)] for d in docs_tokens], dtype=float)
    avg = np.where(lens.mean(0) > 0, lens.mean(0), 1.0)
    out = np.zeros(N)
    for t in dict.fromkeys(query_tokens):
        if one is None:
            df = sum(any(t in d[f] for f in range(F)) for d in docs_tokens)
        else:
            df = sum(t in d[one] for d in docs_tokens)
        if df == 0:
            continue
        idf = math.log1p((N - df + 0.5) / (df + 0.5))
        for i, d in enumerate(docs_tokens):
            ptf = sum(w[f] * d[f].count(t) / (1 - b + b * lens[i, f] / avg[f]) for f in range(F))
            out[i] += idf * ptf * (k1 + 1) / (k1 + ptf) if ptf > 0 else 0.0
    return out


@pytest.fixture(scope="module")
def tokens(catalogue):
    return [[analyze(catalogue[f].iloc[i]) for f in FIELDS] for i in range(len(catalogue))]


def scores_of(index, req, n):
    out = np.zeros(n)
    r = index.search({**req, "size": n})
    pos = {f"P{i:04d}": i for i in range(n)}
    for h in r["hits"]["hits"]:
        out[pos[h["_id"]]] = h["_score"]
    return out, r


@pytest.mark.parametrize("q", ["black running shoes", "wool socks kids", "nike", "levi's jeans blue"])
@pytest.mark.parametrize("weights", [{"product_title": 3, "product_brand": 2, "product_color": 1,
                                      "product_bullet_point": 1, "product_description": 0.5},
                                     {"product_title": 1.0, "product_description": 0.25}])
def test_multi_match_is_bm25f(index, catalogue, tokens, q, weights):
    w = [float(weights.get(f, 0.0)) for f in FIELDS]
    want = reference_bm25f(tokens, analyze(q), w, 0.75, 1.2)
    fields = [f"{f}^{v}" for f, v in weights.items()]
    got, r = scores_of(index, {"query": {"multi_match": {"query": q, "fields": fields}}}, len(catalogue))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6)
    assert r["hits"]["total"] == int((want > 0).sum())


@pytest.mark.parametrize("q", ["black running shoes", "nike", "socks"])
def test_single_field_match_uses_field_idf(index, catalogue, tokens, q):
    w = [1.0, 0, 0, 0, 0]
    want = reference_bm25f(tokens, analyze(q), w, 0.75, 1.2, one=0)
    got, _ = scores_of(index, {"query": {"match": {"product_title": q}}}, len(catalogue))
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_operator_and_and_msm(index, catalogue, tokens):
    q = "black running shoes"
    terms = list(dict.fromkeys(analyze(q)))
    have = np.array([sum(t in d[0] for t in terms) for d in tokens])
    r = index.search({"query": {"match": {"product_title": {"query": q, "operator": "and"}}}, "size": 0})
    assert r["hits"]["total"] == int((have == 3).sum())
    r = index.search({"query": {"match": {"product_title": {"query": q, "minimum_should_match": 2}}}, "size": 0})
    assert r["hits"]["total"] == int((have >= 2).sum())
    r = index.search({"query": {"match": {"product_title": {"query": q, "minimum_should_match": "-34%"}}}, "size": 0})
    assert r["hits"]["total"] == int((have >= 2).sum())        # 3 - floor(3 * 0.34) = 2


def test_unknown_and_stopword_queries(index):
    assert index.search({"query": {"match": {"product_title": "the of and"}}})["hits"]["total"] == 0
    assert index.search({"query": {"match": {"product_title": {"query": "black zzzz", "operator": "and"}}}}
                        )["hits"]["total"] == 0


def test_boost_scales(index):
    a = index.search({"query": {"match": {"product_title": "shoes"}}, "size": 3})["hits"]["hits"]
    b = index.search({"query": {"match": {"product_title": {"query": "shoes", "boost": 2}}}, "size": 3})["hits"]["hits"]
    assert [h["_id"] for h in a] == [h["_id"] for h in b]
    assert np.allclose([2 * h["_score"] for h in a], [h["_score"] for h in b], rtol=1e-6)


def test_analyzer():
    assert analyze("Levi's Running SHOES, the best!") == ["levi", "run", "shoe", "best"]
    assert analyze(None) == [] and analyze(float("nan")) == []
