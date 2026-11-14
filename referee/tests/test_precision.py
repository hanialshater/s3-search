"""The precision comparison's figures and gates on synthetic runs (the GPU runs are the steward's, in Colab)."""
import numpy as np
import pytest

from referee import precision as pr


def run(ids_by_q, scale=1.0, totals=None):
    return {q: (list(ids), np.linspace(10, 1, len(ids)) * scale, (totals or {}).get(q, len(ids))) for q, ids in ids_by_q.items()}


def test_parse_variant():
    assert pr.parse_variant("a:S3_TRITON_BM25_WEIGHTS=float16,S3_TRITON_BM25_SUM=float32") == \
        ("a", {"S3_TRITON_BM25_WEIGHTS": "float16", "S3_TRITON_BM25_SUM": "float32"})
    for bad in ("a", "a:S3_PLAN_CACHE=0", ":S3_TRITON_BM25_SUM=float32"):
        with pytest.raises(ValueError):
            pr.parse_variant(bad)


def test_identical_runs_pass():
    judg = {1: {"a": "E", "b": "S"}, 2: {"c": "E"}}
    base = run({1: ["a", "b", "x"], 2: ["c", "y"]})
    res = pr.compare(base, run({1: ["a", "b", "x"], 2: ["c", "y"]}), judg, k=3)
    assert all(v for k, v in res["gates"].items() if "record" not in k)   # synthetic data is not the record    assert res["figures"]["ndcg10_diff"] == 0 and res["figures"]["overlap_mean"] == 1.0
    assert res["figures"]["score_rel_error_max"] == 0


def test_changes_are_counted():
    judg = {q: {"a": "E"} for q in range(10)}
    base = run({q: ["a", "b", "c"] for q in range(10)})
    var = run({**{q: ["a", "b", "c"] for q in range(9)}, 9: ["b", "a", "z"]}, scale=1.001, totals={9: 99})
    f = pr.compare(base, var, judg, k=3)
    g, fig = f["gates"], f["figures"]
    assert fig["top10_list_changed_share"] == 0.1 and fig["top10_set_changed_share"] == 0.1
    assert fig["queries_with_different_totals"] == 1 and not g["5 match counts equal"]
    assert fig["n_overlap_below_0.95"] == 1 and fig["overlap_below_0.95"][0]["query_id"] == 9
    assert fig["score_rel_error_max"] >= 0.001   # 0.1% on the unchanged queries, more where ranks moved
    assert not g["3 top 10 as a list changed in at most 5%"]


def test_same_bits():
    a = run({1: ["a"]})
    b = {1: (["a"], a[1][1].copy(), a[1][2])}
    assert pr.same_bits(a, b)
    b[1][1][0] = np.nextafter(b[1][1][0], 0)
    assert not pr.same_bits(a, b)
