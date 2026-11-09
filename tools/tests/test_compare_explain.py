import json
import time

import pandas as pd
import pytest

from tools import compare, explain
from tools.common import ApiError, http_search, text_request
from tools.tests.fakes import FakeIndex, serve_fake

J = {"1": {"a": "E", "b": "S", "c": "I"}, "2": {"x": "E", "y": "C"}}


def test_run_file_roundtrip_and_aliases(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"query_id": 1, "ranked_ids": ["a", "b"]}\n{"query_id": "2", "ids": [7]}\n\n')
    assert compare.load_run(p) == {"1": ["a", "b"], "2": ["7"]}
    compare.save_run({"3": ["z"]}, tmp_path / "o.jsonl")
    assert compare.load_run(tmp_path / "o.jsonl") == {"3": ["z"]}


def test_run_file_rejects_bad_rows(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text('{"query_id": 1, "ids": []}\n{"query_id": 1, "ids": []}\n')
    with pytest.raises(ValueError, match="twice"):
        compare.load_run(p)
    p.write_text('{"query_id": 1}\n')
    with pytest.raises(ValueError, match="need query_id"):
        compare.load_run(p)


def test_compare_matches_referee_metrics_exactly():
    from referee.metrics import score_query, score_run
    a = {"1": ["a", "b", "c"], "2": ["y", "x"]}
    b = {"1": ["b", "a", "c"], "2": ["x", "y"]}
    c = compare.compare_runs(a, b, J, label_a="A", label_b="B")
    assert c["mean"]["a"] == score_run(a, J)["mean"] and c["mean"]["b"] == score_run(b, J)["mean"]
    q1 = next(p for p in c["per_query"] if p["query_id"] == "1")
    assert q1["ndcg_a"] == score_query(a["1"], J["1"])["ndcg@10"]
    assert q1["ndcg_delta"] == pytest.approx(q1["ndcg_b"] - q1["ndcg_a"])
    assert q1["ndcg_delta"] < 0  # E moved from rank 1 to 2
    assert c["n_better"] + c["n_worse"] + c["n_same"] == 2


def test_overlap_and_rank_changes():
    c = compare.compare_runs({"1": ["a", "b", "c"]}, {"1": ["c", "a", "d"]}, J, k=3)
    q = c["per_query"][0]
    assert q["overlap"] == pytest.approx(2 / 3)
    ch = {r["id"]: r for r in q["rank_changes"]}
    assert ch["a"]["moved"] == 1 and ch["c"]["moved"] == -2
    assert ch["b"]["rank_b"] is None and ch["d"]["rank_a"] is None


def test_fit_only_unjudged_queries_are_skipped_and_missing_scored_empty():
    c = compare.compare_runs({"1": ["a"], "99": ["q"]}, {"2": ["x"], "99": ["q"]}, J)
    assert c["n_queries"] == 2 and c["skipped_not_judged"] == 1
    assert c["missing_a"] == ["2"] and c["missing_b"] == ["1"]
    assert c["mean"]["b"]["ndcg@10"] is not None


def test_fit_judgments_and_markdown():
    df = pd.DataFrame({"query_id": [1, 1], "product_id": ["a", "b"], "esci_label": ["E", "S"], "query": ["q", "q"]})
    j = compare.fit_judgments(df)
    assert j == {"1": {"a": "E", "b": "S"}}
    md = compare.render_markdown(compare.compare_runs({"1": ["a"]}, {"1": ["b"]}, j, {"1": "red shoes"}))
    assert "Fit-split" in md and "red shoes" in md and "not a score of record" in md


def test_record_run_against_fake_engine_and_compare():
    idx = FakeIndex()
    run = compare.record_run(idx.search, {1: "red shoe", 2: "desk lamp"}, fields=["product_title"], size=20)
    assert set(run) == {"1", "2"} and run["1"]
    judg = {"1": {run["1"][0]: "E"}, "2": {run["2"][0]: "E"}}
    c = compare.compare_runs(run, {q: list(reversed(v)) for q, v in run.items()}, judg)
    assert c["mean"]["a"]["ndcg@10"] == 1.0 and c["mean_overlap"] is not None


# -- explain ---------------------------------------------------------------------------------------------------------
def test_explain_sums_and_root_match():
    idx = FakeIndex()
    req = text_request("red running shoe", ["product_title^2", "product_brand"])
    top = idx.search(req)["hits"]["hits"][3]["_id"]
    e = explain.explain_product(idx.search, req, top)
    assert e["found"] and e["rank"] == 4 and e["ok"] and e["complete"] and e["root_matches_score"]
    assert e["n_checked"] >= 1 and e["leaves"]
    assert sum(l["value"] for l in e["leaves"]) == pytest.approx(e["score"], abs=1e-4)
    assert top in explain.render_text(e)


def test_explain_detects_a_trace_that_does_not_sum():
    idx = FakeIndex(explain_ok=False)
    req = text_request("shoe")
    e = explain.explain_product(idx.search, req, idx.search(req)["hits"]["hits"][0]["_id"])
    assert not e["ok"] and e["n_failed"] == 1 and not e["root_matches_score"]


def test_unknown_combiner_is_unchecked_not_passed():
    tree = {"value": 3.0, "description": "mystery", "details": [{"value": 1.0, "description": "a", "details": []}]}
    (c,) = explain.check_tree(tree)
    assert c["ok"] is None
    prod = {"value": 6.0, "description": "product of:", "details": [{"value": 2.0, "description": "x", "details": []},
                                                                    {"value": 3.0, "description": "y", "details": []}]}
    assert explain.check_tree(prod)[0]["ok"] is True
    lucene = {"value": 6.0, "description": "score(freq=1.0), computed as boost * idf * tf from:",
              "details": [{"value": 2.0, "description": "idf", "details": []},
                          {"value": 3.0, "description": "tf", "details": []}]}
    assert explain.check_tree(lucene)[0]["ok"] is True


def test_explain_not_found_filtered_and_missing_explain():
    idx = FakeIndex()
    req = {"query": {"bool": {"must": [{"match": {"product_title": "shoe"}}],
                              "filter": [{"term": {"category": "Shoes"}}]}}}
    lamp = next(d["product_id"] for d in idx.docs if d["category"] == "Lamps")
    e = explain.explain_product(idx.search, req, lamp)
    assert not e["found"] and "filter" in e["reason"]
    shoe = next(d["product_id"] for d in idx.docs if d["category"] == "Shoes")
    e = explain.explain_product(idx.search, req, shoe)
    assert e["found"] and e["filters_passed"] == [{"term": {"category": "Shoes"}}]
    with pytest.raises(explain.ExplainUnavailable):
        explain.explain_product(lambda r: {"hits": {"total": 1, "hits": [{"_id": "a", "_score": 1.0}]}}, req, "a")


def test_sample_check_over_http_fake():
    srv, url = serve_fake(FakeIndex())
    try:
        t0 = time.time()
        s = explain.sample_check(http_search(url, "esci-sample"), ["red shoe", "blue jacket", "desk lamp"], n=30)
        assert s["n"] == 30 and s["n_ok"] == 30 and s["n_complete"] == 30
        with pytest.raises(ApiError) as ei:
            http_search(url, "esci-sample")({"size": 10001})
        assert ei.value.status == 400 and ei.value.body["error"]["type"] == "invalid_request"
        assert time.time() - t0 < 5
    finally:
        srv.shutdown()


def test_real_engine_skips_until_it_imports():
    api = pytest.importorskip("engine.api", reason="the architect's engine/api.py is not in yet")
    assert hasattr(api, "build_index")
