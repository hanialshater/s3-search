import json

import pytest

from tools import report
from tools.explain import explain_product
from tools.common import text_request
from tools.tests.fakes import FakeIndex

RES = {
    "meta": {"device": "cpu", "dataset": "toy", "split": "fit", "commit": "abc1234"},
    "quality": {"ndcg@10": 0.3041234567, "exact_recall@100": 0.9, "exact_recall@1000": 0.95, "n_queries": 500},
    "latency": {"page1_p50_ms": 20.6, "page1_p99_ms": 33.8, "stages": {"retrieve": {"p50_ms": 5.0}}},
    "throughput": {"qps_at_p99_100ms": 275},
    "cost": {"per_million_queries_usd": 1.25},
    "slices": {"negation": {"ndcg@10": 0.11, "n": 12}, "filtered": {"ndcg@10": 0.4, "n": 30}},
    "worst_queries": [{"query_id": 7, "query": "no <b>red</b> shoes", "ndcg@10": 0.0,
                       "explain": {"value": 3.0, "description": "sum of", "details": [
                           {"value": 3.0, "description": "weight(title:shoe)", "details": []}]}}],
    "extra_thing": 42,
}
PREDS = [{"text": "p95 under 300 ms", "wrongIf": "p95 over 300 ms", "by": "onboarding"}]


def test_every_number_is_the_results_number():
    page = report.render(RES, PREDS)
    assert report.check_report(RES, page) == []
    assert 'data-value="0.3041234567"' in page  # exact, not the rounded display
    assert "0.3041" in page and "extra_thing" in page  # unknown fields are still shown


def test_check_catches_a_changed_number_and_a_missing_one():
    page = report.render(RES)
    bad = page.replace('data-value="0.9"', 'data-value="0.91"')
    assert any("exact_recall@100" in p for p in report.check_report(RES, bad))
    res2 = json.loads(json.dumps(RES))
    res2["quality"]["new_metric"] = 1.0
    assert any("new_metric" in p and "not on the page" in p for p in report.check_report(res2, page))


def test_cpu_is_labelled_development_only_and_gpu_is_not():
    assert "Development only" in report.render(RES) and "development only: CPU" in report.render(RES)
    gpu = {**RES, "meta": {"device": "cuda", "gpu": "A100"}}
    page = report.render(gpu)
    assert "Development only" not in page and "score of record only if" in page
    assert report.is_development({"quality": {}})  # unknown device is treated as development


def test_predictions_slices_recall_and_escaping():
    page = report.render(RES, PREDS)
    assert "p95 under 300 ms" in page and "Wrong if: p95 over 300 ms" in page
    assert "Quality by slice" in page and "negation" in page and "<h2>Recall</h2>" in page
    assert "no <b>red</b>" not in page and "no &lt;b&gt;red&lt;/b&gt; shoes" in page
    assert "weight(title:shoe)" in page


def test_recompute_matches_referee_and_flags_difference():
    from referee.metrics import score_run
    runs = {"1": ["a", "b"], "2": ["y"]}
    j = {"1": {"a": "E", "b": "S"}, "2": {"x": "E", "y": "C"}}
    m = score_run(runs, j)["mean"]
    res = {"quality": {k: v for k, v in m.items() if not k.startswith("n_")}}
    assert all(r["ok"] for r in report.recompute(res, runs, j))
    res["quality"]["ndcg@10"] += 0.01
    assert [r["metric"] for r in report.recompute(res, runs, j) if not r["ok"]] == ["ndcg@10"]
    assert "DIFFERENT" in report.render(res, recomputed=report.recompute(res, runs, j))


def test_recompute_only_the_depths_the_referee_wrote():
    from referee.metrics import score_run
    runs = {"1": ["a", "b"], "2": ["y"]}
    j = {"1": {"a": "E", "b": "S"}, "2": {"x": "E", "y": "C"}, "3": {"z": "E"}}  # query 3 was not run
    m = score_run({q: runs[q] for q in runs}, {q: j[q] for q in runs}, recall_ks=(100,))["mean"]
    res = {"quality": dict(m)}  # k=100 only, with the n_ counts, as bench writes them
    rows = report.recompute(res, runs, j)
    assert {r["metric"] for r in rows} == set(m) and all(r["ok"] for r in rows)
    assert not any("1000" in r["metric"] for r in rows)
    assert report.recompute({"quality": {"latency": 1.0}}, runs, j) == []


def test_top_level_bench_fields_decide_development_and_are_shown():
    res = {"dataset": "esci-sample", "device": "cuda", "split": "fit", "commit": "abc", "quality": {"ndcg@10": 0.2}}
    assert not report.is_development(res) and report.is_development({**res, "device": "cpu"})
    assert "esci-sample" in report.render(res)


def test_recompute_matches_a_real_bench_run_on_toy():
    api = pytest.importorskip("engine.api")
    from referee import bench, loader
    ds = loader.load_dataset("toy")
    runs: dict = {}
    res = bench.run(ds, ds.fit[["query_id", "query"]].drop_duplicates("query_id"), loader.fit_judgments(ds), api,
                    k=100, latency_queries=5, runs_out=runs)
    rows = report.recompute(res, runs, loader.fit_judgments(ds))
    assert rows and all(r["ok"] for r in rows), rows
    assert report.check_report(res, report.render(res, recomputed=rows)) == []


def test_predictions_from_milestones_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"milestones": [{"id": "S4", "predictions": []}, {"id": "S5", "predictions": PREDS}]}))
    assert report.load_predictions(p, "S5") == PREDS


def test_cli_writes_report_and_exit_code(tmp_path):
    r = tmp_path / "results.json"
    r.write_text(json.dumps(RES))
    out = tmp_path / "r.html"
    assert report.main([str(r), "--out", str(out)]) == 0 and "Test-drive report" in out.read_text()


def test_worst_query_explanation_from_the_engine_trace_renders():
    idx = FakeIndex()
    req = text_request("red shoe")
    e = explain_product(idx.search, req, idx.search(req)["hits"]["hits"][0]["_id"])
    res = {"meta": {"device": "cpu"}, "worst_queries": [{"query": "red shoe", "ndcg@10": 0.1, "explain": e["tree"]}]}
    page = report.render(res)
    assert "sum of" in page and report.check_report(res, page) == []
