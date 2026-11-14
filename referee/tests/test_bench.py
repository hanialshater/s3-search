"""The benchmark harness, end to end on toy with the reference index (the engine is not needed to test the harness)."""
import json

from referee import bench, loader, reference


def test_bench_on_toy_writes_results(tmp_path):
    out = tmp_path / "results.json"
    assert bench.main(["--dataset=toy", "--engine=reference", f"--out={out}", "--k=50"]) == 0
    r = json.loads(out.read_text())
    assert r["label"] == "development only" and r["split"] == "fit" and r["n_queries"] == 20
    assert 0 < r["quality"]["ndcg@10"] <= 1 and 0 < r["quality"]["exact_recall@50"] <= 1
    assert r["latency_ms"]["p50"] <= r["latency_ms"]["p99"] and r["throughput"]["qps"] > 0
    assert r["cost_per_million_queries"] is None


def test_bench_scores_only_what_it_is_given():
    """A perfect run scores 1; the harness never reaches for test judgments (it is given the fit split)."""
    ds = loader.load_dataset("toy")
    judg = loader.fit_judgments(ds)

    class Oracle:
        def search(self, r):
            return self.search_batch([r])[0]

        def search_batch(self, reqs):
            out = []
            for r in reqs:
                text = r["query"]["multi_match"]["query"]
                q = int(ds.fit.loc[ds.fit["query"] == text, "query_id"].iat[0])
                ranked = sorted(judg[q], key=lambda p: -{"E": 3, "S": 2, "C": 1, "I": 0}[judg[q][p]])
                out.append({"hits": {"total": len(ranked), "hits": [{"_id": p, "_score": 1.0} for p in ranked]},
                            "took_ms": {"total": 0.0}})
            return out

    queries = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    r = bench.run(ds, queries, judg, reference, index=Oracle(), k=100, price_per_hour=1.0)
    assert abs(r["quality"]["ndcg@10"] - 1.0) < 1e-12 and r["quality"]["exact_recall@100"] == 1.0
    assert r["cost_per_million_queries"] > 0


def test_template_fill():
    t = {"query": {"bool": {"must": [{"match": {"product_title": "{{query}}"}}], "filter": [{"term": {"x": 1}}]}}}
    assert bench.fill(t, "red lamp")["query"]["bool"]["must"][0]["match"]["product_title"] == "red lamp"


def test_results_record_the_engine_switches(monkeypatch):
    """S3_TRITON_BM25, S3_PLAN_CACHE and S3_CUDA_BLOCKING_SYNC are recorded as set, null when unset."""
    from referee import bench as b
    monkeypatch.setenv("S3_TRITON_BM25", "1")
    monkeypatch.delenv("S3_PLAN_CACHE", raising=False)
    monkeypatch.setenv("S3_CUDA_BLOCKING_SYNC", "0")
    assert b.engine_env() == {"S3_TRITON_BM25": "1", "S3_PLAN_CACHE": None, "S3_CUDA_BLOCKING_SYNC": "0"}
