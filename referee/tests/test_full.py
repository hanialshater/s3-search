"""referee full, end to end on toy with the reference index: the scorer reads the test split, and the held-out
subset is scored from the same runs."""
import json

import pandas as pd

from referee import full, loader


def test_full_on_toy(tmp_path, monkeypatch):
    held = tmp_path / "held.parquet"
    test_ids = sorted(loader.load_dataset("toy").test_queries["query_id"])
    pd.DataFrame({"query_id": test_ids[:4]}).to_parquet(held)
    monkeypatch.setenv("S3_SEARCH_HELD_OUT", str(held))
    out = tmp_path / "results.json"
    assert full.main(["--dataset=toy", "--engine=reference", "--device=cpu", "--k=100", f"--out={out}"]) == 0
    r = json.loads(out.read_text())
    assert r["split"] == "test" and r["n_queries"] == 10 and r["held_out"] == {"n_queries": 4}
    assert r["score_of_record"] is False and r["label"] == "development only"
    sub = r["quality_subsets"]["held_out"]
    assert 0 < sub["ndcg@10"] <= 1 and sub["n_ndcg@10"] == 4
    assert r["freshness"]["passed"] and all(r["gates"].values())
    assert {"S3_TRITON_BM25", "S3_TRITON_TOPK", "S3_PLAN_CACHE", "S3_CUDA_BLOCKING_SYNC"} <= set(r["engine_env"])


def test_development_mode_is_labelled_and_never_writes_results_json(tmp_path, capsys):
    assert full.main(["--dataset=toy", "--engine=reference", "--queries=3", "--out", str(tmp_path / "results.json")]) == 2
    out = tmp_path / "results-dev.json"
    assert full.main(["--dataset=toy", "--engine=reference", "--queries=3", "--k=50", f"--out={out}"]) == 0
    r = json.loads(out.read_text())
    assert r["score_of_record"] is False and r["label"].startswith("DEVELOPMENT MODE") and r["n_queries"] == 3
    printed = capsys.readouterr().out
    assert "DEVELOPMENT MODE" in printed and "scoring:" in printed and "NOT a score of record" in printed
