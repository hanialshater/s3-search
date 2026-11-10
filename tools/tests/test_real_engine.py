"""explain against the real engine on `toy`: 100 lexical and 100 rescored hits. Skips until engine.api imports."""
import numpy as np
import pytest

api = pytest.importorskip("engine.api", reason="engine/api.py not importable")
from referee import loader  # noqa: E402

from tools import explain  # noqa: E402
from tools.common import text_request  # noqa: E402

N = 100


@pytest.fixture(scope="module")
def toy():
    ds = loader.load_dataset("toy")
    return ds, api.build_index(ds.catalogue, ds.mapping)


def _pairs(index, ds, request_for, n=N):
    qs = list(dict.fromkeys(ds.fit["query"]))
    pairs = [(q, h["_id"]) for q in qs for h in index.search(request_for(q))["hits"]["hits"][:6]]
    assert len(pairs) >= n
    return [pairs[i] for i in np.random.default_rng(0).permutation(len(pairs))[:n]]


def _all_ok(results):
    bad = [(r["product_id"], [c for c in r.get("checks", []) if c["ok"] is not True][:1]) for r in results
           if not (r["found"] and r["ok"] and r["complete"] and r["root_matches_score"])]
    assert not bad, bad[:2]


def test_100_lexical_traces_sum_to_the_score(toy):
    ds, index = toy
    rf = lambda q: text_request(q, size=10)  # noqa: E731
    results = [explain.explain_product(index.search, rf(q), pid) for q, pid in _pairs(index, ds, rf)]
    assert len(results) == N
    _all_ok(results)  # complete: every node has an op, so nothing is left unchecked


def test_100_rescored_traces_sum_to_the_ranker_score(toy, tmp_path):
    ds, index = toy
    from engine.rank.train import train_ranker
    model = train_ranker(index, ds.fit, lambda q: text_request(q), window=100, max_queries=30)
    path = api.save_ranker(model, tmp_path / "ranker.json", index.feature_names())
    index.add_ranker("tools-test", api.load_ranker(path))  # the file round trip the serving ranker route uses
    resc = lambda q: {**text_request(q, size=20), "rescore": {"window_size": 50, "s3_ranker": {"model": "tools-test"}}}  # noqa: E731
    results = [explain.explain_product(index.search, resc(q), pid) for q, pid in _pairs(index, ds, resc)]
    assert len(results) == N
    _all_ok(results)
    rankers = [r for r in results if r["tree"].get("op") == "ranker"]
    assert rankers, "no hit was inside the rescore window"
    assert all(any("feature" in k for k in r["tree"]["details"]) for r in rankers)
    # the lexical tree behind lex_* features is shown (as `source`) and was itself checked
    assert any(c["path"].endswith("/source") for r in rankers for c in r["checks"])
