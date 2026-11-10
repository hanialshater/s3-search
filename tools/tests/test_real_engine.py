"""explain against the real engine on `toy` (lexical and rescored). Skips until engine.api imports."""
import numpy as np
import pytest

api = pytest.importorskip("engine.api", reason="engine/api.py not importable")
from referee import loader  # noqa: E402

from tools import explain  # noqa: E402
from tools.common import text_request  # noqa: E402


def _has_op(node):
    return "op" in node and all(_has_op(k) for k in node.get("details") or [])


@pytest.fixture(scope="module")
def toy():
    ds = loader.load_dataset("toy")
    return ds, api.build_index(ds.catalogue, ds.mapping)


def _pairs(index, ds, request_for, n):
    qs = list(dict.fromkeys(ds.fit["query"]))[:15]
    pairs = [(q, h["_id"]) for q in qs for h in index.search(request_for(q))["hits"]["hits"][:8]]
    order = np.random.default_rng(0).permutation(len(pairs))[:n]
    return [pairs[i] for i in order]


def test_lexical_traces_check_out(toy):
    ds, index = toy
    results = [explain.explain_product(index.search, text_request(q, size=10), pid)
               for q, pid in _pairs(index, ds, lambda q: text_request(q, size=10), 30)]
    assert all(r["found"] and r["ok"] and r["n_failed"] == 0 for r in results), \
        [(r["product_id"], [c for c in r["checks"] if c["ok"] is False][:1]) for r in results if not r.get("ok")][:2]
    # when the engine labels its nodes with `op`, every node is checkable, so nothing may be left unchecked
    assert all(r["complete"] for r in results if _has_op(r["tree"]))


def test_rescored_traces_check_out(toy):
    ds, index = toy
    from engine.rank.train import train_ranker
    model = train_ranker(index, ds.fit, lambda q: text_request(q), window=100, max_queries=30)
    index.add_ranker("tools-test", model)
    resc = lambda q: {**text_request(q, size=20), "rescore": {"window_size": 50, "s3_ranker": {"model": "tools-test"}}}  # noqa: E731
    results = [explain.explain_product(index.search, resc(q), pid) for q, pid in _pairs(index, ds, resc, 30)]
    assert all(r["found"] and r["ok"] and r["n_failed"] == 0 for r in results)
    assert all(r["complete"] for r in results if _has_op(r["tree"]))
