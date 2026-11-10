"""Explain adds up (the S5 defect 3 fix). Pre-registered prediction: 100 lexical and 100 rescored explanations sum
to the hit's score within 1e-4, on toy. Every node carries a machine-readable ``op``."""
import numpy as np
import pytest

from engine.api import build_index
from engine.rank.train import train_ranker
from referee.loader import load_dataset

TOL = 1e-4
OPS = {"sum", "product", "max", "value", "ranker"}


def check_tree(node, path="root"):
    """Every node has an op; sum/product/max/ranker nodes equal their details' combination within TOL."""
    assert node.get("op") in OPS, (path, node.get("op"))
    kids = node["details"]
    vals = [k["value"] for k in kids]
    if node["op"] in ("sum", "ranker"):
        assert abs(sum(vals) - node["value"]) <= TOL, (path, node["value"], sum(vals))
    elif node["op"] == "product":
        assert abs(float(np.prod(vals)) - node["value"]) <= TOL, path
    elif node["op"] == "max":
        assert abs(max(vals) - node["value"]) <= TOL, path
    if node["op"] == "ranker":
        assert kids[0]["description"] == "ranker base value" and all("feature" in k for k in kids[1:])
    for i, k in enumerate(kids):
        check_tree(k, f"{path}.details[{i}]")
    if "source" in node:
        check_tree(node["source"], f"{path}.source")


@pytest.fixture(scope="module")
def toy():
    ds = load_dataset("toy")
    return ds, build_index(ds.catalogue, ds.mapping)


def texts(ds):
    return sorted(set(ds.fit["query"]) | set(ds.test_queries["query"]))


def lexical_requests(ds):
    text_fields = [f for f, s in ds.mapping["properties"].items() if s["type"] == "text"]
    for t in texts(ds):
        yield {"query": {"match": {text_fields[0]: t}}}
        yield {"query": {"multi_match": {"query": t, "fields": [f"{f}^{1 + i}" for i, f in enumerate(text_fields)]}}}
        yield {"query": {"bool": {"must": [{"match": {text_fields[0]: {"query": t, "boost": 2.0}}}],
                                  "should": [{"multi_match": {"query": t}}, {"range": {"price": {"lte": 50}}}],
                                  "filter": [{"range": {"stock": {"gte": 0}}}], "boost": 0.5}}}


def test_100_lexical_explanations_add_up(toy):
    ds, ix = toy
    n = 0
    for req in lexical_requests(ds):
        r = ix.search({**req, "size": 10, "explain": True})
        for h, e in zip(r["hits"]["hits"], r["explain"]):
            assert e["_id"] == h["_id"]
            assert abs(e["value"] - h["_score"]) <= TOL, (req, e["value"], h["_score"])
            check_tree(e)
            n += 1
    assert n >= 100, n


def test_100_rescored_explanations_add_up(toy):
    ds, ix = toy
    emb = np.stack(ds.catalogue["embedding"].to_numpy())

    def request_for(text):
        top = ix.search({"query": {"multi_match": {"query": text}}, "size": 1})["hits"]["hits"]
        row = int(np.flatnonzero(ds.catalogue["product_id"].astype(str).to_numpy() == top[0]["_id"])[0]) if top else 0
        return {"query": {"hybrid": {"queries": [{"multi_match": {"query": text}},
                                                 {"knn": {"embedding": {"vector": emb[row].tolist(), "k": 40}}}]}}}

    model = train_ranker(ix, ds.fit, request_for, window=100, n_random=10, n_rounds=30, min_leaf=5)
    ix.add_ranker("m", model)
    n = 0
    for t in texts(ds):
        req = {**request_for(t), "rescore": {"window_size": 20, "s3_ranker": {"model": "m"}}, "size": 25,
               "explain": True}
        r = ix.search(req)
        for h, e in zip(r["hits"]["hits"], r["explain"]):
            assert abs(e["value"] - h["_score"]) <= TOL
            check_tree(e)
            if h["sort"][0] == 1:                                  # inside the window: the ranker's score
                assert e["op"] == "ranker" and len(e["details"]) == 1 + len(ix.feature_names())
                lex = next(k for k in e["details"] if k.get("feature") == "lex_score")
                assert lex["source"]["op"] in OPS
                n += 1
    assert n >= 100, n


def test_save_and_load_ranker(tmp_path):
    import json
    from engine.api import load_ranker, save_ranker
    from engine.rank.gbdt import GBDT
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    m = GBDT(n_rounds=10, min_leaf=5).fit(X, X[:, 0])
    p = save_ranker(m, tmp_path / "m.json", ["a", "b", "c"])
    doc = json.loads(p.read_text())
    assert doc["format"] == "s3-ranker" and doc["format_version"] == 1 and doc["model"] == m.to_dict()
    back = load_ranker(p)
    assert np.array_equal(back.predict(X), m.predict(X)) and back.feature_names == ["a", "b", "c"]
    (tmp_path / "bare.json").write_text(json.dumps(m.to_dict()))         # serving's interim format
    assert np.array_equal(load_ranker(tmp_path / "bare.json").predict(X), m.predict(X))
    (tmp_path / "bad.json").write_text(json.dumps({"format": "s3-ranker", "format_version": 2}))
    with pytest.raises(ValueError):
        load_ranker(tmp_path / "bad.json")


def test_gbdt_contributions_are_exact(toy):
    from engine.rank.gbdt import GBDT
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5))
    X[rng.random(500) < 0.2, 1] = np.nan
    y = X[:, 0] * 2 + np.nan_to_num(X[:, 1]) - (X[:, 2] > 0)
    m = GBDT(n_rounds=25, max_depth=3, min_leaf=5).fit(X, y)
    base, c = m.contributions(X)
    assert np.allclose(base + c.sum(1), m.predict(X), atol=1e-5)
    assert np.abs(c[:, 0]).mean() > np.abs(c[:, 4]).mean()     # the informative feature carries the attribution
