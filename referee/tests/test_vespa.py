"""The Vespa quality adapter without Vespa: /search/ and /document/v1 are faked, so the request mapping, the V1
coverage rule, the corpus check and the response shape are checked offline (Vespa itself runs only in Colab)."""
import json

import pandas as pd
import pytest

from referee import full
from referee import vespa as vs


def resp(ids, coverage=None, errors=None, total=None):
    root = {"fields": {"totalCount": total if total is not None else len(ids)},
            "children": [{"relevance": 1.0 / (i + 1), "fields": {"product_id": p}} for i, p in enumerate(ids)],
            "coverage": coverage or {"coverage": 100, "documents": 3, "full": True}}
    if errors:
        root["errors"] = errors
    return {"root": root}


def test_body_title_and_multi(monkeypatch):
    monkeypatch.delenv("S3_SEARCH_VESPA_MODE", raising=False)
    b = vs.body({"query": {"match": {"product_title": "red shoe"}}, "size": 1000})
    assert b["ranking"] == "title" and b["hits"] == 1000 and b["query"] == "red shoe"
    assert 'defaultIndex: "product_title", grammar: "any"' in b["yql"] and "targetHits" not in b["yql"]
    m = vs.body({"query": {"multi_match": {"query": "q", "fields": ["product_title^1.0", "product_brand^0.126"]}}, "size": 10})
    assert m["ranking"] == "bm25w" and m["input.query(w_product_brand)"] == 0.126
    assert m["input.query(w_product_color)"] == 0.0 and m["yql"].count("userInput(@query)") == 2
    monkeypatch.setenv("S3_SEARCH_VESPA_MODE", "weakAnd")
    w = vs.body({"query": {"match": {"product_title": "x"}}, "size": 2000})
    assert 'grammar: "weakAnd", targetHits: 2000' in w["yql"]
    monkeypatch.setenv("S3_SEARCH_VESPA_RANK_PROPERTIES", json.dumps({"bm25(product_title).k1": 0.9}))
    assert vs.body({"query": {"match": {"product_title": "x"}}})["ranking.properties"] == {"bm25(product_title).k1": 0.9}
    for bad in ({"query": {"match": {"product_brand": "x"}}}, {"query": {"bool": {}}},
                {"query": {"multi_match": {"query": "q", "fields": ["nope"]}}},
                {"query": {"multi_match": {"query": "q", "fields": ["product_title"], "type": "cross_fields"}}}):
        with pytest.raises(vs.UnsupportedQuery):
            vs.body(bad)


def test_convert_and_v1():
    r = vs.convert(resp(["a", "b"], total=40), 3.0)
    assert r["hits"]["hits"] == [{"_id": "a", "_score": 1.0}, {"_id": "b", "_score": 0.5}] and r["hits"]["total"] == 40
    for bad in (resp(["a"], coverage={"coverage": 90, "full": False}), resp(["a"], coverage={"coverage": 100, "full": True, "degraded": {"timeout": True}}),
                resp(["a"], errors=[{"code": 12}])):
        with pytest.raises(vs.Degraded):
            vs.convert(bad, 1.0)


def test_build_checks_the_corpus(monkeypatch):
    cat = pd.DataFrame({"product_id": ["a", "b", "c"], "product_title": ["t a", "t b", None]})
    store = {"a": "t a", "b": "t b", "c": ""}
    calls = []

    def fake(method, path, body=None, timeout=60):
        calls.append((method, path))
        if path == "/search/":
            return resp([], total=count[0])
        return {"fields": {"product_title": store[path.rsplit("/", 1)[1]]}}
    count = [3]
    monkeypatch.setattr(vs, "_http", fake)
    idx = vs.build_index(cat, {})
    assert sum(1 for c in calls if c[1].startswith("/document/v1/")) == 3
    store["b"] = "something else"
    with pytest.raises(RuntimeError, match="title"):
        vs.build_index(cat, {})
    count[0] = 2
    with pytest.raises(RuntimeError, match="feed it first"):
        vs.build_index(cat, {})
    monkeypatch.setattr(vs, "_http", lambda *a, **k: resp(["x", "y"]))
    out = idx.search_batch([{"query": {"match": {"product_title": "q"}}}] * 3)
    assert len(out) == 3 and out[0]["hits"]["hits"][0]["_id"] == "x"


def test_full_refuses_bm25_for_vespa(tmp_path, capsys):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"k1": 0.5}))
    assert full.main(["--dataset=toy", "--engine=vespa", f"--bm25={p}", f"--out={tmp_path / 'r.json'}"]) == 2
    assert "vespa" in capsys.readouterr().err
