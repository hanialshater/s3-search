"""The zvec quality adapter without zvec: a fake module records the calls, so the request mapping, the collection's
schema and batches, the fusion choice and the response shape are checked offline (zvec itself runs only in Colab)."""
import json
import sys
import types

import pandas as pd
import pytest

from referee import full, loader
from referee import zvec as zv


class Fake(types.SimpleNamespace):
    pass


def fake_zvec(tmp_path):
    calls = {"insert": [], "query": [], "init": 0}

    class Col:
        def insert(self, docs):
            assert len(docs) <= 1024
            calls["insert"].append(docs)

        def optimize(self):
            calls["optimized"] = True

        def close(self):
            pass

        def query(self, queries, topk, output_fields, reranker=None):
            calls["query"].append({"queries": queries, "topk": topk, "output_fields": output_fields, "reranker": reranker})
            return [Fake(id=f"p{i}", score=1.0 / (i + 1)) for i in range(min(topk, 3))]

    def create_and_open(path, schema):
        import os
        os.makedirs(path, exist_ok=True)
        calls["schema"] = schema
        return Col()

    m = types.ModuleType("zvec")
    m.__version__ = "0.8.1"
    m.FtsIndexParam = lambda tokenizer_name, filters: ("fts", tokenizer_name, tuple(filters))
    m.CollectionSchema = lambda name, fields: Fake(name=name, fields=fields)
    m.FieldSchema = lambda name, dtype, nullable, index_param: Fake(name=name, dtype=dtype, nullable=nullable, index_param=index_param)
    m.DataType = Fake(STRING="STRING")
    m.create_and_open = create_and_open
    m.Doc = lambda id, fields: Fake(id=id, fields=fields)
    m.init = lambda **kw: calls.__setitem__("init", calls["init"] + 1)
    m.CollectionOption = lambda read_only: Fake(read_only=read_only)
    m.open = lambda path, option: (calls.__setitem__("opened", (path, option.read_only)), Col())[1]
    m.Query = lambda field_name, fts: Fake(field=field_name, text=fts.match_string)
    m.Fts = lambda match_string: Fake(match_string=match_string)
    m.WeightedReRanker = lambda weights: ("weighted", tuple(weights))
    m.RrfReRanker = lambda: ("rrf",)
    return m, calls


@pytest.fixture
def z(tmp_path, monkeypatch):
    m, calls = fake_zvec(tmp_path)
    monkeypatch.setitem(sys.modules, "zvec", m)
    monkeypatch.setattr(zv, "_INIT", False)
    monkeypatch.setenv("S3_SEARCH_ZVEC_DIR", str(tmp_path / "zv"))
    return calls


def test_routes():
    assert zv.routes({"query": {"match": {"product_title": "red shoe"}}, "size": 7}) == ("red shoe", [("product_title", 1.0)], 7)
    assert zv.routes({"query": {"match": {"product_title": {"query": "x"}}}})[0] == "x"
    t, f, s = zv.routes({"query": {"multi_match": {"query": "q", "fields": ["product_title^1.0", "product_brand^0.126", "product_color"]}}, "size": 2000})
    assert (t, s) == ("q", 2000) and f == [("product_title", 1.0), ("product_brand", 0.126), ("product_color", 1.0)]
    for bad in ({"query": {"multi_match": {"query": "q", "fields": ["a"], "type": "cross_fields"}}},
                {"query": {"bool": {}}}, {"query": {"match": {"a": {"query": "x", "operator": "and"}}}}, {}):
        with pytest.raises(zv.UnsupportedQuery):
            zv.routes(bad)


def test_build_schema_batches_and_reuse(z):
    ds = loader.load_dataset("toy")
    cat = pd.concat([ds.catalogue] * 6, ignore_index=True)          # over one write batch
    cat["product_id"] = [f"id{i}" for i in range(len(cat))]
    cat.loc[0, "product_brand"] = None
    zv.build_index(cat, ds.mapping)
    text = [f for f, sp in ds.mapping["properties"].items() if sp.get("type") == "text"]
    assert [f.name for f in z["schema"].fields] == text
    assert all(f.index_param == ("fts", "standard", ("lowercase", "stemmer")) and f.nullable for f in z["schema"].fields)
    docs = [d for b in z["insert"] for d in b]
    assert len(docs) == len(cat) and docs[0].fields["product_brand"] == "" and z.get("optimized")
    n = len(z["insert"])
    zv.build_index(cat, ds.mapping)                                    # same settings and count: reused
    assert len(z["insert"]) == n and z["opened"][1] is True and z["init"] == 1


@pytest.mark.parametrize("fusion,want", [("weighted", ("weighted", (1.0, 0.5))), ("rrf", ("rrf",))])
def test_search_maps_fusion_and_shape(z, monkeypatch, fusion, want):
    monkeypatch.setenv("S3_SEARCH_ZVEC_FUSION", fusion)
    ds = loader.load_dataset("toy")
    idx = zv.build_index(ds.catalogue, ds.mapping)
    r = idx.search({"query": {"multi_match": {"query": "red", "fields": ["product_title", "product_brand^0.5"]}}, "size": 1000})
    q = z["query"][-1]
    assert q["topk"] == 1000 and q["output_fields"] == [] and q["reranker"] == want
    assert [x.field for x in q["queries"]] == ["product_title", "product_brand"]
    assert r["hits"]["hits"][0] == {"_id": "p0", "_score": 1.0} and r["hits"]["total"] == 3
    one = idx.search({"query": {"match": {"product_title": "red"}}, "size": 10})
    assert z["query"][-1]["reranker"] is None and len(one["hits"]["hits"]) == 3
    assert idx.search({"query": {"match": {"product_title": "  "}}})["hits"]["hits"] == []
    with pytest.raises(zv.InvalidRequest):
        idx.search({"query": {"match": {"no_such": "x"}}})
    assert len(idx.search_batch([{"query": {"match": {"product_title": "a"}}}] * 3)) == 3
    assert zv.settings()["fusion"] == fusion and zv.settings()["version"] == "0.8.1"


def test_bad_fusion(monkeypatch):
    monkeypatch.setenv("S3_SEARCH_ZVEC_FUSION", "max")
    with pytest.raises(ValueError):
        zv.fusion()


def test_full_refuses_bm25_for_zvec(tmp_path, capsys):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"k1": 0.5}))
    assert full.main(["--dataset=toy", "--engine=zvec", f"--bm25={p}", f"--out={tmp_path / 'r.json'}"]) == 2
    assert "zvec" in capsys.readouterr().err
