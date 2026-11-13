"""BM25F parameters in the referee (steward's review of ba28176, 2026-11-13).

- The reference scorer's per-field b against a value computed by hand, not against the engine (the engine's own test
  compares the two, and both changed in one commit by one author, so the oracle needs a check of its own).
- The reference refuses exactly what INTERFACES section 2 refuses, and, when the engine is importable, the same cases
  as the engine's parser.
- `referee full --bm25`: validated, recorded with its sha256; refused for OpenSearch; never touches what is scored.
"""
import copy
import json
import math

import pandas as pd
import pytest

from referee import freshness, full, loader, opensearch, reference, runtime

MAPPING = {"_meta": {"engine": {"id_field": "product_id", "hot": []}},
           "properties": {"product_id": {"type": "keyword"},
                          "product_title": {"type": "text", "analyzer": "s3_english"},
                          "product_brand": {"type": "text", "analyzer": "s3_english"}}}
CAT = pd.DataFrame({"product_id": ["d0", "d1", "d2"],
                    "product_title": ["red shoe", "blue shoe shoe", "red red red red"],
                    "product_brand": ["red", "zeta", "acme"]})


def _m(bm25):
    m = copy.deepcopy(MAPPING)
    m["_meta"]["engine"]["bm25"] = bm25
    return m


def test_per_field_b_by_hand():
    k1, b_title, b_brand = 0.7, 0.2, 0.9
    idx = reference.build_index(CAT, _m({"k1": k1, "b": {"product_title": b_title}, "b_default": b_brand}))
    r = idx.search({"query": {"multi_match": {"query": "red", "fields": ["product_title^1.0", "product_brand^0.5"]}},
                    "size": 10})
    got = {h["_id"]: h["_score"] for h in r["hits"]["hits"]}
    idf = math.log1p((3 - 2 + 0.5) / (2 + 0.5))               # "red" is in d0 and d2 (any field), N = 3
    avg_title, avg_brand = (2 + 3 + 4) / 3, 1.0

    def part(tf, length, avg, b, w):
        return w * tf / (1 - b + b * length / avg)

    def bm25(p):
        return idf * p * (k1 + 1) / (k1 + p)

    p0 = part(1, 2, avg_title, b_title, 1.0) + part(1, 1, avg_brand, b_brand, 0.5)
    p2 = part(4, 4, avg_title, b_title, 1.0)
    assert set(got) == {"d0", "d2"}
    assert got["d0"] == pytest.approx(bm25(p0), rel=1e-9)
    assert got["d2"] == pytest.approx(bm25(p2), rel=1e-9)
    # one number for every field is the same as a map naming every field with that number
    one = reference.build_index(CAT, _m({"k1": k1, "b": 0.4}))
    both = reference.build_index(CAT, _m({"k1": k1, "b": {"product_title": 0.4, "product_brand": 0.4}}))
    req = {"query": {"multi_match": {"query": "red shoe", "fields": ["product_title", "product_brand"]}}, "size": 10}
    assert one.search(req)["hits"]["hits"] == both.search(req)["hits"]["hits"]


GOOD = [{}, {"k1": 0.036}, {"b": 0.3}, {"k1": 0, "b": 1}, {"b": {"product_title": 0.742}},
        {"b": {"product_title": 0.742}, "b_default": 0.1}, {"b": {}, "b_default": 0.5}]
BAD = [({"k1": -1}, "_meta.engine.bm25.k1"), ({"k1": "x"}, "_meta.engine.bm25.k1"),
       ({"b": 1.5}, "_meta.engine.bm25.b"), ({"b": -0.1}, "_meta.engine.bm25.b"),
       ({"b": {"no_such_field": 0.5}}, "_meta.engine.bm25.b.no_such_field"),
       ({"b": {"product_title": 2}}, "_meta.engine.bm25.b.product_title"),
       ({"b": {"product_title": 0.5}, "b_default": 7}, "_meta.engine.bm25.b_default"),
       ({"b": 0.5, "b_default": 0.5}, "_meta.engine.bm25.b_default"),
       ({"b_default": 0.5}, "_meta.engine.bm25.b_default"),
       ("not an object", "_meta.engine.bm25")]


@pytest.mark.parametrize("bm25", GOOD)
def test_reference_accepts(bm25):
    reference.build_index(CAT, _m(bm25))


@pytest.mark.parametrize("bm25,path", BAD)
def test_reference_refuses(bm25, path):
    with pytest.raises(reference.InvalidMapping) as e:
        reference.build_index(CAT, _m(bm25))
    assert path in str(e.value)


@pytest.mark.parametrize("bm25", GOOD + [b for b, _ in BAD])
def test_engine_accepts_and_refuses_the_same(bm25):
    api = pytest.importorskip("engine.api")
    from engine.errors import InvalidMapping
    try:
        reference.build_index(CAT, _m(bm25))
        ref_ok = True
    except reference.InvalidMapping:
        ref_ok = False
    try:
        api.build_index(CAT, _m(bm25), device="cpu")
        eng_ok = True
    except InvalidMapping:
        eng_ok = False
    assert ref_ok == eng_ok, f"{bm25!r}: reference {'accepts' if ref_ok else 'refuses'}, engine does not"


@pytest.fixture
def quick(monkeypatch):
    """`referee fast`'s budget: the runtime probe and the 10,000-change freshness check are not what these tests are
    about (test_full.py and test_runtime.py cover them), so they are stubbed here."""
    monkeypatch.setattr(runtime, "environment", lambda: {"gpu": None, "driver": None, "cuda_driver_api": None,
                                                         "torch_cuda": None})
    monkeypatch.setattr(freshness, "check_freshness", lambda *a, **k: {"passed": True, "stubbed": True})


def _write(tmp_path, obj, name="p.bm25.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_full_bm25_is_recorded(tmp_path, quick):
    out = tmp_path / "tuned.json"
    p = _write(tmp_path, {"k1": 0.036, "b": {"product_title": 0.742, "product_brand": 0.675}, "b_default": 0.6})
    assert full.main(["--dataset=toy", "--engine=reference", "--device=cpu", "--k=100", f"--bm25={p}",
                      f"--out={out}"]) == 0
    r = json.loads(out.read_text())
    assert r["bm25"] == r["bm25_effective"] == json.loads(p.read_text())
    import hashlib
    assert r["bm25_file"] == {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    # what is scored does not move: the toy test split, every query judged
    toy = loader.load_dataset("toy")
    assert r["split"] == "test" and r["n_queries"] == len(toy.test_queries) and r["score_of_record"] is False


@pytest.mark.parametrize("obj", [{"k1": 0.5, "k_1": 0.3}, {"b": {"product_titel": 0.5}}, [1, 2]])   # values: the reference tests above
def test_full_bm25_refuses_bad_files(tmp_path, obj, capsys, quick):
    p = _write(tmp_path, obj)
    assert full.main(["--dataset=toy", "--engine=reference", "--device=cpu", "--skip-contract", f"--bm25={p}",
                      f"--out={tmp_path / 'r.json'}"]) == 2
    assert not (tmp_path / "r.json").exists()
    assert "--bm25" in capsys.readouterr().err


def test_full_bm25_refused_for_opensearch(tmp_path, capsys):
    p = _write(tmp_path, {"k1": 0.5})
    assert full.main(["--dataset=toy", "--engine=opensearch", f"--bm25={p}", f"--out={tmp_path / 'r.json'}"]) == 2
    assert "opensearch" in capsys.readouterr().err


def test_opensearch_refuses_a_per_field_b_map():
    m = copy.deepcopy(loader.ESCI_MAPPING)
    m["_meta"]["engine"]["bm25"] = {"k1": 0.5, "b": {"product_title": 0.5}}
    with pytest.raises(opensearch.UnsupportedQuery):
        opensearch.index_settings(m)


def test_skipped_contract_is_never_a_score_of_record(tmp_path, quick):
    pytest.importorskip("engine.api")
    out = tmp_path / "r.json"
    full.main(["--dataset=toy", "--engine=engine", "--device=cpu", "--k=50", "--skip-contract", f"--out={out}"])
    r = json.loads(out.read_text())
    assert r["score_of_record"] is False and r["contract_exit_code"] is None
    assert any("contract was skipped" in c for c in r["caveats"])
    assert r["label"] == "development only: the contract was skipped"
    assert "bm25" not in r and r["bm25_effective"] == {"k1": 1.2, "b": 0.75}   # toy's own, recorded without --bm25
