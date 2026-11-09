"""The named datasets: toy (always), esci-sample (skipped where the ESCI files are not downloaded)."""
import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from referee import loader, reference


def test_toy_is_fast_deterministic_and_typed():
    t = time.perf_counter()
    a = loader.load_dataset("toy")
    assert time.perf_counter() - t < 1.0
    b = loader.load_dataset("toy")
    cols = ["product_id", "product_title", "product_brand", "product_color", "product_bullet_point",
            "product_description", "category", "price", "stock"]
    pd.testing.assert_frame_equal(a.catalogue[cols], b.catalogue[cols])
    assert np.array_equal(np.stack(a.catalogue["embedding"]), np.stack(b.catalogue["embedding"]))
    c = a.catalogue
    assert len(c) == 200 and c["product_id"].is_unique
    assert c["price"].dtype == np.float64 and c["stock"].dtype.kind == "i"
    assert pd.api.types.is_string_dtype(c["category"]) or c["category"].dtype == object
    assert (c["stock"] == 0).sum() > 10, "stock 0 must be common, for filters"
    assert c["product_color"].isna().any() and c["product_brand"].isna().any(), "missing values are part of the test"
    assert set(a.mapping["properties"]) >= set(cols[1:] + ["embedding"]) - {"product_id"}
    assert a.mapping["_meta"]["engine"]["id_field"] == "product_id"


def test_toy_judgments_split_and_boundary():
    ds = loader.load_dataset("toy")
    fit_q, test_q = set(ds.fit["query_id"]), set(ds.test_queries["query_id"])
    assert len(fit_q) == 20 and len(test_q) == 10 and not fit_q & test_q
    assert list(ds.test_queries.columns) == ["query_id", "query"]
    assert set(ds.fit["esci_label"]) == {"E", "S", "C", "I"}
    hidden = loader._load_dataset_test_judgments("toy")
    assert set(hidden) == test_q and all(len(j) > 5 for j in hidden.values())
    assert set(ds.catalogue["product_id"]) >= {p for j in hidden.values() for p in j}


def test_reference_bm25f_by_hand():
    """Two fields, weights 2 and 1, global df: worked out in this test from the interfaces' formula."""
    cat = pd.DataFrame({"product_id": ["a", "b", "c"], "t": ["red lamp", "lamp", "blue mug"],
                        "d": ["red red", None, "lamp shade red"]})
    mapping = {"_meta": {"engine": {"id_field": "product_id"}},
               "properties": {"t": {"type": "text"}, "d": {"type": "text"}}}
    idx = reference.build_index(cat, mapping)
    k1, b, n = 1.2, 0.75, 3
    lt, ld = np.array([2, 1, 2.0]), np.array([2, 0, 3.0])
    at, ad = lt.mean(), ld.mean()
    norm_t, norm_d = 1 - b + b * lt / at, 1 - b + b * ld / ad
    idf = lambda df: math.log1p((n - df + 0.5) / (df + 0.5))
    # "red": in a (t1, d2) and c (d1): global df 2. "lamp": a (t1), b (t1), c (d1): df 3.
    exp = np.zeros(3)
    for tf_t, tf_d, df in (([1, 0, 0], [2, 0, 1], 2), ([1, 1, 0], [0, 0, 1], 3)):
        p = 2 * np.array(tf_t) / norm_t + 1 * np.array(tf_d) / norm_d
        exp += idf(df) * p * (k1 + 1) / (k1 + p)
    r = idx.search({"query": {"multi_match": {"query": "red lamps", "fields": ["t^2", "d"]}}})
    got = {h["_id"]: h["_score"] for h in r["hits"]["hits"]}
    assert [h["_id"] for h in r["hits"]["hits"]] == list(np.array(["a", "b", "c"])[np.argsort(-exp, kind="stable")])
    for pid, e in zip("abc", exp):
        assert got[pid] == pytest.approx(e, rel=1e-12)


needs_esci = pytest.mark.skipif(not loader.available("esci-sample"),
                                reason="ESCI not downloaded here (referee.loader.fetch())")


@needs_esci
def test_esci_sample_matches_its_manifest():
    ds = loader.load_dataset("esci-sample")
    m = json.loads(loader.SAMPLE_MANIFEST.read_text())
    assert m["queries"] == {"test": 500, "train": 2000}
    assert ds.fit["query_id"].nunique() == 2000 and len(ds.test_queries) == 500
    assert len(ds.catalogue) == m["products"] and ds.catalogue["product_id"].is_unique
    built = json.loads((loader.SAMPLE_DIR / "manifest.json").read_text())
    assert built["sha256"] == m["sha256"], "esci-sample differs from the committed manifest: not the same draw"
    assert list(ds.test_queries.columns) == ["query_id", "query"]
