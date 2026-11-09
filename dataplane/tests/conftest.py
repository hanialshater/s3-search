from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from dataplane import _engine
from dataplane.kv import Store
from dataplane.tests import fake_engine

MAPPING = {
    "_meta": {"engine": {
        "id_field": "product_id",
        "hot": ["price", "stock"],
        "bm25": {"k1": 1.2, "b": 0.75},
        "embeddings": {"embedding": {"model": "tfidf-svd-128", "inputs": ["product_title", "product_brand",
                                                                           "product_color"]}},
    }},
    "properties": {
        "product_title": {"type": "text", "analyzer": "s3_english"},
        "product_brand": {"type": "text", "analyzer": "s3_english",
                          "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "product_color": {"type": "text", "analyzer": "s3_english",
                          "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
        "product_bullet_point": {"type": "text", "analyzer": "s3_english"},
        "product_description": {"type": "text", "analyzer": "s3_english"},
        "category": {"type": "keyword"},
        "price": {"type": "float"},
        "stock": {"type": "integer"},
        "embedding": {"type": "knn_vector", "dimension": 16,
                      "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
    },
}

BRANDS = ["Acme", "Levi's", "Zeta", "Nordic", "Kumo"]
COLORS = ["red", "blue", "black", "green"]
NOUNS = ["shoe", "jacket", "lamp", "kettle", "mug", "backpack", "cable", "charger"]


def make_catalogue(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        b, c, noun = BRANDS[i % 5], COLORS[i % 4], NOUNS[rng.integers(0, len(NOUNS))]
        rows.append({"product_id": f"P{i:05d}", "product_title": f"{b} {c} {noun} model {i}", "product_brand": b,
                     "product_color": c, "product_bullet_point": f"a {noun} that works",
                     "product_description": f"The {c} {noun} by {b}.", "category": noun,
                     "price": round(float(rng.uniform(1, 400)), 2), "stock": int(rng.integers(0, 50))})
    return pd.DataFrame(rows)


@pytest.fixture
def mapping() -> dict:
    return copy.deepcopy(MAPPING)


@pytest.fixture
def catalogue() -> pd.DataFrame:
    return make_catalogue()


@pytest.fixture
def store(tmp_path, mapping):
    s = Store(tmp_path / "store.sqlite", mapping)
    yield s
    s.close()


@pytest.fixture
def fake():
    with _engine.use(fake_engine.hooks()) as h:
        yield h


def load(store: Store, catalogue: pd.DataFrame) -> tuple[int, int]:
    from dataplane.intake import enrich, validate
    docs = [validate(r, store.mapping) for r in catalogue.to_dict("records")]
    enrich(docs, store.mapping)
    return store.apply({"op": "upsert", "_id": d["product_id"], "doc": d} for d in docs)
