"""The engine's own tests: a synthetic catalogue (the prototype's toy check, extended with filter, facet and sort
columns), built in well under a second. The referee's contract tests use the steward's datasets instead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.backend import available_devices

WORDS = "black running shoes nike leather jacket blue cotton shirt women men kids socks wool levi's jeans".split()
BRANDS = ["Nike", "Levi's", "Adidas", "Acme", None]
COLORS = ["Black", "Blue", "Red", None]
CATS = ["shoes", "apparel", "socks"]


def toy_catalogue(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "product_id": f"P{i:04d}",
            "product_title": " ".join(rng.choice(WORDS, size=rng.integers(1, 7))),
            "product_brand": BRANDS[rng.integers(len(BRANDS))],
            "product_color": COLORS[rng.integers(len(COLORS))],
            "product_bullet_point": " ".join(rng.choice(WORDS, size=rng.integers(0, 6))),
            "product_description": " ".join(rng.choice(WORDS, size=rng.integers(0, 12))),
            "category": CATS[rng.integers(len(CATS))],
            "price": float(np.round(rng.uniform(1, 100), 2)) if rng.random() > 0.1 else None,
            "stock": int(rng.integers(0, 5)),
        })
    return pd.DataFrame(rows)


def toy_mapping(dim: int = 16) -> dict:
    kw = {"keyword": {"type": "keyword", "ignore_above": 256}}
    return {
        "_meta": {"engine": {"id_field": "product_id", "hot": ["price", "stock"],
                             "embeddings": {"embedding": {"model": f"tfidf-svd-{dim}",
                                                          "inputs": ["product_title", "product_brand",
                                                                     "product_color"]}}}},
        "properties": {
            "product_title": {"type": "text"},
            "product_brand": {"type": "text", "fields": kw},
            "product_color": {"type": "text", "fields": kw},
            "product_bullet_point": {"type": "text"},
            "product_description": {"type": "text"},
            "category": {"type": "keyword"},
            "price": {"type": "float"},
            "stock": {"type": "integer"},
            "embedding": {"type": "knn_vector", "dimension": dim,
                          "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
        },
    }


@pytest.fixture(scope="session")
def catalogue():
    return toy_catalogue()


@pytest.fixture(scope="session")
def mapping():
    return toy_mapping()


@pytest.fixture(scope="session")
def segment(catalogue, mapping):
    from engine.build import build_segment
    return build_segment(catalogue, mapping)


@pytest.fixture(scope="session", params=available_devices())
def device(request):
    return request.param


@pytest.fixture(scope="session")
def index(segment, device):
    from engine.api import Index
    return Index(segment, device)
