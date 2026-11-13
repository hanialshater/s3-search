"""Lexical scores are the same bits on every backend (the founder's Colab, torch 2.14 on CPU, differed from numpy at
the last bit on multi_match): the per-field and per-group sums are elementwise adds in a fixed order, not a library
reduction. An empty synonyms list plans exactly as no synonyms."""
import copy

import pytest

from engine.api import available_devices, build_index
from engine.plan.nodes import Lexical
from engine.plan.plan import plan
from referee import loader
from referee.checks import lexical_requests

torch_cpu = pytest.mark.skipif("torch-cpu" not in available_devices(), reason="needs torch")
SYN = ["lamp, light", "boots, shoe", "kettle => mug", "cup, beaker"]


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


def hits(ix, r):
    return [(h["_id"], h["_score"]) for h in ix.search(r)["hits"]["hits"]]


@torch_cpu
@pytest.mark.parametrize("synonyms", [None, [], SYN])
def test_lexical_bits_equal_across_backends(ds, synonyms):
    m = copy.deepcopy(ds.mapping)
    if synonyms is not None:
        m["_meta"]["engine"]["synonyms"] = synonyms
    a, b = build_index(ds.catalogue, m, "cpu"), build_index(ds.catalogue, m, "torch-cpu")
    reqs = lexical_requests(ds)
    reqs += [{"query": {"multi_match": {"query": t, "fields": ["product_title^3", "product_brand^2",
                                                               "product_description"]}}, "size": 50}
             for t in ("acme mugs", "zenith lamp", "kestrel backpack", "orange jacket", "nordic speaker",
                       "light shoe cup")]
    for r in reqs:
        assert hits(a, r) == hits(b, r), r


def test_empty_synonyms_plan_as_none(ds):
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["synonyms"] = []
    plain, empty = build_index(ds.catalogue, ds.mapping), build_index(ds.catalogue, m)
    assert empty._st.mapping.synonyms == {}
    for r in lexical_requests(ds)[:40]:
        assert hits(plain, r) == hits(empty, r)
    r = {"query": {"multi_match": {"query": "acme mugs", "fields": ["product_title^2", "product_brand"]}}}
    assert isinstance(plan(r, empty._st).root, Lexical)
