"""R1: the Rust host path (native/s3_hostpath) prepares exactly what the Python parser, planner and executor would.

Skipped where the extension is not built; built and run on a Colab CPU runtime (colab/hostpath_ci.py)."""
import json

import pytest

from engine import api, hostpath
from referee import loader

from engine.backend import available_devices

FIELDS = ["product_title^1.0", "product_brand^0.126", "product_color^0.14", "product_bullet_point^0.06",
          "product_description^0.036"]


def corpus(texts):
    out = []
    for i, t in enumerate(texts):
        out += [{"query": {"match": {"product_title": t}}},
                {"query": {"multi_match": {"query": t, "fields": FIELDS}}, "size": 2000},
                {"query": {"multi_match": {"query": t}}, "from": 3, "size": 7},
                {"query": {"multi_match": {"query": t, "fields": "product_title", "type": "most_fields"}}},
                {"query": {"match": {"product_title": {"query": t, "operator": "AND", "boost": 2}}}, "size": 50},
                {"query": {"match": {"product_title": {"query": t, "minimum_should_match": ["2", "-1", "50%", " 75% ",
                                                                                              1, -2][i % 6]}}}},
                {"query": {"multi_match": {"query": t, "fields": ["product_title^2", "product_brand",
                                                                  "product_title^0.5"], "tie_breaker": 0.3}},
                 "aggs": {"b": {"terms": {"field": "product_brand.keyword", "size": 5}},
                          "a": {"terms": {"field": "product_brand.keyword"}}}, "track_total_hits": True},
                {"query": {"match": {"product_brand": t}}, "aggregations": {"x": {"terms": {"field": "category"}}}}]
    out += [{"query": {"match": {"product_title": "lamp"}}, "size": 0},                 # from + size 0: Python
            {"query": {"match": {"product_title": "lamp"}}, "size": 9000, "from": 2000},  # over the window: an error
            {"query": {"match": {"product_title": {"query": "lamp", "fuzziness": 1}}}},
            {"query": {"match": {"product_title": "lämp"}}},                             # non-ASCII
            {"query": {"match": {"product_title": "the of"}}},                           # stopwords only: msm 0
            {"query": {"match": {"product_title": "qqqzzzunseenword"}}},                 # not in the table
            {"query": {"bool": {"must": [{"match": {"product_title": "lamp"}}]}}},
            {"query": {"match": {"product_title": 5}}},
            {"query": {"match": {"product_title": "lamp"}}, "size": 10.0},
            {"query": {"match": {"product_title": "lamp"}}, "sort": ["_score"]},
            {"query": {"match": {"category": "lamp"}}}]
    return [json.dumps(r).encode() for r in out]


@pytest.mark.parametrize("name", ["toy", "esci-sample"])
def test_rust_prepare_equals_the_python_planner(name):
    pytest.importorskip("s3_hostpath")
    try:
        ds = loader.load_dataset(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{name}: {e}")
    ix = api.build_index(ds.catalogue, ds.mapping)
    st = ix._st
    hi = hostpath.host_index(st)
    words = hostpath._words(st)
    texts = list(dict.fromkeys(list(ds.fit["query"]) + list(ds.test_queries["query"])))
    payloads = corpus([str(t) for t in texts])
    rust = hi.prepare(payloads)
    py = hostpath.prepare_reference(st, payloads, words)
    served = sorted(p for c in rust["chunks"] for p in c["positions"])
    eligible = sorted(p for c in py["chunks"] for p in c["positions"])
    assert set(served) <= set(eligible), sorted(set(served) - set(eligible))[:10]   # never serves what Python wouldn't
    assert len(served) >= 0.5 * len(eligible)                                       # and serves most of them
    sub = [payloads[p] for p in served]                                             # the same requests, both ways
    a, b = hi.prepare(sub), hostpath.prepare_reference(st, sub, words)
    assert a["fallback"] == b["fallback"] == []
    assert a["ints"] == b["ints"]
    assert a["floats"] == b["floats"]
    assert a["chunks"] == b["chunks"]


def _strip(r):
    return {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}


@pytest.mark.parametrize("device", available_devices())
def test_search_prepared_gives_search_batchs_responses(device):
    """The engine side: a prepared batch's responses equal search_batch's for the same requests, on every device
    (the fused kernel on cuda, the reference elsewhere). The batch is prepared by the oracle, and also by Rust when the
    extension is built."""
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    st = ix._st
    texts = list(dict.fromkeys(list(ds.fit["query"]) + list(ds.test_queries["query"])))
    payloads = corpus([str(t) for t in texts])[:120]
    preps = [hostpath.prepare_reference(st, payloads, hostpath._words(st))]
    try:
        preps.append(hostpath.host_index(st).prepare(payloads))
    except ImportError:
        pass
    for prep in preps:
        positions = prep["eligible"]
        got = ix.search_prepared(prep, hits="columns", ids="bytes")
        assert positions and len(got) == len(positions) and set(positions).isdisjoint(prep["fallback"])
        want = ix.search_batch([json.loads(payloads[p]) for p in positions], hits="columns", ids="bytes")
        assert [_strip(r) for r in got] == [_strip(r) for r in want]
        dicts = ix.search_prepared(prep, hits="dicts", ids="str")
        want_d = ix.search_batch([json.loads(payloads[p]) for p in positions])
        assert [_strip(r) for r in dicts] == [_strip(r) for r in want_d]
        assert sorted(positions + list(prep["fallback"])) == list(range(len(payloads)))
