"""A per-field BM25F ``b`` reaches every scoring path, not only the single-segment lexical one.

`test_bm25_params.py` checks the lexical scores against the referee's reference on one segment. Here, on every
device: the phrase and term-group paths (match_phrase, fuzziness, prefix), base plus delta segments and a merged
segment (the multi-part scorer), the batched executor against one request at a time, and explain against the score.
A segment built with other BM25F parameters is refused rather than scored with the index's.
"""
import copy

import numpy as np
import pytest

api = pytest.importorskip("engine.api")
from engine.build import build_segment, merge_segments  # noqa: E402
from referee import checks, loader  # noqa: E402

PER_FIELD = {"k1": 0.9, "b": {"product_title": 0.3}, "b_default": 0.95}   # title's b differs from the rest
TITLE_ONLY = {"k1": 0.9, "b": 0.3}                                        # the same b for title, as one number


def _with(mapping, bm25):
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["bm25"] = bm25
    return m


def _scores(resp):
    return {h["_id"]: h["_score"] for h in resp["hits"]["hits"]}


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


def _title_requests(ds):
    """Requests that score on product_title only, through each lexical operator."""
    out = []
    for t in list(dict.fromkeys(ds.fit["query"]))[:15]:
        out.append({"query": {"match": {"product_title": t}}})                                   # lexical
        out.append({"query": {"match_phrase": {"product_title": {"query": t, "slop": 2}}}})     # phrase
        out.append({"query": {"match": {"product_title": {"query": t, "fuzziness": 1}}}})      # term groups
        out.append({"query": {"match_bool_prefix": {"product_title": t}}})                     # term groups, prefix
    return [{**r, "size": checks.ALL} for r in out]


@pytest.mark.parametrize("device", api.available_devices())
def test_each_operator_uses_the_fields_own_b(device, ds):
    """On title alone, a map giving title b = 0.3 (others 0.95) scores exactly as one b of 0.3 everywhere."""
    per_field = api.build_index(ds.catalogue, _with(ds.mapping, PER_FIELD), device=device)
    one_b = api.build_index(ds.catalogue, _with(ds.mapping, TITLE_ONLY), device=device)
    other_b = api.build_index(ds.catalogue, _with(ds.mapping, {"k1": 0.9, "b": 0.95}), device=device)
    differs = 0
    for r in _title_requests(ds):
        got = per_field.search(r)
        checks.assert_same_ranking(got, one_b.search(r), f"{device}: {r['query']}", tol=checks.DEVICE_RTOL)
        a, c = _scores(got), _scores(other_b.search(r))
        differs += any(abs(a[d] - c[d]) > 1e-6 for d in a)
    assert differs, "the default b was never the one that mattered: this test would not see a missing b_of"


@pytest.mark.parametrize("device", api.available_devices())
def test_segments_and_merges_score_as_a_rebuild(device, ds):
    """Base plus a delta, and their merge, score multi_match exactly as one segment of the same products."""
    m = _with(ds.mapping, PER_FIELD)
    cat = ds.catalogue
    base, delta = cat.iloc[:140], cat.iloc[140:]
    one = api.build_index(cat, m, device=device)
    two = api.Index(build_segment(base, m, version=1), device)
    two.add_segment(build_segment(delta, m, version=2), version=2)
    merged = api.Index(merge_segments(two.snapshot_segments()[0], m, version=3), device)
    reqs = [r for r in checks.lexical_requests(ds) if "multi_match" in str(r)][:30]
    assert reqs
    for r in reqs:
        r = {**r, "size": checks.ALL}
        exp = one.search(r)
        checks.assert_same_ranking(two.search(r), exp, f"base + delta on {device}: {r['query']}",
                                   tol=checks.DEVICE_RTOL)
        checks.assert_same_ranking(merged.search(r), exp, f"merged on {device}: {r['query']}", tol=checks.DEVICE_RTOL)


@pytest.mark.parametrize("device", api.available_devices())
def test_batch_and_explain_agree(device, ds):
    ix = api.build_index(ds.catalogue, _with(ds.mapping, PER_FIELD), device=device)
    reqs = checks.lexical_requests(ds)[:24] + _title_requests(ds)[:8]
    for r, b in zip(reqs, ix.search_batch(reqs)):
        checks.assert_same_ranking(b, ix.search(r), f"batch vs one on {device}: {r['query']}", tol=0.0)
    for r in checks.lexical_requests(ds)[:24]:
        resp = ix.search({**r, "size": 10, "explain": True})
        for h, e in zip(resp["hits"]["hits"], resp["explain"]):
            assert e["_id"] == h["_id"]
            assert np.isclose(e["value"], h["_score"], rtol=1e-4, atol=1e-5), (r, e["value"], h["_score"])


def test_a_segment_with_other_bm25_parameters_is_refused(ds):
    ix = api.Index(build_segment(ds.catalogue.iloc[:140], _with(ds.mapping, PER_FIELD), version=1))
    other = build_segment(ds.catalogue.iloc[140:], _with(ds.mapping, TITLE_ONLY), version=2)
    with pytest.raises(ValueError, match="BM25F parameters"):
        ix.add_segment(other, version=2)
