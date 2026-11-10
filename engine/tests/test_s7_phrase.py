"""S7-1: positions and match_phrase with slop, against a brute-force phrase matcher over the analysed texts."""
import copy
import itertools
import math

import numpy as np
import pytest

from engine.analysis import analyze_with_positions
from engine.api import Index, UnsupportedQuery, build_index
from engine.segment import read_manifest, write_segment

FIELD = "product_description"
QUERIES = ["black running shoes", "wool socks", "men shirt", "leather jacket blue", "shoes black", "kids the wool",
           "nike", "socks wool kids", "jeans levi's"]


def brute(catalogue, field, query, slop):
    """{doc: freq} by Lucene's rule: some choice of one position per slot with max(v) - min(v) <= slop, where
    v = position - query offset. freq: occurrences for slop 0, else the number of matching start values."""
    qt, qp = analyze_with_positions(query)
    out = {}
    if not qt:
        return out
    offs = [p - qp[0] for p in qp]
    for d, text in enumerate(catalogue[field].tolist()):
        toks, pos = analyze_with_positions(text)
        vals = [[p - o for t, p in zip(toks, pos) if t == q] for q, o in zip(qt, offs)]
        if any(not v for v in vals):
            continue
        starts = set()
        for combo in itertools.product(*vals):
            if max(combo) - min(combo) <= slop:
                starts.add(min(combo))
        if starts:
            out[d] = len(starts)
    return out


@pytest.mark.parametrize("slop", [0, 1, 2])
@pytest.mark.parametrize("q", QUERIES)
def test_phrase_hits_against_brute_force(index, catalogue, q, slop):
    want = brute(catalogue, FIELD, q, slop)
    r = index.search({"query": {"match_phrase": {FIELD: {"query": q, "slop": slop}}}, "size": 200, "sort": ["_doc"]})
    got = [int(h["_id"][1:]) for h in r["hits"]["hits"]]
    assert got == sorted(want) and r["hits"]["total"] == len(want)


@pytest.mark.parametrize("q", QUERIES)
def test_exact_phrase_score_is_bm25_of_the_phrase_frequency(index, catalogue, q):
    st = index._st
    f = st.field_index[FIELD]
    want = brute(catalogue, FIELD, q, 0)
    qt, _ = analyze_with_positions(q)
    idf = sum(float(st.host_idf_field[st.vocab[t], f]) for t in qt) if all(t in st.vocab for t in qt) else 0
    r = index.search({"query": {"match_phrase": {FIELD: q}}, "size": 200})
    for h in r["hits"]["hits"]:
        d = int(h["_id"][1:])
        norm = 1 - 0.75 + 0.75 * float(st.host_rel_len[d, f])
        ptf = want[d] / norm
        assert math.isclose(h["_score"], idf * ptf * 2.2 / (1.2 + ptf), rel_tol=1e-5)


def test_stopwords_leave_gaps():
    assert analyze_with_positions("red the lamp") == (["red", "lamp"], [0, 2])


def test_old_segments_without_positions_still_load(catalogue, mapping, tmp_path):
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["positions"] = False
    ix = build_index(catalogue, m)
    p = ix.save(tmp_path / "nopos")
    assert "text/positions.npy" not in read_manifest(p)["files"]
    back = Index.from_segment(p)
    assert back.search({"query": {"match": {FIELD: "wool"}}})["hits"]["total"] > 0
    with pytest.raises(UnsupportedQuery):
        back.search({"query": {"match_phrase": {FIELD: "wool socks"}}})


def test_positions_round_trip(segment, tmp_path):
    p = write_segment(copy.deepcopy(segment), tmp_path / "seg")
    a, b = Index(segment), Index.from_segment(p)
    for q in QUERIES:
        r = {"query": {"match_phrase": {FIELD: {"query": q, "slop": 1}}}, "size": 50}
        assert a.search(r)["hits"] == b.search(r)["hits"]


def test_phrase_in_bool_and_on_keyword(index, catalogue):
    r = index.search({"query": {"bool": {"must": [{"match_phrase": {FIELD: "wool socks"}}],
                                         "filter": [{"term": {"category": "socks"}}]}}, "size": 200})
    want = {d for d in brute(catalogue, FIELD, "wool socks", 0) if catalogue["category"].iloc[d] == "socks"}
    assert {int(h["_id"][1:]) for h in r["hits"]["hits"]} == want
    r = index.search({"query": {"match_phrase": {"category": "socks"}}, "size": 0})   # a term on a keyword field
    assert r["hits"]["total"] == int((catalogue["category"] == "socks").sum())
