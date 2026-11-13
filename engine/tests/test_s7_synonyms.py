"""S7-6: query-time synonyms from the mapping, against brute force (Lucene's SynonymQuery scoring) on toy."""
import copy
import itertools
import math

import numpy as np
import pytest

from engine.analysis import analyze, analyze_with_positions
from engine.api import InvalidMapping, build_index
from engine.mapping import parse_synonyms

SYN = ["shoes, socks", "tee => shirt", "coat => jacket, jeans", "# a comment", ""]
FIELD = "product_title"


@pytest.fixture(scope="module")
def syn_index(catalogue, mapping):
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["synonyms"] = SYN
    return build_index(catalogue, m)


def test_parse():
    s = parse_synonyms(SYN)
    assert s["shoe"] == ("shoe", "sock") and s["sock"] == ("shoe", "sock")
    assert s["tee"] == ("shirt",) and s["coat"] == ("jacket", "jean") and "shirt" not in s
    with pytest.raises(InvalidMapping):
        parse_synonyms(["ice cream, gelato"])
    for bad in ["lamp =>", "=> light", " , ", "lamp, , light", "a => b => c"]:
        with pytest.raises(InvalidMapping) as e:
            parse_synonyms([bad])
        assert repr(bad.split("#")[0].strip()) in str(e.value) or "=>" in bad


def brute(index, catalogue, q, op="or"):
    """{doc: score} for match on FIELD: each query term is a clause of its synonyms; a synonym clause sums the
    alternatives' tf before saturation, with the lowest idf of its terms (Lucene's SynonymQuery)."""
    st = index._st
    f = st.field_index[FIELD]
    syn = st.mapping.synonyms
    terms = list(dict.fromkeys(analyze(q)))
    out = {}
    for d, text in enumerate(catalogue[FIELD]):
        toks = analyze(text)
        norm = 1 - 0.75 + 0.75 * float(st.host_rel_len[d, f])
        s, n = 0.0, 0
        for t in terms:
            alts = syn.get(t, (t,))
            tf = sum(toks.count(a) for a in alts)
            if not tf:
                continue
            n += 1
            idf = min(float(st.host_idf_field[st.vocab[a], f]) for a in alts if a in st.vocab)
            ptf = tf / norm
            s += idf * ptf * 2.2 / (1.2 + ptf)
        if n and (n == len(terms) if op == "and" else True):
            out[d] = s
    return out


@pytest.mark.parametrize("q", ["tee", "black shoes", "coat", "wool socks", "nike"])
@pytest.mark.parametrize("op", ["or", "and"])
def test_match_with_synonyms(syn_index, catalogue, q, op):
    want = brute(syn_index, catalogue, q, op)
    r = syn_index.search({"query": {"match": {FIELD: {"query": q, "operator": op}}}, "size": 500})
    got = {int(h["_id"][1:]): h["_score"] for h in r["hits"]["hits"]}
    assert set(got) == set(want)
    for d in got:
        assert math.isclose(got[d], want[d], rel_tol=1e-5, abs_tol=1e-6)


def test_replacement_drops_the_original(syn_index, catalogue):
    r = syn_index.search({"query": {"match": {FIELD: "tee"}}, "size": 500})
    want = {i for i, s in enumerate(catalogue[FIELD]) if "shirt" in analyze(s)}
    assert {int(h["_id"][1:]) for h in r["hits"]["hits"]} == want


def test_phrase_with_synonyms(syn_index, catalogue):
    q = "black shoes"
    alts = [("black",), ("shoe", "sock")]
    want = set()
    for d, text in enumerate(catalogue[FIELD]):
        toks, pos = analyze_with_positions(text)
        at = {(t, p) for t, p in zip(toks, pos)}
        if any(all((a, p0 + i) in at for i, a in enumerate(combo))
               for combo in itertools.product(*alts) for p0 in pos):
            want.add(d)
    r = syn_index.search({"query": {"match_phrase": {FIELD: q}}, "size": 500})
    assert {int(h["_id"][1:]) for h in r["hits"]["hits"]} == want


def test_highlight_marks_synonyms(syn_index):
    r = syn_index.search({"query": {"match": {FIELD: "coat"}}, "size": 20,
                          "highlight": {"fields": {FIELD: {"number_of_fragments": 0}}}})
    for h in r["hits"]["hits"]:
        frag = h["highlight"][FIELD][0]
        assert "<em>jacket</em>" in frag or "<em>jeans</em>" in frag


def test_no_synonyms_no_change(index):
    """Without synonyms in the mapping, match plans stay plain lexical clauses (the S7 'nothing changes' rule)."""
    from engine.plan.nodes import Lexical
    from engine.plan.plan import plan
    assert isinstance(plan({"query": {"match": {FIELD: "black shoes"}}}, index._st).root, Lexical)
