"""S7-2: fuzzy, match with fuzziness, prefix and match_bool_prefix, against brute force over the terms dictionary."""
import numpy as np
import pytest

from engine.analysis import analyze
from engine.dictionary import Dictionary, auto_edits
from engine.errors import InvalidRequest, UnsupportedQuery

TEXT = "product_title"


def osa(a, b, transpositions=True):
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if transpositions and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def field_terms(catalogue, field):
    return {t for s in catalogue[field] for t in analyze(s)}


def docs_with(catalogue, field, terms):
    return {i for i, s in enumerate(catalogue[field]) if set(analyze(s)) & set(terms)}


def hit_set(index, req):
    r = index.search({**req, "size": 500})
    return {int(h["_id"][1:]) for h in r["hits"]["hits"]}, r


def test_auto_edits():
    assert [auto_edits("AUTO", "x" * n) for n in (1, 2, 3, 5, 6, 9)] == [0, 0, 1, 1, 2, 2]
    assert auto_edits("AUTO:2,4", "abc") == 1 and auto_edits(1, "ab") == 1 and auto_edits("2", "ab") == 2
    with pytest.raises(UnsupportedQuery):
        auto_edits(3, "abcdef")
    with pytest.raises(InvalidRequest):
        auto_edits("lots", "abc")


@pytest.mark.parametrize("tr", [True, False])
def test_vectorized_distances_equal_brute_force(tr):
    rng = np.random.default_rng(0)
    words = sorted({"".join(rng.choice(list("abcde"), rng.integers(1, 7))) for _ in range(400)})
    d = Dictionary(words)
    for q in ["abc", "bca", "ab", "edcba", "aabb", "c"]:
        for e in (1, 2):
            want = {w for w in words if osa(q, w, tr) <= e}
            got = {words[i] for i in d.fuzzy(q, e, max_expansions=10_000, transpositions=tr)}
            assert got == want, (q, e)
            pl = {w for w in want if w[:1] == q[:1]}
            assert {words[i] for i in d.fuzzy(q, e, prefix_length=1, max_expansions=10_000,
                                              transpositions=tr)} == pl


def test_max_expansions_keeps_the_most_similar():
    d = Dictionary(sorted(["shoe", "shoes", "show", "shot", "sho", "shoxx", "sxoe"]))
    got = [d.terms[i] for i in d.fuzzy("shoe", 2, max_expansions=3)]
    assert got[0] == "shoe" and set(got[1:]) <= {"sho", "shoes", "shot", "show", "sxoe"}
    assert all(osa("shoe", t) == 1 for t in got[1:])


@pytest.mark.parametrize("value,fz", [("shoe", "AUTO"), ("jaket", "AUTO"), ("wol", 1), ("socsk", 2),
                                      ("lether", "AUTO"), ("nikee", 1)])
def test_fuzzy_query_hits(index, catalogue, value, fz):
    terms = [t for t in field_terms(catalogue, TEXT) if osa(value, t) <= auto_edits(fz, value)]
    want = docs_with(catalogue, TEXT, terms)
    got, r = hit_set(index, {"query": {"fuzzy": {TEXT: {"value": value, "fuzziness": fz}}}})
    assert got == want and r["hits"]["total"] == len(want)


@pytest.mark.parametrize("q", ["blak runing shoes", "wool sokcs", "leathr jackt"])
@pytest.mark.parametrize("op", ["or", "and"])
def test_match_with_fuzziness(index, catalogue, q, op):
    vocab = field_terms(catalogue, TEXT)
    toks = list(dict.fromkeys(analyze(q)))
    groups = [{t for t in vocab if osa(tok, t) <= auto_edits("AUTO", tok)} for tok in toks]
    want = set()
    for i, s in enumerate(catalogue[TEXT]):
        have = set(analyze(s))
        n = sum(bool(have & g) for g in groups)
        if (n == len(groups)) if op == "and" else n >= 1:
            want.add(i)
    got, _ = hit_set(index, {"query": {"match": {TEXT: {"query": q, "fuzziness": "AUTO", "operator": op}}}})
    assert got == want


def test_fuzziness_zero_is_a_plain_match(index):
    a = index.search({"query": {"match": {TEXT: {"query": "black shoes", "fuzziness": 0}}}, "size": 50})
    b = index.search({"query": {"match": {TEXT: "black shoes"}}, "size": 50})
    assert a["hits"] == b["hits"]


@pytest.mark.parametrize("value", ["sh", "ja", "wo", "b", "zz"])
def test_prefix_on_text(index, catalogue, value):
    terms = [t for t in field_terms(catalogue, TEXT) if t.startswith(value)]
    want = docs_with(catalogue, TEXT, terms)
    got, r = hit_set(index, {"query": {"prefix": {TEXT: value}}})
    assert got == want and all(h["_score"] == 1.0 for h in r["hits"]["hits"])


@pytest.mark.parametrize("value", ["N", "Ac", "Levi", "x"])
def test_prefix_on_keyword(index, catalogue, value):
    want = {i for i, v in enumerate(catalogue["product_brand"]) if isinstance(v, str) and v.startswith(value)}
    got, _ = hit_set(index, {"query": {"prefix": {"product_brand.keyword": {"value": value, "boost": 2}}}})
    assert got == want


def test_fuzzy_on_keyword(index, catalogue):
    want = {i for i, v in enumerate(catalogue["product_brand"]) if isinstance(v, str) and osa("Nkie", v) <= 1}
    got, _ = hit_set(index, {"query": {"fuzzy": {"product_brand.keyword": {"value": "Nkie", "fuzziness": 1}}}})
    assert got == want and want


@pytest.mark.parametrize("q", ["black running sh", "wool so", "leather j", "kids"])
@pytest.mark.parametrize("op", ["or", "and"])
def test_match_bool_prefix(index, catalogue, q, op):
    toks = analyze(q)
    vocab = sorted(field_terms(catalogue, TEXT))
    last = {t for t in vocab if t.startswith(toks[-1])}
    want = set()
    for i, s in enumerate(catalogue[TEXT]):
        have = set(analyze(s))
        hits = [t in have for t in toks[:-1]] + [bool(have & last)]
        if all(hits) if op == "and" else any(hits):
            want.add(i)
    got, _ = hit_set(index, {"query": {"match_bool_prefix": {TEXT: {"query": q, "operator": op}}}})
    assert got == want
