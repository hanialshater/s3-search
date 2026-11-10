"""S7-5: highlight (plain) and function_score, against brute force on toy."""
import math
import re

import numpy as np
import pytest

from engine.analysis import _WORD, _token, analyze
from engine.errors import InvalidRequest, UnsupportedQuery

EM = re.compile(r"<em>(.*?)</em>")


def brute_mark(text, terms, pre="<em>", post="</em>"):
    out, cur = [], 0
    for m in _WORD.finditer(text.lower()):
        if _token(m.group(0)) in terms:
            out.append(text[cur:m.start()] + pre + text[m.start():m.end()] + post)
            cur = m.end()
    out.append(text[cur:])
    return "".join(out).strip()


@pytest.mark.parametrize("q,field", [("black running shoes", "product_title"), ("wool socks kids", "product_description"),
                                     ("leather jacket", "product_description")])
def test_whole_field_equals_brute_force(index, catalogue, q, field):
    terms = set(analyze(q))
    r = index.search({"query": {"match": {field: q}}, "size": 50,
                      "highlight": {"fields": {field: {"number_of_fragments": 0}}}})
    text_of = dict(zip(catalogue["product_id"], catalogue[field]))
    assert r["hits"]["hits"]
    for h in r["hits"]["hits"]:
        assert h["highlight"][field] == [brute_mark(text_of[h["_id"]], terms)]


@pytest.mark.parametrize("size,n", [(20, 2), (40, 1), (100, 5), (10, 3)])
def test_fragments_contain_a_query_term(index, catalogue, size, n):
    field, q = "product_description", "black wool shoes kids"
    terms = set(analyze(q))
    r = index.search({"query": {"match": {field: q}}, "size": 50, "highlight": {
        "fields": {field: {"fragment_size": size, "number_of_fragments": n}}, "pre_tags": ["<b>"],
        "post_tags": ["</b>"]}})
    text_of = dict(zip(catalogue["product_id"], catalogue[field]))
    for h in r["hits"]["hits"]:
        frags = h["highlight"][field]
        assert 1 <= len(frags) <= n
        for fr in frags:
            words = re.findall(r"<b>(.*?)</b>", fr)
            assert words and all(_token(w.lower()) in terms for w in words)
            assert fr.replace("<b>", "").replace("</b>", "") in text_of[h["_id"]]


def test_require_field_match_and_must_not(index):
    q = {"bool": {"must": [{"match": {"product_title": "shoes"}}],
                  "should": [{"match": {"product_description": "wool"}}],
                  "must_not": [{"match": {"product_description": "leather"}}]}}
    r = index.search({"query": q, "size": 30, "highlight": {"fields": {"product_title": {"number_of_fragments": 0},
                                                                         "product_description": {"number_of_fragments": 0}}}})
    for h in r["hits"]["hits"]:
        for w in EM.findall(" ".join(h["highlight"].get("product_title", []))):
            assert _token(w.lower()) == "shoe"
        for w in EM.findall(" ".join(h["highlight"].get("product_description", []))):
            assert _token(w.lower()) == "wool"                    # never the must_not term
    r = index.search({"query": {"match": {"product_title": "wool"}}, "size": 30, "highlight": {
        "require_field_match": False, "fields": {"product_description": {"number_of_fragments": 0}}}})
    assert any("product_description" in h.get("highlight", {}) for h in r["hits"]["hits"])


def test_highlight_fuzzy_and_phrase_terms(index):
    r = index.search({"query": {"fuzzy": {"product_title": {"value": "shoe", "fuzziness": 1}}}, "size": 20,
                      "highlight": {"fields": {"product_title": {"number_of_fragments": 0}}}})
    assert all(h.get("highlight") for h in r["hits"]["hits"])


# ---- function_score

def fvf(v, factor, mod):
    x = factor * v
    f = {"none": lambda: x, "log": lambda: math.log10(x), "log1p": lambda: math.log10(1 + x),
         "log2p": lambda: math.log10(2 + x), "ln": lambda: math.log(x), "ln1p": lambda: math.log1p(x),
         "ln2p": lambda: math.log(2 + x), "square": lambda: x * x, "sqrt": lambda: math.sqrt(x),
         "reciprocal": lambda: 1 / x}[mod]
    return f()


def scores(index, req):
    r = index.search({**req, "size": 500})
    return {h["_id"]: h["_score"] for h in r["hits"]["hits"]}, r


@pytest.mark.parametrize("mod", ["none", "log1p", "log2p", "ln1p", "ln2p", "square", "sqrt"])
@pytest.mark.parametrize("bm", ["multiply", "replace", "sum", "avg", "max", "min"])
def test_field_value_factor(index, catalogue, mod, bm):
    q = {"match": {"product_title": "black shoes"}}
    base, _ = scores(index, {"query": q})
    req = {"query": {"function_score": {"query": q, "boost_mode": bm, "boost": 2.0,
                                        "field_value_factor": {"field": "stock", "factor": 1.5, "modifier": mod}}}}
    got, r = scores(index, req)
    stock = dict(zip(catalogue["product_id"], catalogue["stock"]))
    assert set(got) == set(base)
    for pid, qs in base.items():
        f = fvf(float(stock[pid]), 1.5, mod)
        s = {"multiply": qs * f, "replace": f, "sum": qs + f, "avg": (qs + f) / 2, "max": max(qs, f),
             "min": min(qs, f)}[bm] * 2.0
        assert math.isclose(got[pid], s, rel_tol=1e-5, abs_tol=1e-6), (pid, got[pid], s)


@pytest.mark.parametrize("sm", ["multiply", "sum", "avg", "first", "max", "min"])
def test_functions_with_filters(index, catalogue, sm):
    q = {"match_all": {}}
    fns = [{"filter": {"term": {"category": "shoes"}}, "weight": 3.0},
           {"filter": {"range": {"stock": {"gte": 2}}}, "field_value_factor": {"field": "stock"}, "weight": 0.5},
           {"filter": {"term": {"category": "socks"}}, "weight": 2.0}]
    got, _ = scores(index, {"query": {"function_score": {"query": q, "functions": fns, "score_mode": sm,
                                                         "max_boost": 4.0}}})
    for _, row in catalogue.iterrows():
        vals, ws = [], []
        if row["category"] == "shoes":
            vals.append(3.0), ws.append(3.0)
        if row["stock"] >= 2:
            vals.append(row["stock"] * 0.5), ws.append(0.5)
        if row["category"] == "socks":
            vals.append(2.0), ws.append(2.0)
        if not vals:
            f = 1.0
        else:
            f = {"multiply": float(np.prod(vals)), "sum": sum(vals), "avg": sum(vals) / sum(ws), "first": vals[0],
                 "max": max(vals), "min": min(vals)}[sm]
        f = min(f, 4.0)
        assert math.isclose(got[row["product_id"]], 1.0 * f, rel_tol=1e-6), (row["product_id"], sm)


def test_missing_and_min_score_and_errors(index, catalogue):
    q = {"function_score": {"field_value_factor": {"field": "price", "missing": 10.0}, "boost_mode": "replace",
                            "min_score": 50.0}}
    got, r = scores(index, {"query": q})
    price = catalogue.set_index("product_id")["price"].fillna(10.0)
    want = {p for p, v in price.items() if np.float32(v) >= 50.0}
    assert set(got) == want
    with pytest.raises(InvalidRequest):                          # a doc without price and no missing
        index.search({"query": {"function_score": {"field_value_factor": {"field": "price"}}}})
    with pytest.raises(InvalidRequest):                          # log of 0 stock: -inf
        index.search({"query": {"function_score": {"field_value_factor": {"field": "stock", "modifier": "log"}}}})
