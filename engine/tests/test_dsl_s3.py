"""S3: the parser (syntax, no index) and the planner (against an index); S3's second prediction, that a request
outside the subset is refused with a named error and never silently answered."""
import pytest

from engine.dsl import ast as A
from engine.dsl.parse import parse
from engine.errors import InvalidRequest, UnsupportedQuery
from engine.plan.nodes import Lexical, Range, Term
from engine.plan.plan import plan

M = {"match": {"product_title": "shoes"}}

UNSUPPORTED = [
    # query types outside the subset
    ({"query": {"fuzzy": {"product_title": {"value": "shoe", "rewrite": "constant_score"}}}},
     "query.fuzzy.product_title.rewrite"),
    ({"query": {"match_phrase_prefix": {"product_title": "re"}}}, "query.match_phrase_prefix"),
    ({"query": {"match_phrase": {"product_title": {"query": "x", "analyzer": "standard"}}}},
     "query.match_phrase.product_title.analyzer"),
    ({"query": {"match_bool_prefix": {"product_title": {"query": "re", "fuzziness": 1}}}},
     "query.match_bool_prefix.product_title.fuzziness"),
    ({"query": {"query_string": {"query": "shoe"}}}, "query.query_string"),
    ({"query": {"simple_query_string": {"query": "shoe"}}}, "query.simple_query_string"),
    ({"query": {"wildcard": {"product_title": "sh*"}}}, "query.wildcard"),
    ({"query": {"prefix": {"product_title": {"value": "sh", "case_insensitive": True}}}},
     "query.prefix.product_title.case_insensitive"),
    ({"query": {"regexp": {"product_title": "sh.*"}}}, "query.regexp"),
    ({"query": {"exists": {"field": "price", "null_value": 0}}}, "query.exists.null_value"),
    ({"query": {"ids": {"values": ["P0001"], "type": "_doc"}}}, "query.ids.type"),
    ({"query": {"constant_score": {"filter": M}}}, "query.constant_score"),
    ({"query": {"dis_max": {"queries": [M]}}}, "query.dis_max"),
    ({"query": {"function_score": {"query": M}}}, "query.function_score"),
    ({"query": {"script_score": {"query": M, "script": {"source": "1"}}}}, "query.script_score"),
    ({"query": {"nested": {"path": "x", "query": M}}}, "query.nested"),
    ({"query": {"more_like_this": {"like": "x"}}}, "query.more_like_this"),
    ({"query": {"bool": {"must": [{"wildcard": {"product_title": "sh*"}}]}}}, "query.bool.must[0].wildcard"),
    ({"query": {"bool": {"filter": {"geo_distance": {}}}}}, "query.bool.filter.geo_distance"),
    # options outside the subset
    ({"query": {"match": {"product_title": {"query": "x", "fuzzy_rewrite": "x"}}}},
     "query.match.product_title.fuzzy_rewrite"),
    ({"query": {"match": {"product_title": {"query": "shoes", "fuzziness": 3}}}}, "query.match.product_title.fuzziness"),
    ({"query": {"fuzzy": {"price": "3"}}}, "query.fuzzy.price"),
    ({"query": {"match": {"product_title": {"query": "x", "analyzer": "standard"}}}},
     "query.match.product_title.analyzer"),
    ({"query": {"match": {"product_title": {"query": "x", "zero_terms_query": "all"}}}},
     "query.match.product_title.zero_terms_query"),
    ({"query": {"multi_match": {"query": "x", "type": "phrase"}}}, "query.multi_match.type"),
    ({"query": {"multi_match": {"query": "x", "type": "bool_prefix"}}}, "query.multi_match.type"),
    ({"query": {"multi_match": {"query": "x", "fields": ["product_*"]}}}, "query.multi_match.fields[0]"),
    ({"query": {"range": {"price": {"gte": 1, "format": "x"}}}}, "query.range.price.format"),
    ({"query": {"range": {"price": {"gte": 1, "relation": "within"}}}}, "query.range.price.relation"),
    ({"query": {"terms": {"category": {"index": "x", "id": "1", "path": "c"}}}}, "query.terms.category"),
    ({"query": {"term": {"category": {"value": "x", "case_insensitive": True}}}},
     "query.term.category.case_insensitive"),
    ({"query": {"knn": {"embedding": {"vector": [0.0] * 16, "k": 3, "max_distance": 1}}}},
     "query.knn.embedding.max_distance"),
    ({"query": {"bool": {"must": [M], "adjust_pure_negative": False}}}, "query.bool.adjust_pure_negative"),
    ({"query": {"range": {"product_title": {"gte": "a"}}}}, "query.range.product_title"),
    ({"query": {"term": {"product_title": "shoe"}}}, "query.term.product_title"),
    # request keys outside the subset
    ({"query": M, "highlight": {"fields": {"product_title": {}}}}, "highlight"),
    ({"query": M, "collapse": {"field": "category", "inner_hits": {"name": "x"}}}, "collapse.inner_hits"),
    ({"query": M, "collapse": {"field": "category"}, "search_after": [1.0, 2]}, "search_after"),
    ({"query": M, "post_filter": {"wildcard": {"category": "x*"}}}, "post_filter.wildcard"),
    ({"query": M, "suggest": {}}, "suggest"),
    ({"query": M, "min_score": 1}, "min_score"),
    ({"query": M, "track_scores": True}, "track_scores"),
    ({"query": M, "timeout": "1s"}, "timeout"),
    ({"query": M, "script_fields": {}}, "script_fields"),
    ({"query": M, "docvalue_fields": ["price"]}, "docvalue_fields"),
    ({"query": M, "stored_fields": ["price"]}, "stored_fields"),
    ({"query": M, "pit": {"id": "x"}}, "pit"),
    ({"query": M, "indices_boost": []}, "indices_boost"),
    ({"query": M, "profile": True}, "profile"),
    # aggregations outside the subset
    ({"aggs": {"h": {"histogram": {"field": "price", "interval": 10}}}}, "aggs.h.histogram"),
    ({"aggs": {"x": {"avg": {"field": "price"}}}}, "aggs.x.avg"),
    ({"aggs": {"x": {"cardinality": {"field": "category"}}}}, "aggs.x.cardinality"),
    ({"aggs": {"x": {"terms": {"field": "category"}, "aggs": {"y": {"terms": {"field": "category"}}}}}}, "aggs.x.aggs"),
    ({"aggs": {"x": {"terms": {"field": "category", "order": {"_key": "asc"}}}}}, "aggs.x.terms.order"),
    ({"aggs": {"x": {"terms": {"field": "category", "min_doc_count": 0}}}}, "aggs.x.terms.min_doc_count"),
    ({"aggs": {"x": {"terms": {"field": "product_title"}}}}, "aggs.x.terms.field"),
    # sort and rescore outside the subset
    ({"sort": [{"price": {"order": "asc", "missing": 0}}]}, "sort[0].price.missing"),
    ({"sort": [{"price": {"order": "asc", "mode": "min"}}]}, "sort[0].price.mode"),
    ({"sort": [{"_script": {"type": "number"}}]}, "sort[0]._script.type"),
    ({"sort": ["product_title"]}, "sort[0]"),
    ({"query": M, "rescore": {"window_size": 5, "query": {"rescore_query": M}}}, "rescore.query"),
]

INVALID = [
    {"from": 9995, "size": 10},
    {"size": -1},
    {"size": "10"},
    {"query": {"match": {"nope": "x"}}},
    {"query": {"match": {}}},
    {"query": {"match": M["match"], "term": {"category": "x"}}},
    {"query": {"term": {"category": "shoes"}}, "search_after": [1]},
    {"query": M, "search_after": [1, 2], "from": 3},
    {"query": {"term": {"stock": 2.5}}},
    {"query": {"term": {"stock": "two"}}},
    {"query": {"range": {"price": {"gte": 1, "gt": 2}}}},
    {"query": {"knn": {"embedding": {"vector": [0.1, 0.2], "k": 3}}}},
    {"query": {"hybrid": {"queries": []}}},
    {"query": M, "rank": {"rrf": {}}},
    {"query": M, "rescore": {"s3_ranker": {"model": "m"}}, "sort": ["_score"]},
    {"query": M, "rescore": {"s3_ranker": {"model": "nope"}}},
    {"query": M, "explain": "yes"},
    {"aggs": {"x": {}}},
]


@pytest.mark.parametrize("req,path", UNSUPPORTED, ids=[p for _, p in UNSUPPORTED])
def test_unsupported_refused_by_name(index, req, path):
    with pytest.raises(UnsupportedQuery) as e:
        index.search(req)
    assert type(e.value) is UnsupportedQuery and e.value.path == path


@pytest.mark.parametrize("req", INVALID)
def test_invalid_refused(index, req):
    with pytest.raises(InvalidRequest):
        index.search(req)


def test_parse_needs_no_index_and_plan_resolves():
    r = parse({"query": {"bool": {"must": [{"match": {"product_title": "Running SHOES"}}],
                                  "filter": [{"range": {"price": {"gt": 9.99}}}]}},
               "sort": [{"price": "desc"}], "aggs": {"c": {"terms": {"field": "category"}}}})
    assert isinstance(r, A.Request) and isinstance(r.query, A.QBool)
    assert r.query.must[0].query == "Running SHOES"           # not analysed yet
    assert r.sort[0].order == "desc" and r.aggs[0].field == "category"
    with pytest.raises(UnsupportedQuery):
        parse({"query": {"wildcard": {"x": "y"}}})
    parse({"query": {"match": {"no_such_field": "x"}}})       # field names are the planner's business


def test_plan_semantics(index):
    st = index._st
    p = plan({"query": {"bool": {"must": [{"match": {"product_title": "Running SHOES"}}],
                                 "filter": [{"range": {"price": {"gt": 9.99}}}, {"match": {"category": "shoes"}}]}}}, st)
    lex = p.root.must[0]
    assert isinstance(lex, Lexical) and lex.terms == ("run", "shoe") and lex.idf == "field" and lex.msm == 1
    rng, term = p.root.filter
    assert isinstance(rng, Range) and rng.lo == float(__import__("numpy").float32(9.99)) and not rng.lo_inc
    assert isinstance(term, Term) and term.values == ("shoes",)   # match on a keyword field is a term
    assert plan(parse({"query": M}), st) == plan({"query": M}, st)
    mm = plan({"query": {"multi_match": {"query": "x", "fields": ["product_title^2", "product_brand"]}}}, st).root
    assert mm.fields == (("product_title", 2.0), ("product_brand", 1.0)) and mm.idf == "global"


def test_float_bounds_compare_as_float32(index, catalogue):
    """OpenSearch rounds a float field's bound to float32; a price equal to the bound is inside gte and lte."""
    price = float(catalogue["price"].dropna().iloc[0])
    pid = catalogue.loc[catalogue["price"] == price, "product_id"].tolist()
    for op in ("gte", "lte"):
        r = index.search({"query": {"bool": {"filter": [{"range": {"price": {op: price}}}]}}, "size": 200})
        assert set(pid) <= {h["_id"] for h in r["hits"]["hits"]}
    r = index.search({"query": {"term": {"price": price}}})
    assert {h["_id"] for h in r["hits"]["hits"]} == set(pid)
