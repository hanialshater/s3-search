"""The subset's edge, as one table (owner: evaluation-steward; makers do not edit it, P6).

Every construct the contract probes, with the S7 board item that adds it (None: outside S7, refused for good), whether
the steward has marked that item landed, the request that probes it, and the fragment the error's `.path` must name.

The contract (`checks.check_scope`) reads only this table:
- not landed (or outside S7): the engine must raise `UnsupportedQuery` naming the path;
- landed: the engine must answer (its correctness is S7-9's conformance families, not this check).
So a construct that starts answering before the steward marks it landed fails `referee fast`, and so does a construct
marked landed that the engine still refuses. S7's commits cannot drift from the table silently.

Synonyms (S7-6) are a mapping setting, not a request construct: `MAPPING_ITEMS` below marks it, and the contract
checks it with mappings (`contract/test_contract.py::test_synonym_mappings`, `s7_checks.check_s7_6`).
"""
from __future__ import annotations

Q = {"match": {"product_title": "lamp"}}

# (name, S7 item or None, landed, path fragment, request)
SCOPE = [
    # S7-1: positions and phrases
    ("match_phrase", "S7-1", True, "match_phrase", {"query": {"match_phrase": {"product_title": "red lamp"}}}),
    ("match_phrase slop", "S7-1", True, "match_phrase",
     {"query": {"match_phrase": {"product_title": {"query": "red lamp", "slop": 1}}}}),
    # S7-2: typo tolerance and autocomplete
    ("fuzzy", "S7-2", True, "fuzzy", {"query": {"fuzzy": {"product_title": "lamq"}}}),
    ("match fuzziness", "S7-2", True, "fuzziness",
     {"query": {"match": {"product_title": {"query": "lamq", "fuzziness": "AUTO"}}}}),
    ("prefix", "S7-2", True, "prefix", {"query": {"prefix": {"product_title": "lam"}}}),
    ("match_bool_prefix", "S7-2", True, "match_bool_prefix",
     {"query": {"match_bool_prefix": {"product_title": "red la"}}}),
    # S7-3: exists, ids, post_filter, collapse on a keyword field
    ("exists", "S7-3", True, "exists", {"query": {"exists": {"field": "product_color"}}}),
    ("ids", "S7-3", True, "ids", {"query": {"ids": {"values": ["T0001", "T0002"]}}}),
    ("post_filter", "S7-3", True, "post_filter", {"query": Q, "post_filter": {"term": {"category": "lamp"}}}),
    ("collapse keyword", "S7-3", True, "collapse", {"query": Q, "collapse": {"field": "category"}}),
    ("collapse numeric", "S7-3", True, "collapse", {"query": Q, "collapse": {"field": "stock"}}),
    # S7-4: metric and bucket aggregations, one level of sub-aggregations under terms
    *[(f"agg {a}", "S7-4", True, a, {"query": Q, "size": 0, "aggs": {"x": {a: {"field": "price"}}}})
      for a in ("min", "max", "avg", "sum", "stats", "value_count")],
    ("agg cardinality", "S7-4", True, "cardinality",
     {"query": Q, "size": 0, "aggs": {"x": {"cardinality": {"field": "category"}}}}),
    ("agg range", "S7-4", True, "range",
     {"size": 0, "aggs": {"x": {"range": {"field": "price", "ranges": [{"to": 50}, {"from": 50}]}}}}),
    ("agg histogram", "S7-4", True, "histogram",
     {"size": 0, "aggs": {"x": {"histogram": {"field": "price", "interval": 50}}}}),
    ("agg terms sub-aggregation", "S7-4", True, "aggs",
     {"size": 0, "aggs": {"c": {"terms": {"field": "category"}, "aggs": {"p": {"avg": {"field": "price"}}}}}}),
    # S7-5: highlight (plain, text fields), function_score with field_value_factor and weight
    ("highlight text", "S7-5", True, "highlight", {"query": Q, "highlight": {"fields": {"product_title": {}}}}),
    ("function_score field_value_factor", "S7-5", True, "function_score",
     {"query": {"function_score": {"query": Q, "field_value_factor": {"field": "stock", "missing": 1}}}}),
    ("function_score weight", "S7-5", True, "function_score",
     {"query": {"function_score": {"query": Q, "functions": [{"filter": {"term": {"category": "lamp"}},
                                                               "weight": 2}]}}}),
    # Outside S7: refused for good
    ("query_string", None, False, "query_string", {"query": {"query_string": {"query": "lamp AND red"}}}),
    ("wildcard", None, False, "wildcard", {"query": {"wildcard": {"product_title": "la*"}}}),
    ("regexp", None, False, "regexp", {"query": {"regexp": {"product_title": "la.p"}}}),
    ("nested", None, False, "nested", {"query": {"nested": {"path": "variants", "query": {"match_all": {}}}}}),
    ("dis_max", None, False, "dis_max", {"query": {"dis_max": {"queries": [Q]}}}),
    ("constant_score", None, False, "constant_score",
     {"query": {"constant_score": {"filter": {"term": {"category": "lamp"}}}}}),
    ("terms lookup", None, False, "terms",
     {"query": {"terms": {"category": {"index": "other", "id": "1", "path": "category"}}}}),
    ("suggest", None, False, "suggest",
     {"query": Q, "suggest": {"s": {"text": "lamq", "term": {"field": "product_title"}}}}),
    ("pit", None, False, "pit", {"query": Q, "pit": {"id": "abc", "keep_alive": "1m"}}),
    ("profile", None, False, "profile", {"query": Q, "profile": True}),
    ("highlight keyword field", None, False, "highlight", {"query": Q, "highlight": {"fields": {"category": {}}}}),
    ("highlight fvh", None, False, "highlight",
     {"query": Q, "highlight": {"type": "fvh", "fields": {"product_title": {}}}}),
    ("collapse text field", None, False, "collapse", {"query": Q, "collapse": {"field": "product_title"}}),
    ("function_score random_score", None, False, "function_score",
     {"query": {"function_score": {"query": Q, "random_score": {}}}}),
    ("function_score script_score", None, False, "function_score",
     {"query": {"function_score": {"query": Q, "script_score": {"script": "1"}}}}),
    ("agg two-level sub-aggregations", None, False, "aggs",
     {"size": 0, "aggs": {"c": {"terms": {"field": "category"}, "aggs": {
         "b": {"terms": {"field": "product_brand.keyword"}, "aggs": {"p": {"avg": {"field": "price"}}}}}}}}),
    ("script_fields", None, False, "script_fields", {"query": Q, "script_fields": {"x": {"script": "1"}}}),
]


# S7 items that are mapping settings, not request constructs: {item: landed}
MAPPING_ITEMS = {"S7-6": True}


def must_refuse(honor_landed: bool = True):
    """Entries the implementation must refuse: everything outside S7, and S7 items not marked landed. The reference
    index implements none of S7, so it is checked with honor_landed=False (it refuses all)."""
    return [e for e in SCOPE if not (honor_landed and e[1] is not None and e[2])]


def landed():
    return [e for e in SCOPE if e[1] is not None and e[2]]


def landed_items():
    """Every landed S7 item, request constructs and mapping settings."""
    return sorted({e[1] for e in landed()} | {i for i, ok in MAPPING_ITEMS.items() if ok})
