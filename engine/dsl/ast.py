"""The parsed request (docs/INTERFACES.md §6): the DSL's own shapes, checked for syntax only. Field names are not yet
resolved and text is not yet analysed; the planner (``engine.plan.plan``) does that against an index.

Every node keeps ``path``, the JSON path of its key, so the planner's errors name the request's key."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Fuzzy:
    """Fuzziness options of match, multi_match and fuzzy (OpenSearch's defaults)."""
    fuzziness: Any = None                # None: exact terms
    prefix_length: int = 0
    max_expansions: int = 50
    transpositions: bool = True


@dataclass(frozen=True)
class QMatch:
    field: str
    query: Any
    operator: str = "or"
    msm: Any = None
    boost: float = 1.0
    path: str = "query.match"
    fuzzy: Fuzzy = Fuzzy()


@dataclass(frozen=True)
class QFuzzy:
    field: str
    value: Any
    fuzzy: Fuzzy = Fuzzy("AUTO")
    boost: float = 1.0
    path: str = "query.fuzzy"


@dataclass(frozen=True)
class QPrefix:
    field: str
    value: str
    boost: float = 1.0
    path: str = "query.prefix"


@dataclass(frozen=True)
class QMatchBoolPrefix:
    field: str
    query: str
    operator: str = "or"
    msm: Any = None
    max_expansions: int = 50
    boost: float = 1.0
    path: str = "query.match_bool_prefix"


@dataclass(frozen=True)
class QMultiMatch:
    query: Any
    fields: tuple | None = None          # ((name, boost or None), ...); None: every text field
    type: str = "best_fields"
    operator: str = "or"
    msm: Any = None
    tie_breaker: float | None = None
    boost: float = 1.0
    path: str = "query.multi_match"
    fuzzy: Fuzzy = Fuzzy()


@dataclass(frozen=True)
class QMatchPhrase:
    field: str
    query: Any
    slop: int = 0
    boost: float = 1.0
    path: str = "query.match_phrase"


@dataclass(frozen=True)
class QBool:
    must: tuple = ()
    should: tuple = ()
    filter: tuple = ()
    must_not: tuple = ()
    msm: Any = None
    boost: float = 1.0
    path: str = "query.bool"


@dataclass(frozen=True)
class QTerm:
    field: str
    values: tuple
    many: bool = False                   # terms (a list) or term (one value)
    boost: float = 1.0
    path: str = "query.term"


@dataclass(frozen=True)
class QRange:
    field: str
    gte: Any = None
    gt: Any = None
    lte: Any = None
    lt: Any = None
    boost: float = 1.0
    path: str = "query.range"


@dataclass(frozen=True)
class QKnn:
    field: str
    vector: tuple
    k: int = 10
    filter: Any = None
    boost: float = 1.0
    path: str = "query.knn"
    method: str = "exact"                 # S9-1: exact | cagra | ivf_pq
    method_parameters: tuple = ()         # S9-1: sorted (name, int) pairs


@dataclass(frozen=True)
class QNeural:
    field: str
    text: str
    k: int = 10
    filter: Any = None
    boost: float = 1.0
    path: str = "query.neural"
    method: str = "exact"
    method_parameters: tuple = ()


@dataclass(frozen=True)
class QNested:
    path_name: str
    query: Any
    score_mode: str = "avg"
    boost: float = 1.0
    path: str = "query.nested"


@dataclass(frozen=True)
class QGeoDistance:
    field: str
    lat: float
    lon: float
    meters: float
    boost: float = 1.0
    path: str = "query.geo_distance"


@dataclass(frozen=True)
class QGeoBox:
    field: str
    top: float
    left: float
    bottom: float
    right: float
    boost: float = 1.0
    path: str = "query.geo_bounding_box"


@dataclass(frozen=True)
class QExists:
    field: str
    boost: float = 1.0
    path: str = "query.exists"


@dataclass(frozen=True)
class QIds:
    values: tuple
    boost: float = 1.0
    path: str = "query.ids"


@dataclass(frozen=True)
class QFunctionScore:
    """function_score (S7-5): ``functions`` are (filter query or None, field_value_factor params or None, weight
    or None)."""
    query: Any
    functions: tuple
    score_mode: str = "multiply"
    boost_mode: str = "multiply"
    max_boost: float | None = None
    min_score: float | None = None
    boost: float = 1.0
    path: str = "query.function_score"


@dataclass(frozen=True)
class Highlight:
    """highlight (S7-5): ``fields`` are (field, fragment_size, number_of_fragments, pre_tag, post_tag)."""
    fields: tuple
    require_field_match: bool = True


@dataclass(frozen=True)
class QMatchAll:
    boost: float = 1.0
    path: str = "query.match_all"


@dataclass(frozen=True)
class QHybrid:
    queries: tuple
    path: str = "query.hybrid"


Query = Union[QNested, QGeoDistance, QGeoBox, QFunctionScore, QExists, QIds, QMatch, QMatchPhrase, QFuzzy, QPrefix, QMatchBoolPrefix, QMultiMatch, QBool, QTerm, QRange, QKnn, QNeural, QMatchAll, QHybrid]


@dataclass(frozen=True)
class Agg:
    """An aggregation: ``kind`` is terms, min, max, avg, sum, stats, value_count, cardinality, range or histogram;
    ``params`` its options as sorted (key, value) pairs; ``sub`` its sub-aggregations (under terms only, one level)."""
    name: str
    field: str
    kind: str = "terms"
    params: tuple = ()
    sub: tuple = ()
    path: str = "aggs"

    def param(self, key, default=None):
        return dict(self.params).get(key, default)


@dataclass(frozen=True)
class Sort:
    field: str                            # "_score", "_doc" or a field name
    order: str | None = None              # None: the field's default (desc for _score, asc otherwise)
    missing: str = "_last"
    path: str = "sort"
    geo: tuple | None = None              # S9-4, _geo_distance: (field, lat, lon, unit)


@dataclass(frozen=True)
class Request:
    query: Query
    aggs: tuple = ()
    sort: tuple | None = None             # None: [_score desc]
    from_: int = 0
    size: int = 10
    search_after: tuple | None = None
    explain: bool = False
    rrf: tuple | None = None              # (rank_constant, window_size) from rank.rrf
    rescore: tuple | None = None          # (window_size, model) from rescore.s3_ranker
    post_filter: Any = None               # a query (S7-3): filters hits after aggregations
    collapse: str | None = None           # a field (S7-3): the top hit per value
    highlight: Any = None                 # Highlight (S7-5)
