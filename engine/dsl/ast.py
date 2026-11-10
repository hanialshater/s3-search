"""The parsed request (docs/INTERFACES.md §6): the DSL's own shapes, checked for syntax only. Field names are not yet
resolved and text is not yet analysed; the planner (``engine.plan.plan``) does that against an index.

Every node keeps ``path``, the JSON path of its key, so the planner's errors name the request's key."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class QMatch:
    field: str
    query: Any
    operator: str = "or"
    msm: Any = None
    boost: float = 1.0
    path: str = "query.match"


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


@dataclass(frozen=True)
class QNeural:
    field: str
    text: str
    k: int = 10
    filter: Any = None
    boost: float = 1.0
    path: str = "query.neural"


@dataclass(frozen=True)
class QMatchAll:
    boost: float = 1.0
    path: str = "query.match_all"


@dataclass(frozen=True)
class QHybrid:
    queries: tuple
    path: str = "query.hybrid"


Query = Union[QMatch, QMatchPhrase, QMultiMatch, QBool, QTerm, QRange, QKnn, QNeural, QMatchAll, QHybrid]


@dataclass(frozen=True)
class Agg:
    name: str
    field: str
    size: int = 10
    path: str = "aggs"


@dataclass(frozen=True)
class Sort:
    field: str                            # "_score", "_doc" or a field name
    order: str | None = None              # None: the field's default (desc for _score, asc otherwise)
    missing: str = "_last"
    path: str = "sort"


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
