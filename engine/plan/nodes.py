"""Plan nodes (docs/INTERFACES.md §6). Frozen, hashable: the executor groups equal shapes into one batch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass(frozen=True)
class Lexical:
    fields: tuple            # ((field, weight), ...) in the index's text-field order
    terms: tuple             # distinct analysed terms, in query order (unknown terms included)
    msm: int                 # how many of `terms` a doc must match (0: the query has no terms, matches nothing)
    idf: str                 # "global" or "field" (one field queried)
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class LexicalGroups:
    """Clause groups (S7-2, S7-6): each group is a tuple of alternative analysed terms that count as one clause
    for ``msm``; ``modes[g]`` is "sum" (fuzzy expansions), "synonym" (one term, Lucene's SynonymQuery) or "const"
    (a prefix: 1.0 where present)."""
    fields: tuple
    groups: tuple
    modes: tuple
    msm: int
    idf: str
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Phrase:
    field: str
    slots: tuple             # per slot, its alternative analysed terms (one term, or a synonym set)
    offsets: tuple           # each slot's position in the query (stopwords leave gaps)
    slop: int = 0
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Dense:
    field: str
    vector: tuple | None
    text: str | None
    k: int
    filter: Any = None
    boost: float = 1.0
    path: str = "query"
    method: str = "exact"                 # S9-1: the requested method; the executor says which ran
    method_parameters: tuple = ()


@dataclass(frozen=True)
class Term:
    field: str
    values: tuple
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Range:
    field: str
    lo: Any = None
    hi: Any = None
    lo_inc: bool = True
    hi_inc: bool = True
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Nested:
    """S9-5: ``query`` is planned against the children of ``nested_path``; parents match when a child does."""
    nested_path: str
    query: Any
    score_mode: str = "avg"
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class GeoDistance:
    field: str
    lat: float
    lon: float
    meters: float
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class GeoBox:
    """Lucene's encoded bounds (S9-4): lat_lo/lat_hi and lon_lo/lon_hi inclusive; ``crosses`` the dateline."""
    field: str
    lat_lo: int
    lat_hi: int
    lon_lo: int
    lon_hi: int
    crosses: bool = False
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class FunctionScore:
    """S7-5: functions are (filter node or None, (field, factor, modifier, missing) or None, weight or None)."""
    query: Any
    functions: tuple
    score_mode: str = "multiply"
    boost_mode: str = "multiply"
    max_boost: float | None = None
    min_score: float | None = None
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Exists:
    field: str
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Ids:
    values: tuple
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class MatchAll:
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Bool:
    must: tuple = ()
    should: tuple = ()
    filter: tuple = ()
    must_not: tuple = ()
    msm: int | None = None
    boost: float = 1.0
    path: str = "query"


@dataclass(frozen=True)
class Hybrid:
    queries: tuple
    rrf_k: int = 60
    window: int = 20_000
    path: str = "query"


Node = Union[Nested, GeoDistance, GeoBox, FunctionScore, Exists, Ids, Lexical, LexicalGroups, Phrase, Dense, Term, Range, MatchAll, Bool, Hybrid]


@dataclass(frozen=True)
class TermsAgg:
    """An aggregation node (S7-4: any kind, not only terms; the name is kept for compatibility)."""
    name: str
    field: str
    size: int = 10
    kind: str = "terms"
    params: tuple = ()
    sub: tuple = ()

    def param(self, key, default=None):
        return dict(self.params).get(key, default)


@dataclass(frozen=True)
class SortKey:
    field: str               # "_score", "_doc" or a field name
    descending: bool
    missing_last: bool = True
    geo: tuple | None = None   # S9-4, _geo_distance: (field, lat, lon, unit metres)


@dataclass(frozen=True)
class Rescore:
    window: int
    model: str


@dataclass(frozen=True)
class Plan:
    root: Node
    aggs: tuple = ()
    sort: tuple = (SortKey("_score", True),)
    from_: int = 0
    size: int = 10
    search_after: tuple | None = None
    rescore: Rescore | None = None
    explain: bool = False
    post_filter: Any = None               # S7-3: a node; hits and total after it, aggregations before it
    collapse: str | None = None           # S7-3: keep the top hit per value of this field
    highlight: Any = None                 # S7-5: engine.dsl.ast.Highlight
