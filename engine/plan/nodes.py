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
class Dense:
    field: str
    vector: tuple | None
    text: str | None
    k: int
    filter: Any = None
    boost: float = 1.0
    path: str = "query"


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


Node = Union[Lexical, Dense, Term, Range, MatchAll, Bool, Hybrid]


@dataclass(frozen=True)
class TermsAgg:
    name: str
    field: str
    size: int = 10


@dataclass(frozen=True)
class SortKey:
    field: str               # "_score", "_doc" or a field name
    descending: bool
    missing_last: bool = True


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
