"""The planner (docs/INTERFACES.md §6): a parsed ``Request`` (``engine.dsl.ast``) onto the operators' ``Plan``
(``engine.plan.nodes``), against one index.

It resolves field names and kinds, analyses query text, works out ``minimum_should_match``, chooses BM25F's idf mode,
applies the sort defaults, and checks what only the index can tell (vector dimensions, an encoder for ``neural``).
Errors keep the request's JSON paths: a field the index lacks is ``InvalidRequest``; a field of the wrong kind for the
clause, ``UnsupportedQuery``.
"""
from __future__ import annotations

import numpy as np

from ..analysis import analyze
from ..dsl import ast as A
from ..dsl.parse import parse, resolve_msm
from ..errors import InvalidRequest, UnsupportedQuery
from .nodes import Bool, Dense, Hybrid, Lexical, MatchAll, Plan, Range, Rescore, SortKey, Term, TermsAgg


class Planner:
    def __init__(self, state):
        self.s = state      # IndexState: .mapping.kind(), .field_index, .text_fields, .vectors, .vector_spec, .encoders

    def _field(self, name, path, kinds):
        k = self.s.mapping.kind(name)
        if k is None:
            raise InvalidRequest(path, f"unknown field {name!r}")
        if k not in kinds:
            raise UnsupportedQuery(path, f"field {name!r} is {k}; this clause needs {' or '.join(kinds)}")
        return k

    # ---- the request
    def request(self, r: A.Request) -> Plan:
        root = self.node(r.query)
        if r.rrf is not None:
            if not isinstance(root, Hybrid):
                raise InvalidRequest("rank", "rank.rrf needs a hybrid query")
            root = Hybrid(root.queries, r.rrf[0], r.rrf[1], root.path)
        aggs = tuple(self.agg(a) for a in r.aggs)
        rescore = Rescore(*r.rescore) if r.rescore else None
        if rescore is not None and r.sort is not None:
            raise InvalidRequest("sort", "rescore cannot be combined with an explicit sort")
        sort = tuple(self.sort_key(s) for s in r.sort) if r.sort is not None else (SortKey("_score", True),)
        if r.search_after is not None:
            n = len(sort) + (1 if rescore else 0) + 1
            if len(r.search_after) != n:
                raise InvalidRequest("search_after", f"expected the last hit's sort list ({n} values)")
        return Plan(root=root, aggs=aggs, sort=sort, from_=r.from_, size=r.size, search_after=r.search_after,
                    rescore=rescore, explain=r.explain)

    # ---- queries
    def node(self, q):
        return getattr(self, "n_" + type(q).__name__)(q)

    def _lexical(self, text, weights, operator, msm, boost, path):
        terms = tuple(dict.fromkeys(analyze(text)))
        n = len(terms)
        need = n if operator == "and" else max(1, resolve_msm(msm, n))
        order = self.s.field_index
        fw = tuple(sorted(weights.items(), key=lambda x: order[x[0]]))
        idf = "field" if sum(1 for _, w in fw if w > 0) == 1 else "global"
        return Lexical(fw, terms, min(need, n), idf, boost, path)

    def n_QMatchAll(self, q):
        return MatchAll(q.boost, q.path)

    def n_QMatch(self, q):
        kind = self._field(q.field, q.path, ("text", "keyword", "numeric"))
        if kind == "keyword":                  # OpenSearch: a match on a keyword or numeric field is a term
            return Term(q.field, (str(q.query),), q.boost, q.path)
        if kind == "numeric":
            return Term(q.field, (self._numeric_value(q.field, q.query, q.path),), q.boost, q.path)
        return self._lexical(str(q.query), {q.field: 1.0}, q.operator, q.msm, q.boost, q.path)

    def n_QMultiMatch(self, q):
        if q.fields is None:
            weights = {f: 1.0 for f in self.s.text_fields}
        else:
            weights = {}
            for i, (f, w) in enumerate(q.fields):
                self._field(f, f"{q.path}.fields[{i}]", ("text",))
                weights[f] = 1.0 if w is None else w
        return self._lexical(q.query, weights, q.operator, q.msm, q.boost, q.path)

    def n_QBool(self, q):
        msm = None if q.msm is None else resolve_msm(q.msm, len(q.should))
        return Bool(tuple(map(self.node, q.must)), tuple(map(self.node, q.should)), tuple(map(self.node, q.filter)),
                    tuple(map(self.node, q.must_not)), msm, q.boost, q.path)

    def _numeric_value(self, field, v, path, bound=None):
        """A term value or range bound as the field's type compares it (OpenSearch's rules): float fields round the
        value to float32; integer fields refuse a fractional term (a fractional bound is allowed: ``gte 2.5`` is
        ``gte 3``); boolean fields take true/false or "true"/"false"."""
        dt = self.s.mapping.numeric[field]
        if dt == "bool":
            if isinstance(v, str) and v.lower() in ("true", "false"):
                return v.lower() == "true"
            if isinstance(v, bool):
                return v
            raise InvalidRequest(path, f"{v!r} is not a boolean")
        if isinstance(v, bool):
            raise InvalidRequest(path, f"{v!r} is not a number")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise InvalidRequest(path, f"{v!r} is not a number") from None
        if dt == "float32":
            return float(np.float32(x))
        if bound is None and x != int(x):
            raise InvalidRequest(path, f"{v!r} has a decimal part; the field is an integer")
        return x

    def n_QTerm(self, q):
        p = f"{q.path}.{q.field}"
        kind = self._field(q.field, p, ("keyword", "numeric"))
        if kind == "keyword":
            vals = tuple(str(v).lower() if isinstance(v, bool) else str(v) for v in q.values)
        else:
            vals = tuple(self._numeric_value(q.field, v, p) for v in q.values)
        return Term(q.field, vals, q.boost, q.path)

    def n_QRange(self, q):
        p = f"{q.path}.{q.field}"
        kind = self._field(q.field, p, ("keyword", "numeric"))
        if kind == "numeric" and self.s.mapping.numeric[q.field] == "bool":
            raise UnsupportedQuery(p, "range on a boolean field")

        def conv(v, k):
            if v is None:
                return None
            if kind == "keyword":
                return str(v)
            return self._numeric_value(q.field, v, f"{p}.{k}", bound=k)
        lo, lo_inc = (conv(q.gte, "gte"), True) if q.gte is not None else (conv(q.gt, "gt"), False)
        hi, hi_inc = (conv(q.lte, "lte"), True) if q.lte is not None else (conv(q.lt, "lt"), False)
        return Range(q.field, lo, hi, lo_inc, hi_inc, q.boost, q.path)

    def _vector_field(self, field, p):
        self._field(field, p, ("vector",))
        if field not in self.s.vectors:
            raise UnsupportedQuery(p, f"field {field!r} has no vectors in this index")

    def n_QKnn(self, q):
        p = f"{q.path}.{q.field}"
        self._vector_field(q.field, p)
        dim = self.s.vector_spec[q.field]["dim"]
        if len(q.vector) != dim:
            raise InvalidRequest(f"{p}.vector", f"a list of {dim} numbers")
        flt = self.node(q.filter) if q.filter is not None else None
        return Dense(q.field, q.vector, None, q.k, flt, q.boost, q.path)

    def n_QNeural(self, q):
        p = f"{q.path}.{q.field}"
        self._vector_field(q.field, p)
        if q.field not in self.s.encoders:
            raise UnsupportedQuery(p, f"field {q.field!r} has no query encoder in this index (given vectors: use knn)")
        flt = self.node(q.filter) if q.filter is not None else None
        return Dense(q.field, None, q.text, q.k, flt, q.boost, q.path)

    def n_QHybrid(self, q):
        return Hybrid(tuple(map(self.node, q.queries)), path=q.path)

    # ---- aggregations and sort
    def agg(self, a: A.Agg) -> TermsAgg:
        self._field(a.field, f"{a.path}.terms.field", ("keyword", "numeric"))
        return TermsAgg(a.name, a.field, a.size)

    def sort_key(self, s: A.Sort) -> SortKey:
        if s.field not in ("_score", "_doc"):
            self._field(s.field, s.path, ("keyword", "numeric"))
        order = s.order or ("desc" if s.field == "_score" else "asc")
        return SortKey(s.field, order == "desc", s.missing == "_last")


def plan(request, state) -> Plan:
    """A request (a dict, or a parsed ``Request``) → ``Plan`` for ``state``."""
    if isinstance(request, dict) or not isinstance(request, A.Request):
        request = parse(request)
    return Planner(state).request(request)
