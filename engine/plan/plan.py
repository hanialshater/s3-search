"""The planner (docs/INTERFACES.md §6): a parsed ``Request`` (``engine.dsl.ast``) onto the operators' ``Plan``
(``engine.plan.nodes``), against one index.

It resolves field names and kinds, analyses query text, works out ``minimum_should_match``, chooses BM25F's idf mode,
applies the sort defaults, and checks what only the index can tell (vector dimensions, an encoder for ``neural``).
Errors keep the request's JSON paths: a field the index lacks is ``InvalidRequest``; a field of the wrong kind for the
clause, ``UnsupportedQuery``.
"""
from __future__ import annotations

import numpy as np

from ..analysis import analyze, analyze_with_positions
from ..dsl import ast as A
from ..dsl.parse import parse, resolve_msm
from ..errors import InvalidRequest, UnsupportedQuery
from .nodes import Bool, Dense, Exists, FunctionScore, GeoBox, GeoDistance, Nested, Ids, LexicalGroups, Phrase, Hybrid, Lexical, MatchAll, Plan, Range, Rescore, SortKey, Term, TermsAgg


class _OutsideNested(Exception):
    """A leaf names a field under a nested path outside a nested query: OpenSearch answers no docs."""


class Planner:
    def __init__(self, state):
        self.s = state      # IndexState: .mapping.kind(), .field_index, .text_fields, .vectors, .vector_spec, .encoders

    def _field(self, name, path, kinds):
        k = self.s.mapping.kind(name)
        if k == "in_nested" and "in_nested" not in kinds:
            if getattr(self, "_leaf", False):
                raise _OutsideNested()
            raise UnsupportedQuery(path, f"field {name!r} is under a nested path: use a nested query or aggregation")
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
        post = self.node(r.post_filter) if r.post_filter is not None else None
        collapse = None
        if r.collapse is not None:
            self._field(r.collapse, "collapse.field", ("keyword", "numeric"))
            if r.search_after is not None:
                raise UnsupportedQuery("search_after", "search_after with collapse is not supported")
            if rescore is not None:
                raise InvalidRequest("collapse", "collapse cannot be combined with rescore (as OpenSearch)")
            collapse = r.collapse
        hl = r.highlight
        if hl is not None:
            for f, *_ in hl.fields:
                self._field(f, f"highlight.fields.{f}", ("text",))
            if not all(p.seg.source for p in self.s.parts):
                raise UnsupportedQuery("highlight", "this index keeps no source text (_meta.engine.source: false)")
        return Plan(root=root, aggs=aggs, sort=sort, from_=r.from_, size=r.size, search_after=r.search_after,
                    rescore=rescore, explain=r.explain, post_filter=post, collapse=collapse, highlight=hl)

    # ---- queries
    def node(self, q):
        if type(q).__name__ in ("QBool", "QHybrid", "QFunctionScore", "QNested"):
            return getattr(self, "n_" + type(q).__name__)(q)
        self._leaf = True
        try:
            return getattr(self, "n_" + type(q).__name__)(q)
        except _OutsideNested:
            return Ids((), 1.0, getattr(q, "path", "query"))      # matches nothing, as OpenSearch
        finally:
            self._leaf = False

    def n_QNested(self, q):
        if self.s.mapping.kind(q.path_name) != "nested":
            raise InvalidRequest(f"{q.path}.path", f"{q.path_name!r} is not a nested field")
        inner = Planner(self.s.nested[q.path_name]).node(q.query)
        return Nested(q.path_name, inner, q.score_mode, q.boost, q.path)

    def _lexical(self, text, weights, operator, msm, boost, path):
        terms = tuple(dict.fromkeys(analyze(text)))
        n = len(terms)
        need = n if operator == "and" else max(1, resolve_msm(msm, n))
        order = self.s.field_index
        fw = tuple(sorted(weights.items(), key=lambda x: order[x[0]]))
        idf = "field" if sum(1 for _, w in fw if w > 0) == 1 else "global"
        syn = self.s.mapping.synonyms
        if syn and any(t in syn for t in terms):         # S7-6: a term with synonyms is one clause of them all
            groups = tuple(syn.get(t, (t,)) for t in terms)
            modes = tuple("synonym" if t in syn else "sum" for t in terms)
            return LexicalGroups(fw, groups, modes, min(need, n), idf, boost, path)
        return Lexical(fw, terms, min(need, n), idf, boost, path)

    def _ordered(self, weights):
        order = self.s.field_index
        fw = tuple(sorted(weights.items(), key=lambda x: order[x[0]]))
        return fw, ("field" if sum(1 for _, w in fw if w > 0) == 1 else "global")

    def _fuzzy_lexical(self, text, weights, operator, msm, fz, boost, path):
        """match / multi_match with fuzziness: one group per analysed query term, its expansions in the queried
        fields' terms dictionary."""
        from ..dictionary import auto_edits, field_mask, text_dictionary
        toks = list(dict.fromkeys(analyze(text)))
        edits = [auto_edits(fz.fuzziness, t, f"{path}.fuzziness") for t in toks]
        if not any(edits):
            return None                                   # no edits anywhere: a plain match
        d = text_dictionary(self.s)
        allowed = field_mask(self.s, [f for f, w in weights.items() if w > 0])
        groups = tuple(tuple(d.terms[i] for i in d.fuzzy(t, e, fz.prefix_length, fz.max_expansions,
                                                         fz.transpositions, allowed)) for t, e in zip(toks, edits))
        n = len(groups)
        need = n if operator == "and" else max(1, resolve_msm(msm, n))
        fw, idf = self._ordered(weights)
        return LexicalGroups(fw, groups, ("sum",) * n, min(need, n), idf, boost, path)

    def n_QFuzzy(self, q):
        from ..dictionary import auto_edits, field_mask, keyword_dictionary, text_dictionary
        p = f"{q.path}.{q.field}"
        kind = self._field(q.field, p, ("text", "keyword"))
        value = str(q.value)                              # term-level: not analysed (as OpenSearch)
        fz = q.fuzzy
        e = auto_edits(fz.fuzziness, value, f"{p}.fuzziness")
        if kind == "keyword":
            kd = keyword_dictionary(self.s.keywords[q.field])
            vals = tuple(kd.terms[i] for i in kd.fuzzy(value, e, fz.prefix_length, fz.max_expansions,
                                                       fz.transpositions))
            return Term(q.field, vals, q.boost, q.path)
        d = text_dictionary(self.s)
        terms = tuple(d.terms[i] for i in d.fuzzy(value, e, fz.prefix_length, fz.max_expansions, fz.transpositions,
                                                   field_mask(self.s, [q.field])))
        return LexicalGroups(((q.field, 1.0),), (terms,), ("sum",), 1, "field", q.boost, q.path)

    def n_QPrefix(self, q):
        from ..dictionary import field_mask, text_dictionary
        p = f"{q.path}.{q.field}"
        kind = self._field(q.field, p, ("text", "keyword"))
        if kind == "keyword":                             # every value that starts with it: a code range
            return Range(q.field, q.value, q.value + "\U0010ffff", True, False, q.boost, q.path)
        d = text_dictionary(self.s)
        terms = tuple(d.terms[i] for i in d.prefix(q.value, field_mask(self.s, [q.field])))
        return LexicalGroups(((q.field, 1.0),), (terms,), ("const",), 1, "field", q.boost, q.path)

    def n_QMatchBoolPrefix(self, q):
        from ..dictionary import field_mask, text_dictionary
        self._field(q.field, q.path, ("text",))
        toks = analyze(q.query)
        if not toks:
            return LexicalGroups(((q.field, 1.0),), (), (), 0, "field", q.boost, q.path)
        d = text_dictionary(self.s)
        last = tuple(d.terms[i] for i in d.prefix(toks[-1], field_mask(self.s, [q.field]), q.max_expansions))
        syn = self.s.mapping.synonyms
        groups = tuple(syn.get(t, (t,)) for t in toks[:-1]) + (last,)
        n = len(groups)
        need = n if q.operator == "and" else max(1, resolve_msm(q.msm, n))
        modes = tuple("synonym" if t in syn else "sum" for t in toks[:-1]) + ("const",)
        return LexicalGroups(((q.field, 1.0),), groups, modes, min(need, n), "field", q.boost, q.path)

    def n_QFunctionScore(self, q):
        fns = []
        for i, (flt, fvf, w) in enumerate(q.functions):
            if fvf is not None:
                p = f"{q.path}.functions[{i}].field_value_factor.field" if len(q.functions) > 1 or flt is not None \
                    else f"{q.path}.field_value_factor.field"
                self._field(fvf[0], p, ("numeric",))
            fns.append((self.node(flt) if flt is not None else None, fvf, w))
        return FunctionScore(self.node(q.query), tuple(fns), q.score_mode, q.boost_mode, q.max_boost, q.min_score,
                             q.boost, q.path)

    def n_QGeoDistance(self, q):
        self._field(q.field, f"{q.path}.{q.field}", ("geo",))
        return GeoDistance(q.field, q.lat, q.lon, q.meters, q.boost, q.path)

    def n_QGeoBox(self, q):
        from ..geo import encode_lat, encode_lat_ceil, encode_lon, encode_lon_ceil
        self._field(q.field, f"{q.path}.{q.field}", ("geo",))
        return GeoBox(q.field, encode_lat_ceil(q.bottom), encode_lat(q.top), encode_lon_ceil(q.left),
                      encode_lon(q.right), q.left > q.right, q.boost, q.path)

    def n_QExists(self, q):
        self._field(q.field, f"{q.path}.field", ("text", "keyword", "numeric", "vector", "geo"))
        return Exists(q.field, q.boost, q.path)

    def n_QIds(self, q):
        return Ids(q.values, q.boost, q.path)

    def n_QMatchAll(self, q):
        return MatchAll(q.boost, q.path)

    def n_QMatch(self, q):
        kind = self._field(q.field, q.path, ("text", "keyword", "numeric"))
        if kind == "keyword":                  # OpenSearch: a match on a keyword or numeric field is a term
            return Term(q.field, (str(q.query),), q.boost, q.path)
        if kind == "numeric":
            return Term(q.field, (self._numeric_value(q.field, q.query, q.path),), q.boost, q.path)
        if q.fuzzy.fuzziness is not None:
            g = self._fuzzy_lexical(str(q.query), {q.field: 1.0}, q.operator, q.msm, q.fuzzy, q.boost, q.path)
            if g is not None:
                return g
        return self._lexical(str(q.query), {q.field: 1.0}, q.operator, q.msm, q.boost, q.path)

    def n_QMatchPhrase(self, q):
        kind = self._field(q.field, q.path, ("text", "keyword", "numeric"))
        if kind == "keyword":                  # OpenSearch: a phrase on a keyword field is a term
            return Term(q.field, (str(q.query),), q.boost, q.path)
        if kind == "numeric":
            return Term(q.field, (self._numeric_value(q.field, q.query, q.path),), q.boost, q.path)
        if not self.s.has_positions:
            raise UnsupportedQuery(q.path, "this index was built without positions (_meta.engine.positions: false)")
        toks, pos = analyze_with_positions(str(q.query))
        slots = tuple(self.s.mapping.synonyms.get(t, (t,)) for t in toks)     # S7-6: synonyms per slot
        offsets = tuple(p - pos[0] for p in pos) if pos else ()
        return Phrase(q.field, slots, offsets, q.slop, q.boost, q.path)

    def n_QMultiMatch(self, q):
        if q.fields is None:
            weights = {f: 1.0 for f in self.s.text_fields}
        else:
            weights = {}
            for i, (f, w) in enumerate(q.fields):
                self._field(f, f"{q.path}.fields[{i}]", ("text",))
                weights[f] = 1.0 if w is None else w
        if q.fuzzy.fuzziness is not None:
            if q.type == "cross_fields":
                raise InvalidRequest(f"{q.path}.fuzziness", "fuzziness is not allowed for type cross_fields "
                                                            "(as OpenSearch)")
            g = self._fuzzy_lexical(q.query, weights, q.operator, q.msm, q.fuzzy, q.boost, q.path)
            if g is not None:
                return g
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
    AGG_FIELDS = {"terms": ("keyword", "numeric"), "value_count": ("keyword", "numeric"),
                  "cardinality": ("keyword", "numeric")}

    def agg(self, a: A.Agg) -> TermsAgg:
        if a.kind == "nested":
            if self.s.mapping.kind(a.field) != "nested":
                raise InvalidRequest(f"{a.path}.nested.path", f"{a.field!r} is not a nested field")
            child = Planner(self.s.nested[a.field])
            return TermsAgg(a.name, a.field, 10, "nested", (), tuple(child.agg(x) for x in a.sub))
        kinds = self.AGG_FIELDS.get(a.kind, ("numeric",))
        self._field(a.field, f"{a.path}.{a.kind}.field", kinds)
        if a.kind not in ("terms", "value_count", "cardinality") and self.s.mapping.numeric.get(a.field) == "bool" \
                and a.kind in ("range", "histogram"):
            raise UnsupportedQuery(f"{a.path}.{a.kind}.field", "range and histogram need a number field")
        return TermsAgg(a.name, a.field, int(a.param("size", 10)), a.kind, a.params,
                        tuple(self.agg(x) for x in a.sub))

    def sort_key(self, s: A.Sort) -> SortKey:
        if s.geo is not None:
            from ..geo import UNITS
            f, lat, lon, unit = s.geo
            self._field(f, f"{s.path}.{f}", ("geo",))
            return SortKey("_geo_distance", s.order == "desc", True, (f, lat, lon, UNITS[unit]))
        if s.field not in ("_score", "_doc"):
            self._field(s.field, s.path, ("keyword", "numeric"))
        order = s.order or ("desc" if s.field == "_score" else "asc")
        return SortKey(s.field, order == "desc", s.missing == "_last")


def plan(request, state) -> Plan:
    """A request (a dict, or a parsed ``Request``) → ``Plan`` for ``state``."""
    if isinstance(request, dict) or not isinstance(request, A.Request):
        request = parse(request)
    return Planner(state).request(request)
