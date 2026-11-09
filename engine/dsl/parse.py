"""The OpenSearch query DSL subset (docs/INTERFACES.md §5) parsed into a ``Plan`` (§6).

S1 parses and plans in one pass: the parser needs the index's field kinds to choose nodes (a ``match`` on a keyword
field is a term). Anything outside the subset raises ``UnsupportedQuery`` with its JSON path; malformed input inside it
raises ``InvalidRequest``.
"""
from __future__ import annotations

import math
import re

from ..analysis import analyze
from ..errors import InvalidRequest, UnsupportedQuery
from ..plan.nodes import Bool, Dense, Hybrid, Lexical, MatchAll, Plan, Range, Rescore, SortKey, Term, TermsAgg

MAX_WINDOW = 10_000
TOP_KEYS = {"query", "aggs", "aggregations", "sort", "from", "size", "search_after", "explain", "_source",
            "track_total_hits", "rank", "rescore"}
QUERY_TYPES = {"match", "multi_match", "bool", "term", "terms", "range", "knn", "match_all", "hybrid", "neural"}


def _int(v, path, lo=0):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise InvalidRequest(path, f"expected an int >= {lo}")
    return v


def _num(v, path):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise InvalidRequest(path, "expected a number")
    return float(v)


def _keys(d, allowed, path):
    if not isinstance(d, dict):
        raise InvalidRequest(path, "expected an object")
    for k in d:
        if k not in allowed:
            raise UnsupportedQuery(f"{path}.{k}", f"{k!r} is not supported here")


def resolve_msm(spec, n: int, path: str) -> int:
    """OpenSearch ``minimum_should_match`` for ``n`` optional clauses: int, negative int, "x%", "-x%"."""
    if spec is None:
        return 1
    if isinstance(spec, bool):
        raise InvalidRequest(path, "minimum_should_match: an int or a percentage")
    if isinstance(spec, int):
        k = spec if spec >= 0 else n + spec
    elif isinstance(spec, str) and re.fullmatch(r"-?\d+%?", spec.strip()):
        s = spec.strip()
        if s.endswith("%"):
            p = int(s[:-1])
            k = math.floor(n * p / 100) if p >= 0 else n - math.floor(n * -p / 100)
        else:
            k = int(s) if int(s) >= 0 else n + int(s)
    else:
        raise UnsupportedQuery(path, f"minimum_should_match {spec!r}: ints and percentages only")
    return max(0, min(n, k))


class Parser:
    def __init__(self, schema):
        self.s = schema       # IndexState: .mapping.kind(), .field_index, .text_fields, .vector_spec, .encoders

    # ---- the request
    def request(self, req: dict) -> Plan:
        if not isinstance(req, dict):
            raise InvalidRequest("$", "the request is an object")
        for k in req:
            if k not in TOP_KEYS:
                raise UnsupportedQuery(k, f"top-level key {k!r} is not supported")
        root = self.query(req.get("query", {"match_all": {}}), "query")
        if "rank" in req:
            _keys(req["rank"], {"rrf"}, "rank")
            if not isinstance(root, Hybrid):
                raise InvalidRequest("rank", "rank.rrf needs a hybrid query")
            rr = req["rank"]["rrf"] or {}
            _keys(rr, {"rank_constant", "window_size"}, "rank.rrf")
            root = Hybrid(root.queries, _int(rr.get("rank_constant", 60), "rank.rrf.rank_constant", 1),
                          _int(rr.get("window_size", root.window), "rank.rrf.window_size", 1), root.path)
        if "aggs" in req and "aggregations" in req:
            raise InvalidRequest("aggs", "give aggs or aggregations, not both")
        aggs = self.aggs(req.get("aggs", req.get("aggregations", {})), "aggs")
        from_ = _int(req.get("from", 0), "from")
        size = _int(req.get("size", 10), "size")
        if from_ + size > MAX_WINDOW:
            raise InvalidRequest("from", f"from + size must be <= {MAX_WINDOW} (use search_after)")
        rescore = self.rescore(req["rescore"]) if "rescore" in req else None
        sort = self.sort(req.get("sort"), "sort")
        after = None
        if "search_after" in req:
            sa = req["search_after"]
            n_keys = len(sort) + (1 if rescore else 0)
            if not isinstance(sa, list) or len(sa) != n_keys + 1:
                raise InvalidRequest("search_after", f"expected the last hit's sort list ({n_keys + 1} values)")
            if from_ not in (0,):
                raise InvalidRequest("from", "search_after needs from 0")
            after = tuple(sa)
        explain = req.get("explain", False)
        if not isinstance(explain, bool):
            raise InvalidRequest("explain", "a bool")
        return Plan(root=root, aggs=aggs, sort=sort, from_=from_, size=size, search_after=after, rescore=rescore,
                    explain=explain)

    # ---- queries
    def query(self, q, path):
        if not isinstance(q, dict) or len(q) != 1:
            raise InvalidRequest(path, "a query is an object with exactly one key")
        (kind, body), = q.items()
        if kind not in QUERY_TYPES:
            raise UnsupportedQuery(f"{path}.{kind}", f"query type {kind!r} is not supported")
        return getattr(self, "q_" + kind)(body, f"{path}.{kind}")

    def q_match_all(self, body, path):
        _keys(body, {"boost"}, path)
        return MatchAll(_num(body.get("boost", 1.0), f"{path}.boost"), path)

    def _field(self, name, path, kinds):
        k = self.s.mapping.kind(name)
        if k is None:
            raise InvalidRequest(path, f"unknown field {name!r}")
        if k not in kinds:
            raise UnsupportedQuery(path, f"field {name!r} is {k}; this needs {'/'.join(kinds)}")
        return k

    def _lexical(self, text, fields, operator, msm, boost, path):
        if not isinstance(text, str):
            if isinstance(text, (int, float)) and not isinstance(text, bool):
                text = str(text)
            else:
                raise InvalidRequest(f"{path}.query", "a string")
        if operator not in ("or", "and", "OR", "AND"):
            raise InvalidRequest(f"{path}.operator", "or | and")
        terms = tuple(dict.fromkeys(analyze(text)))
        n = len(terms)
        need = n if operator.lower() == "and" else max(1, resolve_msm(msm, n, f"{path}.minimum_should_match"))
        order = self.s.field_index
        fw = tuple(sorted(fields.items(), key=lambda x: order[x[0]]))
        idf = "field" if sum(1 for _, w in fw if w > 0) == 1 else "global"
        return Lexical(fw, terms, min(need, n), idf, boost, path)

    def q_match(self, body, path):
        if not isinstance(body, dict) or len(body) != 1:
            raise InvalidRequest(path, "match takes one field")
        (field, spec), = body.items()
        if not isinstance(spec, dict):
            spec = {"query": spec}
        _keys(spec, {"query", "operator", "minimum_should_match", "boost"}, f"{path}.{field}")
        if "query" not in spec:
            raise InvalidRequest(f"{path}.{field}.query", "required")
        kind = self._field(field, f"{path}.{field}", ("text", "keyword", "numeric"))
        boost = _num(spec.get("boost", 1.0), f"{path}.{field}.boost")
        if kind != "text":
            return Term(field, (spec["query"],), boost, path)
        return self._lexical(spec["query"], {field: 1.0}, spec.get("operator", "or"),
                             spec.get("minimum_should_match"), boost, f"{path}.{field}")

    def q_multi_match(self, body, path):
        _keys(body, {"query", "fields", "type", "operator", "minimum_should_match", "tie_breaker", "boost"}, path)
        if body.get("type", "best_fields") not in ("best_fields", "most_fields", "cross_fields"):
            raise UnsupportedQuery(f"{path}.type", f"type {body.get('type')!r}: best_fields, most_fields, cross_fields")
        if "query" not in body:
            raise InvalidRequest(f"{path}.query", "required")
        fields = body.get("fields") or list(self.s.text_fields)
        if not isinstance(fields, list):
            raise InvalidRequest(f"{path}.fields", "a list")
        weights = {}
        for i, f in enumerate(fields):
            m = re.fullmatch(r"([^\^*]+)(?:\^([0-9.]+))?", str(f))
            if not m:
                raise UnsupportedQuery(f"{path}.fields[{i}]", f"{f!r}: field names with an optional ^boost")
            self._field(m.group(1), f"{path}.fields[{i}]", ("text",))
            weights[m.group(1)] = float(m.group(2)) if m.group(2) else 1.0
        return self._lexical(body["query"], weights, body.get("operator", "or"), body.get("minimum_should_match"),
                             _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_bool(self, body, path):
        _keys(body, {"must", "should", "filter", "must_not", "minimum_should_match", "boost"}, path)
        parts = {}
        for occ in ("must", "should", "filter", "must_not"):
            v = body.get(occ, [])
            v = [v] if isinstance(v, dict) else v
            if not isinstance(v, list):
                raise InvalidRequest(f"{path}.{occ}", "a query or a list of queries")
            parts[occ] = tuple(self.query(c, f"{path}.{occ}[{i}]") for i, c in enumerate(v))
        msm = None
        if "minimum_should_match" in body:
            msm = resolve_msm(body["minimum_should_match"], len(parts["should"]), f"{path}.minimum_should_match")
        return Bool(parts["must"], parts["should"], parts["filter"], parts["must_not"], msm,
                    _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_term(self, body, path):
        if not isinstance(body, dict) or len(body) != 1:
            raise InvalidRequest(path, "term takes one field")
        (field, spec), = body.items()
        if isinstance(spec, dict):
            _keys(spec, {"value", "boost"}, f"{path}.{field}")
            if "value" not in spec:
                raise InvalidRequest(f"{path}.{field}.value", "required")
            value, boost = spec["value"], _num(spec.get("boost", 1.0), f"{path}.{field}.boost")
        else:
            value, boost = spec, 1.0
        self._field(field, f"{path}.{field}", ("keyword", "numeric"))
        return Term(field, (value,), boost, path)

    def q_terms(self, body, path):
        if not isinstance(body, dict):
            raise InvalidRequest(path, "an object")
        boost = _num(body.get("boost", 1.0), f"{path}.boost")
        fields = [k for k in body if k != "boost"]
        if len(fields) != 1:
            raise InvalidRequest(path, "terms takes one field")
        field = fields[0]
        values = body[field]
        if not isinstance(values, list):
            raise UnsupportedQuery(f"{path}.{field}", "a list of values (terms lookup is not supported)")
        self._field(field, f"{path}.{field}", ("keyword", "numeric"))
        return Term(field, tuple(values), boost, path)

    def q_range(self, body, path):
        if not isinstance(body, dict) or len(body) != 1:
            raise InvalidRequest(path, "range takes one field")
        (field, spec), = body.items()
        _keys(spec, {"gte", "gt", "lte", "lt", "boost"}, f"{path}.{field}")
        kind = self._field(field, f"{path}.{field}", ("keyword", "numeric"))
        if kind == "numeric" and self.s.mapping.numeric[field] == "bool":
            raise UnsupportedQuery(f"{path}.{field}", "range on a boolean field")
        conv = (lambda v, p: _num(v, p)) if kind == "numeric" else (lambda v, p: str(v))
        lo = hi = None
        lo_inc = hi_inc = True
        if "gte" in spec and "gt" in spec or "lte" in spec and "lt" in spec:
            raise InvalidRequest(f"{path}.{field}", "give one lower and one upper bound at most")
        if "gte" in spec:
            lo = conv(spec["gte"], f"{path}.{field}.gte")
        if "gt" in spec:
            lo, lo_inc = conv(spec["gt"], f"{path}.{field}.gt"), False
        if "lte" in spec:
            hi = conv(spec["lte"], f"{path}.{field}.lte")
        if "lt" in spec:
            hi, hi_inc = conv(spec["lt"], f"{path}.{field}.lt"), False
        return Range(field, lo, hi, lo_inc, hi_inc, _num(spec.get("boost", 1.0), f"{path}.{field}.boost"), path)

    def q_knn(self, body, path):
        if not isinstance(body, dict) or len(body) != 1:
            raise InvalidRequest(path, "knn takes one field")
        (field, spec), = body.items()
        p = f"{path}.{field}"
        _keys(spec, {"vector", "k", "filter", "boost"}, p)
        self._field(field, p, ("vector",))
        if field not in self.s.vectors:
            raise UnsupportedQuery(p, f"field {field!r} has no vectors in this index")
        vec = spec.get("vector")
        if not isinstance(vec, list) or len(vec) != self.s.vector_spec[field]["dim"]:
            raise InvalidRequest(f"{p}.vector", f"a list of {self.s.vector_spec[field]['dim']} numbers")
        k = _int(spec.get("k", 10), f"{p}.k", 1)
        flt = self.query(spec["filter"], f"{p}.filter") if "filter" in spec else None
        return Dense(field, tuple(float(x) for x in vec), None, k, flt, _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_neural(self, body, path):
        if not isinstance(body, dict) or len(body) != 1:
            raise InvalidRequest(path, "neural takes one field")
        (field, spec), = body.items()
        p = f"{path}.{field}"
        _keys(spec, {"query_text", "k", "filter", "boost", "model_id"}, p)
        self._field(field, p, ("vector",))
        if field not in self.s.encoders:
            raise UnsupportedQuery(p, f"field {field!r} has no encoder in this index")
        if not isinstance(spec.get("query_text"), str):
            raise InvalidRequest(f"{p}.query_text", "a string")
        k = _int(spec.get("k", 10), f"{p}.k", 1)
        flt = self.query(spec["filter"], f"{p}.filter") if "filter" in spec else None
        return Dense(field, None, spec["query_text"], k, flt, _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_hybrid(self, body, path):
        _keys(body, {"queries"}, path)
        qs = body.get("queries")
        if not isinstance(qs, list) or not qs:
            raise InvalidRequest(f"{path}.queries", "a non-empty list")
        return Hybrid(tuple(self.query(q, f"{path}.queries[{i}]") for i, q in enumerate(qs)), path=path)

    # ---- aggregations, sort, rescore
    def aggs(self, body, path):
        if not isinstance(body, dict):
            raise InvalidRequest(path, "an object")
        out = []
        for name, spec in body.items():
            p = f"{path}.{name}"
            if not isinstance(spec, dict) or set(spec) != {"terms"}:
                bad = next((k for k in (spec if isinstance(spec, dict) else {}) if k != "terms"), None)
                raise UnsupportedQuery(f"{p}.{bad}" if bad else p, "only terms aggregations, not nested")
            t = spec["terms"]
            _keys(t, {"field", "size"}, f"{p}.terms")
            field = t.get("field")
            self._field(field, f"{p}.terms.field", ("keyword", "numeric"))
            out.append(TermsAgg(name, field, _int(t.get("size", 10), f"{p}.terms.size", 1)))
        return tuple(out)

    def sort(self, body, path):
        if body is None:
            return (SortKey("_score", True),)
        items = body if isinstance(body, list) else [body]
        out = []
        for i, it in enumerate(items):
            p = f"{path}[{i}]"
            if isinstance(it, str):
                name, spec = it, {}
            elif isinstance(it, dict) and len(it) == 1:
                (name, spec), = it.items()
                spec = {"order": spec} if isinstance(spec, str) else spec
                _keys(spec, {"order", "missing"}, p)
            else:
                raise InvalidRequest(p, "a field name or {field: {order}}")
            if name not in ("_score", "_doc"):
                self._field(name, p, ("keyword", "numeric"))
            default = "desc" if name == "_score" else "asc"
            order = spec.get("order", default)
            if order not in ("asc", "desc"):
                raise InvalidRequest(f"{p}.order", "asc | desc")
            missing = spec.get("missing", "_last")
            if missing not in ("_last", "_first"):
                raise UnsupportedQuery(f"{p}.missing", "_last | _first")
            out.append(SortKey(name, order == "desc", missing == "_last"))
        return tuple(out)

    def rescore(self, body):
        _keys(body, {"window_size", "s3_ranker"}, "rescore")
        r = body.get("s3_ranker")
        if not isinstance(r, dict) or not isinstance(r.get("model"), str):
            raise UnsupportedQuery("rescore", "rescore needs s3_ranker: {model}")
        return Rescore(_int(body.get("window_size", 1000), "rescore.window_size", 1), r["model"])


def parse(request: dict, schema) -> Plan:
    return Parser(schema).request(request)
