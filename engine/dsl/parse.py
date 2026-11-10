"""The OpenSearch query DSL subset (docs/INTERFACES.md §5), parsed into a ``Request`` (``engine.dsl.ast``), §6.

Syntax only: keys, shapes and value types. No index is needed; field names and text are resolved by the planner.
Anything outside the subset raises ``UnsupportedQuery`` naming its JSON path (an unknown key, query type,
aggregation type or option); malformed input inside it raises ``InvalidRequest``. Nothing is silently ignored except
``_source`` and ``track_total_hits``, which do not change the hits (totals are always exact; serving adds
``_source``).
"""
from __future__ import annotations

import math
import re

from ..errors import InvalidRequest, UnsupportedQuery
from .ast import (Agg, QBool, QMatchPhrase, QHybrid, QKnn, QMatch, QMatchAll, QMultiMatch, QNeural, QRange, QTerm, Request, Sort)

MAX_WINDOW = 10_000
TOP_KEYS = {"query", "aggs", "aggregations", "sort", "from", "size", "search_after", "explain", "_source",
            "track_total_hits", "rank", "rescore"}
QUERY_TYPES = ("match", "multi_match", "bool", "term", "terms", "range", "knn", "match_all", "hybrid", "neural",
               "match_phrase")
MULTI_MATCH_TYPES = ("best_fields", "most_fields", "cross_fields")


def _int(v, path, lo=0):
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise InvalidRequest(path, f"expected an int >= {lo}")
    return v


def _num(v, path):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise InvalidRequest(path, "expected a finite number")
    return float(v)


def _obj(d, path):
    if not isinstance(d, dict):
        raise InvalidRequest(path, "expected an object")
    return d


def _keys(d, allowed, path):
    _obj(d, path)
    for k in d:
        if k not in allowed:
            raise UnsupportedQuery(f"{path}.{k}", f"{k!r} is not supported here")


def _one_field(body, path, what):
    _obj(body, path)
    if len(body) != 1:
        raise InvalidRequest(path, f"{what} takes exactly one field")
    (field, spec), = body.items()
    if not isinstance(field, str) or not field:
        raise InvalidRequest(path, "a field name")
    return field, spec


def _msm(v, path):
    if v is None:
        return None
    if isinstance(v, bool) or not (isinstance(v, int) or (isinstance(v, str) and re.fullmatch(r"\s*-?\d+%?\s*", v))):
        raise UnsupportedQuery(path, f"minimum_should_match {v!r}: ints and percentages only")
    return v


def resolve_msm(spec, n: int, path: str = "minimum_should_match") -> int:
    """OpenSearch ``minimum_should_match`` for ``n`` optional clauses: int, negative int, "x%", "-x%"; 1 if None."""
    if spec is None:
        return 1
    if isinstance(spec, int):
        k = spec if spec >= 0 else n + spec
    else:
        s = spec.strip()
        if s.endswith("%"):
            p = int(s[:-1])
            k = math.floor(n * p / 100) if p >= 0 else n - math.floor(n * -p / 100)
        else:
            k = int(s) if int(s) >= 0 else n + int(s)
    return max(0, min(n, k))


def _scalar(v, path):
    if isinstance(v, (str, int, float, bool)) and not (isinstance(v, float) and not math.isfinite(v)):
        return v
    raise InvalidRequest(path, "a string, number or boolean")


class Parser:
    # ---- the request
    def request(self, req: dict) -> Request:
        if not isinstance(req, dict):
            raise InvalidRequest("$", "the request is an object")
        for k in req:
            if k not in TOP_KEYS:
                raise UnsupportedQuery(k, f"top-level key {k!r} is not supported")
        query = self.query(req["query"], "query") if "query" in req else QMatchAll(path="query")
        rrf = None
        if "rank" in req:
            _keys(req["rank"], {"rrf"}, "rank")
            rr = req["rank"].get("rrf")
            rr = {} if rr is None else rr
            _keys(rr, {"rank_constant", "window_size"}, "rank.rrf")
            rrf = (_int(rr.get("rank_constant", 60), "rank.rrf.rank_constant", 1),
                   _int(rr.get("window_size", 20_000), "rank.rrf.window_size", 1))
        if "aggs" in req and "aggregations" in req:
            raise InvalidRequest("aggs", "give aggs or aggregations, not both")
        akey = "aggs" if "aggs" in req else "aggregations"
        aggs = self.aggs(req.get(akey, {}), akey)
        from_ = _int(req.get("from", 0), "from")
        size = _int(req.get("size", 10), "size")
        if from_ + size > MAX_WINDOW:
            raise InvalidRequest("from", f"from + size must be <= {MAX_WINDOW} (use search_after)")
        sort = self.sort(req["sort"], "sort") if "sort" in req else None
        rescore = self.rescore(req["rescore"]) if "rescore" in req else None
        after = None
        if "search_after" in req:
            sa = req["search_after"]
            if not isinstance(sa, list) or not sa:
                raise InvalidRequest("search_after", "the last hit's sort list")
            if from_ != 0:
                raise InvalidRequest("from", "search_after needs from 0")
            after = tuple(sa)
        explain = req.get("explain", False)
        if not isinstance(explain, bool):
            raise InvalidRequest("explain", "a bool")
        if not isinstance(req.get("track_total_hits", True), (bool, int)):
            raise InvalidRequest("track_total_hits", "a bool or an int (totals are always exact)")
        return Request(query=query, aggs=aggs, sort=sort, from_=from_, size=size, search_after=after,
                       explain=explain, rrf=rrf, rescore=rescore)

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
        return QMatchAll(_num(body.get("boost", 1.0), f"{path}.boost"), path)

    def _text(self, v, path):
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise InvalidRequest(path, "a string")
        return v if isinstance(v, str) else str(v)

    def _operator(self, v, path):
        if not isinstance(v, str) or v.lower() not in ("or", "and"):
            raise InvalidRequest(path, "or | and")
        return v.lower()

    def q_match(self, body, path):
        field, spec = _one_field(body, path, "match")
        p = f"{path}.{field}"
        if not isinstance(spec, dict):
            return QMatch(field, _scalar(spec, p), path=p)
        _keys(spec, {"query", "operator", "minimum_should_match", "boost"}, p)
        if "query" not in spec:
            raise InvalidRequest(f"{p}.query", "required")
        return QMatch(field, _scalar(spec["query"], f"{p}.query"), self._operator(spec.get("operator", "or"),
                                                                                    f"{p}.operator"),
                      _msm(spec.get("minimum_should_match"), f"{p}.minimum_should_match"),
                      _num(spec.get("boost", 1.0), f"{p}.boost"), p)

    def q_match_phrase(self, body, path):
        field, spec = _one_field(body, path, "match_phrase")
        p = f"{path}.{field}"
        if not isinstance(spec, dict):
            return QMatchPhrase(field, _scalar(spec, p), path=p)
        _keys(spec, {"query", "slop", "boost"}, p)
        if "query" not in spec:
            raise InvalidRequest(f"{p}.query", "required")
        return QMatchPhrase(field, _scalar(spec["query"], f"{p}.query"), _int(spec.get("slop", 0), f"{p}.slop"),
                            _num(spec.get("boost", 1.0), f"{p}.boost"), p)

    def q_multi_match(self, body, path):
        _keys(body, {"query", "fields", "type", "operator", "minimum_should_match", "tie_breaker", "boost"}, path)
        t = body.get("type", "best_fields")
        if t not in MULTI_MATCH_TYPES:
            raise UnsupportedQuery(f"{path}.type", f"type {t!r}: {', '.join(MULTI_MATCH_TYPES)}")
        if "query" not in body:
            raise InvalidRequest(f"{path}.query", "required")
        fields = None
        if "fields" in body:
            fl = body["fields"]
            if isinstance(fl, str):
                fl = [fl]
            if not isinstance(fl, list) or not fl:
                raise InvalidRequest(f"{path}.fields", "a non-empty list")
            out = []
            for i, f in enumerate(fl):
                m = re.fullmatch(r"([^\^*?]+)(?:\^([0-9]*\.?[0-9]+))?", f) if isinstance(f, str) else None
                if not m:
                    raise UnsupportedQuery(f"{path}.fields[{i}]", f"{f!r}: field names with an optional ^boost "
                                                                  "(no wildcards)")
                out.append((m.group(1), float(m.group(2)) if m.group(2) else None))
            fields = tuple(out)
        tb = body.get("tie_breaker")
        return QMultiMatch(self._text(body["query"], f"{path}.query"), fields, t,
                           self._operator(body.get("operator", "or"), f"{path}.operator"),
                           _msm(body.get("minimum_should_match"), f"{path}.minimum_should_match"),
                           None if tb is None else _num(tb, f"{path}.tie_breaker"),
                           _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_bool(self, body, path):
        _keys(body, {"must", "should", "filter", "must_not", "minimum_should_match", "boost"}, path)
        parts = {}
        for occ in ("must", "should", "filter", "must_not"):
            v = body.get(occ, [])
            if isinstance(v, dict):
                parts[occ] = (self.query(v, f"{path}.{occ}"),)
                continue
            if not isinstance(v, list):
                raise InvalidRequest(f"{path}.{occ}", "a query or a list of queries")
            parts[occ] = tuple(self.query(c, f"{path}.{occ}[{i}]") for i, c in enumerate(v))
        return QBool(parts["must"], parts["should"], parts["filter"], parts["must_not"],
                     _msm(body.get("minimum_should_match"), f"{path}.minimum_should_match"),
                     _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_term(self, body, path):
        _obj(body, path)
        boost = 1.0
        field, spec = _one_field(body, path, "term")
        p = f"{path}.{field}"
        if isinstance(spec, dict):
            _keys(spec, {"value", "boost"}, p)
            if "value" not in spec:
                raise InvalidRequest(f"{p}.value", "required")
            value, boost = spec["value"], _num(spec.get("boost", 1.0), f"{p}.boost")
        else:
            value = spec
        return QTerm(field, (_scalar(value, p),), False, boost, path)

    def q_terms(self, body, path):
        _obj(body, path)
        fields = [k for k in body if k != "boost"]
        if len(fields) != 1:
            raise InvalidRequest(path, "terms takes exactly one field")
        field = fields[0]
        values = body[field]
        if isinstance(values, dict):
            raise UnsupportedQuery(f"{path}.{field}", "terms lookup is not supported; give a list of values")
        if not isinstance(values, list):
            raise InvalidRequest(f"{path}.{field}", "a list of values")
        vals = tuple(_scalar(v, f"{path}.{field}[{i}]") for i, v in enumerate(values))
        return QTerm(field, vals, True, _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_range(self, body, path):
        field, spec = _one_field(body, path, "range")
        p = f"{path}.{field}"
        _keys(spec, {"gte", "gt", "lte", "lt", "boost"}, p)
        if ("gte" in spec and "gt" in spec) or ("lte" in spec and "lt" in spec):
            raise InvalidRequest(p, "at most one lower and one upper bound")
        b = {k: (None if spec.get(k) is None else _scalar(spec[k], f"{p}.{k}")) for k in ("gte", "gt", "lte", "lt")}
        return QRange(field, b["gte"], b["gt"], b["lte"], b["lt"], _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_knn(self, body, path):
        field, spec = _one_field(body, path, "knn")
        p = f"{path}.{field}"
        _keys(spec, {"vector", "k", "filter", "boost"}, p)
        vec = spec.get("vector")
        if not isinstance(vec, list) or not vec or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                                       for x in vec):
            raise InvalidRequest(f"{p}.vector", "a non-empty list of numbers")
        flt = self.query(spec["filter"], f"{p}.filter") if "filter" in spec else None
        return QKnn(field, tuple(float(x) for x in vec), _int(spec.get("k", 10), f"{p}.k", 1), flt,
                    _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_neural(self, body, path):
        field, spec = _one_field(body, path, "neural")
        p = f"{path}.{field}"
        _keys(spec, {"query_text", "k", "filter", "boost", "model_id"}, p)
        if not isinstance(spec.get("query_text"), str):
            raise InvalidRequest(f"{p}.query_text", "a string")
        flt = self.query(spec["filter"], f"{p}.filter") if "filter" in spec else None
        return QNeural(field, spec["query_text"], _int(spec.get("k", 10), f"{p}.k", 1), flt,
                       _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_hybrid(self, body, path):
        _keys(body, {"queries"}, path)
        qs = body.get("queries")
        if not isinstance(qs, list) or not qs:
            raise InvalidRequest(f"{path}.queries", "a non-empty list")
        return QHybrid(tuple(self.query(q, f"{path}.queries[{i}]") for i, q in enumerate(qs)), path)

    # ---- aggregations, sort, rescore
    def aggs(self, body, path):
        _obj(body, path)
        out = []
        for name, spec in body.items():
            p = f"{path}.{name}"
            _obj(spec, p)
            for k in spec:
                if k != "terms":
                    raise UnsupportedQuery(f"{p}.{k}", "only terms aggregations, not nested"
                                           if k in ("aggs", "aggregations") else f"aggregation {k!r} is not supported")
            if "terms" not in spec:
                raise InvalidRequest(p, "an aggregation needs a type (terms)")
            t = spec["terms"]
            _keys(t, {"field", "size"}, f"{p}.terms")
            if not isinstance(t.get("field"), str):
                raise InvalidRequest(f"{p}.terms.field", "required: a field name")
            out.append(Agg(name, t["field"], _int(t.get("size", 10), f"{p}.terms.size", 1), p))
        return tuple(out)

    def sort(self, body, path):
        items = body if isinstance(body, list) else [body]
        out = []
        for i, it in enumerate(items):
            p = f"{path}[{i}]"
            if isinstance(it, str):
                name, spec = it, {}
            elif isinstance(it, dict) and len(it) == 1:
                (name, spec), = it.items()
                spec = {"order": spec} if isinstance(spec, str) else spec
                _keys(spec, {"order", "missing"}, f"{p}.{name}")
            else:
                raise InvalidRequest(p, "a field name or {field: {order}}")
            order = spec.get("order")
            if order is not None and order not in ("asc", "desc"):
                raise InvalidRequest(f"{p}.{name}.order", "asc | desc")
            missing = spec.get("missing", "_last")
            if missing not in ("_last", "_first"):
                raise UnsupportedQuery(f"{p}.{name}.missing", "_last | _first (a custom missing value is not "
                                                              "supported)")
            out.append(Sort(name, order, missing, p))
        return tuple(out)

    def rescore(self, body):
        _keys(body, {"window_size", "s3_ranker"}, "rescore")
        r = body.get("s3_ranker")
        if r is None:
            raise UnsupportedQuery("rescore", "rescore needs s3_ranker: {model} (query rescorers are not supported)")
        _keys(r, {"model"}, "rescore.s3_ranker")
        if not isinstance(r.get("model"), str):
            raise InvalidRequest("rescore.s3_ranker.model", "a model name")
        return (_int(body.get("window_size", 1000), "rescore.window_size", 1), r["model"])


def parse(request: dict, schema=None) -> Request:
    """``request`` → ``Request``. ``schema`` is accepted for the S1 signature and ignored: parsing needs no index."""
    return Parser().request(request)
