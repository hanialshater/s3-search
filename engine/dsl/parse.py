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
from .ast import (Agg, Fuzzy, Highlight, QNested, QGeoBox, QGeoDistance, QFunctionScore, QBool, QExists, QIds, QFuzzy, QMatchBoolPrefix, QMatchPhrase, QPrefix, QHybrid, QKnn, QMatch, QMatchAll, QMultiMatch, QNeural, QRange, QTerm, Request, Sort)

MAX_WINDOW = 10_000
TOP_KEYS = {"query", "aggs", "aggregations", "sort", "from", "size", "search_after", "explain", "_source",
            "post_filter", "collapse", "highlight",
            "track_total_hits", "rank", "rescore"}
QUERY_TYPES = ("match", "multi_match", "bool", "term", "terms", "range", "knn", "match_all", "hybrid", "neural",
               "match_phrase", "fuzzy", "prefix", "match_bool_prefix", "exists", "ids",
               "function_score", "geo_distance", "geo_bounding_box", "nested")
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


FUZZY_KEYS = {"fuzziness", "prefix_length", "max_expansions", "transpositions"}


def _fuzzy_opts(spec, p, default=None):
    fz = spec.get("fuzziness", default)
    if fz is not None and not (isinstance(fz, (str, int)) and not isinstance(fz, bool)):
        raise InvalidRequest(f"{p}.fuzziness", "AUTO, AUTO:low,high, 0, 1 or 2")
    tr = spec.get("transpositions", True)
    if not isinstance(tr, bool):
        raise InvalidRequest(f"{p}.transpositions", "a bool")
    return Fuzzy(fz, _int(spec.get("prefix_length", 0), f"{p}.prefix_length"),
                 _int(spec.get("max_expansions", 50), f"{p}.max_expansions", 1), tr)


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
        post = self.query(req["post_filter"], "post_filter") if "post_filter" in req else None
        collapse = None
        if "collapse" in req:
            _keys(req["collapse"], {"field"}, "collapse")
            if not isinstance(req["collapse"].get("field"), str):
                raise InvalidRequest("collapse.field", "required: a field name")
            collapse = req["collapse"]["field"]
        hl = self.highlight(req["highlight"]) if "highlight" in req else None
        return Request(query=query, aggs=aggs, sort=sort, from_=from_, size=size, search_after=after,
                       explain=explain, rrf=rrf, rescore=rescore, post_filter=post, collapse=collapse, highlight=hl)

    HL_KEYS = {"fragment_size", "number_of_fragments", "pre_tags", "post_tags", "type"}

    def highlight(self, body):
        _keys(body, {"fields", "require_field_match"} | self.HL_KEYS, "highlight")
        top = self._hl_opts(body, "highlight", (100, 5, "<em>", "</em>"))
        fields = body.get("fields")
        if isinstance(fields, list):
            items = []
            for i, x in enumerate(fields):
                if not isinstance(x, dict) or len(x) != 1:
                    raise InvalidRequest(f"highlight.fields[{i}]", "an object with one field")
                items += list(x.items())
        elif isinstance(fields, dict):
            items = list(fields.items())
        else:
            raise InvalidRequest("highlight.fields", "required: an object of fields")
        out = []
        for f, spec in items:
            p = f"highlight.fields.{f}"
            if "*" in f:
                raise UnsupportedQuery(p, "wildcard field names are not supported")
            spec = spec or {}
            _keys(spec, self.HL_KEYS, p)
            out.append((f,) + self._hl_opts(spec, p, top))
        rfm = body.get("require_field_match", True)
        if not isinstance(rfm, bool):
            raise InvalidRequest("highlight.require_field_match", "a bool")
        return Highlight(tuple(out), rfm)

    def _hl_opts(self, spec, p, default):
        if spec.get("type", "plain") != "plain":
            raise UnsupportedQuery(f"{p}.type", "only the plain highlighter")
        size = _int(spec.get("fragment_size", default[0]), f"{p}.fragment_size")
        n = _int(spec.get("number_of_fragments", default[1]), f"{p}.number_of_fragments")
        tags = []
        for k, d in (("pre_tags", default[2]), ("post_tags", default[3])):
            v = spec.get(k)
            if v is None:
                tags.append(d)
            elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                tags.append(v[0])
            else:
                raise InvalidRequest(f"{p}.{k}", "a non-empty list of strings")
        return (size, n, tags[0], tags[1])

    # ---- queries
    def query(self, q, path):
        if not isinstance(q, dict) or len(q) != 1:
            raise InvalidRequest(path, "a query is an object with exactly one key")
        (kind, body), = q.items()
        if kind not in QUERY_TYPES:
            raise UnsupportedQuery(f"{path}.{kind}", f"query type {kind!r} is not supported")
        return getattr(self, "q_" + kind)(body, f"{path}.{kind}")

    def q_nested(self, body, path):
        _keys(body, {"path", "query", "score_mode", "boost"}, path)
        if not isinstance(body.get("path"), str):
            raise InvalidRequest(f"{path}.path", "required: the nested field")
        if "query" not in body:
            raise InvalidRequest(f"{path}.query", "required")
        sm = body.get("score_mode", "avg")
        if sm not in ("avg", "sum", "max", "min", "none"):
            raise InvalidRequest(f"{path}.score_mode", "avg | sum | max | min | none")
        return QNested(body["path"], self.query(body["query"], f"{path}.query"), sm,
                       _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def _geo_field(self, body, path, reserved):
        from ..geo import BadGeo, parse_point  # noqa: F401
        _obj(body, path)
        fields = [k for k in body if k not in reserved]
        if len(fields) != 1:
            raise InvalidRequest(path, "exactly one geo_point field")
        return fields[0]

    def q_geo_distance(self, body, path):
        from ..geo import BadGeo, parse_distance, parse_point
        reserved = {"distance", "distance_type", "boost", "validation_method", "_name", "ignore_unmapped"}
        f = self._geo_field(body, path, reserved)
        for k in ("validation_method", "_name", "ignore_unmapped"):
            if k in body:
                raise UnsupportedQuery(f"{path}.{k}", f"{k!r} is not supported")
        if body.get("distance_type", "arc") != "arc":
            raise UnsupportedQuery(f"{path}.distance_type", "only arc distances")
        if "distance" not in body:
            raise InvalidRequest(f"{path}.distance", "required")
        try:
            lat, lon = parse_point(body[f])
            meters = parse_distance(body["distance"])
        except BadGeo as e:
            raise InvalidRequest(f"{path}.{f}", str(e)) from None
        return QGeoDistance(f, lat, lon, meters, _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_geo_bounding_box(self, body, path):
        from ..geo import BadGeo, parse_point
        reserved = {"boost", "validation_method", "type", "_name", "ignore_unmapped"}
        f = self._geo_field(body, path, reserved)
        for k in ("validation_method", "type", "_name", "ignore_unmapped"):
            if k in body:
                raise UnsupportedQuery(f"{path}.{k}", f"{k!r} is not supported")
        box = _obj(body[f], f"{path}.{f}")
        try:
            if {"top_left", "bottom_right"} == set(box):
                top, left = parse_point(box["top_left"])
                bottom, right = parse_point(box["bottom_right"])
            elif {"top_right", "bottom_left"} == set(box):
                top, right = parse_point(box["top_right"])
                bottom, left = parse_point(box["bottom_left"])
            elif {"top", "left", "bottom", "right"} == set(box):
                top, left, bottom, right = (float(box[k]) for k in ("top", "left", "bottom", "right"))
                parse_point({"lat": top, "lon": left}), parse_point({"lat": bottom, "lon": right})
            else:
                raise UnsupportedQuery(f"{path}.{f}", "top_left/bottom_right, top_right/bottom_left, or "
                                                      "top/left/bottom/right (no wkt)")
        except BadGeo as e:
            raise InvalidRequest(f"{path}.{f}", str(e)) from None
        if top < bottom:
            raise InvalidRequest(f"{path}.{f}", f"top {top} is below bottom {bottom}")
        return QGeoBox(f, top, left, bottom, right, _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_exists(self, body, path):
        _keys(body, {"field", "boost"}, path)
        if not isinstance(body.get("field"), str):
            raise InvalidRequest(f"{path}.field", "required: a field name")
        return QExists(body["field"], _num(body.get("boost", 1.0), f"{path}.boost"), path)

    def q_ids(self, body, path):
        _keys(body, {"values", "boost"}, path)
        vals = body.get("values")
        if not isinstance(vals, list):
            raise InvalidRequest(f"{path}.values", "a list of ids")
        return QIds(tuple(str(_scalar(v, f"{path}.values[{i}]")) for i, v in enumerate(vals)),
                    _num(body.get("boost", 1.0), f"{path}.boost"), path)

    SCORE_MODES = ("multiply", "sum", "avg", "first", "max", "min")
    BOOST_MODES = ("multiply", "replace", "sum", "avg", "max", "min")
    MODIFIERS = ("none", "log", "log1p", "log2p", "ln", "ln1p", "ln2p", "square", "sqrt", "reciprocal")

    def _function(self, f, p):
        _keys(f, {"filter", "field_value_factor", "weight"}, p)
        if "field_value_factor" not in f and "weight" not in f:
            raise InvalidRequest(p, "a function needs field_value_factor or weight")
        flt = self.query(f["filter"], f"{p}.filter") if "filter" in f else None
        fvf = None
        if "field_value_factor" in f:
            v = f["field_value_factor"]
            q = f"{p}.field_value_factor"
            _keys(v, {"field", "factor", "modifier", "missing"}, q)
            if not isinstance(v.get("field"), str):
                raise InvalidRequest(f"{q}.field", "required: a field name")
            mod = v.get("modifier", "none")
            if mod not in self.MODIFIERS:
                raise InvalidRequest(f"{q}.modifier", " | ".join(self.MODIFIERS))
            fvf = (v["field"], _num(v.get("factor", 1.0), f"{q}.factor"), mod,
                   None if v.get("missing") is None else _num(v["missing"], f"{q}.missing"))
        w = None if "weight" not in f else _num(f["weight"], f"{p}.weight")
        return (flt, fvf, w)

    def q_function_score(self, body, path):
        fn_keys = {"field_value_factor", "weight"}
        _keys(body, {"query", "functions", "score_mode", "boost_mode", "max_boost", "min_score", "boost"} | fn_keys,
              path)
        query = self.query(body["query"], f"{path}.query") if "query" in body else QMatchAll(path=f"{path}.query")
        if "functions" in body:
            if fn_keys & set(body):
                raise InvalidRequest(path, "give functions, or one function at the top level, not both")
            fl = body["functions"]
            if not isinstance(fl, list):
                raise InvalidRequest(f"{path}.functions", "a list of functions")
            functions = tuple(self._function(f, f"{path}.functions[{i}]") for i, f in enumerate(fl))
        else:
            top = {k: body[k] for k in fn_keys if k in body}
            functions = (self._function(top, path),) if top else ()
        sm, bm = body.get("score_mode", "multiply"), body.get("boost_mode", "multiply")
        if sm not in self.SCORE_MODES:
            raise InvalidRequest(f"{path}.score_mode", " | ".join(self.SCORE_MODES))
        if bm not in self.BOOST_MODES:
            raise InvalidRequest(f"{path}.boost_mode", " | ".join(self.BOOST_MODES))
        mb = None if body.get("max_boost") is None else _num(body["max_boost"], f"{path}.max_boost")
        ms = None if body.get("min_score") is None else _num(body["min_score"], f"{path}.min_score")
        return QFunctionScore(query, functions, sm, bm, mb, ms, _num(body.get("boost", 1.0), f"{path}.boost"), path)

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
        _keys(spec, {"query", "operator", "minimum_should_match", "boost"} | FUZZY_KEYS, p)
        if "query" not in spec:
            raise InvalidRequest(f"{p}.query", "required")
        return QMatch(field, _scalar(spec["query"], f"{p}.query"), self._operator(spec.get("operator", "or"),
                                                                                    f"{p}.operator"),
                      _msm(spec.get("minimum_should_match"), f"{p}.minimum_should_match"),
                      _num(spec.get("boost", 1.0), f"{p}.boost"), p, _fuzzy_opts(spec, p))

    def q_fuzzy(self, body, path):
        field, spec = _one_field(body, path, "fuzzy")
        p = f"{path}.{field}"
        if not isinstance(spec, dict):
            return QFuzzy(field, _scalar(spec, p), Fuzzy("AUTO"), 1.0, path)
        _keys(spec, {"value", "boost"} | FUZZY_KEYS, p)
        if "value" not in spec:
            raise InvalidRequest(f"{p}.value", "required")
        return QFuzzy(field, _scalar(spec["value"], f"{p}.value"), _fuzzy_opts(spec, p, "AUTO"),
                      _num(spec.get("boost", 1.0), f"{p}.boost"), path)

    def q_prefix(self, body, path):
        field, spec = _one_field(body, path, "prefix")
        p = f"{path}.{field}"
        if not isinstance(spec, dict):
            return QPrefix(field, str(_scalar(spec, p)), 1.0, path)
        _keys(spec, {"value", "boost"}, p)
        if "value" not in spec:
            raise InvalidRequest(f"{p}.value", "required")
        return QPrefix(field, str(_scalar(spec["value"], f"{p}.value")), _num(spec.get("boost", 1.0), f"{p}.boost"),
                       path)

    def q_match_bool_prefix(self, body, path):
        field, spec = _one_field(body, path, "match_bool_prefix")
        p = f"{path}.{field}"
        if not isinstance(spec, dict):
            return QMatchBoolPrefix(field, self._text(spec, p), path=p)
        _keys(spec, {"query", "operator", "minimum_should_match", "max_expansions", "boost"}, p)
        if "query" not in spec:
            raise InvalidRequest(f"{p}.query", "required")
        return QMatchBoolPrefix(field, self._text(spec["query"], f"{p}.query"),
                                self._operator(spec.get("operator", "or"), f"{p}.operator"),
                                _msm(spec.get("minimum_should_match"), f"{p}.minimum_should_match"),
                                _int(spec.get("max_expansions", 50), f"{p}.max_expansions", 1),
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
        _keys(body, {"query", "fields", "type", "operator", "minimum_should_match", "tie_breaker", "boost"}
              | FUZZY_KEYS, path)
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
                           _num(body.get("boost", 1.0), f"{path}.boost"), path, _fuzzy_opts(body, path))

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
    AGG_KEYS = {"terms": {"field", "size"}, "min": {"field"}, "max": {"field"}, "avg": {"field"}, "sum": {"field"},
                "stats": {"field"}, "value_count": {"field"}, "cardinality": {"field"},
                "range": {"field", "ranges"}, "histogram": {"field", "interval", "offset", "min_doc_count"},
                "nested": {"path"}}

    def aggs(self, body, path, depth=0):
        _obj(body, path)
        out = []
        for name, spec in body.items():
            p = f"{path}.{name}"
            _obj(spec, p)
            kinds = [k for k in spec if k not in ("aggs", "aggregations")]
            for k in kinds:
                if k not in self.AGG_KEYS:
                    raise UnsupportedQuery(f"{p}.{k}", f"aggregation {k!r} is not supported")
            if len(kinds) != 1:
                raise InvalidRequest(p, "an aggregation has exactly one type")
            kind = kinds[0]
            subkey = "aggs" if "aggs" in spec else "aggregations" if "aggregations" in spec else None
            if kind == "nested":                         # S9-5: a bucket of children; its sub-aggregations
                t = spec[kind]                           # see the children (terms under it may nest one level)
                _keys(t, {"path"}, f"{p}.nested")
                if not isinstance(t.get("path"), str):
                    raise InvalidRequest(f"{p}.nested.path", "required: the nested field")
                if depth > 0:
                    raise UnsupportedQuery(f"{p}.nested", "a nested aggregation at the top level only")
                sub = self.aggs(spec[subkey], f"{p}.{subkey}", 0) if subkey else ()
                out.append(Agg(name, t["path"], "nested", (), sub, p))
                continue
            if subkey is not None and (kind != "terms" or depth > 0):
                raise UnsupportedQuery(f"{p}.{subkey}", "sub-aggregations only under terms, one level")
            t = spec[kind]
            _keys(t, self.AGG_KEYS[kind], f"{p}.{kind}")
            if not isinstance(t.get("field"), str):
                raise InvalidRequest(f"{p}.{kind}.field", "required: a field name")
            params = {}
            if kind == "terms":
                params["size"] = _int(t.get("size", 10), f"{p}.terms.size", 1)
            elif kind == "range":
                rs = t.get("ranges")
                if not isinstance(rs, list) or not rs:
                    raise InvalidRequest(f"{p}.range.ranges", "a non-empty list of {from, to, key}")
                ranges = []
                for i, r in enumerate(rs):
                    _keys(r, {"from", "to", "key"}, f"{p}.range.ranges[{i}]")
                    lo = None if r.get("from") is None else _num(r["from"], f"{p}.range.ranges[{i}].from")
                    hi = None if r.get("to") is None else _num(r["to"], f"{p}.range.ranges[{i}].to")
                    key = r.get("key")
                    if key is not None and not isinstance(key, str):
                        raise InvalidRequest(f"{p}.range.ranges[{i}].key", "a string")
                    ranges.append((lo, hi, key))
                params["ranges"] = tuple(ranges)
            elif kind == "histogram":
                if "interval" not in t:
                    raise InvalidRequest(f"{p}.histogram.interval", "required")
                iv = _num(t["interval"], f"{p}.histogram.interval")
                if iv <= 0:
                    raise InvalidRequest(f"{p}.histogram.interval", "must be > 0")
                params.update(interval=iv, offset=_num(t.get("offset", 0.0), f"{p}.histogram.offset"),
                              min_doc_count=_int(t.get("min_doc_count", 0), f"{p}.histogram.min_doc_count"))
            sub = self.aggs(spec[subkey], f"{p}.{subkey}", depth + 1) if subkey else ()
            out.append(Agg(name, t["field"], kind, tuple(sorted(params.items())), sub, p))
        return tuple(out)

    def sort(self, body, path):
        items = body if isinstance(body, list) else [body]
        out = []
        for i, it in enumerate(items):
            p = f"{path}[{i}]"
            if isinstance(it, str):
                name, spec = it, {}
            elif isinstance(it, dict) and len(it) == 1 and "_geo_distance" in it:
                out.append(self._geo_sort(it["_geo_distance"], f"{p}._geo_distance"))
                continue
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

    def _geo_sort(self, spec, p):
        """``_geo_distance`` sort (S9-4): {field: point, "order", "unit", "distance_type": "arc"}."""
        from ..geo import UNITS, BadGeo, parse_point
        reserved = {"order", "unit", "distance_type", "mode", "ignore_unmapped", "validation_method"}
        _obj(spec, p)
        for k in ("mode", "ignore_unmapped", "validation_method"):
            if k in spec:
                raise UnsupportedQuery(f"{p}.{k}", f"{k!r} is not supported")
        if spec.get("distance_type", "arc") != "arc":
            raise UnsupportedQuery(f"{p}.distance_type", "only arc distances")
        fields = [k for k in spec if k not in reserved]
        if len(fields) != 1:
            raise InvalidRequest(p, "exactly one geo_point field")
        f = fields[0]
        pt = spec[f]
        if isinstance(pt, list) and pt and isinstance(pt[0], (list, dict, str)):
            if len(pt) != 1:
                raise UnsupportedQuery(f"{p}.{f}", "one point (several points are not supported)")
            pt = pt[0]
        try:
            lat, lon = parse_point(pt)
        except BadGeo as e:
            raise InvalidRequest(f"{p}.{f}", str(e)) from None
        unit = spec.get("unit", "m")
        if unit not in UNITS:
            raise InvalidRequest(f"{p}.unit", f"one of {sorted(set(UNITS))}")
        order = spec.get("order", "asc")
        if order not in ("asc", "desc"):
            raise InvalidRequest(f"{p}.order", "asc | desc")
        return Sort("_geo_distance", order, "_last", p, (f, lat, lon, unit))

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
