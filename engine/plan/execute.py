"""The batched executor (docs/INTERFACES.md §6): retrieve, filter, fuse, rank, facet, page.

Lexical leaves of every plan in a batch that share fields and idf mode are scored in one ``bm25f`` call; dense
leaves on one field share one matmul. The rest of each tree is elementwise work on ``[N]`` rows.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from ..errors import InvalidRequest, UnsupportedQuery
from .. import ops
from .. import ann
from .nodes import Bool, Dense, Exists, FunctionScore, GeoBox, GeoDistance, Nested, Ids, LexicalGroups, Phrase, Hybrid, Lexical, MatchAll, Plan, Range, Term

INF = float("inf")


def walk(node):
    yield node
    if isinstance(node, Bool):
        for c in node.must + node.should + node.filter + node.must_not:
            yield from walk(c)
    elif isinstance(node, Hybrid):
        for c in node.queries:
            yield from walk(c)
    elif isinstance(node, Dense) and node.filter is not None:
        yield from walk(node.filter)
    elif isinstance(node, FunctionScore):
        yield from walk(node.query)
        for flt, _, _ in node.functions:
            if flt is not None:
                yield from walk(flt)


def _modify(b, mod: str, x):
    """field_value_factor's modifier (Lucene's: log and its variants are base 10, ln natural)."""
    import math
    if mod == "none":
        return x
    if mod in ("log", "log1p", "log2p"):
        add = {"log": 0.0, "log1p": 1.0, "log2p": 2.0}[mod]
        return b.log1p(x + add - 1.0) / math.log(10.0)
    if mod in ("ln", "ln1p", "ln2p"):
        add = {"ln": 0.0, "ln1p": 1.0, "ln2p": 2.0}[mod]
        return b.log1p(x + add - 1.0)
    if mod == "square":
        return x * x
    if mod == "sqrt":
        return b.where(x >= 0, x, float("nan")) ** 0.5
    if mod == "reciprocal":
        return 1.0 / x
    raise ValueError(mod)



def _knn_methods(b, plan, outcome=None) -> dict:
    """S9-1, §5: ``{field: {"requested", "ran"}}`` for the plan's knn and neural clauses (with ``reason`` when an
    approximate method ran as exact), a list in request order when several share a field. ``outcome``: node ->
    (ran, reason) as executed."""
    out = {}
    roots = [plan.root] + ([plan.post_filter] if plan.post_filter is not None else [])
    for n in (m for r in roots for m in walk(r)):
        if isinstance(n, Dense):
            r, why = (outcome or {}).get(n) or ann.status(b, n.method)
            e = {"requested": n.method, "ran": r}
            if why is not None:
                e["reason"] = why
            if n.field not in out:
                out[n.field] = e
            elif isinstance(out[n.field], list):
                out[n.field].append(e)
            else:
                out[n.field] = [out[n.field], e]
    return out

class Timer:
    def __init__(self, b):
        self.b, self.laps, self.t0 = b, defaultdict(float), time.perf_counter()
        self.t = self.t0

    def lap(self, name):
        self.b.synchronize()
        now = time.perf_counter()
        self.laps[name] += (now - self.t) * 1000.0
        self.t = now

    def total(self):
        return (time.perf_counter() - self.t0) * 1000.0


class Executor:
    def __init__(self, st, rankers=None):
        self.st, self.b = st, st.b
        self.rankers = rankers if rankers is not None else {}
        self.knn_outcome = {}

    # ---- retrieval, batched across plans
    def _prefetch(self, plans, timer):
        st, b = self.st, self.b
        lex = defaultdict(list)
        dense = defaultdict(list)
        for p in plans:
            roots = [p.root] + ([p.post_filter] if p.post_filter is not None else [])
            for n in (m for r in roots for m in walk(r)):
                if isinstance(n, Lexical):
                    lex[(n.fields, n.idf)].append(n)
                elif isinstance(n, Dense):
                    dense[n.field].append(n)
        self.lex_cache, self.dense_cache, self.dense_query = {}, {}, {}
        for (fields, idf_mode), nodes in lex.items():
            nodes = list(dict.fromkeys(nodes))
            F = len(st.text_fields)
            w = np.zeros(F, np.float32)
            for f, wt in fields:
                w[st.field_index[f]] = wt
            bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)
            f_i = next(st.field_index[f] for f, wt in fields if wt > 0) if idf_mode == "field" else None
            idfs = [part.idf_field[:, f_i] if f_i is not None else part.idf for part in st.parts]
            terms_pp = [[[part.vocab[t] for t in n.terms if t in part.vocab] for n in nodes] for part in st.parts]
            if len(st.parts) == 1:
                scores, nmatch = ops.bm25f(b, st.parts[0], terms_pp[0], w, bb, st.mapping.k1, idfs[0])
            else:                                       # all segments' postings into one pass (S8-2c)
                scores, nmatch = ops.bm25f_multi(b, st.parts, terms_pp, w, bb, st.mapping.k1, idfs, st.N)
            for i, n in enumerate(nodes):
                self.lex_cache[n] = (scores[i], nmatch[i])
        timer.lap("retrieve")
        for field, nodes in dense.items():
            nodes = list(dict.fromkeys(nodes))
            vecs = []
            for n in nodes:
                if n.vector is not None:
                    vecs.append(np.asarray(n.vector, np.float32))
                else:
                    vecs.append(st.encoders[field].encode_queries([n.text])[0])
            q = np.stack(vecs)
            if st.vector_spec[field].get("space_type", "cosinesimil") == "cosinesimil":
                norm = np.linalg.norm(q, axis=1, keepdims=True)
                q = q / np.where(norm > 0, norm, 1.0)
            if any(n.text is not None for n in nodes):
                timer.lap("encode")
            qd = b.asarray(q, "float32")
            sims = None
            for part in st.parts:
                sp = ops.dense_scores(b, part.vectors[field], qd)
                if part.identity:
                    sims = sp
                    continue
                if sims is None:
                    sims = b.zeros((len(nodes), st.N), "float32")
                if len(part.live_local):
                    sims[:, b.asarray(part.live_slot, "int64")] = sp[:, b.asarray(part.live_local, "int64")]
            for i, n in enumerate(nodes):
                self.dense_cache[n] = sims[i]
                self.dense_query[n] = qd[i]
        if dense:
            timer.lap("retrieve")

    # ---- the tree
    def eval(self, node, ctx):
        """``(score f32[N], match bool[N])``; scores are 0 where match is false."""
        b, st = self.b, self.st
        N = st.N
        if isinstance(node, Lexical):
            s, nm = self.lex_cache[node]
            if node.msm == 0:
                m = b.full(N, False, "bool")
            else:
                m = nm >= node.msm
            out = (b.where(m, s * node.boost, 0.0), m)
            ctx.setdefault("lexical", (node, s, m))
        elif isinstance(node, Phrase):
            out = self._phrase(node)
        elif isinstance(node, LexicalGroups):
            out = self._groups(node)
        elif isinstance(node, MatchAll):
            out = (b.full(N, node.boost, "float32"), b.full(N, True, "bool"))
        elif isinstance(node, FunctionScore):
            out = self._function_score(node, ctx)
        elif isinstance(node, Nested):
            out = self._nested(node)
        elif isinstance(node, GeoDistance):
            from ..geo import haversine_m, lat_col, lon_col
            f = node.field
            d = haversine_m(b, st.numerics[lat_col(f)], st.numerics[lon_col(f)], node.lat, node.lon)
            m = ~st.missing[lat_col(f)] & (d <= node.meters)
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, GeoBox):
            from ..geo import lat_col, lon_col
            f = node.field
            la, lo = st.numerics[lat_col(f)], st.numerics[lon_col(f)]
            m = ~st.missing[lat_col(f)] & (la >= node.lat_lo) & (la <= node.lat_hi)
            if node.crosses:
                m = m & ((lo >= node.lon_lo) | (lo <= node.lon_hi))
            else:
                m = m & (lo >= node.lon_lo) & (lo <= node.lon_hi)
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, Exists):
            m = st.exists_mask(node.field)
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, Ids):
            m = b.full(N, False, "bool")
            slots = [st.pos[i] for i in node.values if i in st.pos]
            if slots:
                m[b.asarray(np.asarray(slots, np.int64), "int64")] = True
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, Term):
            m = self._term(node)
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, Range):
            m = self._range(node)
            out = (b.where(m, node.boost, 0.0), m)
        elif isinstance(node, Dense):
            sims = self.dense_cache[node]
            valid = st.live
            if node.filter is not None:
                valid = valid & self.eval(node.filter, ctx)[1]
            tried, why = ann.status(b, node.method)
            if tried != "exact":                           # S9-1: the docs cuVS finds, scored exactly
                try:
                    cand, hows = ann.candidates(b, st, node.field, node.method, dict(node.method_parameters),
                                                self.dense_query[node], node.k, valid)
                except ann.Fallback as e:                  # cuVS could not answer: exact, with the reason
                    tried, why = "exact", str(e)
                else:
                    valid = cand
                    tried, why = ann.label(node.method, hows)
            self.knn_outcome[node] = (tried, why)
            idx, _ = b.topk(sims.reshape(1, N), node.k, valid.reshape(1, N))
            idx = idx[0]
            idx = idx[idx >= 0]
            m = b.full(N, False, "bool")
            m[b.astype(idx, "int64")] = True
            if st.vector_spec[node.field].get("space_type", "cosinesimil") == "cosinesimil":
                sc = (1.0 + sims) / 2.0
            else:
                sc = b.where(sims >= 0, sims + 1.0, 1.0 / (1.0 - sims))
            out = (b.where(m, sc * node.boost, 0.0), m)
            ctx.setdefault("dense", (node, sims, m))
        elif isinstance(node, Bool):
            out = self._bool(node, ctx)
        elif isinstance(node, Hybrid):
            out = self._hybrid(node, ctx)
        else:  # pragma: no cover
            raise UnsupportedQuery(getattr(node, "path", "query"), f"no operator for {type(node).__name__}")
        if ctx.get("explain") is not None:
            ctx["explain"][id(node)] = out
        return out

    def _groups(self, node):
        b, st = self.b, self.st
        N, F = st.N, len(st.text_fields)
        if node.msm == 0 or not node.groups:
            return b.zeros(N, "float32"), b.full(N, False, "bool")
        w = np.zeros(F, np.float32)
        for f, wt in node.fields:
            w[st.field_index[f]] = wt
        bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)
        if node.idf == "field":
            f_i = next(st.field_index[f] for f, wt in node.fields if wt > 0)
            idf_h = st.host_idf_field[:, f_i]
        else:
            f_i, idf_h = None, st.host_idf
        group_idf = [min((float(idf_h[st.gid(t)]) for t in g if st.gid(t) is not None), default=0.0)
                     for g in node.groups]
        score = count = None
        for part in st.parts:
            pidf = part.idf_field[:, f_i] if node.idf == "field" else part.idf
            groups = [[part.vocab[t] for t in g if t in part.vocab] for g in node.groups]
            sc, present = ops.bm25f_groups(b, part, groups, list(node.modes), w, bb, st.mapping.k1, pidf, group_idf)
            sc = ops.sum_lr(sc, 0) if sc.shape[0] else b.zeros(st.N, "float32")
            score = sc if score is None else score + sc
            count = present if count is None else count | present
        m = b.sum(b.astype(count, "int32"), axis=0) >= node.msm
        return b.where(m, score * node.boost, 0.0), m

    def _phrase(self, node):
        from ..ops.phrase import phrase, phrase_scores
        b, st = self.b, self.st
        f_i = st.field_index[node.field]
        if not node.slots:
            return b.zeros(st.N, "float32"), b.full(st.N, False, "bool")
        freq = m = None
        for part in st.parts:
            slots = [[part.vocab[t] for t in alts if t in part.vocab] for alts in node.slots]
            fq, mm = phrase(b, part, f_i, slots, list(node.offsets), node.slop)
            freq, m = (fq, mm) if freq is None else (freq + fq, m | mm)
        idf_sum = sum(max((float(st.host_idf_field[st.gid(t), f_i]) for t in alts if st.gid(t) is not None),
                          default=0.0)
                      for alts in node.slots)
        sc = phrase_scores(b, st, f_i, freq, idf_sum, st.mapping.k1, st.mapping.b_of(st.text_fields[f_i]))
        return b.where(m, sc * node.boost, 0.0), m

    def _child(self, path):
        ex = getattr(self, "_children", {}).get(path)
        if ex is None:
            ex = Executor(self.st.nested[path], self.rankers)
            self.__dict__.setdefault("_children", {})[path] = ex
        return ex

    def _nested(self, node):
        """S9-5: the inner query on the children, rolled up to their parents (block join) by score_mode."""
        b, st = self.b, self.st
        ex = self._child(node.nested_path)
        ex._prefetch([Plan(root=node.query)], Timer(b))
        cs, cm = ex.eval(node.query, {"explain": None})
        cm = cm & ex.st.live
        par = st.nested_parent(node.nested_path)
        keys = par[cm]
        N = st.N
        cnt = b.segment_count(keys, N)
        m = cnt > 0
        v = cs[cm]
        sm = node.score_mode
        if sm == "none":
            s = b.zeros(N, "float32")
        elif sm in ("sum", "avg"):
            tot = b.astype(b.segment_sum(keys, v, N), "float64")
            s = tot if sm == "sum" else tot / b.astype(b.where(m, cnt, 1), "float64")
        elif sm == "max":
            s = b.segment_max(keys, v, N)
        else:
            s = b.segment_min(keys, v, N)
        s = b.astype(b.where(m, s, 0.0), "float32") * node.boost
        return b.where(m, s, 0.0), m

    def _function_score(self, node, ctx):
        """function_score (S7-5), as OpenSearch combines it: each function applies where its filter matches; the
        applying functions combine by score_mode (a doc no function applies to gets 1), capped at max_boost; the
        result combines with the query's score by boost_mode, times boost; min_score drops docs below it."""
        b, st = self.b, self.st
        N = st.N
        q, m = self.eval(node.query, ctx)
        q = b.astype(q, "float64")
        vals, applies, weights = [], [], []
        for flt, fvf, w in node.functions:
            app = m if flt is None else m & self.eval(flt, ctx)[1]
            v = b.full(N, 1.0, "float64")
            if fvf is not None:
                field, factor, mod, missing = fvf
                col = b.astype(st.numerics[field], "float64")
                miss = st.missing[field]
                if missing is None:
                    if int(b.count_nonzero(app & miss)):
                        raise InvalidRequest(node.path, f"Missing value for field [{field}] (give missing)")
                    x = col * factor
                else:
                    x = b.where(miss, missing, col) * factor
                v = _modify(b, mod, x)
                bad = app & (b.isnan(v) | (v < 0) | (v == float("inf")))
                if int(b.count_nonzero(bad)):
                    raise InvalidRequest(node.path, "field_value_factor gives a negative, infinite or NaN score")
            if w is not None:
                v = v * w
            vals.append(v)
            applies.append(app)
            weights.append(1.0 if w is None else w)
        f = b.full(N, 1.0, "float64")
        if vals:
            any_app = applies[0]
            for a in applies[1:]:
                any_app = any_app | a
            sm = node.score_mode
            if sm == "multiply":
                for v, a in zip(vals, applies):
                    f = b.where(a, f * v, f)
            elif sm in ("sum", "avg"):
                tot, wsum = b.zeros(N, "float64"), b.zeros(N, "float64")
                for v, a, w in zip(vals, applies, weights):
                    tot = b.where(a, tot + v, tot)
                    wsum = b.where(a, wsum + w, wsum)
                f = tot if sm == "sum" else b.where(wsum > 0, tot / b.where(wsum > 0, wsum, 1.0), 1.0)
            elif sm == "first":
                done = b.full(N, False, "bool")
                for v, a in zip(vals, applies):
                    f = b.where(a & ~done, v, f)
                    done = done | a
            else:                                         # max, min
                best = b.full(N, float("-inf") if sm == "max" else float("inf"), "float64")
                for v, a in zip(vals, applies):
                    best = b.where(a & ((v > best) if sm == "max" else (v < best)), v, best)
                f = best
            f = b.where(any_app, f, 1.0)
            if node.max_boost is not None:
                f = b.where(f > node.max_boost, node.max_boost, f)
        bm = node.boost_mode
        if bm == "multiply":
            s = q * f
        elif bm == "replace":
            s = f
        elif bm == "sum":
            s = q + f
        elif bm == "avg":
            s = (q + f) / 2.0
        elif bm == "max":
            s = b.where(q > f, q, f)
        else:
            s = b.where(q < f, q, f)
        s = b.astype(s * node.boost, "float32")
        if node.min_score is not None:
            m = m & (s >= node.min_score)
        return b.where(m, s, 0.0), m

    def _bool(self, node, ctx):
        b, N = self.b, self.st.N
        score = b.zeros(N, "float32")
        match = b.full(N, True, "bool")
        for c in node.must:
            s, m = self.eval(c, ctx)
            score, match = score + s, match & m
        if node.should:
            msm = node.msm if node.msm is not None else (0 if (node.must or node.filter) else 1)
            cnt = b.zeros(N, "int32")
            for c in node.should:
                s, m = self.eval(c, ctx)
                score = score + s
                cnt = cnt + b.astype(m, "int32")
            if msm > 0:
                match = match & (cnt >= msm)
        for c in node.filter:
            match = match & self.eval(c, ctx)[1]
        for c in node.must_not:
            match = match & ~self.eval(c, ctx)[1]
        if not node.must and not node.should:
            score = b.zeros(N, "float32")
        return b.where(match, score * node.boost, 0.0), match

    def _term(self, node):
        b, st = self.b, self.st
        f = node.field
        if f in st.keywords:
            kw = st.keywords[f]
            vals = [str(v) for v in node.values if v is not None and str(v)]
            codes = [int(c) for c in kw.lookup(vals) if c > 0]
            return ops.term_mask(b, kw.codes, np.asarray(codes or [-1], np.int64))
        dt = st.num_dtype[f]
        vals = list(node.values)              # the planner converted them to the field's type
        cast = "int32" if dt == "bool" else "float64"
        col = b.astype(st.numerics[f], cast)
        target = b.asarray(np.asarray(vals, dtype=np.float64).astype(cast), cast)
        return ops.term_mask(b, col, target, st.missing[f])

    def _range(self, node):
        b, st = self.b, self.st
        f = node.field
        if f in st.keywords:
            kw = st.keywords[f]
            vals = kw.values[1:]
            lo = hi = None
            if node.lo is not None:
                lo = int(np.searchsorted(vals, node.lo, side="left" if node.lo_inc else "right")) + 1
            if node.hi is not None:
                hi = int(np.searchsorted(vals, node.hi, side="right" if node.hi_inc else "left"))
            m = ops.range_mask(b, kw.codes, lo, hi, True, True)
            return m & (kw.codes > 0)
        col = b.astype(st.numerics[f], "float64")
        return ops.range_mask(b, col, node.lo, node.hi, node.lo_inc, node.hi_inc, st.missing[f])

    def _hybrid(self, node, ctx):
        b, st = self.b, self.st
        N = st.N
        lists = []
        for c in node.queries:
            s, m = self.eval(c, ctx)
            idx, _ = b.topk(s.reshape(1, N), min(node.window, N), (m & st.live).reshape(1, N))
            lists.append(idx)
        score, ranks = ops.rrf(b, lists, N, node.rrf_k)
        score, rank_rows = score[0], [r[0] for r in ranks]
        ctx["hybrid"] = (node, score, rank_rows)
        match = score > 0
        return score, match

    def _facet_codes(self, field):
        """Codes (0 = missing) and the key of each code, for a terms aggregation. Keyword fields have them
        stored; numeric fields get them from the column's current values (hot fields change), keys sorted
        ascending so that code order is key order. Integer fields give int keys, booleans bools."""
        b, st = self.b, self.st
        if field in st.keywords:
            kw = st.keywords[field]
            return kw.codes, kw.values
        return st.value_codes(field)

    # ---- aggregations (S7-4)
    def _agg(self, a, mask):
        from .. import aggs as A
        b, st = self.b, self.st
        if a.kind == "nested":                          # S9-5: the children of the matching parents
            ex = self._child(a.field)
            par = st.nested_parent(a.field)
            cm = ex.st.live & (par >= 0) & mask[b.where(par >= 0, par, 0)]
            out = {"doc_count": int(b.count_nonzero(cm))}
            for x in a.sub:
                out[x.name] = ex._agg(x, cm)
            return out
        if a.kind == "terms":
            codes, keys_of = self._facet_codes(a.field)
            counts = ops.facet_terms(b, codes, mask.reshape(1, -1), len(keys_of))
            (top, with_value), = ops.top_buckets(b, counts, a.size)
            buckets = []
            for c, n in top:
                bk = {"key": keys_of[c], "doc_count": n}
                if a.sub:
                    sub_mask = mask & (codes == c)
                    for x in a.sub:
                        bk[x.name] = self._agg(x, sub_mask)
                buckets.append(bk)
            return {"doc_count_error_upper_bound": 0, "sum_other_doc_count": with_value - sum(c for _, c in top),
                    "buckets": buckets}
        if a.kind in ("value_count", "cardinality") and a.field in st.keywords:
            codes = st.keywords[a.field].codes
            return A.keyword_metric(b, a.kind, codes, mask, st.keywords[a.field].n)
        vals, miss = st.numerics[a.field], st.missing[a.field]
        present = mask & ~miss
        if a.kind == "cardinality":
            codes, keys_of = st.value_codes(a.field)
            return A.keyword_metric(b, "cardinality", codes, present, len(keys_of))
        if a.kind == "range":
            return A.range_agg(b, vals, present, a.param("ranges"))
        if a.kind == "histogram":
            return A.histogram(b, vals, present, a.param("interval"), a.param("offset"), a.param("min_doc_count"))
        return A.metric(b, a.kind, vals, present)

    # ---- sort keys
    def _keys(self, plan, score, tier=None):
        b, st = self.b, self.st
        keys = []
        if tier is not None:
            keys.append((tier, True))
        for k in plan.sort:
            if k.field == "_score":
                keys.append((score, k.descending))
                continue
            if k.field == "_doc":
                keys.append((b.astype(b.arange(st.N), "float64"), k.descending))
                continue
            if k.geo is not None:                       # distance in the unit; missing: +inf (as OpenSearch)
                from ..geo import haversine_m, lat_col, lon_col
                f, lat, lon, unit = k.geo
                d = haversine_m(b, st.numerics[lat_col(f)], st.numerics[lon_col(f)], lat, lon) / unit
                keys.append((b.where(st.missing[lat_col(f)], INF, d), k.descending))
                continue
            sentinel = (-INF if k.descending else INF) if k.missing_last else (INF if k.descending else -INF)
            if k.field in st.keywords:
                kw = st.keywords[k.field]
                col = b.astype(kw.codes, "float64")
                miss = kw.codes == 0
            else:
                col = b.astype(st.numerics[k.field], "float64")
                miss = st.missing[k.field]
            keys.append((b.where(miss, sentinel, col), k.descending))
        return keys

    def _after_to_keys(self, plan, after, has_tier):
        st = self.st
        out = []
        vals = list(after)
        if has_tier:
            out.append(float(vals.pop(0)))
        for k, v in zip(plan.sort, vals[:-1]):
            if k.field in ("_score", "_doc"):
                out.append(float(v))
                continue
            if k.geo is not None:
                out.append(INF if v is None else float(v))
                continue
            sentinel = (-INF if k.descending else INF) if k.missing_last else (INF if k.descending else -INF)
            if v is None:
                out.append(sentinel)
            elif k.field in st.keywords:
                kw = st.keywords[k.field]
                s = str(v)
                c = int(kw.lookup([s])[0]) if s else -1
                out.append(float(c) if c > 0 else
                           float(np.searchsorted(kw.values[1:], s)) + 0.5)
            else:
                out.append(float(v))
        out.append(int(vals[-1]))
        return tuple(out)

    def _sort_values(self, plan, row, doc, has_tier):
        st = self.st
        out = []
        vals = list(row)
        if has_tier:
            out.append(int(vals.pop(0)))
        for k, v in zip(plan.sort, vals):
            if k.field == "_score":
                out.append(float(v))
            elif k.field == "_doc":
                out.append(int(v))
            elif k.geo is not None:
                out.append(None if math.isinf(v) else float(v))
            elif math.isinf(v):
                out.append(None)
            elif k.field in st.keywords:
                out.append(st.keywords[k.field].values[int(v)])
            elif st.num_dtype[k.field] == "bool":
                out.append(bool(v))
            elif st.num_dtype[k.field].startswith("int"):
                out.append(int(v))
            else:
                out.append(float(v))
        out.append(int(doc))
        return out

    def _sort_columns(self, plan, rows, first_offset, has_tier):
        """``_sort_values`` for a whole page at once (S6-7, INTERFACES §5 "Columnar hits"): one column per position of
        a hit's ``sort`` list, ``{"values", "missing"}``, so that expanding them gives exactly ``_sort_values``."""
        st = self.st
        out = []
        j = 0
        if has_tier:
            out.append({"values": rows[:, 0].astype(np.int64), "missing": None})
            j = 1
        for k in plan.sort:
            v = rows[:, j]
            j += 1
            miss = None
            if k.field == "_score":
                vals = v.astype(np.float64)
            elif k.field == "_doc":
                vals = v.astype(np.int64)
            else:
                inf = np.isinf(v)
                miss = inf if inf.any() else None
                safe = np.where(inf, 0.0, v)
                if k.geo is not None:
                    vals = safe.astype(np.float64)
                elif k.field in st.keywords:
                    kv = st.keywords[k.field].values
                    vals = [kv[c] for c in safe.astype(np.int64).tolist()]
                elif st.num_dtype[k.field] == "bool":
                    vals = safe.astype(bool)
                elif st.num_dtype[k.field].startswith("int"):
                    vals = safe.astype(np.int64)
                else:
                    vals = safe.astype(np.float64)
            out.append({"values": vals, "missing": miss})
        out.append({"values": first_offset.astype(np.int64), "missing": None})
        return out

    # ---- the ranker
    def _rescore(self, plan, score, match, ctx, timer):
        from ..rank.features import candidate_features
        b, st = self.b, self.st
        ranker = self.rankers.get(plan.rescore.model)
        if ranker is None:
            raise InvalidRequest("rescore.s3_ranker.model", f"no ranker named {plan.rescore.model!r} on this index")
        N = st.N
        idx, _ = b.topk(score.reshape(1, N), min(plan.rescore.window, N), match.reshape(1, N))
        X, cand = candidate_features(self, ctx, idx)
        timer.lap("features")
        ok = cand[0] >= 0
        sel = cand[0][ok]
        Xd = X[0][ok]
        if hasattr(ranker, "predict_backend"):            # on the device: no copy to the host
            pred = ranker.predict_backend(b, Xd)
        else:                                             # any other model: through the host
            pred = b.asarray(np.asarray(ranker.predict(b.to_numpy(Xd)), np.float32), "float32")
        if ctx.get("explain") is not None:                # explain reads features on the host
            sel_h = b.to_numpy(sel)
            ctx["rescore"] = {"ranker": ranker, "row_of": {int(d): i for i, d in enumerate(sel_h)},
                              "X": np.asarray(b.to_numpy(Xd), np.float32), "pred": b.to_numpy(pred)}
        new = b.copy(score)
        tier = b.zeros(N, "float32")
        new[sel] = pred
        tier[sel] = 1.0
        timer.lap("rank")
        return new, tier

    # ---- one batch (S6-6): each plan's match set and scores first; then one count, the terms facets and the
    # plain score-sorted pages for the whole batch, each in one transfer; the rest per plan, as before.
    BATCH_ROWS = 64                                   # plans stacked per [B, N] step (bounds the device memory)
    BATCHED = True                                    # False: every plan on the per-plan path (the reference)

    def run(self, plans: list[Plan], columns: bool = False) -> list[dict]:
        b, st = self.b, self.st
        timer = Timer(b)
        shared_t0 = time.perf_counter()
        self._prefetch(plans, timer)
        shared = dict(timer.laps)
        shared_ms = (time.perf_counter() - shared_t0) * 1000.0
        states = []
        for plan in plans:
            t = Timer(b)
            ctx = {"explain": {} if plan.explain else None}
            self.knn_outcome = {}
            score, match = self.eval(plan.root, ctx)
            match = match & st.live
            score = b.where(match, score, 0.0)
            t.lap("filter" if not isinstance(plan.root, Hybrid) else "fuse")
            hits_mask = match
            if plan.post_filter is not None:             # aggregations see the query's matches, hits these
                hits_mask = match & self.eval(plan.post_filter, {"explain": None})[1]
            tier = None
            batched = self.BATCHED and plan.rescore is not None and not plan.explain
            if plan.rescore is not None and not batched:  # explain reads the rescore's host copy: per plan
                score, tier = self._rescore(plan, score, match, ctx, t)
            states.append({"plan": plan, "t": t, "ctx": ctx, "score": score, "match": match, "hits_mask": hits_mask,
                           "tier": tier, "knn": _knn_methods(b, plan, self.knn_outcome), "rescore_later": batched})
        self._batched_rescore(states)
        totals, pages = self._batched_totals_and_pages(states)
        aggs = self._batched_terms(states)
        out = []
        for i, sx in enumerate(states):
            plan, t, score, tier = sx["plan"], sx["t"], sx["score"], sx["tier"]
            total = totals.get(i)
            resp = {}
            if plan.aggs:
                t0 = time.perf_counter()
                done = aggs.get(i, {})
                resp["aggregations"] = {a.name: done[a.name] if a.name in done else self._agg(a, sx["match"])
                                        for a in plan.aggs}
                t.laps["facet"] += sx.pop("facet_ms", 0.0) + (time.perf_counter() - t0) * 1000.0
            t.t = time.perf_counter()
            group = None
            if plan.collapse is not None:
                group, group_keys = self._facet_codes(plan.collapse)
            want_columns = columns and group is None and plan.highlight is None and not plan.explain
            if i in pages:                               # the batched page: same arrays as ops.page_arrays
                ids_a, rows_a, extra_a = pages[i]
                t.laps["page"] += sx.pop("page_ms", 0.0)
            elif want_columns:
                keys = self._keys(plan, score, tier)
                after = self._after_to_keys(plan, plan.search_after, tier is not None) if plan.search_after else None
                ids_a, rows_a, extra_a, total = ops.page_arrays(b, keys, sx["hits_mask"], plan.from_, plan.size,
                                                                after, extra=[score, st.first_offset_dev],
                                                                tiekey=st.first_offset_dev, count=sx["hits_mask"])
            else:
                keys = self._keys(plan, score, tier)
                after = self._after_to_keys(plan, plan.search_after, tier is not None) if plan.search_after else None
                extra_cols = [score, st.first_offset_dev] + ([group] if group is not None else [])
                ids, rows, extra, total = ops.page(b, keys, sx["hits_mask"], plan.from_, plan.size, after,
                                                   extra=extra_cols, tiekey=st.first_offset_dev, collapse=group,
                                                   count=sx["hits_mask"])
                ids_a = None
            if want_columns:
                cols = {"_id": st.ids[ids_a].tolist(), "_score": extra_a[:, 0].astype(np.float32),
                        "sort": self._sort_columns(plan, rows_a, extra_a[:, 1], tier is not None)}
                t.lap("page")
                resp = {"hits": {"total": total, "columns": cols}, **resp}
                if sx["knn"]:
                    resp["knn"] = sx["knn"]
                out.append(self._took(resp, shared, shared_ms, t))
                continue
            if ids_a is not None:                        # a batched page, in the dict form
                ids, rows, extra = ids_a.tolist(), rows_a.tolist(), extra_a.tolist()
            score_h = [e[0] for e in extra]
            hits = [{"_id": str(st.ids[d]), "_score": float(e[0]),
                     "sort": self._sort_values(plan, r, int(e[1]), tier is not None)}
                    for d, e, r in zip(ids, extra, rows)]
            if group is not None:                        # OpenSearch: the collapse key in "fields"
                for h, e in zip(hits, extra):
                    c = int(e[2])
                    h["fields"] = {plan.collapse: [None if c == 0 else group_keys[c]]}
            t.lap("page")
            resp = {"hits": {"total": total, "hits": hits}, **resp}
            if sx["knn"]:
                resp["knn"] = sx["knn"]
            if plan.highlight is not None:
                from ..highlight import highlight_hits
                for h, hl in zip(hits, highlight_hits(st, plan, ids)):
                    if hl:
                        h["highlight"] = hl
            if plan.explain:
                from ..explain import explain_hits
                resp["explain"] = explain_hits(self, plan, sx["ctx"], ids, score_h)
            out.append(self._took(resp, shared, shared_ms, t))
        return out

    @staticmethod
    def _took(resp, shared, shared_ms, t):
        took = {k: round(v, 3) for k, v in shared.items()}
        for k, v in t.laps.items():
            took[k] = round(took.get(k, 0.0) + v, 3)
        took["total"] = round(shared_ms + t.total(), 3)
        resp["took_ms"] = took
        return resp

    def _batched_rescore(self, states):
        """S6-8: the rescored plans of the batch (without explain), per ranker and chunk: one top-k window over the
        stacked scores, the candidates' features in one gather, one ranker call over every plan's candidates, and
        the new scores and tiers scattered back. Row q equals ``_rescore`` of that plan alone: its window is the
        first ``window`` of a larger top-k (score descending, then doc id), its features are its own row's, and the
        ranker scores rows independently."""
        from ..rank.features import candidate_features_batch
        b, st = self.b, self.st
        N = st.N
        by_model = defaultdict(list)
        for i, sx in enumerate(states):
            if sx["rescore_later"]:
                by_model[sx["plan"].rescore.model].append(i)
        for model, idxs in by_model.items():
            ranker = self.rankers.get(model)
            if ranker is None:
                raise InvalidRequest("rescore.s3_ranker.model", f"no ranker named {model!r} on this index")
            for ch in self._chunks(idxs):
                t0 = time.perf_counter()
                G = len(ch)
                wins = [min(states[i]["plan"].rescore.window, N) for i in ch]
                S = b.stack([states[i]["score"] for i in ch], 0)
                M = b.stack([states[i]["match"] for i in ch], 0)
                idx, _ = b.topk(S, max(wins), M)
                W = idx.shape[1]
                inside = b.arange(W).reshape(1, W) < b.asarray(wins, "int64").reshape(G, 1)
                idx = b.where(inside, b.astype(idx, "int64"), -1)
                X, cand = candidate_features_batch(self, [states[i]["ctx"] for i in ch], idx)
                b.synchronize()
                t1 = time.perf_counter()
                ok = cand >= 0
                rr, cc = b.nonzero(ok)
                sel = cand[rr, cc]
                Xd = X[ok]
                if hasattr(ranker, "predict_backend"):    # on the device: no copy to the host
                    pred = ranker.predict_backend(b, Xd)
                else:                                     # any other model: through the host
                    pred = b.asarray(np.asarray(ranker.predict(b.to_numpy(Xd)), np.float32), "float32")
                new = b.copy(S)
                tier = b.zeros((G, N), "float32")
                new[rr, sel] = pred
                tier[rr, sel] = 1.0
                b.synchronize()
                t2 = time.perf_counter()
                for g, i in enumerate(ch):
                    states[i]["score"], states[i]["tier"] = new[g], tier[g]
                    states[i]["t"].laps["features"] += (t1 - t0) * 1000.0 / G
                    states[i]["t"].laps["rank"] += (t2 - t1) * 1000.0 / G

    def _chunks(self, idx):
        for k in range(0, len(idx), self.BATCH_ROWS):
            yield idx[k:k + self.BATCH_ROWS]

    def _batched_terms(self, states):
        """``{plan index: {agg name: result}}`` for the plain terms aggregations (no sub-aggregations), grouped by
        field and size: one bincount and one top-buckets transfer per group and chunk, the same buckets as ``_agg``."""
        b = self.b
        groups = defaultdict(list)
        for i, sx in enumerate(states):
            for a in sx["plan"].aggs:
                if self.BATCHED and a.kind == "terms" and not a.sub:
                    groups[(a.field, a.size)].append((i, a))
        out = defaultdict(dict)
        for (field, size), items in groups.items():
            t0 = time.perf_counter()
            codes, keys_of = self._facet_codes(field)
            for ch in self._chunks(items):
                mask = b.stack([states[i]["match"] for i, _ in ch], 0)
                counts = ops.facet_terms(b, codes, mask, len(keys_of))
                for (i, a), (top, with_value) in zip(ch, ops.top_buckets(b, counts, size)):
                    out[i][a.name] = {"doc_count_error_upper_bound": 0,
                                      "sum_other_doc_count": with_value - sum(c for _, c in top),
                                      "buckets": [{"key": keys_of[c], "doc_count": n} for c, n in top]}
            ms = (time.perf_counter() - t0) * 1000.0 / len(items)
            for i, _ in items:
                states[i]["facet_ms"] = states[i].get("facet_ms", 0.0) + ms
        return out

    def _batched_totals_and_pages(self, states):
        """``({plan index: hits.total}, {plan index: (doc ids, key values, extra values)})`` for the plans sorted by
        score alone (descending; no tier, collapse or search_after): per chunk of them, one count over the stacked hit
        masks and one top-k over the stacked scores, brought to the host in one transfer (the other plans' totals ride
        on their own page's transfer). A row's first ``from + size`` of a larger top-k are its own top-k (the order is score descending, then doc id), so each
        page equals ``ops.page_arrays``'s."""
        b, st = self.b, self.st

        def is_plain(sx):
            p = sx["plan"]
            return (sx["tier"] is None and p.collapse is None and not p.search_after and len(p.sort) == 1
                    and p.sort[0].field == "_score" and p.sort[0].descending and p.from_ + p.size > 0
                    and str(sx["score"].dtype).endswith("float32"))

        totals, pages = {}, {}
        for ch in self._chunks([i for i, sx in enumerate(states) if self.BATCHED and is_plain(sx)]):
            t0 = time.perf_counter()
            G = len(ch)
            V = b.stack([states[i]["hits_mask"] for i in ch], 0)
            S = b.stack([states[i]["score"] for i in ch], 0)
            idx, _ = b.topk(S, max(states[i]["plan"].from_ + states[i]["plan"].size for i in ch), V)
            need = idx.shape[1]                           # min(the largest from + size, N)
            idx = b.astype(idx, "int64")
            safe = b.where(idx >= 0, idx, 0)
            rows = b.arange(G).reshape(G, 1)
            buf = b.stack([b.astype(idx, "float64"), b.astype(S[rows, safe], "float64"),
                           b.astype(st.first_offset_dev[safe], "float64")], 2)         # [G, need, 3]
            host = b.to_numpy(b.concatenate([b.astype(b.count_nonzero(V, axis=1), "float64"), buf.reshape(-1)], 0))
            pg = host[G:].reshape(G, need, 3)                # one transfer for the chunk: totals, then pages
            ms = (time.perf_counter() - t0) * 1000.0 / G
            for r, i in enumerate(ch):
                p = states[i]["plan"]
                totals[i] = int(host[r])
                h = pg[r, p.from_:p.from_ + p.size]
                h = h[h[:, 0] >= 0]
                pages[i] = (h[:, 0].astype("int64"), h[:, 1:2], h[:, 1:3])
                states[i]["page_ms"] = ms
        return totals, pages
