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
from .nodes import Bool, Dense, Hybrid, Lexical, MatchAll, Plan, Range, Term

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

    # ---- retrieval, batched across plans
    def _prefetch(self, plans, timer):
        st, b = self.st, self.b
        lex = defaultdict(list)
        dense = defaultdict(list)
        for p in plans:
            for n in walk(p.root):
                if isinstance(n, Lexical):
                    lex[(n.fields, n.idf)].append(n)
                elif isinstance(n, Dense):
                    dense[n.field].append(n)
        self.lex_cache, self.dense_cache = {}, {}
        for (fields, idf_mode), nodes in lex.items():
            nodes = list(dict.fromkeys(nodes))
            F = len(st.text_fields)
            w = np.zeros(F, np.float32)
            for f, wt in fields:
                w[st.field_index[f]] = wt
            bb = np.full(F, st.mapping.b, np.float32)
            if idf_mode == "field":
                f_i = next(st.field_index[f] for f, wt in fields if wt > 0)
                idf = st.idf_field[:, f_i]
            else:
                idf = st.idf
            terms = [[st.vocab[t] for t in n.terms if t in st.vocab] for n in nodes]
            scores, nmatch = ops.bm25f(b, st, terms, w, bb, st.mapping.k1, idf)
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
            sims = ops.dense_scores(b, st.vectors[field], b.asarray(q, "float32"))
            for i, n in enumerate(nodes):
                self.dense_cache[n] = sims[i]
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
        elif isinstance(node, MatchAll):
            out = (b.full(N, node.boost, "float32"), b.full(N, True, "bool"))
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
            codes = [kw.code_of[str(v)] for v in node.values if v is not None and str(v) in kw.code_of and str(v)]
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
        dt = st.num_dtype[field]
        vals = b.to_numpy(st.numerics[field])
        miss = b.to_numpy(st.missing[field])
        uniq = np.unique(vals[~miss])
        codes = np.where(miss, 0, np.searchsorted(uniq, vals) + 1).astype(np.int64)
        conv = bool if dt == "bool" else int if dt.startswith("int") else float
        return b.asarray(codes, "int64"), [None] + [conv(v) for v in uniq]

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
            sentinel = (-INF if k.descending else INF) if k.missing_last else (INF if k.descending else -INF)
            if v is None:
                out.append(sentinel)
            elif k.field in st.keywords:
                kw = st.keywords[k.field]
                s = str(v)
                out.append(float(kw.code_of[s]) if s in kw.code_of and s else
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
        cand_h = b.to_numpy(cand)[0]
        ok = cand_h >= 0
        Xh = np.asarray(b.to_numpy(X)[0][ok], np.float32)
        pred = np.asarray(ranker.predict(Xh), np.float32)
        if ctx.get("explain") is not None:
            ctx["rescore"] = {"ranker": ranker, "row_of": {int(d): i for i, d in enumerate(cand_h[ok])}, "X": Xh,
                              "pred": pred}
        sel = b.asarray(cand_h[ok], "int64")
        new = b.copy(score)
        tier = b.zeros(N, "float32")
        if len(pred):
            new[sel] = b.asarray(pred, "float32")
            tier[sel] = 1.0
        timer.lap("rank")
        return new, tier

    # ---- one batch
    def run(self, plans: list[Plan]) -> list[dict]:
        b, st = self.b, self.st
        timer = Timer(b)
        shared_t0 = time.perf_counter()
        self._prefetch(plans, timer)
        shared = dict(timer.laps)
        shared_ms = (time.perf_counter() - shared_t0) * 1000.0
        out = []
        for plan in plans:
            t = Timer(b)
            ctx = {"explain": {} if plan.explain else None}
            score, match = self.eval(plan.root, ctx)
            match = match & st.live
            score = b.where(match, score, 0.0)
            t.lap("filter" if not isinstance(plan.root, Hybrid) else "fuse")
            total = int(b.count_nonzero(match))
            tier = None
            if plan.rescore is not None:
                score, tier = self._rescore(plan, score, match, ctx, t)
            resp = {}
            if plan.aggs:
                aggs = {}
                for a in plan.aggs:
                    codes, keys_of = self._facet_codes(a.field)
                    counts = ops.facet_terms(b, codes, match.reshape(1, -1), len(keys_of))
                    top = ops.top_buckets(b, counts, a.size)[0]
                    with_value = int(b.to_numpy(counts)[0][1:].sum())
                    aggs[a.name] = {"doc_count_error_upper_bound": 0,
                                    "sum_other_doc_count": with_value - sum(c for _, c in top),
                                    "buckets": [{"key": keys_of[c], "doc_count": n} for c, n in top]}
                resp["aggregations"] = aggs
                t.lap("facet")
            keys = self._keys(plan, score, tier)
            after = self._after_to_keys(plan, plan.search_after, tier is not None) if plan.search_after else None
            ids, rows = ops.page(b, keys, match, plan.from_, plan.size, after)
            score_h = b.to_numpy(score[b.asarray(np.asarray(ids, np.int64), "int64")]) if ids else []
            hits = [{"_id": str(st.ids[d]), "_score": float(s),
                     "sort": self._sort_values(plan, r, d, tier is not None)}
                    for d, s, r in zip(ids, score_h, rows)]
            t.lap("page")
            resp = {"hits": {"total": total, "hits": hits}, **resp}
            if plan.explain:
                from ..explain import explain_hits
                resp["explain"] = explain_hits(self, plan, ctx, ids, score_h)
            took = {k: round(v, 3) for k, v in shared.items()}
            for k, v in t.laps.items():
                took[k] = round(took.get(k, 0.0) + v, 3)
            took["total"] = round(shared_ms + t.total(), 3)
            resp["took_ms"] = took
            out.append(resp)
        return out
