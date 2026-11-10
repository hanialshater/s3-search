"""Explain (docs/INTERFACES.md §5): a Lucene-shaped tree per hit, ``{op, value, description, details}``.

``op`` says how a node's value comes from its ``details`` (machine-readable; ``description`` is for people):
- ``"sum"``: the sum of the details' values;
- ``"product"``: the product of the details' values;
- ``"max"``: the largest detail value;
- ``"value"``: a leaf, or a value computed by the formula in ``description``; its details (if any) are inputs, not
  terms of a sum;
- ``"ranker"``: a learned ranker's score, ``base + each feature's contribution``; the details are the base and one
  node per feature (``feature``, ``feature_value``), which sum to the value.
A node may carry ``source``: the tree a value came from (for example the lexical tree behind ``lex_score``); it is not
part of the arithmetic.

Lexical clauses are broken down per term and per field, recomputed on the host from the segment arrays.
"""
from __future__ import annotations

import numpy as np

from .plan.nodes import Bool, Dense, Phrase, Hybrid, Lexical, MatchAll, Range, Term


def _node(op, value, description, details=(), **extra):
    out = {"op": op, "value": float(value), "description": description, "details": list(details)}
    out.update(extra)
    return out


def _boosted(value, boost, inner):
    """``inner`` (whose value is the unboosted score) times ``boost``, or ``inner`` itself when boost is 1."""
    if boost == 1.0:
        inner["value"] = float(value)
        return inner
    return _node("product", value, f"boost {boost} x", [inner, _node("value", boost, "boost")])


def _lexical(ex, node, doc):
    st = ex.st
    k1, bpar = st.mapping.k1, st.mapping.b
    terms, total = [], 0.0
    f_one = next(st.field_index[f] for f, w in node.fields if w > 0) if node.idf == "field" else None
    part, loc = st.owner(doc)                       # the segment that holds this slot's live doc
    for t in node.terms:
        tid, gid = part.vocab.get(t), st.vocab.get(t)
        if tid is None:
            continue
        lo, hi = int(part.host_term_ptr[tid]), int(part.host_term_ptr[tid + 1])
        j = lo + int(np.searchsorted(part.host_doc_ids[lo:hi], loc))
        if j >= hi or part.host_doc_ids[j] != loc:
            continue
        idf = float(st.host_idf_field[gid, f_one] if f_one is not None else st.host_idf[gid])
        per_field, ptf = [], 0.0
        for f, w in node.fields:
            fi = st.field_index[f]
            tf = int(part.host_tf[j, fi])
            if w <= 0 or tf == 0:
                continue
            norm = max(1.0 - bpar + bpar * float(st.host_rel_len[doc, fi]), 1e-6)
            c = w * tf / norm
            ptf += c
            per_field.append(_node("value", c, f"{f}: weight {w} * tf {tf} / norm {norm:.4g}"))
        if ptf <= 0:
            continue
        v = idf * ptf * (k1 + 1.0) / (k1 + ptf)
        total += v
        terms.append(_node("value", v, f"term {t!r}: idf {idf:.4g} * pseudo_tf * (k1 + 1) / (k1 + pseudo_tf), "
                                       f"pseudo_tf {ptf:.4g} = the sum of the field parts below",
                           [_node("sum", ptf, "pseudo_tf", per_field)]))
    return _node("sum", total, f"BM25F (k1 {k1}, b {bpar}): sum over query terms", terms)


def explain_node(ex, node, ctx, doc):
    res = ctx["explain"].get(id(node))
    b = ex.b
    value = float(b.to_numpy(res[0][doc])) if res is not None else 0.0
    matched = bool(b.to_numpy(res[1][doc])) if res is not None else False
    if isinstance(node, Lexical):
        if not matched:
            return _node("value", 0.0, "BM25F: no match (operator or minimum_should_match not met)")
        return _boosted(value, node.boost, _lexical(ex, node, doc))
    if isinstance(node, Bool):
        if not matched:
            return _node("value", 0.0, "bool: no match")
        if not node.must and not node.should:
            return _node("value", 0.0, "bool with only filter/must_not clauses: score 0")
        kids = [explain_node(ex, c, ctx, doc) for c in node.must + node.should]
        inner = _node("sum", sum(k["value"] for k in kids), "bool: sum of must and should", kids)
        return _boosted(value, node.boost, inner)
    if isinstance(node, Hybrid):
        h_node, _, rank_rows = ctx["hybrid"]
        kids = []
        for i, (q, rr) in enumerate(zip(node.queries, rank_rows)):
            r = int(b.to_numpy(rr[doc]))
            if r > 0:
                kids.append(_node("value", 1.0 / (node.rrf_k + r), f"1 / ({node.rrf_k} + rank {r}) in query {i}",
                                  source=explain_node(ex, q, ctx, doc)))
        return _node("sum", value, f"RRF (rank_constant {node.rrf_k}): sum over the lists that hold the doc", kids)
    if isinstance(node, Phrase):
        return _node("value", value, f"phrase on {node.field} (slop {node.slop}): idf sum * freq * (k1 + 1) / "
                                     f"(k1 + freq / norm) x boost {node.boost}")
    if isinstance(node, Dense):
        return _node("value", value, f"knn on {node.field}: (1 + cos) / 2 x boost {node.boost}")
    if isinstance(node, (Term, Range, MatchAll)):
        return _node("value", value, f"{type(node).__name__.lower()}: constant 1.0 x boost {node.boost}")
    return _node("value", value, type(node).__name__)


def _rescored(ex, plan, ctx, doc, score, lexical_tree):
    rs = ctx.get("rescore")
    row = rs["row_of"].get(int(doc)) if rs else None
    if row is None:                                        # below the window: the first-stage score
        return _node("value", score, f"below the rescore window {plan.rescore.window}: first-stage score",
                     source=lexical_tree)
    ranker, x = rs["ranker"], rs["X"][row]
    from .rank.features import feature_names_for
    names = feature_names_for(ex.st)
    if not hasattr(ranker, "contributions"):
        return _node("value", score, f"s3_ranker {plan.rescore.model!r} (no contributions available)",
                     source=lexical_tree)
    base, contrib = ranker.contributions(x.reshape(1, -1))
    details = [_node("value", base[0], "ranker base value")]
    for i, n in enumerate(names):
        fv = float(x[i])
        extra = {"feature": n, "feature_value": None if np.isnan(fv) else fv}
        if n.startswith("lex_"):
            extra["source"] = lexical_tree
        details.append(_node("value", contrib[0, i], f"{n} = {fv:.6g}", **extra))
    return _node("ranker", score, f"s3_ranker {plan.rescore.model!r}: base + feature contributions "
                                  f"(window {plan.rescore.window})", details)


def explain_hits(ex, plan, ctx, ids, scores):
    out = []
    for d, s in zip(ids, scores):
        e = explain_node(ex, plan.root, ctx, int(d))
        if plan.rescore is not None:
            e = _rescored(ex, plan, ctx, int(d), float(s), e)
        e["_id"] = str(ex.st.ids[d])
        out.append(e)
    return out
