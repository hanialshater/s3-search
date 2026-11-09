"""Explain (docs/INTERFACES.md §5): a Lucene-shaped tree per hit, ``{value, description, details}``. Lexical clauses
are broken down per term and per field, recomputed on the host from the segment arrays."""
from __future__ import annotations

import numpy as np

from .plan.nodes import Bool, Dense, Hybrid, Lexical, MatchAll, Range, Term


def _lexical(ex, node, doc):
    st = ex.st
    k1 = st.mapping.k1
    bpar = st.mapping.b
    details = []
    total = 0.0
    if node.idf == "field":
        f_one = next(st.field_index[f] for f, w in node.fields if w > 0)
    for t in node.terms:
        tid = st.vocab.get(t)
        if tid is None:
            continue
        lo, hi = int(st.host_term_ptr[tid]), int(st.host_term_ptr[tid + 1])
        j = lo + int(np.searchsorted(st.host_doc_ids[lo:hi], doc))
        if j >= hi or st.host_doc_ids[j] != doc:
            continue
        idf = float(st.host_idf_field[tid, f_one] if node.idf == "field" else st.host_idf[tid])
        per_field, ptf = [], 0.0
        for f, w in node.fields:
            fi = st.field_index[f]
            tf = int(st.host_tf[j, fi])
            if w <= 0 or tf == 0:
                continue
            norm = max(1.0 - bpar + bpar * float(st.host_rel_len[doc, fi]), 1e-6)
            c = w * tf / norm
            ptf += c
            per_field.append({"value": c, "description": f"{f}: weight {w} * tf {tf} / norm {norm:.4g}",
                              "details": []})
        if ptf <= 0:
            continue
        v = idf * ptf * (k1 + 1.0) / (k1 + ptf)
        total += v
        details.append({"value": v, "description": f"term {t!r}: idf {idf:.4g} * pseudo_tf * (k1 + 1) / (k1 + "
                                                   f"pseudo_tf), pseudo_tf {ptf:.4g}", "details": per_field})
    return {"value": total * node.boost, "description": f"BM25F (k1 {k1}, b {bpar}) x boost {node.boost}",
            "details": details}


def explain_node(ex, node, ctx, doc):
    res = ctx["explain"].get(id(node))
    value = float(ex.b.to_numpy(res[0][doc])) if res is not None else 0.0
    if isinstance(node, Lexical):
        out = _lexical(ex, node, doc)
        out["value"] = value
        return out
    if isinstance(node, Bool):
        kids = [explain_node(ex, c, ctx, doc) for c in node.must + node.should]
        return {"value": value, "description": f"sum of (bool) x boost {node.boost}", "details": kids}
    if isinstance(node, Hybrid):
        kids = [explain_node(ex, c, ctx, doc) for c in node.queries]
        return {"value": value, "description": f"RRF (rank_constant {node.rrf_k}) over {len(kids)} queries",
                "details": kids}
    if isinstance(node, Dense):
        return {"value": value, "description": f"knn on {node.field}: (1 + cos) / 2 x boost {node.boost}",
                "details": []}
    if isinstance(node, (Term, Range, MatchAll)):
        return {"value": value, "description": f"{type(node).__name__.lower()} (constant) x boost {node.boost}",
                "details": []}
    return {"value": value, "description": type(node).__name__, "details": []}


def explain_hits(ex, plan, ctx, ids, scores):
    out = []
    for d, s in zip(ids, scores):
        e = explain_node(ex, plan.root, ctx, int(d))
        if plan.rescore is not None:
            e = {"value": float(s), "description": f"s3_ranker {plan.rescore.model!r} over the top "
                                                    f"{plan.rescore.window}", "details": [e]}
        e["_id"] = str(ex.st.ids[d])
        out.append(e)
    return out
