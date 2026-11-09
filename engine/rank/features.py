"""The ranker's features for one plan's candidates, from what the executor computed (docs/INTERFACES.md §5 rescore)."""
from __future__ import annotations

import numpy as np

from .. import ops
from ..analysis import analyze
from ..plan.nodes import Dense, Lexical


def feature_names_for(st) -> list[str]:
    return ops.feature_names(st.text_fields, st.match_fields())


def candidate_features(ex, ctx, cand):
    """``cand int32[1,W]`` (-1 pads) → ``(X f32[1,W,n_features], cand int64[1,W])``.

    The lexical and dense components come from the plan's hybrid query (its first lexical and first dense
    sub-query), or from the first lexical / dense node evaluated when there is no hybrid; a missing component gives
    zero scores and absent ranks."""
    b, st = ex.b, ex.st
    N = st.N
    zeros = b.zeros((1, N), "float32")
    cand = b.astype(cand, "int64")
    lex = dense = zeros
    lex_rank = dense_rank = zeros
    rrf = zeros
    lex_node = None
    if "hybrid" in ctx:
        node, rrf_row, rank_rows = ctx["hybrid"]
        rrf = rrf_row.reshape(1, N)
        for q, rr in zip(node.queries, rank_rows):
            if isinstance(q, Lexical) and lex_node is None:
                lex_node = q
                lex = ex.lex_cache[q][0].reshape(1, N)
                lex_rank = rr.reshape(1, N)
            elif isinstance(q, Dense) and dense is zeros:
                dense = ex.dense_cache[q].reshape(1, N)
                dense_rank = rr.reshape(1, N)
    else:
        if "lexical" in ctx:
            lex_node, s, m = ctx["lexical"]
            lex = b.where(m, s, 0.0).reshape(1, N)
        if "dense" in ctx:
            _, sims, _ = ctx["dense"]
            dense = sims.reshape(1, N)
    if lex_node is not None:
        _, nm = ex.lex_cache[lex_node]
        n_match = int(b.count_nonzero(nm >= max(1, lex_node.msm))) if lex_node.msm else 0
        q_tokens = set(lex_node.terms)
        q_len = len(lex_node.terms)
    else:
        n_match, q_tokens, q_len = 0, set(), 0
    mf = st.match_fields()
    match_codes = [[st.keywords[f].matched_codes(q_tokens)] for f in mf]
    kw_codes = [st.keywords[f].codes for f in mf]
    X = ops.gather_features(b, cand, lex, lex_rank, dense, dense_rank, rrf, match_codes, kw_codes,
                            st.static_features(), [q_len], [n_match])
    return X, cand
