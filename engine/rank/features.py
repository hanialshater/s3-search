"""The ranker's features for one plan's candidates, from what the executor computed (docs/INTERFACES.md §5 rescore)."""
from __future__ import annotations

import math

import numpy as np

from .. import ops
from ..analysis import analyze
from ..plan.nodes import Dense, Lexical


def feature_names_for(st) -> list[str]:
    return ops.feature_names(st.text_fields, st.match_fields())


def _log1p_table(b, st):
    """``f32[N + 1]`` on the device: ``float32(math.log1p(n))`` for every possible match count n (cached per state)."""
    cache = st.__dict__.setdefault("_log1p_counts", {})
    key = (b.name, st.N)
    if key not in cache:
        table = np.fromiter((math.log1p(n) for n in range(st.N + 1)), np.float64, st.N + 1).astype(np.float32)
        cache[key] = b.asarray(table, "float32")
    return cache[key]


def _components(ex, ctx):
    """One plan's feature inputs: its lexical, lexical-rank, dense, dense-rank and RRF rows (f32[N] each, zeros
    when absent), the lexical node (or None) and its match-count mask (or None)."""
    b, st = ex.b, ex.st
    N = st.N
    zeros = b.zeros(N, "float32")
    lex = dense = zeros
    lex_rank = dense_rank = zeros
    rrf = zeros
    lex_node = None
    dense_set = False
    if "hybrid" in ctx:
        node, rrf_row, rank_rows = ctx["hybrid"]
        rrf = rrf_row.reshape(N)
        for q, rr in zip(node.queries, rank_rows):
            if isinstance(q, Lexical) and lex_node is None:
                lex_node = q
                lex = ex.lex_cache[q][0].reshape(N)
                lex_rank = rr.reshape(N)
            elif isinstance(q, Dense) and not dense_set:
                dense_set = True
                dense = ex.dense_cache[q].reshape(N)
                dense_rank = rr.reshape(N)
    else:
        if "lexical" in ctx:
            lex_node, sc, m = ctx["lexical"]
            lex = b.where(m, sc, 0.0).reshape(N)
        if "dense" in ctx:
            _, sims, _ = ctx["dense"]
            dense = sims.reshape(N)
    nm = None
    if lex_node is not None and lex_node.msm:
        nm = ex.lex_cache[lex_node][1] >= max(1, lex_node.msm)
    return (lex, lex_rank, dense, dense_rank, rrf), lex_node, nm


def candidate_features(ex, ctx, cand):
    """``cand int32[1,W]`` (-1 pads) → ``(X f32[1,W,n_features], cand int64[1,W])``.

    The lexical and dense components come from the plan's hybrid query (its first lexical and first dense
    sub-query), or from the first lexical / dense node evaluated when there is no hybrid; a missing component gives
    zero scores and absent ranks."""
    return candidate_features_batch(ex, [ctx], cand)


def candidate_features_batch(ex, ctxs, cand):
    """S6-8: ``candidate_features`` for several plans at once. ``ctxs``: one plan context per row of ``cand
    int32[Q,W]`` (-1 pads) → ``(X f32[Q,W,n_features], cand int64[Q,W])``; row q equals ``candidate_features`` of
    ``ctxs[q]`` alone. The lexical match counts come to the host in one transfer."""
    b, st = ex.b, ex.st
    cand = b.astype(cand, "int64")
    comps, nodes, nms = zip(*(_components(ex, c) for c in ctxs))
    bufs = [b.stack([c[k] for c in comps], 0) for k in range(5)]           # five f32[Q, N]
    # The match counts stay on the device (no transfer for rescoring, INTERFACES §0). Their log1p feature is read
    # from a table of the host's math.log1p of 0..N in float32, so it has the same bits as the host path's.
    zero = b.count_nonzero(b.zeros(1, "bool"))
    counts = b.astype(b.stack([b.count_nonzero(m) if m is not None else zero for m in nms], 0), "int64")
    n_match = _log1p_table(b, st)[counts]
    q_tokens = [set(n.terms) if n is not None else set() for n in nodes]
    q_len = [len(n.terms) if n is not None else 0 for n in nodes]
    mf = st.match_fields()
    match_codes = [[st.keywords[f].matched_codes(t) for t in q_tokens] for f in mf]
    kw_codes = [st.keywords[f].codes for f in mf]
    kw_sizes = [len(st.keywords[f].values) for f in mf]
    X = ops.gather_features(b, cand, *bufs, match_codes, kw_codes, st.static_features(), q_len, n_match, kw_sizes)
    return X, cand
