"""The ranker's feature gather (the prototype's ``GpuEngine.stage1`` with ``complete=True``): every candidate's
lexical and dense scores read from the full buffers, its ranks in each list, RRF, query-value matches on keyword
fields, static document features and query features. NaN marks "absent" (a rank of a doc missing from a list)."""
from __future__ import annotations

import math

FEATURE_BASE = ["lex_score", "lex_norm", "lex_rank_log", "dense_sim", "dense_gap", "dense_rank_log", "rrf", "in_both"]


def feature_names(text_fields, match_fields) -> list[str]:
    names = FEATURE_BASE + [f"match_{f}" for f in match_fields] + [f"len_{f}" for f in text_fields]
    if match_fields:
        names.append(f"pop_log_{match_fields[0]}")
    return names + ["q_len", "n_match_log"]


def gather_features(b, cand, lex, lex_rank, dense, dense_rank, rrf, match_codes, kw_codes, static, q_len, n_match,
                    kw_sizes=None):
    """``cand int64[Q,W]`` (-1 pads). ``lex, lex_rank, dense, dense_rank, rrf``: f32[Q,N] buffers (ranks 1-based, 0 when
    absent). ``match_codes``: per match field, per query, the matching codes (host int64 arrays); ``kw_codes``: per
    match field, the docs' codes on the device. ``static``: f32[N,S]. ``q_len``, ``n_match``: host lists per query.
    Returns ``f32[Q,W,n_features]``."""
    nan = float("nan")
    Q, W = cand.shape
    safe = b.where(cand >= 0, cand, 0)
    g = lambda buf: b.take_along(buf, safe, axis=1)
    lex_f, lr, dense_f, dr, rrf_f = g(lex), g(lex_rank), g(dense), g(dense_rank), g(rrf)
    in_l, in_d = lr > 0, dr > 0
    lex_top = _row_max(b, lex)
    top = _masked_row_max(b, dense_f, in_d)
    dense_top = b.where(top > float("-inf"), top, nan)
    cols = [lex_f, lex_f / lex_top, b.where(in_l, b.log1p(lr), nan), dense_f, dense_f - dense_top,
            b.where(in_d, b.log1p(dr), nan), rrf_f, b.astype(in_l & in_d, "float32")]
    for f, (codes_dev, per_q) in enumerate(zip(kw_codes, match_codes)):
        # S6-11: every query's matching codes in one transfer and one isin, keyed (query, code), instead of one each.
        # ``kw_sizes``: each field's number of codes (all codes are below it), so no device read is needed.
        cc = b.astype(codes_dev[safe], "int64")                        # [Q, W]
        span = int(kw_sizes[f]) if kw_sizes is not None else 1 + int(b.to_numpy(codes_dev).max(initial=0))
        flat = [q * span + int(c) for q in range(Q) for c in per_q[q]]
        if flat:
            keys = b.arange(Q).reshape(Q, 1) * span + cc
            cols.append(b.astype(b.isin(keys, b.asarray(flat, "int64")), "float32"))
        else:
            cols.append(b.zeros((Q, W), "float32"))
    st = static[safe]                                                  # [Q,W,S]
    if isinstance(n_match, (list, tuple)):                            # host counts
        qf = b.asarray([[float(q_len[q]), math.log1p(float(n_match[q]))] for q in range(Q)], "float32")
    else:                                     # S6-8: log1p(n_match) already, f32[Q] on the device
        qf = b.stack([b.asarray([float(x) for x in q_len], "float32"), b.astype(n_match, "float32")], 1)
    X = b.concatenate([b.stack(cols, 2), st, qf.reshape(Q, 1, 2) + b.zeros((Q, W, 2), "float32")], axis=2)
    return X


def _row_max(b, buf):
    """``f32[Q,1]``: each row's maximum, NaN when it is not positive (the prototype's ``lex_top``)."""
    idx, val = b.topk(buf, 1)
    return b.where(val > 0, val, float("nan"))


def _masked_row_max(b, x, m):
    idx, val = b.topk(x, 1, m)
    return val
