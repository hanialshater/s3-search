"""The ranker's training recipe, fixed (docs/BOARD.md S1-5).

The prototype labelled the *top* unjudged candidates of each query as negatives (``NEG_TOP``), and unjudged does not
mean irrelevant: those are the products most like the judged good ones. The fix trains on the judged candidates with
their labels plus a uniform random sample of the unjudged candidates as label 0, and never the top unjudged ones.
"""
from __future__ import annotations

import numpy as np

from .gbdt import GBDT

LABEL_INT = {"E": 3, "S": 2, "C": 1, "I": 0}


def select_rows(cand: np.ndarray, judged: np.ndarray, rng: np.random.Generator, n_random: int) -> np.ndarray:
    """Positions in ``cand`` to train on: every judged candidate, and ``n_random`` unjudged ones drawn uniformly
    from all unjudged candidates (not from the top)."""
    valid = cand >= 0
    unjudged = np.flatnonzero(~judged & valid)
    rnd = rng.choice(unjudged, size=min(n_random, len(unjudged)), replace=False) if len(unjudged) else unjudged
    return np.sort(np.concatenate([np.flatnonzero(judged & valid), rnd]))


def build_training(index, fit, request_for, window: int = 1000, n_random: int = 20, seed: int = 13,
                   max_queries: int | None = None):
    """``fit``: DataFrame[query_id, query, product_id, esci_label] (the loader's fit split). ``request_for(text)``:
    the request whose candidates the ranker will rescore (a hybrid query). Returns ``(X, y, group)``."""
    rng = np.random.default_rng(seed)
    Xs, ys, gs = [], [], []
    for g, (qid, grp) in enumerate(fit.groupby("query_id", sort=True)):
        if max_queries is not None and g >= max_queries:
            break
        text = str(grp["query"].iloc[0])
        cand_ids, X = index.candidate_features(request_for(text), window)
        lab_of = dict(zip(grp["product_id"].astype(str), grp["esci_label"].map(LABEL_INT)))
        judged = np.array([c in lab_of for c in cand_ids], bool)
        if not judged.any():
            continue
        sel = select_rows(np.arange(len(cand_ids)), judged, rng, n_random)
        Xs.append(X[sel])
        ys.append(np.array([lab_of.get(cand_ids[i], 0) for i in sel], np.float32))
        gs.append(np.full(len(sel), g))
    if not Xs:
        raise ValueError("no fit query had a judged product among its candidates")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def train_ranker(index, fit, request_for, window: int = 1000, n_random: int = 20, seed: int = 13,
                 max_queries: int | None = None, **gbdt) -> GBDT:
    X, y, _ = build_training(index, fit, request_for, window, n_random, seed, max_queries)
    return GBDT(**gbdt).fit(X, y)
