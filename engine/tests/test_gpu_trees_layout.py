"""The XGBoost export's tree layout, checked on CPU without XGBoost or a GPU.

XGBoost's GPU predictor steps to ``left_child + (x >= threshold)``, so it needs every right child right after its left
child. The engine's trees are depth-first; exported as they were, XGBoost on cuda took the wrong branch on about 94% of
rows (6.57 off at worst) while its CPU predictor was exact. Here the exported model is walked the GPU predictor's way.
"""
import numpy as np

from engine.rank.gpu_trees import to_xgboost_json
from engine.tests.test_s9_gpu_optional import gbdt_and_rows


def _walk_like_the_gpu(doc, X):
    learner = doc["learner"]
    base = float(learner["learner_model_param"]["base_score"])
    out = np.full(len(X), base, np.float64)
    for t in learner["gradient_booster"]["model"]["trees"]:
        lc, cond, feat, dl = t["left_children"], t["split_conditions"], t["split_indices"], t["default_left"]
        for r in range(len(X)):
            i = 0
            while lc[i] != -1:
                x = X[r, feat[i]]
                i = (lc[i] if dl[i] else lc[i] + 1) if np.isnan(x) else lc[i] + int(not (np.float32(x) < np.float32(cond[i])))
            out[r] += cond[i]
    return out.astype(np.float32)


def test_every_right_child_follows_its_left_child():
    m, _ = gbdt_and_rows()
    for t in to_xgboost_json(m, 18)["learner"]["gradient_booster"]["model"]["trees"]:
        lc, rc, parents = t["left_children"], t["right_children"], t["parents"]
        assert parents[0] == 2147483647
        for i, (lo, hi) in enumerate(zip(lc, rc)):
            if lo == -1:
                assert hi == -1
                continue
            assert hi == lo + 1 and parents[lo] == parents[hi] == i and lo > i
        assert int(t["tree_param"]["num_nodes"]) == len(lc)


def test_the_gpu_walk_equals_the_engine():
    m, X = gbdt_and_rows()
    X = X[:2000]
    got = _walk_like_the_gpu(to_xgboost_json(m, X.shape[1]), X)
    assert np.abs(got - m.predict(X)).max() <= 1e-5
