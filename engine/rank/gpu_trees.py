"""GBDT inference on the GPU (S9-3): the engine's GBDT exported to XGBoost's JSON model and run by XGBoost (on
``cuda`` when there is one) or by cuML's FIL, from the same saved model. Both are optional imports
(``requirements-gpu.txt``); without them the engine's own device walk (``GBDT.predict_backend``) runs.

The export is exact: the engine splits on bins (``bin <= t`` with ``bin = 1 + count(edge <= x)``, NaN in bin 0), which
is ``x < edges[t - 1]`` with NaN going left; XGBoost compares float32 features with float32 thresholds, so each
threshold is the smallest float32 not below the edge (no float32 lies strictly between them, so the decisions are
the same). Leaves carry the engine's values (learning rate included) and ``base_score`` its base.
"""
from __future__ import annotations

import json

import numpy as np


def _ceil32(x: float) -> float:
    f = np.float32(x)
    if float(f) < x:
        f = np.nextafter(f, np.float32(np.inf))
    return float(f)


def to_xgboost_json(model, n_features: int) -> dict:
    trees = []
    for ti, t in enumerate(model.trees):
        feat, thr_bin = t["feature"], t["bin"]
        left, right, value = t["left"], t["right"], t["value"]
        n = len(value)
        parents = [2147483647] * n
        conds, lefts, rights, splits = [], [], [], []
        for i in range(n):
            f = int(feat[i])
            if f < 0:                                         # a leaf
                conds.append(float(np.float32(value[i])))
                lefts.append(-1), rights.append(-1), splits.append(0)
                continue
            b = int(thr_bin[i])
            e = model.edges[f]
            thr = -np.inf if b == 0 else (_ceil32(float(e[b - 1])) if b - 1 < len(e) else np.inf)
            conds.append(float(thr) if np.isfinite(thr) else (-3.4028234663852886e38 if thr < 0 else 3.4028234663852886e38))
            lefts.append(int(left[i])), rights.append(int(right[i])), splits.append(f)
            parents[int(left[i])] = i
            parents[int(right[i])] = i
        trees.append({"base_weights": [float(np.float32(v)) for v in value], "categories": [], "categories_nodes": [],
                      "categories_segments": [], "categories_sizes": [], "default_left": [1 if s >= 0 else 0 for s in lefts],
                      "id": ti, "left_children": lefts, "loss_changes": [0.0] * n, "parents": parents,
                      "right_children": rights, "split_conditions": conds, "split_indices": splits,
                      "split_type": [0] * n, "sum_hessian": [1.0] * n,
                      "tree_param": {"num_deleted": "0", "num_feature": str(n_features), "num_nodes": str(n),
                                     "size_leaf_vector": "1"}})
    return {"learner": {"attributes": {}, "feature_names": [], "feature_types": [],
                        "gradient_booster": {"model": {"gbtree_model_param": {"num_parallel_tree": "1",
                                                                               "num_trees": str(len(trees))},
                                                       "iteration_indptr": list(range(len(trees) + 1)),
                                                       "tree_info": [0] * len(trees), "trees": trees},
                                             "name": "gbtree"},
                        "learner_model_param": {"base_score": repr(float(model.base)), "boost_from_average": "0",
                                                "num_class": "0", "num_feature": str(n_features), "num_target": "1"},
                        "objective": {"name": "reg:squarederror", "reg_loss_param": {"scale_pos_weight": "1"}}},
            "version": [2, 1, 0]}


class GpuTrees:
    """A ranker that runs a GBDT through XGBoost (``runtime="xgboost"``) or cuML FIL (``runtime="fil"``), on the
    GPU when the index is on ``cuda``. ``predict_backend`` keeps the features on the device (DLPack)."""

    def __init__(self, model, n_features: int, runtime: str = "xgboost", device: str = "cpu"):
        self.model, self.runtime, self.device = model, runtime, device
        self.n_features = n_features
        doc = json.dumps(to_xgboost_json(model, n_features)).encode()
        if runtime == "xgboost":
            import xgboost as xgb                              # optional: requirements-gpu.txt
            self.booster = xgb.Booster()
            self.booster.load_model(bytearray(doc))
            self.booster.set_param({"device": "cuda" if device == "cuda" else "cpu"})
        elif runtime == "fil":
            import os
            import tempfile
            from cuml import ForestInference                   # optional: requirements-gpu.txt
            with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as fh:
                fh.write(doc)
            try:
                self.fil = ForestInference.load(fh.name, model_type="xgboost_json", is_classifier=False)
            finally:
                os.unlink(fh.name)
        else:
            raise ValueError("runtime: xgboost | fil")

    def predict(self, X) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, np.float32))
        if self.runtime == "xgboost":
            return np.asarray(self.booster.inplace_predict(X), np.float32)
        return np.asarray(self.fil.predict(X), np.float32).ravel()

    def predict_backend(self, b, X):
        if b.name == "cuda":
            import cupy as cp
            import torch
            Xc = cp.from_dlpack(torch.utils.dlpack.to_dlpack(X.to(torch.float32).contiguous()))
            out = self.booster.inplace_predict(Xc) if self.runtime == "xgboost" else self.fil.predict(Xc)
            return torch.utils.dlpack.from_dlpack(cp.asarray(out, cp.float32).ravel().toDlpack())
        return b.asarray(self.predict(b.to_numpy(X)), "float32")

    def contributions(self, X):
        return self.model.contributions(X)
