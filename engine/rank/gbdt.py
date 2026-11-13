"""A small histogram gradient-boosted tree ensemble in numpy: the CPU stand-in for the prototype's XGBoost ranker.

Squared loss on graded labels (E=3, S=2, C=1, I=0), so it is pointwise where XGBoost ran a ranking objective; it is
documented as a stand-in (docs/decisions/2026-11-09-s1-engine.md). NaN (an absent feature) gets its own lowest bin.
Deterministic: no sampling."""
from __future__ import annotations

import numpy as np


class GBDT:
    def __init__(self, n_rounds=100, learning_rate=0.1, max_depth=4, min_leaf=20, n_bins=32, l2=1.0):
        self.n_rounds, self.lr, self.max_depth = n_rounds, learning_rate, max_depth
        self.min_leaf, self.n_bins, self.l2 = min_leaf, n_bins, l2
        self.edges: list[np.ndarray] = []
        self.trees: list[dict] = []
        self.base = 0.0

    def _bin(self, X):
        X = np.asarray(X, np.float64)
        out = np.zeros(X.shape, np.int32)
        for f, e in enumerate(self.edges):
            col = X[:, f]
            ok = ~np.isnan(col)
            out[ok, f] = 1 + np.searchsorted(e, col[ok], side="right")
        return out

    def fit(self, X, y) -> "GBDT":
        X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
        n, F = X.shape
        self.edges = []
        for f in range(F):
            col = X[:, f][~np.isnan(X[:, f])]
            e = np.unique(np.quantile(col, np.linspace(0, 1, self.n_bins + 1)[1:-1])) if len(col) else np.zeros(0)
            self.edges.append(e)
        B = self.n_bins + 1
        Xb = self._bin(X)
        self.base = float(y.mean()) if n else 0.0
        pred = np.full(n, self.base)
        self.trees = []
        flat = Xb + np.arange(F)[None, :] * B
        for _ in range(self.n_rounds):
            g = pred - y
            tree = {"feature": [], "bin": [], "left": [], "right": [], "value": []}
            leaf_of = np.zeros(n, np.int64)
            self._grow(tree, 0, np.arange(n), flat, Xb, g, F, B, leaf_of)
            vals = np.asarray(tree["value"])
            pred += vals[leaf_of]
            self.trees.append({k: np.asarray(v) for k, v in tree.items()})
        return self

    def _new(self, tree, value):
        for k in ("feature", "bin", "left", "right"):
            tree[k].append(-1)
        tree["value"].append(value)
        return len(tree["value"]) - 1

    def _grow(self, tree, depth, idx, flat, Xb, g, F, B, leaf_of):
        G, H = g[idx].sum(), float(len(idx))
        node = self._new(tree, -self.lr * G / (H + self.l2))
        if depth >= self.max_depth or len(idx) < 2 * self.min_leaf:
            leaf_of[idx] = node
            return node
        hg = np.bincount(flat[idx].ravel(), weights=np.repeat(g[idx], F), minlength=F * B).reshape(F, B)
        hh = np.bincount(flat[idx].ravel(), minlength=F * B).reshape(F, B).astype(np.float64)
        GL, HL = np.cumsum(hg, axis=1)[:, :-1], np.cumsum(hh, axis=1)[:, :-1]
        GR, HR = G - GL, H - HL
        gain = GL ** 2 / (HL + self.l2) + GR ** 2 / (HR + self.l2) - G ** 2 / (H + self.l2)
        gain[(HL < self.min_leaf) | (HR < self.min_leaf)] = -np.inf
        f, t = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not np.isfinite(gain[f, t]) or gain[f, t] <= 1e-12:
            leaf_of[idx] = node
            return node
        go_left = Xb[idx, f] <= t
        tree["feature"][node], tree["bin"][node] = int(f), int(t)
        left = self._grow(tree, depth + 1, idx[go_left], flat, Xb, g, F, B, leaf_of)
        right = self._grow(tree, depth + 1, idx[~go_left], flat, Xb, g, F, B, leaf_of)
        tree["left"][node], tree["right"][node] = left, right
        return node

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X)
        if len(X) == 0:
            return np.zeros(0, np.float32)
        Xb = self._bin(X)
        out = np.full(len(X), self.base)
        rows = np.arange(len(X))
        for t in self.trees:
            node = np.zeros(len(X), np.int64)
            for _ in range(self.max_depth + 1):
                f = t["feature"][node]
                inner = f >= 0
                if not inner.any():
                    break
                go_left = Xb[rows, np.where(inner, f, 0)] <= t["bin"][node]
                node = np.where(inner, np.where(go_left, t["left"][node], t["right"][node]), node)
            out += t["value"][node]
        return out.astype(np.float32)

    def _device_arrays(self, b):
        """The trees as padded node arrays on ``b``'s device (cached per backend): feature, bin, left, right
        int64[T, M] and value float64[T, M]; the edges as float64[F, E] padded with +inf."""
        cache = self.__dict__.setdefault("_dev", {})
        key = (b.name, getattr(b, "accumulate", None))
        if key not in cache:
            T = len(self.trees)
            M = max((len(t["value"]) for t in self.trees), default=1)
            arr = {k: np.full((max(T, 1), M), -1 if k != "value" else 0, np.int64 if k != "value" else np.float64)
                   for k in ("feature", "bin", "left", "right", "value")}
            for i, t in enumerate(self.trees):
                for k in arr:
                    arr[k][i, :len(t[k])] = t[k]
            E = max((len(e) for e in self.edges), default=0)
            edges = np.full((len(self.edges), max(E, 1)), np.inf)
            for f, e in enumerate(self.edges):
                edges[f, :len(e)] = e
            dev = {k: b.asarray(v, "float64" if k == "value" else "int64") for k, v in arr.items()}
            dev["edges"] = b.asarray(edges, "float64")
            dev["real"] = b.asarray(np.isfinite(edges), "bool")
            cache[key] = dev
        return cache[key]

    def predict_backend(self, b, X):
        """``predict`` on the device: ``X`` float32[n, F] on ``b``'s device → float32[n] there. Binning and tree
        walks are gathers over the node arrays, one step per depth level; trees are added one at a time in float64,
        in the same order as ``predict``, so both give the same float32 values."""
        n = X.shape[0]
        if n == 0 or not self.trees:
            return b.full(n, self.base, "float32")
        d = self._device_arrays(b)
        x = b.astype(X, "float64")
        nan = b.isnan(x)
        ge = (x.reshape(n, x.shape[1], 1) >= d["edges"].reshape(1, *d["edges"].shape)) & \
            d["real"].reshape(1, *d["real"].shape)                                    # real edges <= x
        xb = b.where(nan, 0, b.sum(b.astype(ge, "int64"), axis=2) + 1)
        # Every tree walks at once (S6-5): node is int64[n, T], an index into the flattened [T * M] node arrays, so
        # a depth level is a few gathers whatever the number of trees (one tree at a time was T x depth small
        # kernels: 100.7 ms at batch 1 on an A100).
        T, M = d["feature"].shape
        feat, thr = d["feature"].reshape(-1), d["bin"].reshape(-1)
        left, right = d["left"].reshape(-1), d["right"].reshape(-1)
        base = (b.arange(T) * M).reshape(1, T)
        rows = b.arange(n).reshape(n, 1)
        node = b.zeros((n, T), "int64")
        for _ in range(self.max_depth + 1):
            at = base + node
            f = feat[at]
            inner = f >= 0
            go_left = xb[rows, b.where(inner, f, 0)] <= thr[at]
            node = b.where(inner, b.where(go_left, left[at], right[at]), node)
        leaf = d["value"].reshape(-1)[base + node]                                  # float64[n, T]
        out = b.full(n, self.base, "float64")
        for t in range(len(self.trees)):              # added one tree at a time, in predict's order: the same bits
            out = out + leaf[:, t]
        return b.astype(out, "float32")

    def contributions(self, X):
        """Exact per-feature contributions (Saabas path attribution): ``(base float64[n], contrib float64[n, F])``
        with ``base + contrib.sum(1) == predict(X)`` up to float rounding. Each split on a sample's path adds the
        change in node value to the feature it splits on; ``base`` is the model's base plus every tree's root
        value."""
        X = np.asarray(X)
        n, F = X.shape
        contrib = np.zeros((n, F), np.float64)
        base = np.full(n, self.base, np.float64)
        if n == 0:
            return base, contrib
        Xb = self._bin(X)
        rows = np.arange(n)
        for t in self.trees:
            node = np.zeros(n, np.int64)
            base += t["value"][0]
            for _ in range(self.max_depth + 1):
                f = t["feature"][node]
                inner = f >= 0
                if not inner.any():
                    break
                fi = np.where(inner, f, 0)
                go_left = Xb[rows, fi] <= t["bin"][node]
                child = np.where(inner, np.where(go_left, t["left"][node], t["right"][node]), node)
                delta = np.where(inner, t["value"][child] - t["value"][node], 0.0)
                np.add.at(contrib, (rows, fi), delta)
                node = child
        return base, contrib

    def to_dict(self) -> dict:
        return {"params": dict(n_rounds=self.n_rounds, learning_rate=self.lr, max_depth=self.max_depth,
                               min_leaf=self.min_leaf, n_bins=self.n_bins, l2=self.l2),
                "base": self.base, "edges": [e.tolist() for e in self.edges],
                "trees": [{k: v.tolist() for k, v in t.items()} for t in self.trees]}

    @classmethod
    def from_dict(cls, d) -> "GBDT":
        m = cls(**d["params"])
        m.base, m.edges = d["base"], [np.asarray(e) for e in d["edges"]]
        m.trees = [{k: np.asarray(v) for k, v in t.items()} for t in d["trees"]]
        return m
