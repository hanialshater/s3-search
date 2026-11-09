"""Encoders for dense retrieval (docs/INTERFACES.md §2). ``tfidf-svd-<d>``: TF-IDF over the ``s3_english`` analyzer,
reduced to ``d`` dimensions by a randomized SVD, in numpy only (no scikit-learn). It is fitted on the catalogue at
build time, on the host, so it is identical on every device. Vectors are L2-normalised."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .analysis import analyze

_CHUNK = 262_144


def _sparse_matmul(rows, cols, vals, X, n_out, by="rows"):
    """``A @ X`` (by="rows": group by row) or ``A.T @ X`` (by="cols") for a COO matrix sorted by the grouping key."""
    key, src = (rows, cols) if by == "rows" else (cols, rows)
    out = np.zeros((n_out, X.shape[1]), dtype=np.float64)
    for s in range(0, len(key), _CHUNK):
        k, c, v = key[s:s + _CHUNK], src[s:s + _CHUNK], vals[s:s + _CHUNK]
        contrib = v[:, None] * X[c]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        out[k[starts]] += np.add.reduceat(contrib, starts, axis=0)
    return out


class TfidfSvdEncoder:
    def __init__(self, dim: int = 128, max_features: int = 100_000, seed: int = 0):
        self.dim, self.max_features, self.seed = dim, max_features, seed
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, np.float32)
        self.components = np.zeros((0, dim), np.float32)

    @property
    def name(self):
        return f"tfidf-svd-{self.dim}"

    def _coo(self, texts):
        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            cnt = Counter(self.vocab[w] for w in analyze(t) if w in self.vocab)
            if not cnt:
                continue
            c = np.fromiter(cnt.keys(), np.int64, len(cnt))
            v = np.fromiter(cnt.values(), np.float64, len(cnt)) * self.idf[c]
            v /= np.linalg.norm(v)
            rows.append(np.full(len(c), i, np.int64)), cols.append(c), vals.append(v)
        if not rows:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)

    def fit(self, texts) -> "TfidfSvdEncoder":
        texts = list(texts)
        n = len(texts)
        df = Counter()
        for t in texts:
            df.update(set(analyze(t)))
        terms = sorted(df, key=lambda w: (-df[w], w))[: self.max_features]
        terms.sort()
        self.vocab = {w: i for i, w in enumerate(terms)}
        d = np.array([df[w] for w in terms], dtype=np.float64)
        self.idf = (np.log((1 + n) / (1 + d)) + 1.0).astype(np.float32)
        V = len(terms)
        rows, cols, vals = self._coo(texts)
        self.components = np.zeros((V, self.dim), np.float32)
        l = min(self.dim + 10, n, V)
        if l == 0 or len(vals) == 0:
            return self
        rng = np.random.default_rng(self.seed)
        by_col = np.argsort(cols, kind="stable")
        rc, cc, vc = rows[by_col], cols[by_col], vals[by_col]
        Y = _sparse_matmul(rows, cols, vals, rng.standard_normal((V, l)), n, "rows")
        for _ in range(3):                                  # power iterations
            Qm, _ = np.linalg.qr(Y)
            Z, _ = np.linalg.qr(_sparse_matmul(rc, cc, vc, Qm, V, "cols"))
            Y = _sparse_matmul(rows, cols, vals, Z, n, "rows")
        Qm, _ = np.linalg.qr(Y)
        Bt = _sparse_matmul(rc, cc, vc, Qm, V, "cols")      # (Q^T A)^T, V x l
        U, S, _ = np.linalg.svd(Bt, full_matrices=False)    # right singular vectors of A are U here
        k = min(self.dim, U.shape[1])
        comp = U[:, :k]
        comp *= np.where(comp[np.abs(comp).argmax(axis=0), np.arange(k)] < 0, -1.0, 1.0)  # deterministic sign
        self.components[:, :k] = comp
        return self

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        rows, cols, vals = self._coo(texts)
        out = _sparse_matmul(rows, cols, vals, self.components.astype(np.float64), len(texts), "rows")
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norm > 0, norm, 1.0)).astype(np.float32)

    encode_docs = encode
    encode_queries = encode

    def to_arrays(self) -> dict:
        terms = sorted(self.vocab, key=self.vocab.get)
        return {"terms": np.array(terms, dtype=object), "idf": self.idf, "components": self.components,
                "dim": np.int64(self.dim)}

    @classmethod
    def from_arrays(cls, a: dict) -> "TfidfSvdEncoder":
        enc = cls(dim=int(a["dim"]))
        enc.vocab = {str(w): i for i, w in enumerate(a["terms"])}
        enc.idf, enc.components = np.asarray(a["idf"], np.float32), np.asarray(a["components"], np.float32)
        return enc


def get_encoder(name: str):
    """A fresh (unfitted) encoder by name. ``tfidf-svd-<d>``; ``bge-small-en-v1.5`` needs sentence-transformers."""
    m = re.fullmatch(r"tfidf-svd-(\d+)", name)
    if m:
        return TfidfSvdEncoder(dim=int(m.group(1)))
    raise ValueError(f"unknown encoder {name!r} (v1: tfidf-svd-<d>)")


def doc_texts(catalogue, inputs) -> list[str]:
    """The text an encoder reads for each product: its input columns joined by ' | ' (the prototype's form)."""
    parts = [catalogue[c].fillna("").astype(str) for c in inputs]
    out = parts[0]
    for p in parts[1:]:
        out = out + " | " + p
    return out.tolist()
