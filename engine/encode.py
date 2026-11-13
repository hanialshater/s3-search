"""Encoders for dense retrieval (docs/INTERFACES.md §2). ``tfidf-svd-<d>``: TF-IDF over the ``s3_english`` analyzer,
reduced to ``d`` dimensions by a randomized SVD, in numpy only (no scikit-learn). It is fitted on the catalogue at
build time, on the host, so it is identical on every device. Vectors are L2-normalised."""
from __future__ import annotations

import re

import numpy as np

from .analysis import analyze

def _sparse_matmul(rows, cols, vals, X, n_out, by="rows"):
    """``A @ X`` (by="rows") or ``A.T @ X`` (by="cols") for a COO matrix: one ``bincount`` per column of X (a
    contiguous gather and a float64 sum in entry order, so deterministic), which moves far less memory than
    gathering whole rows of X."""
    key, src = (rows, cols) if by == "rows" else (cols, rows)
    XT = np.ascontiguousarray(np.asarray(X, np.float64).T)
    out = np.empty((X.shape[1], n_out), np.float64)
    for j in range(X.shape[1]):
        out[j] = np.bincount(key, weights=vals * XT[j][src], minlength=n_out)
    return out.T


def _orthonormal(Y):
    """An orthonormal basis of Y's columns: Cholesky QR, twice (CholeskyQR2: a small Gram matrix and a triangular
    solve instead of a tall LAPACK QR); Householder QR if the Gram matrix is not positive definite."""
    Q = Y
    for _ in range(2):
        G = Q.T @ Q
        try:
            L = np.linalg.cholesky(G)
        except np.linalg.LinAlgError:
            return np.linalg.qr(Y)[0]
        Q = Q @ np.linalg.inv(L).T                    # L is l x l (small): Q L^-T
    return Q


def _token_ids(texts, word_id: dict, assign):
    """Analyse every text once: (doc int64[T], term id int64[T]) for its tokens, the same tokens as
    ``engine.analysis.analyze``. ``word_id`` caches raw word -> id (-1: stopword or unknown); ``assign(token)``
    gives a token's id (or -1) the first time a word is seen."""
    from itertools import chain

    from .analysis import _WORD, _token
    words_per_doc = [_WORD.findall(str(t).lower()) if t is not None and t == t and type(t).__name__ != "NAType"
                     else [] for t in texts]
    n = np.fromiter((len(w) for w in words_per_doc), np.int64, len(words_per_doc))
    words = list(chain.from_iterable(words_per_doc))
    get = word_id.get
    ids = [get(w, -2) for w in words]
    if -2 in ids:
        for i, x in enumerate(ids):
            if x == -2:
                w = words[i]
                t = _token(w)
                ids[i] = word_id[w] = -1 if t is None else assign(t)
    ids = np.asarray(ids, np.int64)
    docs = np.repeat(np.arange(len(n), dtype=np.int64), n)
    keep = ids >= 0
    return docs[keep], ids[keep]


class TfidfSvdEncoder:
    def __init__(self, dim: int = 128, max_features: int = 100_000, seed: int = 0, fit: str = "numpy"):
        """``fit`` (S9-2): "numpy" (the reference, the default), "gpu" (alias "cuml": the sparse products and the
        orthonormalisation on the GPU with CuPy, the same algorithm and random start), or "auto" (gpu when CuPy and
        CUDA are present, else numpy). The fitted arrays are the same format either way, and encoding queries uses
        them on any device."""
        self.dim, self.max_features, self.seed = dim, max_features, seed
        self.fit_backend = fit
        self.fitted_with = None
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, np.float32)
        self.components = np.zeros((0, dim), np.float32)
        self._word_id: dict[str, int] = {}

    @property
    def name(self):
        return f"tfidf-svd-{self.dim}"

    def _coo_from_ids(self, docs, ids, n):
        """L2-normalised TF-IDF rows as COO, sorted by (row, col)."""
        V = max(len(self.vocab), 1)
        if len(ids) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
        uk, cnt = np.unique(docs * V + ids, return_counts=True)
        rows, cols = uk // V, uk % V
        vals = cnt.astype(np.float64) * self.idf[cols]
        norm = np.sqrt(np.bincount(rows, weights=vals * vals, minlength=n))
        return rows, cols, vals / norm[rows]

    def _coo(self, texts):
        texts = list(texts)
        vocab = self.vocab
        docs, ids = _token_ids(texts, self._word_id, lambda t: vocab.get(t, -1))
        return self._coo_from_ids(docs, ids, len(texts))

    def fit_transform(self, texts) -> np.ndarray:
        """``fit`` then ``encode`` of the same texts, analysing them once."""
        rows, cols, vals, n = self._fit(texts)
        return self._project(rows, cols, vals, n)

    def fit(self, texts) -> "TfidfSvdEncoder":
        self._fit(texts)
        return self

    def _fit(self, texts):
        from ._blas import limit_to_affinity
        limit_to_affinity()
        texts = list(texts)
        n = len(texts)
        prov: dict[str, int] = {}
        word_prov: dict[str, int] = {}
        docs, pids = _token_ids(texts, word_prov, lambda t: prov.setdefault(t, len(prov)))
        T = max(len(prov), 1)
        names = sorted(prov, key=prov.get)                      # provisional id -> token
        df = np.bincount(np.unique(docs * T + pids) % T, minlength=len(prov)) if len(pids) else np.zeros(len(prov))
        top = sorted(range(len(prov)), key=lambda i: (-df[i], names[i]))[: self.max_features]
        terms = sorted(names[i] for i in top)
        self.vocab = {w: i for i, w in enumerate(terms)}
        remap = np.full(len(prov) + 1, -1, np.int64)
        for w, i in self.vocab.items():
            remap[prov[w]] = i
        d = np.array([df[prov[w]] for w in terms], dtype=np.float64)
        self.idf = (np.log((1 + n) / (1 + d)) + 1.0).astype(np.float32)
        self._word_id = {w: (int(remap[x]) if x >= 0 else -1) for w, x in word_prov.items()}
        fid = remap[pids] if len(pids) else pids
        keep = fid >= 0
        rows, cols, vals = self._coo_from_ids(docs[keep], fid[keep], n)
        V = len(terms)
        self.components = np.zeros((V, self.dim), np.float32)
        l = min(self.dim + 10, n, V)
        if l == 0 or len(vals) == 0:
            return rows, cols, vals, n
        rng = np.random.default_rng(self.seed)
        omega = rng.standard_normal((V, l))
        if _gpu_fit_wanted(self.fit_backend):
            self.components[:, :] = _svd_gpu(rows, cols, vals, n, V, omega, self.dim)
            self.fitted_with = "gpu"
            return rows, cols, vals, n
        self.fitted_with = "numpy"
        rc, cc, vc = rows, cols, vals                       # bincount needs no sort by column
        Y = _sparse_matmul(rows, cols, vals, omega, n, "rows")
        for _ in range(3):                                  # power iterations
            Qm = _orthonormal(Y)
            Z = _orthonormal(_sparse_matmul(rc, cc, vc, Qm, V, "cols"))
            Y = _sparse_matmul(rows, cols, vals, Z, n, "rows")
        Qm = _orthonormal(Y)
        Bt = _sparse_matmul(rc, cc, vc, Qm, V, "cols")      # (Q^T A)^T, V x l
        w, Vv = np.linalg.eigh(Bt.T @ Bt)                   # Bt = U S Vv^T: U = Bt Vv / S (a small eigenproblem)
        order = np.argsort(-w, kind="stable")
        w, Vv = np.clip(w[order], 0, None), Vv[:, order]
        S = np.sqrt(w)
        k = min(self.dim, int((S > S[0] * 1e-10).sum()) if len(S) and S[0] > 0 else 0)
        comp = (Bt @ Vv[:, :k]) / S[:k]
        comp *= np.where(comp[np.abs(comp).argmax(axis=0), np.arange(k)] < 0, -1.0, 1.0)  # deterministic sign
        self.components[:, :k] = comp
        return rows, cols, vals, n

    def _project(self, rows, cols, vals, n):
        # Only the rows of the components that these texts use, in float64: a query touches a handful of the V
        # rows, and copying all of them (V x d float64, twice) per call was most of a query's encode time (S6-4).
        # The same values are summed in the same order, so the result is bit-for-bit the full product's.
        used, local = np.unique(cols, return_inverse=True)
        sub = self.components[used].astype(np.float64)
        out = _sparse_matmul(rows, local.reshape(-1).astype(np.int64), vals, sub, n, "rows")
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norm > 0, norm, 1.0)).astype(np.float32)

    def encode(self, texts) -> np.ndarray:
        texts = list(texts)
        rows, cols, vals = self._coo(texts)
        return self._project(rows, cols, vals, len(texts))

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
        enc._word_id = {}
        enc.idf, enc.components = np.asarray(a["idf"], np.float32), np.asarray(a["components"], np.float32)
        return enc


def _gpu_fit_wanted(fit: str) -> bool:
    if fit == "numpy":
        return False
    if fit not in ("gpu", "cuml", "auto"):
        raise ValueError("fit: numpy | gpu | cuml | auto")
    try:
        import cupy as cp
        ok = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:                                       # no CuPy, no driver, no GPU
        ok = False
    if not ok and fit != "auto":
        from .errors import DeviceUnavailable
        raise DeviceUnavailable(f"fit={fit!r} needs CuPy and a CUDA GPU (requirements-gpu.txt)")
    return ok


def _svd_gpu(rows, cols, vals, n, V, omega, dim):
    """The numpy fit's randomized SVD (same start, 3 power iterations), on the GPU: CuPy sparse products and QR."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    A = csp.csr_matrix((cp.asarray(vals), (cp.asarray(rows), cp.asarray(cols))), shape=(n, V))
    At = A.T.tocsr()
    Y = A @ cp.asarray(omega)
    for _ in range(3):
        Q, _ = cp.linalg.qr(Y)
        Z, _ = cp.linalg.qr(At @ Q)
        Y = A @ Z
    Q, _ = cp.linalg.qr(Y)
    Bt = At @ Q
    w, Vv = cp.linalg.eigh(Bt.T @ Bt)
    order = cp.argsort(-w)
    w, Vv = cp.clip(w[order], 0, None), Vv[:, order]
    S = cp.sqrt(w)
    s0 = float(S[0]) if len(S) else 0.0
    k = min(dim, int((S > s0 * 1e-10).sum())) if s0 > 0 else 0
    comp = (Bt @ Vv[:, :k]) / S[:k]
    comp = cp.asnumpy(comp)
    comp *= np.where(comp[np.abs(comp).argmax(axis=0), np.arange(k)] < 0, -1.0, 1.0)
    out = np.zeros((V, dim), np.float32)
    out[:, :k] = comp
    return out


def get_encoder(name: str, fit: str = "numpy"):
    """A fresh (unfitted) encoder by name. ``tfidf-svd-<d>``; ``fit``: numpy | gpu | cuml | auto (S9-2)."""
    m = re.fullmatch(r"tfidf-svd-(\d+)", name)
    if m:
        return TfidfSvdEncoder(dim=int(m.group(1)), fit=fit)
    raise ValueError(f"unknown encoder {name!r} (v1: tfidf-svd-<d>)")


def doc_texts(catalogue, inputs) -> list[str]:
    """The text an encoder reads for each product: its input columns joined by ' | ' (the prototype's form)."""
    parts = [catalogue[c].fillna("").astype(str) for c in inputs]
    out = parts[0]
    for p in parts[1:]:
        out = out + " | " + p
    return out.tolist()
