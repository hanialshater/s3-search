"""Encoders for intake's enrichment (docs/INTERFACES.md §2): the pluggable hook.

An encoder is any object with ``name: str``, ``dim: int`` and ``encode(texts: list[str]) -> float32[n, dim]``.
An encoder that must be fitted also has ``fit(texts)``, ``to_arrays()`` and ``from_arrays(arrays)`` (the engine's
``TfidfSvdEncoder``): intake fits it once per store, on the first batch that needs it, and keeps the fitted arrays in
the store (``store.blob("encoder.<field>")``), so later batches, the change feed and the indexer use the same one.

Resolution of a model name (``get_encoder``): a factory registered here with ``register(name, factory)``; else the
engine's encoder (``engine.encode.get_encoder``, v1 ``tfidf-svd-<d>``), so a segment can answer ``neural`` queries;
else ``HashingEncoder``, a deterministic development encoder (signed feature hashing; ``knn`` only).

The text an encoder reads is the engine's form (``engine.encode.doc_texts``): the input fields joined by ``" | "``,
a missing field as ``""``.
"""
from __future__ import annotations

import hashlib
import io
import re
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from dataplane.mapping import is_missing

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    """Signed feature hashing of lowercased words and word bigrams into ``dim`` buckets (blake2b, so the same
    text gives the same vector in every process)."""

    def __init__(self, dim: int = 128):
        self.dim = int(dim)
        self.name = f"hash-{self.dim}"
        self._cache: dict[str, tuple[int, float]] = {}

    def _slot(self, tok: str) -> tuple[int, float]:
        s = self._cache.get(tok)
        if s is None:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            s = (h % self.dim, 1.0 if (h >> 63) & 1 else -1.0)
            if len(self._cache) < 1_000_000:
                self._cache[tok] = s
        return s

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = _WORD.findall(t.lower())
            row = out[i]
            for tok in words + [a + " " + b for a, b in zip(words, words[1:])]:
                j, sign = self._slot(tok)
                row[j] += sign
        return out


_REGISTRY: dict[str, Callable[[int], Encoder]] = {}


def register(name: str, factory: Callable[[int], Encoder]) -> None:
    _REGISTRY[name] = factory


def get_encoder(model: str, dim: int) -> Encoder:
    """A fresh encoder for ``model``: registered, else the engine's, else the hashing encoder."""
    factory = _REGISTRY.get(model)
    if factory:
        return factory(dim)
    try:
        from engine.encode import get_encoder as engine_encoder
        enc = engine_encoder(model)
    except (ImportError, ValueError):
        return HashingEncoder(dim)
    return enc if int(enc.dim) == dim else HashingEncoder(dim)


def needs_fit(enc) -> bool:
    """True for a fittable encoder that has not been fitted (the engine's encoders: an empty vocabulary)."""
    return hasattr(enc, "fit") and hasattr(enc, "to_arrays") and not getattr(enc, "vocab", True)


def doc_text(doc: dict, inputs: list[str]) -> str:
    return " | ".join("" if is_missing(doc.get(i)) else str(doc[i]) for i in inputs)


# ---- the fitted encoder, kept in the store -------------------------------------------------------------------------

def _key(field_name: str) -> str:
    return f"encoder.{field_name}"


def save_fitted(store, field_name: str, enc) -> None:
    a = enc.to_arrays()
    buf = io.BytesIO()
    np.savez(buf, terms=np.asarray([str(t) for t in a["terms"]], dtype=str), idf=np.asarray(a["idf"], np.float32),
             components=np.asarray(a["components"], np.float32), dim=np.int64(a["dim"]))
    store.set_blob(_key(field_name), buf.getvalue())


def fitted_arrays(store, field_name: str) -> dict | None:
    """The fitted encoder arrays stored for a vector field (``terms``, ``idf``, ``components``, ``dim``), or None:
    the form ``engine.segment.SegmentArrays.encoders`` holds."""
    raw = store.blob(_key(field_name))
    if raw is None:
        return None
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {"terms": np.asarray(z["terms"].tolist(), dtype=object), "idf": z["idf"],
                "components": z["components"], "dim": np.int64(z["dim"])}


def load_fitted(store, field_name: str, like) -> object | None:
    """An encoder of ``like``'s class restored from the store, or None when the store holds none."""
    arrays = fitted_arrays(store, field_name)
    if arrays is None or not hasattr(type(like), "from_arrays"):
        return None
    return type(like).from_arrays(arrays)
