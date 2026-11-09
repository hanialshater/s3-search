"""Encoders for intake's enrichment (docs/INTERFACES.md §2): the pluggable hook.

An encoder is any object with ``name: str``, ``dim: int`` and ``encode(texts: list[str]) -> float32[n, dim]``.
Intake L2-normalises what it returns. Register one by name with ``register(name, factory)``; ``factory(dim)``
returns the encoder.

The default, ``HashingEncoder``, is deterministic, needs no fitting and runs on CPU in microseconds per document, so
intake and its tests stay fast. It is the development encoder, not a quality one: the quality encoders
(``tfidf-svd-128``, ``bge-small-en-v1.5``) are the engine's (``engine.encode``), and a deployment that wants them
passes one in (``ingest(..., encoder=...)``).
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Protocol, runtime_checkable

import numpy as np

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
    """The registered encoder for ``model``; the hashing encoder when none is registered under that name."""
    factory = _REGISTRY.get(model)
    return factory(dim) if factory else HashingEncoder(dim)
