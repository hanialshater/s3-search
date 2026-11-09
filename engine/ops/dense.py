"""Exact dense retrieval: inner product over L2-normalised vectors, top-k with ties by doc id."""
from __future__ import annotations


def dense_scores(b, emb, q):
    """``f32[Q,N]`` inner products (cosine on normalised vectors)."""
    return b.matmul(q, emb.T)


def dense_exact(b, emb, q, k: int, mask=None):
    """``(ids i32[Q,k], sims f32[Q,k])`` of the k nearest documents among ``mask``; -1 pads."""
    return b.topk(dense_scores(b, emb, q), k, mask)
