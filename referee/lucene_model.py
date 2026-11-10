"""Lucene's single-field BM25 as OpenSearch 2.19 computes it (LegacyBM25Similarity, which keeps the (k1 + 1) factor),
to explain the conformance suite's score differences between the engine (the prototype's formula) and OpenSearch.

Three differences from the prototype's single-field BM25:
1. N is the number of documents **that have the field** (Lucene's docCount), not the catalogue size;
2. avgdl is averaged over those documents (sumTotalTermFreq / docCount), not over every document;
3. a document's field length is stored in one byte (SmallFloat.intToByte4): exact below 24 tokens, lossy above.

Owner: evaluation-steward (P6).
"""
from __future__ import annotations

import math

from referee.analysis import analyze

_FREE = 24  # 255 - longToInt4(Integer.MAX_VALUE)


def _long_to_int4(i: int) -> int:
    nbits = i.bit_length()
    if nbits < 4:
        return i
    shift = nbits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i: int) -> int:
    bits, shift = i & 0x07, (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


def lucene_length(n: int) -> int:
    """A field length after Lucene's one-byte norm encoding and decoding."""
    if n < _FREE:
        return n
    return _FREE + _int4_to_long(_long_to_int4(n - _FREE))


def match_scores(texts, query, k1=1.2, b=0.75, quantise=True, field_stats=True):
    """{row: score} for a single-field `match` (operator or), Lucene-style."""
    toks = [analyze(t) if t is not None and t == t else None for t in texts]
    has = [t for t in toks if t is not None and len(t) > 0] if field_stats else [t or [] for t in toks]
    n = len(has)
    avg = sum(len(t) for t in has) / n if n else 1.0
    out = {}
    for term in dict.fromkeys(analyze(query)):
        df = sum(term in t for t in has)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, t in enumerate(toks):
            if not t or term not in t:
                continue
            tf = t.count(term)
            dl = lucene_length(len(t)) if quantise else len(t)
            out[i] = out.get(i, 0.0) + idf * (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / avg))
    return out
