"""Text analysis (docs/INTERFACES.md §2). One analyzer, ``s3_english``, ported from the prototype
(baseline/esci_gpu_vs_opensearch.ipynb): lowercase, ``[a-z0-9]+('[a-z0-9]+)?`` words, strip ``'s`` and ``'``, English
stopwords (Lucene's list), Porter stemmer (``snowballstemmer``)."""
from __future__ import annotations

import re

import snowballstemmer

ANALYZERS = ("s3_english",)
ANALYZER_VERSION = "s3_english@1"

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
EN_STOP = frozenset("a an and are as at be but by for if in into is it no not of on or such "
                    "that the their then there these they this to was will with".split())
_stemmer = snowballstemmer.stemmer("porter")
_STEM_CACHE: dict[str, str] = {}


_WORD_CACHE: dict[str, str | None] = {}     # a raw word -> its token, or None (a stopword or empty)


def _token(w: str):
    """One regex word to its token (or None): strip a trailing 's, then ', drop stopwords, stem. Cached per word."""
    t = _WORD_CACHE.get(w, 0)
    if t == 0:
        x = w[:-2] if w.endswith("'s") else w
        x = x.replace("'", "")
        if not x or x in EN_STOP:
            t = None
        else:
            t = _STEM_CACHE.get(x)
            if t is None:
                t = _STEM_CACHE[x] = _stemmer.stemWord(x)
        _WORD_CACHE[w] = t
    return t


def _missing(text) -> bool:
    return text is None or (isinstance(text, float) and text != text) or type(text).__name__ == "NAType"


def analyze_with_positions(text, analyzer: str = "s3_english") -> tuple[list[str], list[int]]:
    """Tokens of ``text`` and their positions. Every word takes a position, stopwords included (they are dropped but
    leave a gap, as Lucene's stop filter does with position increments), so ``"red the lamp"`` gives red@0, lamp@2.
    The tokens equal ``analyze(text)``."""
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}")
    if _missing(text):
        return [], []
    toks = [_token(w) for w in _WORD.findall(str(text).lower())]
    pos = [i for i, t in enumerate(toks) if t is not None]
    return [toks[i] for i in pos], pos


def analyze(text, analyzer: str = "s3_english") -> list[str]:
    """Tokens of ``text`` in order (duplicates kept). ``None`` and NaN analyse to no tokens."""
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}")
    if _missing(text):
        return []
    return [t for t in map(_token, _WORD.findall(str(text).lower())) if t is not None]




def words_of(values):
    """Build-time analysis, pass 1 (the build speed-ups, 2026-11-11): the regex words of a list of strings, as
    ``(distinct words in first-seen order, int32 index into them per word in order, int64 word count per value)``.
    Module level, so build workers can run it."""
    from itertools import chain
    import numpy as np
    findall = _WORD.findall
    per = [findall(v.lower()) for v in values]
    lens = np.fromiter((len(w) for w in per), np.int64, len(per))
    words = list(chain.from_iterable(per))
    del per
    index = {w: i for i, w in enumerate(dict.fromkeys(words))}
    flat = np.fromiter(map(index.__getitem__, words), np.int32, len(words))
    return list(index), flat, lens


def tokens_of(words):
    """Pass 2: each word's token (or None): exactly ``analyze_with_positions``'s."""
    return [_token(w) for w in words]
