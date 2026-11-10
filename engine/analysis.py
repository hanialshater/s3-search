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


def analyze_with_positions(text, analyzer: str = "s3_english") -> tuple[list[str], list[int]]:
    """Tokens of ``text`` and their positions. Every word takes a position, stopwords included (they are dropped but
    leave a gap, as Lucene's stop filter does with position increments), so ``"red the lamp"`` gives red@0, lamp@2.
    ``[t for t in tokens]`` equals ``analyze(text)``."""
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}")
    if text is None or (isinstance(text, float) and text != text) or type(text).__name__ == "NAType":
        return [], []
    toks, pos = [], []
    for i, w in enumerate(_WORD.findall(str(text).lower())):
        if w.endswith("'s"):
            w = w[:-2]
        w = w.replace("'", "")
        if not w or w in EN_STOP:
            continue
        s = _STEM_CACHE.get(w)
        if s is None:
            s = _STEM_CACHE[w] = _stemmer.stemWord(w)
        toks.append(s)
        pos.append(i)
    return toks, pos


def analyze(text, analyzer: str = "s3_english") -> list[str]:
    """Tokens of ``text`` in order (duplicates kept). ``None`` and NaN analyse to no tokens."""
    if analyzer not in ANALYZERS:
        raise ValueError(f"unknown analyzer {analyzer!r}")
    if text is None or (isinstance(text, float) and text != text) or type(text).__name__ == "NAType":
        return []
    out = []
    for w in _WORD.findall(str(text).lower()):
        if w.endswith("'s"):
            w = w[:-2]
        w = w.replace("'", "")
        if not w or w in EN_STOP:
            continue
        s = _STEM_CACHE.get(w)
        if s is None:
            s = _STEM_CACHE[w] = _stemmer.stemWord(w)
        out.append(s)
    return out
