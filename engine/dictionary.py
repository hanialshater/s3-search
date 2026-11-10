"""The terms dictionary (S7-2): fuzzy and prefix expansion over a field's indexed terms, as OpenSearch defines them.

This is query-side setup on the host (like analysis): it turns one query term into a short list of indexed terms,
which the operators then score on the device. A text field's dictionary is the terms with ``df_field > 0``; a keyword
field's is its sorted values.

- Fuzziness: ``AUTO`` (= ``AUTO:3,6``: 0 edits for terms of 1-2 characters, 1 for 3-5, 2 for 6 and more),
  ``AUTO:low,high``, or 0, 1, 2. Edit distance is Levenshtein, with adjacent transpositions counting 1 when
  ``transpositions`` (the default; optimal string alignment, as Lucene's automata). ``prefix_length`` leading
  characters must match exactly (default 0). At most ``max_expansions`` terms (default 50) are kept, the most
  similar first (Lucene's ``1 - edits / min(len)``), ties by term order.
- Prefix: every term that starts with the value (``prefix``: no limit; ``match_bool_prefix``: the first
  ``max_expansions`` in term order).
"""
from __future__ import annotations

import re
from bisect import bisect_left

import numpy as np

MAX_EDITS = 2


def auto_edits(spec, term: str, path: str = "fuzziness") -> int:
    from .errors import InvalidRequest, UnsupportedQuery
    n = len(term)
    if isinstance(spec, bool):
        raise InvalidRequest(path, "AUTO, AUTO:low,high, 0, 1 or 2")
    if isinstance(spec, (int, float)) or (isinstance(spec, str) and re.fullmatch(r"\d+(\.0+)?", spec)):
        d = int(float(spec))
        if float(spec) != d or d < 0:
            raise InvalidRequest(path, "AUTO, AUTO:low,high, 0, 1 or 2")
        if d > MAX_EDITS:
            raise UnsupportedQuery(path, "at most 2 edits (as OpenSearch)")
        return d
    if isinstance(spec, str):
        m = re.fullmatch(r"AUTO(?::(\d+),(\d+))?", spec.strip(), re.I)
        if m:
            lo, hi = (int(m.group(1)), int(m.group(2))) if m.group(1) else (3, 6)
            return 0 if n < lo else 1 if n < hi else 2
    raise InvalidRequest(path, "AUTO, AUTO:low,high, 0, 1 or 2")


def _encode(words, L):
    out = np.full((len(words), max(L, 1)), -1, np.int32)
    for i, w in enumerate(words):
        out[i, :len(w)] = [ord(c) for c in w]
    return out


def osa_distances(query: str, cands: np.ndarray, L: int, transpositions: bool = True) -> np.ndarray:
    """Edit distances from ``query`` to each row of ``cands`` (code points, all of length ``L``), vectorized over the
    rows: Levenshtein, plus adjacent transpositions (optimal string alignment) when asked."""
    n = cands.shape[0]
    q = [ord(c) for c in query]
    m = len(q)
    prev2 = None
    prev = np.tile(np.arange(L + 1, dtype=np.int32), (n, 1))
    for i in range(1, m + 1):
        cur = np.empty((n, L + 1), np.int32)
        cur[:, 0] = i
        for j in range(1, L + 1):
            cost = (cands[:, j - 1] != q[i - 1]).astype(np.int32)
            v = np.minimum(np.minimum(prev[:, j] + 1, cur[:, j - 1] + 1), prev[:, j - 1] + cost)
            if transpositions and i > 1 and j > 1:
                t = (cands[:, j - 1] == q[i - 2]) & (cands[:, j - 2] == q[i - 1])
                v = np.where(t, np.minimum(v, prev2[:, j - 2] + 1), v)
            cur[:, j] = v
        prev2, prev = prev, cur
    return prev[:, L]


class Dictionary:
    """Sorted terms with a mask of which are in the field(s) asked for; buckets by length for fuzzy search."""

    def __init__(self, terms: list[str]):
        self.terms = terms                                       # sorted
        self._by_len = None

    def _buckets(self):
        if self._by_len is None:
            by = {}
            for i, t in enumerate(self.terms):
                by.setdefault(len(t), []).append(i)
            self._by_len = {L: (np.asarray(ids, np.int64), _encode([self.terms[i] for i in ids], L))
                            for L, ids in by.items()}
        return self._by_len

    def fuzzy(self, term: str, edits: int, prefix_length: int = 0, max_expansions: int = 50,
              transpositions: bool = True, allowed=None) -> list[int]:
        """Term indices within ``edits`` of ``term``, most similar first, at most ``max_expansions``."""
        if edits == 0:
            i = bisect_left(self.terms, term)
            ok = i < len(self.terms) and self.terms[i] == term and (allowed is None or allowed[i])
            return [i] if ok else []
        found = []
        pre = [ord(c) for c in term[:prefix_length]]
        for L in range(max(0, len(term) - edits), len(term) + edits + 1):
            if L not in self._buckets():
                continue
            ids, codes = self._buckets()[L]
            keep = np.ones(len(ids), bool) if allowed is None else allowed[ids]
            if pre:
                if L < len(pre):
                    continue
                keep &= (codes[:, :len(pre)] == pre).all(axis=1)
            if not keep.any():
                continue
            d = osa_distances(term, codes[keep], L, transpositions)
            sel = d <= edits
            for i, e in zip(ids[keep][sel], d[sel]):
                found.append((int(i), int(e)))
        sim = lambda ie: 1.0 - ie[1] / max(1, min(len(term), len(self.terms[ie[0]])))   # noqa: E731
        found.sort(key=lambda ie: (-sim(ie), self.terms[ie[0]]))
        return [i for i, _ in found[:max_expansions]]

    def prefix(self, value: str, allowed=None, limit: int | None = None) -> list[int]:
        lo = bisect_left(self.terms, value)
        hi = bisect_left(self.terms, value + "\U0010ffff")
        ids = [i for i in range(lo, hi) if allowed is None or allowed[i]]
        return ids if limit is None else ids[:limit]


def text_dictionary(st) -> Dictionary:
    """The index's vocabulary (sorted, over all segments), cached on the index state until its next layout."""
    d = getattr(st, "_dictionary", None)
    if d is None:
        vl = st.vocab_list
        order = sorted(range(len(vl)), key=vl.__getitem__)
        d = st._dictionary = Dictionary([vl[i] for i in order])
        d.gid_of = np.asarray(order, np.int64)             # dictionary position -> global term id
    return d


def field_mask(st, fields) -> np.ndarray:
    """bool[V]: terms indexed in any of ``fields`` of a live document."""
    cols = [st.field_index[f] for f in fields]
    by_gid = (np.asarray(st.host_df_field)[:, cols] > 0).any(axis=1)
    return by_gid[text_dictionary(st).gid_of]               # in the dictionary's (sorted) order


def keyword_dictionary(kw) -> Dictionary:
    d = getattr(kw, "_dictionary", None)
    if d is None:
        d = kw._dictionary = Dictionary(list(kw.values[1:]))     # code = index + 1
    return d
