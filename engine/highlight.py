"""Highlight (S7-5): OpenSearch's plain highlighter on text fields, from the raw text a segment keeps (``src/``).

- The terms: every analysed term of the query's clauses on that field (``require_field_match``, the default) or on
  any field, except under ``must_not``; a fuzzy, prefix or synonym clause contributes its expansions.
- A word is highlighted when its analysed token is one of them (the same analyzer as the index), wrapped in the
  pre and post tags (default ``<em>``, ``</em>``).
- Fragments: the text is cut at the token that ends past each multiple of ``fragment_size`` characters (Lucene's
  SimpleFragmenter, default 100); each is scored by its distinct highlighted terms; the best ``number_of_fragments``
  (default 5) are returned in text order, and only fragments with at least one highlighted term.
  ``number_of_fragments: 0`` returns the whole text as one fragment.
"""
from __future__ import annotations

from .analysis import _WORD, _token
from .plan.nodes import Bool, Dense, Hybrid, Lexical, LexicalGroups, Phrase


def query_terms(node, out: dict, negated: bool = False):
    """``{field: set(terms)}`` of the clauses under ``node`` (must_not excluded)."""
    if negated:
        return out
    if isinstance(node, Lexical):
        for f, w in node.fields:
            if w > 0:
                out.setdefault(f, set()).update(node.terms)
    elif isinstance(node, LexicalGroups):
        for f, w in node.fields:
            if w > 0:
                for g in node.groups:
                    out.setdefault(f, set()).update(g)
    elif isinstance(node, Phrase):
        for alts in node.slots:
            out.setdefault(node.field, set()).update(alts)
    elif isinstance(node, Bool):
        for c in node.must + node.should + node.filter:
            query_terms(c, out)
    elif isinstance(node, Hybrid):
        for c in node.queries:
            query_terms(c, out)
    elif isinstance(node, Dense):
        pass
    elif hasattr(node, "query"):                         # function_score
        query_terms(node.query, out)
    return out


def highlight_text(text: str, terms: set, fragment_size: int = 100, number_of_fragments: int = 5,
                   pre: str = "<em>", post: str = "</em>") -> list[str]:
    if not text or not terms:
        return []
    low = text.lower()
    marks = []                                            # (start, end, token) of highlighted words
    bounds = [0]
    k = 1
    for m in _WORD.finditer(low):
        if fragment_size > 0 and number_of_fragments > 0 and m.end() >= fragment_size * k:
            bounds.append(m.start())
            k += 1
        t = _token(m.group(0))
        if t is not None and t in terms:
            marks.append((m.start(), m.end(), t))
    if not marks:
        return []
    if number_of_fragments == 0:
        bounds = [0]
    bounds = sorted(set(bounds)) + [len(text)]
    frags = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], bounds[i + 1]
        inside = [x for x in marks if lo <= x[0] < hi]
        if inside:
            frags.append((len({x[2] for x in inside}), i, lo, hi, inside))
    if number_of_fragments:
        best = sorted(frags, key=lambda f: (-f[0], f[1]))[:number_of_fragments]
        frags = sorted(best, key=lambda f: f[1])
    out = []
    for _, _, lo, hi, inside in frags:
        parts, cur = [], lo
        for s, e, _ in inside:
            parts.append(text[cur:s] + pre + text[s:e] + post)
            cur = e
        parts.append(text[cur:hi])
        out.append("".join(parts).strip())
    return out


def highlight_hits(st, plan, slots) -> list[dict]:
    """Per hit (slot): ``{field: [fragments]}`` for the fields asked for (fields without a fragment left out)."""
    h = plan.highlight
    per_field = query_terms(plan.root, {})
    any_field = set().union(*per_field.values()) if per_field else set()
    out = []
    for slot in slots:
        part, loc = st.owner(int(slot))
        got = {}
        for f, size, n, pre, post in h.fields:
            terms = per_field.get(f, set()) if h.require_field_match else any_field
            text = part.seg.source.get(f, [None] * (loc + 1))[loc]
            frags = highlight_text(text or "", terms, size, n, pre, post)
            if frags:
                got[f] = frags
        out.append(got)
    return out
