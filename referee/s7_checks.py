"""Correctness of S7's constructs as they land, by brute force over `toy` (owner: evaluation-steward, P6).

`referee/scope.py` says which S7 items have landed; `check_landed_correct` runs this file's check for every landed
item, and fails for a landed item that has no check yet (the steward writes it when marking the item landed). The
semantics are OpenSearch's (Lucene's), since S7-9 compares with OpenSearch:

- S7-1 `match_phrase`: the analysed query tokens at consecutive positions in one field; stop words keep their
  position, so a removed stop word leaves a gap. `slop` n: two terms match when |(p2 - p1) - 1| <= n (so slop 2 also
  matches the two words swapped). Checked: the match set and total.
- S7-2 `fuzzy` (the value is not analysed), `match` with `fuzziness` (each analysed token), `prefix` (not analysed),
  `match_bool_prefix` (all tokens but the last as terms, the last as a prefix; any may match). Fuzziness AUTO: 0
  edits for 1-2 characters, 1 for 3-5, 2 for more; Damerau distance (an adjacent transposition is one edit);
  prefix_length 0. Checked: the match set and total.

Scores are not checked here; S7-9's OpenSearch families compare them.
"""
from __future__ import annotations

from referee.analysis import analyze, analyze_positions

ALL = 10_000
FIELDS = ["product_title", "product_bullet_point", "product_description"]


def _osa(a: str, b: str, cap: int) -> int:
    """Optimal string alignment distance (Damerau with adjacent transpositions), capped at cap + 1."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
        if min(prev) > cap:
            return cap + 1
    return prev[-1]


def auto_edits(term: str) -> int:
    return 0 if len(term) <= 2 else (1 if len(term) <= 5 else 2)


def _ids(index, req):
    r = index.search({**req, "size": ALL})
    return r["hits"]["total"], {h["_id"] for h in r["hits"]["hits"]}


def _expect(ds, field, pred):
    return {pid for pid, text in zip(ds.catalogue["product_id"], ds.catalogue[field]) if pred(text)}


def _phrase_pred(query, slop):
    q = [t for t, _ in analyze_positions(query)]
    qpos = [p for _, p in analyze_positions(query)]

    def pred(text):
        toks = analyze_positions(text)
        if not q:
            return False
        if len(q) == 1:
            return any(t == q[0] for t, _ in toks)
        if len(q) == 2 and slop:
            want = qpos[1] - qpos[0]
            p1 = [p for t, p in toks if t == q[0]]
            p2 = [p for t, p in toks if t == q[1]]
            return any(abs((b - a) - want) <= slop for a in p1 for b in p2 if a != b)
        positions = {}
        for t, p in toks:
            positions.setdefault(t, set()).add(p)
        offs = [p - qpos[0] for p in qpos]
        return any(all(start + o in positions.get(t, ()) for t, o in zip(q, offs)) for start in positions.get(q[0], ()))
    return pred


def check_s7_1(index, ds):
    problems = []
    cases = [("product_title", "red lamp", 0), ("product_title", "portable gray", 0), ("product_title", "lamp red", 0),
             ("product_title", "lamp red", 2), ("product_title", "kestrel portable", 1),
             ("product_bullet_point", "drinks hot", 0), ("product_bullet_point", "keeps drinks hot", 0),
             ("product_description", "made steel", 0), ("product_description", "made steel", 1),
             ("product_description", "made of steel", 0), ("product_bullet_point", "camping hiking", 0),
             ("product_bullet_point", "camping hiking", 1), ("product_title", "zzzz lamp", 0)]
    for field, text, slop in cases:
        spec = {"query": text, "slop": slop} if slop else text
        total, got = _ids(index, {"query": {"match_phrase": {field: spec}}})
        exp = _expect(ds, field, _phrase_pred(text, slop))
        if got != exp or total != len(exp):
            problems.append(f"match_phrase {field}:{text!r} slop {slop}: {len(got)} hits (total {total}), "
                            f"expected {len(exp)}; missing {sorted(exp - got)[:3]}, extra {sorted(got - exp)[:3]}")
    return problems


def _terms_of(text):
    return set(analyze(text))


def check_s7_2(index, ds):
    problems = []

    def run(label, req, field, pred):
        total, got = _ids(index, req)
        exp = _expect(ds, field, pred)
        if got != exp or total != len(exp):
            problems.append(f"{label}: {len(got)} hits (total {total}), expected {len(exp)}; "
                            f"missing {sorted(exp - got)[:3]}, extra {sorted(got - exp)[:3]}")

    for value in ("lamq", "bots", "kettel", "wooll", "acme", "zzzzzz", "gry"):
        e = auto_edits(value)
        run(f"fuzzy {value!r}", {"query": {"fuzzy": {"product_title": value}}}, "product_title",
            lambda t, v=value, e=e: any(_osa(v, w, e) <= e for w in _terms_of(t)))
    for text in ("red lamq", "blu ketle", "portabel speakr", "the"):
        toks = list(dict.fromkeys(analyze(text)))
        run(f"match fuzziness {text!r}", {"query": {"match": {"product_title": {"query": text, "fuzziness": "AUTO"}}}},
            "product_title",
            lambda t, q=toks: any(_osa(a, w, auto_edits(a)) <= auto_edits(a) for a in q for w in _terms_of(t)))
    for value in ("lam", "ke", "port", "zz"):
        run(f"prefix {value!r}", {"query": {"prefix": {"product_title": value}}}, "product_title",
            lambda t, v=value: any(w.startswith(v) for w in _terms_of(t)))
    for text in ("red la", "portable ke", "acme", "lamps"):
        toks = analyze(text)
        run(f"match_bool_prefix {text!r}", {"query": {"match_bool_prefix": {"product_title": text}}}, "product_title",
            lambda t, q=toks: bool(q) and (bool(set(q[:-1]) & _terms_of(t)) or
                                           any(w.startswith(q[-1]) for w in _terms_of(t))))
    return problems


CHECKS = {"S7-1": check_s7_1, "S7-2": check_s7_2}


def check_landed_correct(index, ds):
    from referee import scope
    problems = []
    for item in sorted({e[1] for e in scope.landed()}):
        if item not in CHECKS:
            problems.append(f"{item} is marked landed but referee/s7_checks.py has no correctness check for it")
            continue
        problems += CHECKS[item](index, ds)
    assert not problems, "; ".join(problems[:8]) + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
