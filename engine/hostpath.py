"""R1 of the founder's CPU-headroom plan: the fused lexical path's per-request host work in Rust.

This module is the Python side of ``native/s3_hostpath`` (a pyo3 extension, optional):

- ``host_index(st)`` builds the Rust ``HostIndex`` for an index state. Its inputs are the analysis table (each word
  to the token engine/analysis.py gives it, or None), the vocabulary and postings offsets, the text fields' b, the
  keyword fields' code counts, the synonym tokens, and the postings chunk limit.
- ``prepare_reference(st, payloads)`` is the oracle. It builds the same output as ``HostIndex.prepare`` from the
  Python parser and planner and the executor's own eligibility rule (``Executor._fused_plans``). The tests compare the
  two byte for byte.

The analysis table holds the words of the index's source text when the segment keeps it, and otherwise the
vocabulary's tokens. ``extra_words`` adds more, for example a query log. A query word missing from the table sends
its request back to Python, so the table changes how often the Rust path serves, never what it returns.
"""
from __future__ import annotations

import json
import struct

import numpy as np

from .analysis import _WORD, _token, analyze
from .dsl.parse import parse, resolve_msm  # noqa: F401  (resolve_msm: the oracle mirrors the planner through plan())
from .plan.nodes import Lexical

BATCH_ROWS = 64


def _words(st, extra_words=()):
    part = st.parts[0]
    src = getattr(part.seg, "source", None) or {}
    words = dict.fromkeys(extra_words)
    for f in st.text_fields:
        for text in src.get(f, ()):
            if text:
                words.update(dict.fromkeys(_WORD.findall(str(text).lower())))
    if not src:
        words.update(dict.fromkeys(part.vocab_list))
    return list(words)


def host_index(st, extra_words=()):
    """The Rust ``HostIndex`` for ``st`` (one segment). Raises ImportError when the extension is not built."""
    import s3_hostpath
    from .ops import lexical as L
    part = st.parts[0]
    words = _words(st, extra_words)
    return s3_hostpath.HostIndex(
        words, [_token(w) for w in words], list(part.vocab_list),
        np.ascontiguousarray(part.host_term_ptr, np.int64).tobytes(), list(st.text_fields),
        [float(np.float32(st.mapping.b_of(f))) for f in st.text_fields],
        {f: len(kw.values) for f, kw in st.keywords.items()}, sorted(st.mapping.synonyms),
        int(L.MAX_POSTINGS_PER_CHUNK))


def prepare_reference(st, payloads, words=None):
    """``HostIndex.prepare``'s output computed in Python: the oracle. ``words``: the analysis table's words (a
    request with a word outside it falls back, as in Rust)."""
    from .plan.execute import Executor
    from .plan.plan import plan as make_plan
    part = st.parts[0]
    table = set(words) if words is not None else None
    plans, served = [], []
    for pos, body in enumerate(payloads):
        try:
            req = json.loads(body)
            p = make_plan(req, st)
        except Exception:  # noqa: BLE001  (any error: Python's path answers it)
            plans.append(None)
            continue
        plans.append(p)
        served.append(pos)
    ex = Executor(st)
    fused = {pos for pos in served if _eligible(ex, plans[pos], part) and _shape_ok(payloads[pos], table)}
    fallback = [pos for pos in range(len(payloads)) if pos not in fused]
    F = len(st.text_fields)
    bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)
    keys, members = [], []
    info = {}
    for pos in sorted(fused):
        p = plans[pos]
        w = np.zeros(F, np.float32)
        for f, wt in p.root.fields:
            w[st.field_index[f]] = wt
        pos_w = [st.field_index[f] for f, wt in p.root.fields if wt > 0]
        idf_field = min(pos_w) if p.root.idf == "field" else -1
        key = (w.view(np.uint32).tobytes(), idf_field)
        info[pos] = (p, w, idf_field)
        if key in keys:
            members[keys.index(key)].append(pos)
        else:
            keys.append(key)
            members.append([pos])
    ints, floats, chunks = [], [], []
    ptr = part.host_term_ptr
    for g, mem in enumerate(members):
        for c0 in range(0, len(mem), BATCH_ROWS):
            ch = mem[c0:c0 + BATCH_ROWS]
            rs, rl, qlo, qhi, msm, rt, facets = [], [], [], [], [], [], []
            for pos in ch:
                p = info[pos][0]
                qlo.append(len(rs))
                for t in p.root.terms:
                    if t in part.vocab:
                        tid = part.vocab[t]
                        rs.append(int(ptr[tid]))
                        rl.append(int(ptr[tid + 1] - ptr[tid]))
                        rt.append(tid)
                qhi.append(len(rs))
                msm.append(int(p.root.msm))
                for a in p.aggs:
                    if a.field not in facets:
                        facets.append(a.field)
            Q = len(ch)
            sizes = [len(st.keywords[f].values) for f in facets]
            offs = [int(x) for x in np.cumsum([0] + [Q * s for s in sizes])[:-1]] if facets else []
            parts = [rs or [0], rl or [0], qlo, qhi, msm, offs or [0], sizes or [1], rt or [0]]
            int_at, int_len = [], []
            for x in parts:
                int_at.append(len(ints))
                int_len.append(len(x))
                ints += x
            float_at = len(floats)
            w = info[ch[0]][1]
            floats += list(w) + list(bb) + list(np.float32(1.0) - bb) + [np.float32(info[pos][0].root.boost)
                                                                         for pos in ch]
            chunks.append({"group": g, "positions": ch, "int_at": int_at, "int_len": int_len, "float_at": float_at,
                           "idf_field": keys[g][1], "facets": facets,
                           "from": [info[pos][0].from_ for pos in ch], "size": [info[pos][0].size for pos in ch],
                           "aggs": [[(a.name, a.field, a.size) for a in info[pos][0].aggs] for pos in ch],
                           "n_terms": [sum(1 for t in info[pos][0].root.terms if t in part.vocab) for pos in ch]})
    return {"fallback": fallback, "ints": struct.pack(f"<{len(ints)}q", *ints),
            "floats": np.asarray(floats, np.float32).astype("<f4").tobytes(), "chunks": chunks, "n_text_fields": F}


def _eligible(ex, p, part):
    """``Executor._fused_plans``'s rule for one plan (the batch, segment and accumulation conditions aside), with
    terms aggregations on keyword fields only (their keys come from the HostIndex)."""
    from .ops import lexical as L
    r = p.root
    if not (isinstance(r, Lexical) and r.msm >= 1 and not p.explain and p.post_filter is None
            and p.rescore is None and p.collapse is None and not p.search_after and p.highlight is None
            and len(p.sort) == 1 and p.sort[0].field == "_score" and p.sort[0].descending
            and p.from_ + p.size > 0 and all(a.kind == "terms" and not a.sub and a.field in ex.st.keywords
                                             for a in p.aggs)):
        return False
    ptr = part.host_term_ptr
    n_post = sum(int(ptr[part.vocab[t] + 1] - ptr[part.vocab[t]]) for t in r.terms if t in part.vocab)
    return n_post <= L.MAX_POSTINGS_PER_CHUNK


def _shape_ok(body, table):
    """The request shapes and texts the Rust side accepts (its own rule, mirrored): an ASCII query text whose words
    are all in the analysis table, and only the keys and value types it handles."""
    try:
        req = json.loads(body)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(req, dict) or set(req) - {"query", "size", "from", "aggs", "aggregations", "track_total_hits"}:
        return False
    q = req.get("query")
    if not isinstance(q, dict) or len(q) != 1:
        return False
    (kind, body_q), = q.items()
    text = None
    if kind == "match" and isinstance(body_q, dict) and len(body_q) == 1:
        (_, spec), = body_q.items()
        if isinstance(spec, str):
            text = spec
        elif isinstance(spec, dict) and not set(spec) - {"query", "operator", "minimum_should_match", "boost"}:
            text = spec.get("query")
            if not _msm_ok(spec.get("minimum_should_match")):
                return False
    elif kind == "multi_match" and isinstance(body_q, dict):
        if set(body_q) - {"query", "fields", "type", "operator", "minimum_should_match", "tie_breaker", "boost"}:
            return False
        text = body_q.get("query")
        if not _msm_ok(body_q.get("minimum_should_match")):
            return False
    if not isinstance(text, str) or not text.isascii():
        return False
    if table is not None and any(w not in table for w in _WORD.findall(text.lower())):
        return False
    akey = "aggs" if "aggs" in req else "aggregations"
    for spec in (req.get(akey) or {}).values():
        if not (isinstance(spec, dict) and set(spec) == {"terms"} and isinstance(spec["terms"], dict)
                and not set(spec["terms"]) - {"field", "size"}):
            return False
    return True


def _msm_ok(v):
    return v is None or (isinstance(v, int) and not isinstance(v, bool)) or (
        isinstance(v, str) and all(c in " -%0123456789" for c in v))


__all__ = ["host_index", "prepare_reference", "analyze"]
