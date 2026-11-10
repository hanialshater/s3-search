"""The loaded index (docs/INTERFACES.md §4): an ordered list of segments (S8) on a device, with global statistics over
live documents, docs in slots ordered by ``first_offset``, and the host-side dictionaries.

- A ``Part`` is one segment: its postings, positions and vectors on the device, its live bits and its own columns.
  Its postings' doc ids are mapped to global slots (``doc_ids``) and postings of dead docs are masked (``post_live``),
  so the operators run on a part as they ran on a whole segment.
- ``IndexState`` holds the slot-ordered per-doc columns (live, numerics, keyword codes, relative lengths, ids,
  first_offset) and the global vocabulary and statistics. ``_relayout`` recomputes them after any structural change
  (add, replace, remove, delete): the slots are the live docs in ascending ``first_offset`` order, which is a
  rebuild's doc order, so ties broken by slot are ties broken by ``first_offset``.
"""
from __future__ import annotations

import copy as _copy
from collections import Counter, defaultdict

import numpy as np

from .analysis import analyze
from .backend import Backend, OverBudget
from .encode import TfidfSvdEncoder
from .mapping import Mapping, parse_mapping
from .segment import NumericColumn, SegmentArrays, device_nbytes

BATCH = 64            # the executor's lexical batch; it sets the working set the memory plan reserves


class KeywordState:
    def __init__(self, b: Backend, codes: np.ndarray, values: list[str]):
        self.codes_host = np.asarray(codes, np.int32)
        self.codes = b.asarray(self.codes_host, "int64")
        self.values = list(values)
        self.code_of = {v: i for i, v in enumerate(self.values)}
        self._toks = None

    @property
    def n(self) -> int:
        return len(self.values)

    def tokens(self):
        """Analysed token sets per code and an inverted index token -> codes (for the match features)."""
        if self._toks is None:
            toks = [frozenset(analyze(v)) for v in self.values]
            inv = defaultdict(list)
            for c, ts in enumerate(toks):
                if c:
                    for t in ts:
                        inv[t].append(c)
            self._toks = (toks, inv)
        return self._toks

    def matched_codes(self, q_tokens: set) -> np.ndarray:
        """Codes whose analysed tokens all appear in the query (the prototype's brand_match/color_match)."""
        toks, inv = self.tokens()
        cands = {c for t in q_tokens for c in inv.get(t, ())}
        return np.array(sorted(c for c in cands if toks[c] and toks[c] <= q_tokens), dtype=np.int64)


def _idf(N, df):
    return np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)


def _unpack(words, n):
    from .backend.numpy_backend import NumpyBackend
    return NumpyBackend().unpack_bits(np.asarray(words), n).copy()


def _pack(mask):
    from .backend.numpy_backend import NumpyBackend
    return NumpyBackend().pack_bits(np.asarray(mask, bool))


class Part:
    """One segment of the index."""

    def __init__(self, seg: SegmentArrays, b: Backend):
        self.b, self.seg = b, seg
        self.manifest = seg.manifest
        self.segment_id = str(seg.manifest.get("segment_id")
                              or f"seg-{seg.manifest.get('version', 0)}-mem{id(seg) & 0xffffffff:08x}")
        self.version = int(seg.manifest.get("version", 0))
        self.n = seg.n_docs
        self.ids = np.asarray(seg.ids, dtype=object)
        self.local_of = {d: i for i, d in enumerate(self.ids)}
        self.first_offset = np.asarray(seg.first_offset, np.int64)
        t = seg.text
        self.fields = list(t.fields)
        self.vocab_list = list(t.vocab)
        self.vocab = {w: i for i, w in enumerate(self.vocab_list)}
        self.host_term_ptr, self.host_doc_ids = np.asarray(t.term_ptr), np.asarray(t.doc_ids)
        self.host_tf, self.host_field_len = np.asarray(t.tf), np.asarray(t.field_len)
        self.term_of_posting = np.repeat(np.arange(len(self.vocab_list), dtype=np.int64), np.diff(self.host_term_ptr))
        self.term_ptr = b.asarray(self.host_term_ptr, "int64")
        self.doc_ids_local = b.asarray(self.host_doc_ids, "int64")
        self.doc_ids = self.doc_ids_local                 # mapped to slots by the index's layout
        self.post_live = None
        self.tf = b.asarray(self.host_tf, "uint8")
        live = seg.live if seg.live is not None else np.full((self.n + 31) // 32, 0xFFFFFFFF, np.uint32)
        self.local_live = _unpack(live, self.n)
        self.numerics, self.missing = {}, {}
        for f, col in seg.numerics.items():
            self.numerics[f] = b.asarray(np.array(col.values))
            self.missing[f] = b.unpack_bits(np.asarray(col.missing), self.n)
        self.keyword_codes = {f: np.asarray(c.codes, np.int64) for f, c in seg.keywords.items()}
        self.keyword_values = {f: list(c.values) for f, c in seg.keywords.items()}
        self.vectors = {f: b.asarray(np.asarray(v, np.float32), "float32") for f, v in seg.vectors.items()}
        self.host_pos_ptr, self.host_positions = t.pos_ptr, t.positions
        self._positions = None
        self.device_nbytes = device_nbytes(seg)
        # set by the index's layout
        self.N = 0
        self.rel_len = None
        self.slot_of_local = np.full(self.n, -1, np.int64)
        self.live_local = self.live_slot = np.zeros(0, np.int64)
        self.identity = False
        self.gid = np.zeros(0, np.int64)

    def positions(self):
        """``(pos_ptr int64[P+1], positions int32[T])`` on the device, moved there on first use (a phrase)."""
        if self._positions is None:
            b = self.b
            self._positions = (b.asarray(np.asarray(self.host_pos_ptr), "int64"),
                               b.asarray(np.asarray(self.host_positions).astype(np.int32), "int32"))
        return self._positions

    def snapshot(self) -> SegmentArrays:
        """The segment as it is now: references to its immutable arrays, copies of its live bits and hot columns."""
        seg = _copy.copy(self.seg)
        seg.manifest = dict(self.seg.manifest)
        seg.live = _pack(self.local_live)
        seg.numerics = {f: NumericColumn(values=np.asarray(self.b.to_numpy(self.numerics[f])).copy(),
                                         missing=_pack(self.b.to_numpy(self.missing[f])))
                        for f in self.numerics}
        return seg


class IndexState:
    def __init__(self, segs, b: Backend, check_budget: bool = True, version: int | None = None,
                 newest_wins: bool = True):
        if isinstance(segs, SegmentArrays):
            segs = [segs]
        if not segs:
            raise ValueError("an index needs at least one segment")
        self.b = b
        self.check_budget = check_budget
        self.parts: list[Part] = []
        self.encoders: dict = {}
        first = Part(segs[0], b)
        self.manifest = first.manifest
        self.mapping: Mapping = parse_mapping(self.manifest["mapping"])
        self.text_fields = list(first.fields)
        self.field_index = {f: i for i, f in enumerate(self.text_fields)}
        self.vector_spec = dict(self.manifest.get("vector_fields", {}))
        self._attach(first)
        for s in segs[1:]:
            self._attach(Part(s, b))
        self.version = int(version if version is not None else max(p.version for p in self.parts))
        if newest_wins:
            self._newest_wins()
        self._relayout()

    # ---- structure
    def _attach(self, p: Part):
        self.check_part(p)
        for f, e in p.seg.encoders.items():
            self.encoders.setdefault(f, TfidfSvdEncoder.from_arrays(e))
        self.parts.append(p)

    def check_part(self, p: Part):
        """Refuse a segment that does not fit this index: other fields, or vectors from another encoder."""
        if not self.parts:
            return
        ref = self.parts[0]
        if p.fields != self.text_fields:
            raise ValueError(f"segment {p.segment_id}: text fields {p.fields} differ from the index's")
        if set(p.seg.numerics) != set(ref.seg.numerics) or set(p.seg.keywords) != set(ref.seg.keywords):
            raise ValueError(f"segment {p.segment_id}: its fields differ from the index's")
        for f, e in p.seg.encoders.items():
            mine = self.encoders.get(f)
            if mine is not None:
                theirs = TfidfSvdEncoder.from_arrays(e)
                if not (np.array_equal(mine.components, theirs.components) and np.array_equal(mine.idf, theirs.idf)):
                    raise ValueError(f"segment {p.segment_id}: its {f!r} encoder differs from the index's")
        for f, spec in p.manifest.get("vector_fields", {}).items():
            r = ref.manifest.get("vector_fields", {}).get(f)
            if r is not None and r.get("model") != spec.get("model"):
                raise ValueError(f"segment {p.segment_id}: {f!r} vectors from {spec.get('model')!r}, "
                                 f"the index's from {r.get('model')!r}")

    def _newest_wins(self):
        """When segments are loaded without a commit, a key live in several keeps only its newest copy."""
        seen = set()
        for p in reversed(self.parts):
            for i in np.flatnonzero(p.local_live):
                k = p.ids[i]
                if k in seen:
                    p.local_live[i] = False
                else:
                    seen.add(k)

    def _relayout(self):
        """Slots, global statistics and slot-ordered columns from the parts' live docs."""
        b = self.b
        F = len(self.text_fields)
        pi, li, fo = [], [], []
        for k, p in enumerate(self.parts):
            loc = np.flatnonzero(p.local_live)
            pi.append(np.full(len(loc), k, np.int64)), li.append(loc.astype(np.int64)), fo.append(p.first_offset[loc])
        pi, li, fo = (np.concatenate(x) for x in (pi, li, fo))
        order = np.lexsort((li, pi, fo))
        pi, li, fo = pi[order], li[order], fo[order]
        ids = np.array([self.parts[k].ids[i] for k, i in zip(pi.tolist(), li.tolist())], dtype=object)
        if len(set(ids.tolist())) != len(ids):
            dup = next(x for x, c in Counter(ids.tolist()).items() if c > 1)
            raise ValueError(f"_id {dup!r} is live in two segments (tombstone the older copy)")
        N = len(ids)
        self.N = self.n_live = N
        self.slot_part, self.slot_local, self.ids, self.first_offset = pi, li, ids, fo
        self.pos = {d: s for s, d in enumerate(ids.tolist())}
        self.known = set()
        for p in self.parts:
            self.known.update(p.local_of)
        # the global vocabulary and the statistics over live docs
        if len(self.parts) == 1:
            vocab = self.parts[0].vocab_list
        else:
            vocab = sorted(set().union(*(p.vocab_list for p in self.parts)))
        self.vocab_list = vocab
        self.vocab = self.parts[0].vocab if len(self.parts) == 1 else {w: i for i, w in enumerate(vocab)}
        G = len(vocab)
        df = np.zeros(G, np.float64)
        df_field = np.zeros((G, F), np.float64)
        field_len = np.zeros((N, F), np.int32)
        for k, p in enumerate(self.parts):
            p.gid = (np.arange(len(p.vocab_list), dtype=np.int64) if len(self.parts) == 1 else
                     np.fromiter((self.vocab[w] for w in p.vocab_list), np.int64, len(p.vocab_list)))
            slots_k = np.flatnonzero(pi == k)
            p.slot_of_local = np.full(p.n, -1, np.int64)
            p.slot_of_local[li[slots_k]] = slots_k
            post_live = p.local_live[p.host_doc_ids] if len(p.host_doc_ids) else np.zeros(0, bool)
            g = p.gid[p.term_of_posting[post_live]]
            df += np.bincount(g, minlength=G)
            tfl = p.host_tf[post_live] > 0
            for f in range(F):
                df_field[:, f] += np.bincount(g[tfl[:, f]], minlength=G)
            field_len[slots_k] = p.host_field_len[li[slots_k]]
        self.host_df, self.host_df_field = df, df_field
        self.host_idf = _idf(N, df)
        self.host_idf_field = _idf(N, df_field) if F else np.zeros((G, 0), np.float32)
        avg = ((field_len.sum(axis=0, dtype=np.int64) / N).astype(np.float32) if N else np.zeros(F, np.float32))
        self.host_field_len = field_len
        self.host_rel_len = (field_len / np.where(avg > 0, avg, 1.0)).astype(np.float32)
        self.rel_len = b.asarray(self.host_rel_len, "float32")
        self.idf = b.asarray(self.host_idf, "float32")
        self.idf_field = b.asarray(self.host_idf_field, "float32")
        for p in self.parts:
            p.N, p.rel_len = N, self.rel_len
            p.host_idf, p.host_idf_field = self.host_idf[p.gid], self.host_idf_field[p.gid]
            p.idf, p.idf_field = b.asarray(p.host_idf, "float32"), b.asarray(p.host_idf_field, "float32")
            p.identity = len(self.parts) == 1 and p.n == N
            if p.identity:
                p.doc_ids, p.post_live = p.doc_ids_local, None
            else:
                mapped = b.asarray(p.slot_of_local, "int64")[p.doc_ids_local] if len(p.host_doc_ids) \
                    else p.doc_ids_local
                p.post_live = mapped >= 0
                p.doc_ids = b.where(p.post_live, mapped, 0)
            p.live_local = np.flatnonzero(p.slot_of_local >= 0)
            p.live_slot = p.slot_of_local[p.live_local]
        # slot-ordered columns
        self.live = b.full(N, True, "bool")
        self.numerics, self.missing, self.num_dtype = {}, {}, {}
        for f in self.parts[0].numerics:
            self.num_dtype[f] = str(np.asarray(self.parts[0].seg.numerics[f].values).dtype)
            self.numerics[f] = self._gather(lambda p, f=f: p.numerics[f], self.num_dtype[f])
            self.missing[f] = self._gather(lambda p, f=f: p.missing[f], "bool")
        self.keywords = {}
        for f in self.parts[0].keyword_values:
            if len(self.parts) == 1:
                p = self.parts[0]
                self.keywords[f] = KeywordState(b, p.keyword_codes[f][li], p.keyword_values[f])
                continue
            values = [""] + sorted(set().union(*(p.keyword_values[f][1:] for p in self.parts)))
            idx = {v: i for i, v in enumerate(values)}
            codes = np.zeros(N, np.int64)
            for k, p in enumerate(self.parts):
                remap = np.fromiter((idx[v] for v in p.keyword_values[f]), np.int64, len(p.keyword_values[f]))
                slots_k = np.flatnonzero(pi == k)
                codes[slots_k] = remap[p.keyword_codes[f][li[slots_k]]]
            self.keywords[f] = KeywordState(b, codes, values)
        self.vectors = {f: True for f in self.parts[0].vectors if all(f in p.vectors for p in self.parts)}
        self.device_nbytes = sum(p.device_nbytes for p in self.parts)
        self.work_nbytes = BATCH * max(N, 1) * 4 * 2
        if self.check_budget and self.device_nbytes + self.work_nbytes > b.memory_budget():
            raise OverBudget(f"index needs {self.device_nbytes + self.work_nbytes} bytes on {b.name}; "
                             f"budget {b.memory_budget()}")
        self._static = None
        self._value_codes = {}
        self._dictionary = None

    def _gather(self, get, dtype):
        """A slot-ordered device column from each part's local column."""
        b = self.b
        if len(self.parts) == 1 and self.parts[0].identity:
            return b.copy(get(self.parts[0]))
        out = b.zeros(self.N, dtype)
        for p in self.parts:
            if len(p.live_local):
                out[b.asarray(p.live_slot, "int64")] = get(p)[b.asarray(p.live_local, "int64")]
        return out

    @property
    def has_positions(self) -> bool:
        return all(p.host_positions is not None for p in self.parts)

    def owner(self, slot: int):
        """``(part, local doc)`` of a slot."""
        return self.parts[int(self.slot_part[slot])], int(self.slot_local[slot])

    def part(self, segment_id: str) -> Part:
        for p in self.parts:
            if p.segment_id == segment_id:
                return p
        raise KeyError(f"no segment {segment_id!r} in this index")

    # ---- the updater's hooks
    def patch(self, field: str, ids, values) -> None:
        """Ids must be known (the caller checks); deleted ids are skipped."""
        pairs = [(self.pos[i], v) for i, v in zip(ids, values) if i in self.pos]
        if not pairs:
            return
        rows = np.array([r for r, _ in pairs], np.int64)
        vals = [v for _, v in pairs]
        miss = np.array([v is None or (isinstance(v, float) and v != v) for v in vals], bool)
        filled = np.array([0 if m else v for v, m in zip(vals, miss)]).astype(self.num_dtype[field])
        b = self.b
        self.numerics[field][b.asarray(rows, "int64")] = b.asarray(filled)
        self.missing[field][b.asarray(rows, "int64")] = b.asarray(miss)
        for k, p in enumerate(self.parts):                  # the parts keep them for snapshots and merges
            sel = self.slot_part[rows] == k
            if sel.any():
                loc = b.asarray(self.slot_local[rows[sel]], "int64")
                p.numerics[field][loc] = b.asarray(filled[sel])
                p.missing[field][loc] = b.asarray(miss[sel])
        self._value_codes.pop(field, None)          # refreshed on next use, for this field only

    def tombstone(self, ids, parts=None) -> int:
        """Mark live copies of ``ids`` dead (in ``parts`` only, if given), without a relayout. Unknown or dead ids
        are ignored. Returns how many copies died."""
        n = 0
        for i in ids:
            s = self.pos.get(i)
            if s is not None:
                p, loc = self.owner(s)
                if (parts is None or p in parts) and p.local_live[loc]:
                    p.local_live[loc] = False
                    n += 1
        return n

    def delete(self, ids) -> None:
        """Ids must be known (the caller checks); already-deleted ids are a no-op. Statistics are live-only, so a
        delete relays out the index (equal to a rebuild)."""
        if self.tombstone(ids):
            self._relayout()

    def value_codes(self, field: str):
        """For a numeric field: ``(codes int64[N] on the device, 0 = missing; keys list)``, codes in ascending value
        order. Computed on the device once per layout and field, and again only after a patch to that field; the
        distinct values come to the host once (they are the facet keys)."""
        hit = self._value_codes.get(field)
        if hit is None:
            b = self.b
            vals, miss = self.numerics[field], self.missing[field]
            uniq = b.unique(vals[~miss])
            codes = b.where(miss, 0, b.searchsorted(uniq, vals) + 1)
            dt = self.num_dtype[field]
            conv = bool if dt == "bool" else int if dt.startswith("int") else float
            hit = self._value_codes[field] = (b.astype(codes, "int64"), [None] + [conv(v) for v in b.to_numpy(uniq)])
        return hit

    def static_features(self):
        """[N, n_static] float32: relative length per text field, then log popularity of the first match field."""
        if self._static is None:
            cols = [self.host_rel_len]
            mf = self.match_fields()
            if mf:
                kw = self.keywords[mf[0]]
                cnt = np.bincount(kw.codes_host, minlength=kw.n).astype(np.float32)
                cnt[0] = 0
                cols.append(np.log1p(cnt[kw.codes_host])[:, None])
            self._static = self.b.asarray(np.concatenate(cols, axis=1).astype(np.float32), "float32")
        return self._static

    def match_fields(self) -> list[str]:
        eng = (self.mapping.raw.get("_meta") or {}).get("engine") or {}
        given = (eng.get("features") or {}).get("match")
        if given is not None:
            return [f for f in given if f in self.keywords]
        return [f for f in self.keywords if "." in f and f.split(".")[0] in self.field_index]
