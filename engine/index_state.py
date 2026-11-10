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
    """A keyword field's global dictionary (values[0] == "", the rest sorted: code order is value order) and its
    codes per slot (host and device)."""

    def __init__(self, b: Backend, codes: np.ndarray, values: list[str], codes_dev=None):
        self.codes_host = np.asarray(codes, np.int32)
        self.codes = codes_dev if codes_dev is not None else b.asarray(self.codes_host, "int64")
        self.values = list(values)
        self._sorted = np.array(self.values[1:], dtype=object)
        self._code_of = None
        self._toks = None

    def clone(self, b) -> "KeywordState":
        """A copy whose codes may be changed without touching this one (for the copy-on-write index, S8-2c)."""
        c = _copy.copy(self)
        c.codes_host = self.codes_host.copy()
        c.codes = b.copy(self.codes)
        c.values = list(self.values)
        return c

    @property
    def code_of(self) -> dict:
        if self._code_of is None:
            self._code_of = {v: i for i, v in enumerate(self.values)}
        return self._code_of

    def lookup(self, vals) -> np.ndarray:
        """Codes of ``vals`` (-1 for a value not in the dictionary), by binary search: no dictionary rebuild."""
        vals = np.array([str(v) for v in vals], dtype=object)
        if not len(vals) or not len(self._sorted):
            return np.full(len(vals), -1, np.int64)
        pos = np.searchsorted(self._sorted, vals)
        ok = pos < len(self._sorted)
        ok[ok] = self._sorted[pos[ok]] == vals[ok]
        return np.where(ok, pos + 1, -1).astype(np.int64)

    def insert(self, b, new_values) -> np.ndarray:
        """Add values (not yet present) to the sorted dictionary; returns old code -> new code, and remaps the
        codes in place."""
        new = np.array(sorted(set(new_values)), dtype=object)
        at = np.searchsorted(self._sorted, new)
        self._sorted = np.insert(self._sorted, at, new)
        self.values = [""] + self._sorted.tolist()
        old = np.arange(1, len(self._sorted) - len(new) + 1)
        remap = np.concatenate([[0], old + np.searchsorted(at, old - 1, side="right")]).astype(np.int64)
        self.codes_host = remap[self.codes_host.astype(np.int64)].astype(np.int32)
        self.codes = b.asarray(remap, "int64")[self.codes]
        self._code_of = None
        self._toks = None
        return remap

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
        self.present = {f: b.unpack_bits(np.asarray(v), self.n) for f, v in seg.present.items()}
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
        self.gid_dev = None
        self._doc_post = None
        self.doc_postings(np.zeros(0, np.int64))           # build the doc -> postings view now, at load (S8-2b)

    def doc_postings(self, locs: np.ndarray) -> np.ndarray:
        """The postings (indices) of local docs ``locs``, through a doc -> postings view built on first use."""
        if self._doc_post is None:
            order = np.argsort(self.host_doc_ids, kind="stable")
            ptr = np.zeros(self.n + 1, np.int64)
            ptr[1:] = np.cumsum(np.bincount(self.host_doc_ids, minlength=self.n))
            self._doc_post = (ptr, order)
        ptr, order = self._doc_post
        locs = np.asarray(locs, np.int64)
        n = ptr[locs + 1] - ptr[locs]
        idx = np.arange(int(n.sum()), dtype=np.int64) + np.repeat(ptr[locs] - (np.cumsum(n) - n), n)
        return order[idx]

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
        # S9-5: the children of each nested path are an index of their own, one part per parent part
        self.nested = {}
        for path in self.mapping.nested:
            self.nested[path] = IndexState([p.seg.nested[path] for p in self.parts], b, check_budget=False,
                                           version=self.version, newest_wins=False)
        self._sync_nested()

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
        self.first_offset_dev = b.asarray(fo, "int64")
        self._exists = {}
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
        self.G = len(vocab)                     # this state's terms: ids >= G belong to a newer state (S8-2c)
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
        self.field_len_sum = field_len.sum(axis=0, dtype=np.int64)
        self.host_live = np.ones(N, bool)
        self.max_fo = int(fo.max()) if N else -1
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
            p.gid_dev = b.asarray(p.gid, "int64")
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

    def exists_mask(self, field: str):
        """bool[N]: docs with a value in ``field`` (S7-3). Text: at least one token; keyword: a value (an empty or
        over-long value counts as none); numeric: not missing; vector: a non-zero vector."""
        b = self.b
        kind = self.mapping.kind(field)
        if kind == "numeric":
            return ~self.missing[field]
        if kind == "geo":
            return ~self.missing[f"{field}#lat"]
        if field not in self._exists:
            if kind in ("text", "keyword") and all(field in p.present for p in self.parts):
                m = self._gather(lambda p: p.present[field], "bool")      # the stored value (segments since S7-9)
            elif kind == "text":
                m = b.asarray(self.host_field_len[:, self.field_index[field]] > 0, "bool")
            elif kind == "keyword":
                m = self.keywords[field].codes > 0
            else:
                m = self._gather(lambda p: b.sum(p.vectors[field] * p.vectors[field], axis=1) > 0, "bool")
            self._exists[field] = m
        return self._exists[field]

    @property
    def has_positions(self) -> bool:
        return all(p.host_positions is not None for p in self.parts)

    def gid(self, term: str):
        """The global id of ``term`` in this state, or None (S8-2c: the shared vocabulary may hold newer terms)."""
        g = self.vocab.get(term)
        return g if g is not None and g < self.G else None

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
        """Ids must be known (the caller checks); already-deleted ids are a no-op. Statistics are live-only and
        updated incrementally (equal to a rebuild)."""
        self.delete_incremental(ids)

    # ---- nested documents (S9-5)
    def _sync_nested(self) -> None:
        """Children die with their parents: every child whose parent copy is dead is tombstoned."""
        self._nested_parent = {}
        for path, child in getattr(self, "nested", {}).items():
            slots = []
            for pp, cp in zip(self.parts, child.parts):
                par = np.asarray(child.b.to_numpy(cp.numerics["_parent"]), np.int64) if cp.n else np.zeros(0, np.int64)
                dead = cp.local_live & ~pp.local_live[par] if cp.n else np.zeros(0, bool)
                if dead.any():
                    slots.extend(cp.slot_of_local[np.flatnonzero(dead)].tolist())
            if slots:
                child._kill(slots)
                child._finish()

    def nested_parent(self, path: str):
        """int64[Nc] on the device: each child slot's parent slot (-1 for a hole)."""
        hit = self._nested_parent.get(path) if hasattr(self, "_nested_parent") else None
        if hit is None:
            child = self.nested[path]
            out = np.full(child.N, -1, np.int64)
            for k, (pp, cp) in enumerate(zip(self.parts, child.parts)):
                sel = np.flatnonzero((child.slot_part == k) & child.host_live)
                if len(sel):
                    par = np.asarray(child.b.to_numpy(cp.numerics["_parent"]), np.int64)[child.slot_local[sel]]
                    out[sel] = pp.slot_of_local[par]
            if not hasattr(self, "_nested_parent"):
                self._nested_parent = {}
            hit = self._nested_parent[path] = self.b.asarray(out, "int64")
        return hit

    def snapshot_part(self, k: int) -> SegmentArrays:
        seg = self.parts[k].snapshot()
        if getattr(self, "nested", None):
            seg.nested = {path: child.snapshot_part(k) for path, child in self.nested.items()}
        return seg

    # ---- copy-on-write (S8-2c): a writer changes a clone and the Index swaps one reference
    def clone(self, field: str | None = None) -> "IndexState":
        """A copy a writer may change while searches keep using this one. ``field``: only that hot column will
        change (a patch), so only it is copied; otherwise every structure a structural change touches in place is
        copied (the parts are shallow copies with their own live bits, slot maps and posting masks), and the
        immutable arrays (postings, positions, vectors, segment columns) stay shared."""
        b = self.b
        c = _copy.copy(self)
        c.numerics, c.missing = dict(self.numerics), dict(self.missing)
        c._value_codes = dict(self._value_codes)
        c.parts = [_copy.copy(p) for p in self.parts]
        for p in c.parts:
            p.numerics, p.missing = dict(p.numerics), dict(p.missing)
        if field is not None:
            c.numerics[field], c.missing[field] = b.copy(self.numerics[field]), b.copy(self.missing[field])
            for p in c.parts:
                p.numerics[field], p.missing[field] = b.copy(p.numerics[field]), b.copy(p.missing[field])
            c._value_codes.pop(field, None)
            return c
        c._gen = object()                         # parts copy their live bits on first write in this clone
        c.host_df, c.host_df_field = self.host_df.copy(), self.host_df_field.copy()
        c.field_len_sum = self.field_len_sum.copy()
        c.host_live, c.live = self.host_live.copy(), b.copy(self.live)
        c.slot_part, c.slot_local = self.slot_part.copy(), self.slot_local.copy()
        c.first_offset, c.ids = self.first_offset.copy(), self.ids.copy()
        c.host_field_len = self.host_field_len.copy()
        c.numerics = {f: b.copy(v) for f, v in self.numerics.items()}
        c.missing = {f: b.copy(v) for f, v in self.missing.items()}
        c.keywords = {f: kw.clone(b) for f, kw in self.keywords.items()}
        c.pos = dict(self.pos)
        # the vocabulary and the known ids are append-only and shared: a state reads only ids < its own G
        c.encoders = dict(self.encoders)
        c._exists = dict(self._exists)
        c.nested = {p: child.clone() for p, child in getattr(self, "nested", {}).items()}
        c._nested_parent = {}
        return c

    # ---- incremental changes (S8-2b): no full relayout per refresh
    def _kill(self, slots) -> None:
        """Tombstone the live docs in ``slots`` (an int array): their parts' live bits, the postings mask, the
        global statistics (their terms and lengths subtracted), the slot's live bit and the _id map."""
        slots = np.asarray(sorted(set(int(s) for s in slots)), np.int64)
        if not len(slots):
            return
        b, F = self.b, len(self.text_fields)
        G = len(self.vocab_list)
        for k in np.unique(self.slot_part[slots]):
            p = self.parts[int(k)]
            if getattr(p, "_gen", None) is not getattr(self, "_gen", None) or getattr(self, "_gen", None) is None:
                p.local_live = p.local_live.copy()          # copy-on-write: older states keep theirs
                p.slot_of_local = p.slot_of_local.copy()
                if p.post_live is not None:
                    p.post_live = b.copy(p.post_live)
                p._gen = getattr(self, "_gen", None)
            sel = slots[self.slot_part[slots] == k]
            locs = self.slot_local[sel]
            posts = p.doc_postings(locs)
            g = p.gid[p.term_of_posting[posts]]
            self.host_df -= np.bincount(g, minlength=G)
            tfl = p.host_tf[posts] > 0
            for f in range(F):
                self.host_df_field[:, f] -= np.bincount(g[tfl[:, f]], minlength=G)
            self.field_len_sum -= p.host_field_len[locs].sum(axis=0, dtype=np.int64)
            p.local_live[locs] = False
            p.slot_of_local[locs] = -1
            if p.post_live is None:
                p.post_live = b.full(len(p.host_doc_ids), True, "bool")
            if len(posts):
                p.post_live[b.asarray(posts, "int64")] = False
            p.identity = False
            p.live_local = np.flatnonzero(p.slot_of_local >= 0)
            p.live_slot = p.slot_of_local[p.live_local]
        self.host_live[slots] = False
        self.live[b.asarray(slots, "int64")] = False
        for s in slots.tolist():
            self.pos.pop(self.ids[s], None)
        self.n_live -= len(slots)

    def _append_terms(self, p: "Part") -> None:
        """Global term ids are append-only: the new segment's unseen terms get the next ids."""
        new = [w for w in p.vocab_list if w not in self.vocab]
        if new:
            base = len(self.vocab_list)
            if self.vocab is self.parts[0].vocab or isinstance(self.vocab_list, list) and \
                    self.vocab_list is self.parts[0].vocab_list:
                self.vocab, self.vocab_list = dict(self.vocab), list(self.vocab_list)
            for i, w in enumerate(new):
                self.vocab[w] = base + i
            self.vocab_list.extend(new)
            self.G = len(self.vocab_list)
            F = len(self.text_fields)
            self.host_df = np.concatenate([self.host_df, np.zeros(len(new))])
            self.host_df_field = np.concatenate([self.host_df_field, np.zeros((len(new), F))])
        p.gid = np.fromiter((self.vocab[w] for w in p.vocab_list), np.int64, len(p.vocab_list))
        p.gid_dev = self.b.asarray(p.gid, "int64")

    def _place(self, p: "Part", reuse: dict, append: bool) -> bool:
        """Give the live docs of a new part their slots: ``reuse`` (_id -> slot) for edited keys and merged docs,
        new slots at the end for the rest (``append``), or dead when not ``append``. Returns False when appending
        would break first_offset order (the caller then relays out in full)."""
        b, F = self.b, len(self.text_fields)
        live = np.flatnonzero(p.local_live)
        ids = p.ids
        reuse_loc = np.array([i for i in live.tolist() if ids[i] in reuse], np.int64)
        rest = np.array([i for i in live.tolist() if ids[i] not in reuse], np.int64)
        if len(rest) and not append:
            p.local_live[rest] = False
            rest = rest[:0]
        if len(rest):
            fo = p.first_offset[rest]
            if (np.diff(fo) <= 0).any() or fo[0] <= self.max_fo:
                return False
        k = len(self.parts)
        self._append_terms(p)
        self.parts.append(p)
        self.known.update(p.local_of)
        reuse_slot = np.array([reuse[ids[i]] for i in reuse_loc.tolist()], np.int64)
        N0, a = self.N, len(rest)
        app_slot = np.arange(N0, N0 + a, dtype=np.int64)
        # grow the slot-ordered columns
        if a:
            self.N = N0 + a
            self.slot_part = np.concatenate([self.slot_part, np.full(a, k, np.int64)])
            self.slot_local = np.concatenate([self.slot_local, rest])
            self.first_offset = np.concatenate([self.first_offset, p.first_offset[rest]])
            self.first_offset_dev = b.concatenate([self.first_offset_dev, b.asarray(p.first_offset[rest], "int64")])
            self.max_fo = int(p.first_offset[rest][-1])
            self.ids = np.concatenate([self.ids, np.asarray(ids[rest], dtype=object)])
            self.host_live = np.concatenate([self.host_live, np.ones(a, bool)])
            self.live = b.concatenate([self.live, b.full(a, True, "bool")])
            self.host_field_len = np.concatenate([self.host_field_len, p.host_field_len[rest]])
            for f in self.numerics:
                self.numerics[f] = b.concatenate([self.numerics[f], p.numerics[f][b.asarray(rest, "int64")]])
                self.missing[f] = b.concatenate([self.missing[f], p.missing[f][b.asarray(rest, "int64")]])
        if len(reuse_loc):
            self.slot_part[reuse_slot] = k
            self.slot_local[reuse_slot] = reuse_loc
            self.host_live[reuse_slot] = True
            rs, rl = b.asarray(reuse_slot, "int64"), b.asarray(reuse_loc, "int64")
            self.live[rs] = True
            self.host_field_len[reuse_slot] = p.host_field_len[reuse_loc]
            for f in self.numerics:
                self.numerics[f][rs] = p.numerics[f][rl]
                self.missing[f][rs] = p.missing[f][rl]
        slot_of = np.full(p.n, -1, np.int64)
        slot_of[reuse_loc] = reuse_slot
        slot_of[rest] = app_slot
        p.slot_of_local = slot_of
        mapped = b.asarray(slot_of, "int64")[p.doc_ids_local] if len(p.host_doc_ids) else p.doc_ids_local
        p.post_live = mapped >= 0
        p.doc_ids = b.where(p.post_live, mapped, 0)
        p.live_local = np.flatnonzero(slot_of >= 0)
        p.live_slot = slot_of[p.live_local]
        p.identity = False
        for i, s in zip(p.ids[p.live_local].tolist(), p.live_slot.tolist()):
            self.pos[i] = s
        # statistics: the new live docs' terms and lengths
        G = len(self.vocab_list)
        post_live = p.local_live[p.host_doc_ids] if len(p.host_doc_ids) else np.zeros(0, bool)
        g = p.gid[p.term_of_posting[post_live]]
        self.host_df += np.bincount(g, minlength=G)
        tfl = p.host_tf[post_live] > 0
        for f in range(F):
            self.host_df_field[:, f] += np.bincount(g[tfl[:, f]], minlength=G)
        self.field_len_sum += p.host_field_len[p.live_local].sum(axis=0, dtype=np.int64)
        self.n_live += len(p.live_local)
        # keyword dictionaries: the delta's values map into the global dictionary by binary search; values never
        # seen before are inserted in order and the old codes remapped (code order stays value order)
        for f, kw in self.keywords.items():
            vals = p.keyword_values[f]
            local = kw.lookup(vals)
            local[0] = 0
            if (local[1:] < 0).any():
                kw.insert(b, [v for v, c in zip(vals[1:], local[1:]) if c < 0])
                local = kw.lookup(vals)
                local[0] = 0
            if a:
                add = local[p.keyword_codes[f][rest]]
                kw.codes_host = np.concatenate([kw.codes_host, add.astype(np.int32)])
                kw.codes = b.concatenate([kw.codes, b.asarray(add, "int64")])
            if len(reuse_loc):
                c = local[p.keyword_codes[f][reuse_loc]]
                kw.codes_host[reuse_slot] = c
                kw.codes[b.asarray(reuse_slot, "int64")] = b.asarray(c, "int64")
        return True

    def _finish(self) -> None:
        """After incremental changes: idf, relative lengths and the parts' views of them, and the caches."""
        b = self.b
        N = self.n_live
        self.host_idf = _idf(N, self.host_df)
        self.host_idf_field = _idf(N, self.host_df_field) if len(self.text_fields) else self.host_idf_field
        self.idf = b.asarray(self.host_idf, "float32")
        self.idf_field = b.asarray(self.host_idf_field, "float32")
        avg = (self.field_len_sum / N).astype(np.float32) if N else np.zeros(len(self.text_fields), np.float32)
        self.host_rel_len = (self.host_field_len / np.where(avg > 0, avg, 1.0)).astype(np.float32)
        self.rel_len = b.asarray(self.host_rel_len, "float32")
        for p in self.parts:
            p.N, p.rel_len = self.N, self.rel_len
            p.idf, p.idf_field = self.idf[p.gid_dev], self.idf_field[p.gid_dev]
            p.host_idf, p.host_idf_field = self.host_idf[p.gid], self.host_idf_field[p.gid]
        self.vectors = {f: True for f in self.parts[0].vectors if all(f in p.vectors for p in self.parts)}
        self.device_nbytes = sum(p.device_nbytes for p in self.parts)
        self.work_nbytes = BATCH * max(self.N, 1) * 4 * 2
        self._static = None
        self._value_codes = {}
        self._dictionary = None
        self._exists = {}
        if self.N - self.n_live > max(1024, self.N // 2):   # too many holes: compact (renumbers slots in order)
            self._relayout()

    def add_part(self, p: "Part", delete_ids=()) -> None:
        """S8-2b: add a segment incrementally. Edited keys (same first_offset) keep their slot; new keys take new
        slots at the end; tombstones subtract their docs from the statistics."""
        self.check_part(p)
        for f, e in p.seg.encoders.items():
            self.encoders.setdefault(f, TfidfSvdEncoder.from_arrays(e))
        reuse = {}
        for i in np.flatnonzero(p.local_live).tolist():
            s = self.pos.get(p.ids[i])
            if s is not None and self.first_offset[s] == p.first_offset[i]:
                reuse[p.ids[i]] = s
        self._kill([self.pos[i] for i in set(delete_ids) | set(p.ids.tolist()) if i in self.pos])
        if not self._place(p, reuse, append=True):
            if p not in self.parts:
                self.parts.append(p)
            self._relayout()
        else:
            self._finish()
        for path, child in getattr(self, "nested", {}).items():   # S9-5: the new children, then the sync
            child.add_part(Part(p.seg.nested[path], self.b))
        self._sync_nested()

    def replace_parts(self, old: list, p: "Part | None", delete_ids=()) -> None:
        self._before_replace = list(self.parts)
        try:
            self._replace_parts(old, p, delete_ids)
        finally:
            del self._before_replace

    def _replace_parts(self, old: list, p: "Part | None", delete_ids=()) -> None:
        """S8-2b: replace segments by their merge (or drop them): the merged docs keep their slots; the
        statistics change only by docs that were in one and not the other."""
        captured = {}
        for q in old:
            for loc, s in zip(q.live_local.tolist(), q.live_slot.tolist()):
                captured[q.ids[loc]] = s
        if p is not None:
            for i in delete_ids:
                loc = p.local_of.get(i)
                if loc is not None:
                    p.local_live[loc] = False
            reuse = {i: s for i, s in captured.items()
                     if i in p.local_of and p.local_live[p.local_of[i]] and
                     self.first_offset[s] == p.first_offset[p.local_of[i]]}
        self._kill(list(captured.values()))
        keep = [q for q in self.parts if q not in old]
        remap = np.full(len(self.parts) + 1, -1, np.int64)
        for new_k, q in enumerate(keep):
            remap[self.parts.index(q)] = new_k
        live_parts = self.slot_part[self.host_live]
        assert (remap[live_parts] >= 0).all()
        self.slot_part = np.where(self.host_live, remap[self.slot_part], 0)
        self.parts = keep
        old_k = [k for k, q in enumerate(self._before_replace) if q in old] if hasattr(self, "_before_replace") else []
        if p is not None:
            self.check_part(p)
            if not self._place(p, reuse, append=False):
                self._relayout()
            else:
                self._finish()
        else:
            if not self.parts:
                raise ValueError("an index needs at least one segment")
            self._finish()
        for path, child in getattr(self, "nested", {}).items():
            child_old = [child.parts[k] for k in old_k]
            child.replace_parts(child_old, Part(p.seg.nested[path], self.b) if p is not None else None)
        self._sync_nested()

    def delete_incremental(self, ids) -> None:
        slots = [self.pos[i] for i in ids if i in self.pos]
        if slots:
            self._kill(slots)
            self._finish()
            self._sync_nested()

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
                cnt = np.bincount(kw.codes_host[self.host_live], minlength=kw.n).astype(np.float32)
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
