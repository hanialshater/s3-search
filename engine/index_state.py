"""The loaded index: a segment's arrays on a device, plus the host-side dictionaries (docs/INTERFACES.md §4)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .analysis import analyze
from .backend import Backend, OverBudget
from .encode import TfidfSvdEncoder
from .mapping import Mapping, parse_mapping
from .segment import SegmentArrays, device_nbytes

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


class IndexState:
    def __init__(self, seg: SegmentArrays, b: Backend, check_budget: bool = True):
        self.b, self.seg = b, seg
        self.mapping: Mapping = parse_mapping(seg.manifest["mapping"])
        self.manifest = seg.manifest
        self.version = int(seg.manifest.get("version", 0))
        self.N = seg.n_docs
        self.device_nbytes = device_nbytes(seg)
        self.work_nbytes = BATCH * max(self.N, 1) * 4 * 2
        if check_budget and self.device_nbytes + self.work_nbytes > b.memory_budget():
            raise OverBudget(f"segment needs {self.device_nbytes + self.work_nbytes} bytes on {b.name}; "
                             f"budget {b.memory_budget()}")
        self.ids = np.asarray(seg.ids, dtype=object)
        self.pos = {d: i for i, d in enumerate(self.ids)}
        t = seg.text
        self.text_fields = list(t.fields)
        self.field_index = {f: i for i, f in enumerate(self.text_fields)}
        self.vocab = {w: i for i, w in enumerate(t.vocab)}
        self.host_term_ptr, self.host_doc_ids = np.asarray(t.term_ptr), np.asarray(t.doc_ids)
        self.host_tf, self.host_field_len = np.asarray(t.tf), np.asarray(t.field_len)
        N, F = self.N, len(self.text_fields)
        idf_of = lambda df: np.log1p((N - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.host_idf = idf_of(np.asarray(t.df, np.float64))
        self.host_idf_field = idf_of(np.asarray(t.df_field, np.float64)) if F else np.zeros((len(t.vocab), 0), np.float32)
        avg = np.asarray(t.avg_len, np.float32)
        self.host_rel_len = (self.host_field_len / np.where(avg > 0, avg, 1.0)).astype(np.float32)
        self.term_ptr = b.asarray(self.host_term_ptr, "int64")
        self.doc_ids = b.asarray(self.host_doc_ids, "int64")
        self.tf = b.asarray(self.host_tf, "uint8")
        self.rel_len = b.asarray(self.host_rel_len, "float32")
        self.idf = b.asarray(self.host_idf, "float32")
        self.idf_field = b.asarray(self.host_idf_field, "float32")
        self.keywords = {f: KeywordState(b, c.codes, c.values) for f, c in seg.keywords.items()}
        self.numerics, self.missing, self.num_dtype = {}, {}, {}
        for f, col in seg.numerics.items():
            self.num_dtype[f] = str(np.asarray(col.values).dtype)
            self.numerics[f] = b.asarray(np.array(col.values))
            self.missing[f] = b.unpack_bits(np.asarray(col.missing), N)
        self.vectors = {f: b.asarray(np.asarray(v, np.float32), "float32") for f, v in seg.vectors.items()}
        self.vector_spec = dict(seg.manifest.get("vector_fields", {}))
        self.encoders = {f: TfidfSvdEncoder.from_arrays(e) for f, e in seg.encoders.items()}
        live = seg.live if seg.live is not None else np.full((N + 31) // 32, 0xFFFFFFFF, np.uint32)
        self.live = b.unpack_bits(np.asarray(live), N)
        self._static = None
        self._value_codes: dict = {}

    # ---- the updater's hooks
    def patch(self, field: str, ids, values) -> None:
        """Ids must be known (the caller checks); deleted ids are skipped."""
        rows = np.array([self.pos[i] for i in ids], dtype=np.int64)
        vals = list(values)
        if len(rows):
            alive = np.asarray(self.b.to_numpy(self.live[self.b.asarray(rows, "int64")]), bool)
            rows, vals = rows[alive], [v for v, a in zip(vals, alive) if a]
        if not len(rows):
            return
        miss = np.array([v is None or (isinstance(v, float) and v != v) for v in vals], bool)
        filled = np.array([0 if m else v for v, m in zip(vals, miss)]).astype(self.num_dtype[field])
        r = self.b.asarray(rows, "int64")
        self.numerics[field][r] = self.b.asarray(filled)
        self.missing[field][r] = self.b.asarray(miss)
        self._value_codes.pop(field, None)          # refreshed on next use, for this field only

    def value_codes(self, field: str):
        """For a numeric field: ``(codes int64[N] on the device, 0 = missing; keys list)``, codes in ascending value
        order. Computed on the device once per segment and field, and again only after a patch to that field; the
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

    def delete(self, ids) -> None:
        """Ids must be known (the caller checks); already-deleted ids are a no-op."""
        rows = [self.pos[i] for i in ids]
        if rows:
            self.live[self.b.asarray(np.array(rows, np.int64), "int64")] = False

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
