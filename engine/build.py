"""The reference segment builder (docs/INTERFACES.md §4): a catalogue DataFrame and a mapping to ``SegmentArrays``.
Ported from the prototype's ``build_postings``. The data plane's indexer calls it on a store snapshot."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from itertools import chain

import numpy as np
import pandas as pd

from .analysis import _WORD, ANALYZER_VERSION, _token
from .encode import doc_texts, get_encoder
from .errors import InvalidMapping
from .mapping import Mapping, parse_mapping
from .segment import KeywordColumn, NumericColumn, SegmentArrays, TextArrays

FIRST_OFFSET_COLUMN = "_first_offset"   # optional catalogue column (the store's snapshot carries it)


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v)) if not isinstance(v, (list, tuple, np.ndarray)) else False
    except (TypeError, ValueError):
        return False


def resolve_id_field(m: Mapping, catalogue: pd.DataFrame) -> str:
    if m.id_field in catalogue.columns:
        return m.id_field
    for alt in ("_id", "product_id"):
        if alt in catalogue.columns:
            return alt
    raise InvalidMapping("_meta.engine.id_field", f"{m.id_field!r} is not a catalogue column")


POS_BITS = 24                    # a position is (field << 24) | position-in-field, uint32


def build_text(catalogue: pd.DataFrame, fields: list[str], positions: bool = True) -> TextArrays:
    N, F = len(catalogue), len(fields)
    term_of: dict[str, int] = {}
    word_tid: dict[str, int] = {}                 # a raw word -> its term id, or -1 (stopword); this build only
    per_field = []
    field_len = np.zeros((N, F), dtype=np.int32)
    for f_i, f in enumerate(fields):
        col = catalogue[f].tolist() if f in catalogue.columns else [None] * N
        # One regex pass per document (C), then one dictionary pass over all the field's words: the same tokens as
        # engine.analysis.analyze_with_positions, without a Python call per token.
        words_per_doc = [[] if _is_missing(t) else _WORD.findall(str(t).lower()) for t in col]
        n_words = np.fromiter((len(w) for w in words_per_doc), np.int64, N)
        words = list(chain.from_iterable(words_per_doc))
        del words_per_doc
        get = word_tid.get
        ids = [get(w, -2) for w in words]
        if -2 in ids:
            for i, x in enumerate(ids):
                if x == -2:
                    w = words[i]
                    t = _token(w)
                    x = word_tid[w] = -1 if t is None else term_of.setdefault(t, len(term_of))
                    ids[i] = x
        ids = np.asarray(ids, np.int64)
        doc_of_word = np.repeat(np.arange(N, dtype=np.int64), n_words)
        keep = ids >= 0
        start = np.concatenate([[0], np.cumsum(n_words)[:-1]]) if N else np.zeros(0, np.int64)
        pos = (np.arange(len(ids), dtype=np.int64) - start[doc_of_word])[keep] if positions else None
        terms, docs = ids[keep], doc_of_word[keep]
        field_len[:, f_i] = np.bincount(docs, minlength=N)
        per_field.append((terms, docs, np.minimum(pos, (1 << POS_BITS) - 1) if positions else None))
    vocab = sorted(term_of)
    V = len(vocab)
    remap = np.zeros(max(V, 1), np.int64)
    remap[[term_of[t] for t in vocab]] = np.arange(V)
    # Every token's posting key (term * N + doc), field-major and in position order within each field; one stable
    # sort by key groups the tokens by posting with (field, position) ascending inside, which gives the postings,
    # tf per field and the positions in one pass.
    key_all = np.concatenate([remap[t] * N + d for t, d, _ in per_field]) if per_field else np.zeros(0, np.int64)
    field_all = np.concatenate([np.full(len(t), f_i, np.int64) for f_i, (t, _, _) in enumerate(per_field)]) \
        if per_field else np.zeros(0, np.int64)
    order = np.argsort(key_all, kind="stable")
    sk = key_all[order]
    first = np.r_[True, sk[1:] != sk[:-1]] if len(sk) else np.zeros(0, bool)
    starts = np.flatnonzero(first)
    postings = sk[starts]
    P = len(postings)
    p_of_tok = np.cumsum(first) - 1
    tf = np.minimum(np.bincount(p_of_tok * F + field_all[order], minlength=P * F), 255).astype(np.uint8)
    tf = tf.reshape(P, F)
    terms_of_p, docs_of_p = (postings // N, postings % N) if N else (postings, postings)
    df = np.bincount(terms_of_p, minlength=V).astype(np.int32)
    df_field = np.stack([np.bincount(terms_of_p[tf[:, f] > 0], minlength=V) for f in range(F)], axis=1) \
        if F else np.zeros((V, 0))
    term_ptr = np.zeros(V + 1, dtype=np.int64)
    term_ptr[1:] = np.cumsum(df)
    avg_len = field_len.mean(axis=0).astype(np.float32) if N else np.zeros(F, np.float32)
    pos_ptr = pos_vals = None
    if positions:
        # Positions, CSR over postings: for posting p, positions[pos_ptr[p]:pos_ptr[p+1]] holds
        # (field << 24) | position, ascending (by field, then position).
        code = np.concatenate([(np.int64(f_i) << POS_BITS) | pos for f_i, (_, _, pos) in enumerate(per_field)]) \
            if per_field else np.zeros(0, np.int64)
        pos_vals = code[order].astype(np.uint32)
        pos_ptr = np.r_[starts, len(sk)].astype(np.int64)
    return TextArrays(fields=list(fields), vocab=vocab, term_ptr=term_ptr, doc_ids=docs_of_p.astype(np.int32), tf=tf,
                      field_len=field_len, df=df, df_field=df_field.astype(np.int32), avg_len=avg_len,
                      pos_ptr=pos_ptr, positions=pos_vals)


def keyword_column(series: pd.Series, ignore_above: int) -> KeywordColumn:
    vals = []
    for v in series.tolist():
        if _is_missing(v):
            vals.append("")
        elif isinstance(v, (list, tuple, np.ndarray)):
            raise InvalidMapping(str(series.name), "keyword fields are single-valued in v1")
        else:
            s = str(v)
            vals.append(s if len(s) <= ignore_above else "")
    uniq = sorted(set(vals) - {""})
    code_of = {v: i + 1 for i, v in enumerate(uniq)}
    codes = np.fromiter((code_of.get(v, 0) for v in vals), np.int32, len(vals))
    return KeywordColumn(codes=codes, values=[""] + uniq)


def numeric_column(series: pd.Series, dtype: str, pack) -> NumericColumn:
    """Accepts numpy dtypes, object columns with None, and pandas' nullable Int32, Int64, Float32, boolean."""
    missing = np.asarray(series.isna().to_numpy(), dtype=bool)
    if dtype == "bool":
        values = np.array([bool(v) if not m else False for v, m in zip(series.tolist(), missing)], dtype=bool)
    else:
        num = pd.to_numeric(series.astype(object).where(~missing, 0))
        values = np.asarray(num.to_numpy(dtype="float64" if dtype == "float32" else "int64"), dtype=dtype)
    return NumericColumn(values=values, missing=pack(missing))


def pack_bits(mask: np.ndarray) -> np.ndarray:
    from .backend.numpy_backend import NumpyBackend
    return NumpyBackend().pack_bits(mask)


def build_segment(catalogue: pd.DataFrame, mapping: dict, version: int = 0,
                  vector_models: dict[str, str] | None = None) -> SegmentArrays:
    """Doc ids are the catalogue's row order (the store's snapshot is ordered by ``first_offset``).

    Only mapped columns (and the id column) are read; others, including the store's ``_``-prefixed columns, are
    ignored, except ``_first_offset`` (kept per doc; 0..N-1 when absent). ``version`` goes to ``manifest["version"]``;
    a caller may also set ``seg.manifest["version"]`` before ``write_segment``, which writes whatever it holds.
    ``vector_models``: for vectors given as a catalogue column, the name of the encoder that produced them (else
    ``catalogue.attrs["vector_models"]``, else ``"given"``); it is recorded as ``vector_fields.<f>.model``.
    """
    m = parse_mapping(mapping)
    catalogue = catalogue.reset_index(drop=True)
    id_col = resolve_id_field(m, catalogue)
    ids = catalogue[id_col].astype(str).to_numpy(dtype=object)
    if len(set(ids)) != len(ids):
        dup = pd.Series(ids)[pd.Series(ids).duplicated()].iloc[0]
        raise ValueError(f"duplicate _id {dup!r} in the catalogue")
    N = len(catalogue)
    first_offset = (catalogue[FIRST_OFFSET_COLUMN].to_numpy(np.int64) if FIRST_OFFSET_COLUMN in catalogue.columns
                    else np.arange(N, dtype=np.int64))
    text = build_text(catalogue, m.text, positions=m.positions)
    seg = SegmentArrays(manifest={}, ids=ids, first_offset=first_offset, text=text,
                        live=pack_bits(np.ones(N, bool)))
    for f, spec in m.keyword.items():
        src = spec["source"]
        series = catalogue[src] if src in catalogue.columns else pd.Series([None] * N, name=src)
        seg.keywords[f] = keyword_column(series, spec["ignore_above"])
    for f, dt in m.numeric.items():
        series = catalogue[f] if f in catalogue.columns else pd.Series([None] * N, dtype=float)
        seg.numerics[f] = numeric_column(series, dt, pack_bits)
    vector_fields = {}
    for f, spec in m.vector.items():
        model = None
        if f in catalogue.columns:
            # Given vectors (the store's): None or NaN rows become zero vectors (similarity 0 to every query).
            vec = np.zeros((N, spec["dim"]), np.float32)
            for i, v in enumerate(catalogue[f].tolist()):
                if not _is_missing(v):
                    a = np.asarray(v, np.float32).ravel()
                    if a.shape[0] != spec["dim"]:
                        raise InvalidMapping(f"properties.{f}.dimension", f"row {i} has {a.shape[0]} dimensions")
                    vec[i] = a
            model = (vector_models or {}).get(f) or (catalogue.attrs.get("vector_models") or {}).get(f) or "given"
        elif f in m.embeddings:
            enc = get_encoder(m.embeddings[f]["model"])
            texts = doc_texts(catalogue, m.embeddings[f].get("inputs") or m.text[:1])
            vec = enc.fit_transform(texts)
            model = enc.name
            seg.encoders[f] = enc.to_arrays()
        else:
            continue
        if vec.shape[1] != spec["dim"]:
            raise InvalidMapping(f"properties.{f}.dimension", f"vectors have {vec.shape[1]} dimensions")
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        if spec["space_type"] == "cosinesimil":
            vec = vec / np.where(norm > 0, norm, 1.0)
        seg.vectors[f] = vec.astype(np.float32)
        vector_fields[f] = {"dim": spec["dim"], "dtype": "float32", "model": model, "space_type": spec["space_type"],
                            "encoder": f in seg.encoders}
    raw = json.dumps(m.raw, sort_keys=True, default=str)
    ident = hashlib.sha256((raw + "\x00".join(ids[:1000]) + f"|{N}|{version}|{len(text.vocab)}").encode())
    seg.manifest = {
        "version": int(version), "segment_id": f"seg-{int(version)}-{ident.hexdigest()[:8]}",
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "analyzer": ANALYZER_VERSION,
        "mapping": m.raw, "mapping_sha256": hashlib.sha256(raw.encode()).hexdigest(), "id_field": id_col,
        "text_fields": list(m.text), "keyword_fields": list(m.keyword), "numeric_fields": dict(m.numeric),
        "vector_fields": vector_fields, "hot": list(m.hot), "bm25": {"k1": m.k1, "b": m.b},
    }
    return seg
