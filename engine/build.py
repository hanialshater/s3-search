"""The reference segment builder (docs/INTERFACES.md §4): a catalogue DataFrame and a mapping to ``SegmentArrays``.
Ported from the prototype's ``build_postings``. The data plane's indexer calls it on a store snapshot."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

from .analysis import ANALYZER_VERSION, tokens_of, words_of
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


def _column(catalogue: pd.DataFrame, f: str, N: int) -> pd.Series:
    return catalogue[f] if f in catalogue.columns else pd.Series([None] * N, dtype=object, name=f)


def _missing_mask(series: pd.Series) -> np.ndarray:
    """``_is_missing`` of every value, vectorised: None, NaN, NA and NaT are missing; lists, tuples, arrays and
    dicts are not."""
    if series.dtype == object:
        return np.asarray(pd.isna(series.to_numpy(dtype=object)), bool)
    return np.asarray(series.isna().to_numpy(), bool)


def _strings(series: pd.Series, miss: np.ndarray) -> np.ndarray:
    """object[N]: ``str(v)`` of every value, ``""`` where missing (vectorised for string columns)."""
    if series.dtype != object:
        try:
            if pd.api.types.is_string_dtype(series.dtype):
                return series.to_numpy(dtype=object, na_value="")
        except TypeError:
            pass
    vals = series.to_numpy(dtype=object)
    out = np.empty(len(vals), dtype=object)
    out[miss] = ""
    keep = ~miss
    out[keep] = [v if type(v) is str else str(v) for v in vals[keep]]
    return out


def build_workers() -> int:
    """Processes for build-time analysis: the CPUs this process may use (``S3_BUILD_WORKERS`` overrides; 1 or 0:
    none). The output does not depend on it."""
    env = os.environ.get("S3_BUILD_WORKERS")
    if env is not None:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                # not Linux
        return max(1, os.cpu_count() or 1)


PARALLEL_MIN_CHARS = 4_000_000      # below this, worker start-up costs more than it saves

_POOL_INPUT = None                  # what the forked workers read (set only while a pool runs)


def _words_slice(task):
    f_i, a, b = task
    return words_of(_POOL_INPUT[f_i][a:b])


def _tokens_slice(task):
    a, b = task
    return tokens_of(_POOL_INPUT[a:b])


GPU_MODULES = ("cupy", "cuvs", "cuml", "rmm", "numba.cuda", "pycuda")


def cuda_maybe_started() -> str | None:
    """Why CUDA may already be started in this process (a forked child of it can hang or crash), or None. Torch is
    asked (``torch.cuda.is_initialized``); a loaded GPU library (CuPy, cuVS, cuML, RMM, Numba's CUDA, PyCUDA)
    counts as started, since those give no such check."""
    import sys
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_initialized():
                return "torch has initialised CUDA"
        except Exception:
            pass
    for name in GPU_MODULES:
        if name in sys.modules:
            return f"{name} is loaded"
    return None


_NOTICED = set()


def _notice(why: str) -> None:
    if why not in _NOTICED:
        _NOTICED.add(why)
        import sys
        print(f"engine.build: analysing in this process, not in workers ({why}); the segment is the same",
              file=sys.stderr)


def _pool_ok(workers: int, chars: int) -> bool:
    """Workers are forked (they inherit their input, nothing is pickled to them, and no caller's ``__main__`` is
    re-imported), and only from a process that is safe to fork: one Python thread (forking a threaded process, a
    server, can deadlock the child) and CUDA not started (a forked child of a CUDA process can hang or crash).
    Otherwise the analysis runs in this process, with a one-line notice; the output is the same either way."""
    import multiprocessing as mp
    import threading
    if workers <= 1 or chars < PARALLEL_MIN_CHARS:
        return False
    if "fork" not in mp.get_all_start_methods():
        why = "no fork on this platform"
    elif threading.active_count() > 1:
        why = f"{threading.active_count()} Python threads"
    else:
        why = cuda_maybe_started()
    if why is not None:
        _notice(why)
        return False
    return True


def _run(fn, data, tasks, workers):
    """``fn`` over ``tasks`` in a pool of forked workers, the results in task order. ``multiprocessing.Pool`` forks
    all its workers before it starts its own threads, so no fork happens in a threaded process."""
    import multiprocessing as mp
    global _POOL_INPUT
    _POOL_INPUT = data
    try:
        with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
            return pool.map(fn, tasks, chunksize=1)
    finally:
        _POOL_INPUT = None


def _analyse_fields(per_field: list[list[str]]):
    """The words of each field's distinct values and every distinct word's token, in two passes that run in worker
    processes for a large catalogue (each pass in chunks, merged back in order, so the result does not depend on the
    number of workers): ``([(word ids int64 per regex word, word count int64 per value) per field], tokens)``, where
    ``tokens[w]`` is word ``w``'s token or None (a stopword)."""
    workers = build_workers()
    chars = sum(len(v) for vals in per_field for v in vals)
    par = _pool_ok(workers, chars)
    if par:
        n_chunks = workers * 4
        tasks = []
        for f_i, vals in enumerate(per_field):
            share = max(1, int(round(n_chunks * sum(len(v) for v in vals) / max(chars, 1))))
            step = max(1, -(-len(vals) // share))
            tasks += [(f_i, a, a + step) for a in range(0, len(vals), step)] or [(f_i, 0, 0)]
        parts = _run(_words_slice, per_field, tasks, workers)
    else:
        tasks = [(f_i, 0, len(vals)) for f_i, vals in enumerate(per_field)]
        parts = [words_of(vals) for vals in per_field]
    word_of: dict[str, int] = {}                          # every distinct word, first seen in task order
    pieces = [([], []) for _ in per_field]
    for (f_i, _, _), (uw, flat, lens) in zip(tasks, parts):
        gid = np.fromiter((word_of.setdefault(w, len(word_of)) for w in uw), np.int64, len(uw))
        pieces[f_i][0].append(gid[flat] if len(flat) else np.zeros(0, np.int64))
        pieces[f_i][1].append(lens)
    words = list(word_of)
    del word_of
    if par and len(words) > 1:
        step = -(-len(words) // (workers * 4))
        tokens = [t for part in _run(_tokens_slice, words, [(a, a + step) for a in range(0, len(words), step)],
                                     workers) for t in part]
    else:
        tokens = tokens_of(words)
    out = [(np.concatenate(f) if f else np.zeros(0, np.int64), np.concatenate(ln) if ln else np.zeros(0, np.int64))
           for f, ln in pieces]
    return out, tokens


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
    per_field = []
    field_len = np.zeros((N, F), dtype=np.int32)
    # Each field's distinct values are analysed once (brands and colours repeat), in worker processes for a large
    # catalogue; a doc's words are its value's, gathered with numpy. The same tokens as
    # engine.analysis.analyze_with_positions (a missing value and "" have no words).
    coded = []
    for f in fields:
        series = _column(catalogue, f, N)
        codes, uniq = pd.factorize(_strings(series, _missing_mask(series)))
        coded.append((np.asarray(codes, np.int64), list(uniq)))
    analysed, tokens = _analyse_fields([u for _, u in coded])
    word_tid = np.fromiter((-1 if t is None else term_of.setdefault(t, len(term_of)) for t in tokens), np.int64,
                           len(tokens))                    # a word -> its term id, or -1 (a stopword)
    for f_i, ((codes, _), (u_flat, u_lens)) in enumerate(zip(coded, analysed)):
        u_ids = word_tid[u_flat]
        u_ptr = np.concatenate([[0], np.cumsum(u_lens)]).astype(np.int64)
        n_words = u_lens[codes] if N else np.zeros(0, np.int64)
        total = int(n_words.sum())
        first_word = np.cumsum(n_words) - n_words
        ids = u_ids[np.repeat(u_ptr[codes] - first_word, n_words) + np.arange(total, dtype=np.int64)] \
            if total else np.zeros(0, np.int64)
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


def _keyword_strings(series: pd.Series, ignore_above: int):
    """(``str`` values with "" for missing and over ``ignore_above``, the missing mask, the kept mask)."""
    miss = _missing_mask(series)
    if series.dtype == object and any(isinstance(v, (list, tuple, np.ndarray))
                                      for v in series.to_numpy(dtype=object)[~miss]):
        raise InvalidMapping(str(series.name), "keyword fields are single-valued in v1")
    strs = _strings(series, miss)
    lens = np.fromiter(map(len, strs), np.int64, len(strs))
    kept = ~miss & (lens <= ignore_above)
    return np.where(kept, strs, ""), miss, kept


def keyword_column(series: pd.Series, ignore_above: int) -> KeywordColumn:
    vals, _, _ = _keyword_strings(series, ignore_above)
    codes, uniq = pd.factorize(vals)
    uniq = list(uniq)
    values = sorted(set(uniq) - {""})
    code_of = {v: i + 1 for i, v in enumerate(values)}
    table = np.fromiter((code_of.get(v, 0) for v in uniq), np.int32, len(uniq))
    codes = table[codes] if len(codes) else np.zeros(0, np.int32)
    return KeywordColumn(codes=np.asarray(codes, np.int32), values=[""] + values)


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
    source = {}
    if m.source:                                          # raw text for highlight (S7-5), host only
        for f in m.text:
            series = _column(catalogue, f, N)
            miss = _missing_mask(series)
            source[f] = np.where(miss, None, _strings(series, miss)).astype(object)
    seg = SegmentArrays(manifest={}, ids=ids, first_offset=first_offset, text=text,
                        live=pack_bits(np.ones(N, bool)))
    for f, spec in m.keyword.items():
        src = spec["source"]
        series = catalogue[src] if src in catalogue.columns else pd.Series([None] * N, name=src)
        seg.keywords[f] = keyword_column(series, spec["ignore_above"])
    for f, dt in m.numeric.items():
        series = catalogue[f] if f in catalogue.columns else pd.Series([None] * N, dtype=float)
        seg.numerics[f] = numeric_column(series, dt, pack_bits)
    from .geo import BadGeo, encode_column, lat_col, lon_col
    for f in m.geo:                                       # S9-4: Lucene's encoding, as hidden int32 columns
        col = catalogue[f].tolist() if f in catalogue.columns else [None] * N
        try:
            lat_e, lon_e, miss = encode_column(col)
        except BadGeo as e:
            raise InvalidMapping(f"properties.{f}", f"a value is not a geo_point: {e}") from None
        seg.numerics[lat_col(f)] = NumericColumn(values=lat_e, missing=pack_bits(miss))
        seg.numerics[lon_col(f)] = NumericColumn(values=lon_e, missing=pack_bits(miss))
    for path, child_raw in m.nested.items():              # S9-5: one row per nested object
        col = catalogue[path].tolist() if path in catalogue.columns else [None] * N
        rows = []
        for d, objs in enumerate(col):
            if _is_missing(objs) if not isinstance(objs, (list, tuple, np.ndarray, dict)) else False:
                continue
            objs = [objs] if isinstance(objs, dict) else list(objs)
            if len(objs) >= 1 << 16:
                raise InvalidMapping(f"properties.{path}", "at most 65,535 nested objects per document")
            for i, o in enumerate(objs):
                if not isinstance(o, dict):
                    raise InvalidMapping(f"properties.{path}", f"row {d}: nested values are objects")
                r = {f"{path}.{k}": v for k, v in o.items()}
                r.update(_child_id=f"{ids[d]}\x1f{i}", _parent=d, _first_offset=(int(first_offset[d]) << 16) | i)
                rows.append(r)
        props = [k for k in child_raw["properties"] if k != "_parent"]
        child = pd.DataFrame(rows, columns=["_child_id", "_parent", "_first_offset"] + props) if rows else \
            pd.DataFrame({c: pd.Series([], dtype=object) for c in ["_child_id", "_first_offset"] + props}
                         ).assign(_parent=pd.Series([], dtype="int64"), _first_offset=pd.Series([], dtype="int64"))
        seg.nested[path] = build_segment(child, child_raw, version=version)
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
            enc = get_encoder(m.embeddings[f]["model"], m.embeddings[f].get("fit", "numpy"))
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
    seg.source = source
    # which docs have a value, for exists: the stored value, not its tokens (a value of stop words or punctuation
    # still exists, as OpenSearch counts it); a keyword over ignore_above is not indexed, so it does not exist
    for f in m.text:
        seg.present[f] = pack_bits(~_missing_mask(_column(catalogue, f, N)))
    for f, spec in m.keyword.items():
        seg.present[f] = pack_bits(_keyword_strings(_column(catalogue, spec["source"], N), spec["ignore_above"])[2])
    seg.manifest = {
        "version": int(version), "segment_id": f"seg-{int(version)}-{ident.hexdigest()[:8]}",
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "analyzer": ANALYZER_VERSION,
        "mapping": m.raw, "mapping_sha256": hashlib.sha256(raw.encode()).hexdigest(), "id_field": id_col,
        "text_fields": list(m.text), "keyword_fields": list(m.keyword), "numeric_fields": {f: str(np.asarray(c.values).dtype) for f, c in seg.numerics.items()},
        "geo_fields": list(m.geo), "nested_paths": list(m.nested),
        "vector_fields": vector_fields, "hot": list(m.hot),
        "bm25": {"k1": m.k1, "b": dict(m.field_b), "b_default": m.b} if m.field_b else {"k1": m.k1, "b": m.b},
    }
    return seg


def merge_segments(segs: list[SegmentArrays], mapping: dict | None = None, version: int | None = None) -> SegmentArrays:
    """One segment of the live docs of ``segs`` (each with its current live bits and hot columns, e.g. from
    ``Index.snapshot_segments``), ordered by ``first_offset``: the postings, positions and vectors are merged as
    arrays (no re-analysis, so no source text is needed). ``version``: the merged segment's (default: the inputs'
    largest). Its statistics are its own; an index recomputes global ones."""
    from .backend.numpy_backend import NumpyBackend
    nb = NumpyBackend()
    if not segs:
        raise ValueError("merge_segments needs at least one segment")
    first = segs[0]
    lives = [nb.unpack_bits(np.asarray(s.live if s.live is not None else nb.pack_bits(np.ones(s.n_docs, bool))),
                            s.n_docs) for s in segs]
    si = np.concatenate([np.full(int(l.sum()), k, np.int64) for k, l in enumerate(lives)])
    li = np.concatenate([np.flatnonzero(l) for l in lives]).astype(np.int64)
    fo = np.concatenate([np.asarray(s.first_offset, np.int64)[np.flatnonzero(l)] for s, l in zip(segs, lives)])
    order = np.lexsort((li, si, fo))
    si, li, fo = si[order], li[order], fo[order]
    N = len(si)
    ids = np.array([segs[k].ids[i] for k, i in zip(si.tolist(), li.tolist())], dtype=object)
    if len(set(ids.tolist())) != N:
        raise ValueError("merge_segments: an _id is live in two inputs")
    new_of = []
    for k, s in enumerate(segs):
        m = np.full(s.n_docs, -1, np.int64)
        sel = np.flatnonzero(si == k)
        m[li[sel]] = sel
        new_of.append(m)
    F = len(first.text.fields)
    vocab = sorted(set().union(*(s.text.vocab for s in segs)))
    gid_of = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    keys, tfs, pos_starts, pos_counts, pos_all = [], [], [], [], []
    with_pos = all(s.text.positions is not None for s in segs)
    pos_base = 0
    for k, s in enumerate(segs):
        t = s.text
        gid = np.fromiter((gid_of[w] for w in t.vocab), np.int64, len(t.vocab))
        term = np.repeat(np.arange(len(t.vocab), dtype=np.int64), np.diff(np.asarray(t.term_ptr)))
        doc = new_of[k][np.asarray(t.doc_ids, np.int64)]
        keep = doc >= 0
        keys.append(gid[term[keep]] * max(N, 1) + doc[keep])
        tfs.append(np.asarray(t.tf)[keep])
        if with_pos:
            pp = np.asarray(t.pos_ptr, np.int64)
            pos_starts.append(pp[:-1][keep] + pos_base)
            pos_counts.append(np.diff(pp)[keep])
            pos_all.append(np.asarray(t.positions))
            pos_base += len(t.positions)
    key = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    o = np.argsort(key, kind="stable")
    key = key[o]
    tf = np.concatenate(tfs)[o] if tfs else np.zeros((0, F), np.uint8)
    terms_of_p, docs_of_p = (key // max(N, 1), key % max(N, 1))
    df = np.bincount(terms_of_p, minlength=V).astype(np.int32)
    df_field = np.stack([np.bincount(terms_of_p[tf[:, f] > 0], minlength=V) for f in range(F)], axis=1) \
        if F else np.zeros((V, 0))
    term_ptr = np.zeros(V + 1, np.int64)
    term_ptr[1:] = np.cumsum(df)
    field_len = np.zeros((N, F), np.int32)
    for k, s in enumerate(segs):
        sel = np.flatnonzero(si == k)
        field_len[sel] = np.asarray(s.text.field_len)[li[sel]]
    avg_len = (field_len.sum(axis=0, dtype=np.int64) / N).astype(np.float32) if N else np.zeros(F, np.float32)
    pos_ptr = positions = None
    if with_pos:
        starts, counts = np.concatenate(pos_starts)[o], np.concatenate(pos_counts)[o]
        _, idx = nb.csr_expand(np.r_[0, np.cumsum(counts)], np.arange(len(counts)), np.zeros(len(counts), np.int64))
        seg_start = np.repeat(starts - np.r_[0, np.cumsum(counts)][:-1], counts)
        positions = np.concatenate(pos_all)[idx + seg_start] if len(idx) else np.zeros(0, np.uint32)
        pos_ptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    text = TextArrays(fields=list(first.text.fields), vocab=vocab, term_ptr=term_ptr,
                      doc_ids=docs_of_p.astype(np.int32), tf=tf, field_len=field_len, df=df,
                      df_field=df_field.astype(np.int32), avg_len=avg_len, pos_ptr=pos_ptr,
                      positions=None if positions is None else positions.astype(np.uint32))
    out = SegmentArrays(manifest=dict(first.manifest), ids=ids, first_offset=fo, text=text,
                        live=pack_bits(np.ones(N, bool)))

    def gather(get, dtype):
        col = np.zeros(N, dtype)
        for k, s in enumerate(segs):
            sel = np.flatnonzero(si == k)
            col[sel] = np.asarray(get(s))[li[sel]]
        return col
    for f in first.keywords:
        values = [""] + sorted(set().union(*(s.keywords[f].values[1:] for s in segs)))
        idx = {v: i for i, v in enumerate(values)}
        codes = np.zeros(N, np.int32)
        for k, s in enumerate(segs):
            remap = np.fromiter((idx[v] for v in s.keywords[f].values), np.int64, len(s.keywords[f].values))
            sel = np.flatnonzero(si == k)
            codes[sel] = remap[np.asarray(s.keywords[f].codes, np.int64)[li[sel]]]
        out.keywords[f] = KeywordColumn(codes=codes, values=values)
    for f, col in first.numerics.items():
        values = gather(lambda s: s.numerics[f].values, np.asarray(col.values).dtype)
        miss = gather(lambda s: nb.unpack_bits(np.asarray(s.numerics[f].missing), s.n_docs), bool)
        out.numerics[f] = NumericColumn(values=values, missing=pack_bits(miss))
    for f, v in first.vectors.items():
        D = np.asarray(v).shape[1]
        vec = np.zeros((N, D), np.float32)
        for k, s in enumerate(segs):
            sel = np.flatnonzero(si == k)
            vec[sel] = np.asarray(s.vectors[f], np.float32)[li[sel]]
        out.vectors[f] = vec
    for s in segs:
        for f, e in s.encoders.items():
            out.encoders.setdefault(f, e)
    for path in first.nested:                             # S9-5: children of live parents, re-pointed
        import copy as _cp
        kids = []
        for k, s in enumerate(segs):
            c = _cp.copy(s.nested[path])
            c.numerics = dict(c.numerics)
            par = np.asarray(c.numerics["_parent"].values, np.int64)
            newp = new_of[k][par] if len(par) else par
            live = nb.unpack_bits(np.asarray(c.live), c.n_docs) & (newp >= 0)
            c.numerics["_parent"] = NumericColumn(values=np.where(newp >= 0, newp, 0).astype(np.int64),
                                                  missing=c.numerics["_parent"].missing)
            c.live = pack_bits(live)
            kids.append(c)
        out.nested[path] = merge_segments(kids, version=version)
    if all(s.present for s in segs):
        for f in first.present:
            col = np.zeros(N, bool)
            for k, s in enumerate(segs):
                sel = np.flatnonzero(si == k)
                col[sel] = nb.unpack_bits(np.asarray(s.present[f]), s.n_docs)[li[sel]]
            out.present[f] = pack_bits(col)
    if all(s.source for s in segs):
        for f in first.source:
            col = np.empty(N, dtype=object)
            for k, s in enumerate(segs):
                sel = np.flatnonzero(si == k)
                col[sel] = np.asarray(s.source[f], dtype=object)[li[sel]]
            out.source[f] = col
    if mapping is not None:
        out.manifest["mapping"] = mapping
    out.manifest["version"] = int(version if version is not None else max(int(s.manifest.get("version", 0))
                                                                           for s in segs))
    out.manifest.pop("segment_id", None)
    out.manifest.pop("files", None)
    out.manifest.pop("checksum", None)
    out.manifest.pop("source_fields", None)
    out.manifest.pop("present_fields", None)
    return out
