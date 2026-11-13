"""Response bytes for the HTTP front (S6-7): columnar hits encoded without a dict per hit.

``encode(obj) -> bytes`` is exactly ``json.dumps(expand(obj), separators=(",", ":")).encode()``, where ``expand``
turns every search response that carries ``hits.columns`` (docs/INTERFACES.md §5, "Columnar hits") back into the dict
form. A response in the dict form, or anything else, goes through ``json.dumps`` unchanged. So the HTTP response is the
same bytes whichever form the engine answered in; only the cost differs. At depth 2,000 the dict form spent its time in
2,000 dicts and in ``json.dumps`` walking them; here each column is formatted once (``float.__repr__`` and
``encode_basestring_ascii`` are what ``json.dumps`` itself calls) and the hits are joined as one string.

``expand_columns(resp)`` is the dict form of one response (for ``_source``, which adds a field per hit, and for tests).
"""
from __future__ import annotations

import json
import os
from json.encoder import encode_basestring_ascii as _esc
from typing import Any

_PLACEHOLDER = "\x00s3-hits\x00"
_SEP = (",", ":")


def _is_columnar(resp: Any) -> bool:
    return isinstance(resp, dict) and isinstance(resp.get("hits"), dict) and "columns" in resp["hits"]


def _floats(values) -> list[str]:
    """``json.dumps``'s text for each float: ``float.__repr__``, and NaN, Infinity, -Infinity as it writes them."""
    import numpy as np
    arr = np.asarray(values)
    arr = arr.astype(np.float64) if arr.dtype != np.float64 else arr
    out = list(map(float.__repr__, arr.tolist()))
    if not np.isfinite(arr).all():
        for i in np.flatnonzero(~np.isfinite(arr)).tolist():
            v = arr[i]
            out[i] = "NaN" if v != v else ("Infinity" if v > 0 else "-Infinity")
    return out


def _column_text(col: dict, h: int) -> list[str]:
    """One sort column as JSON text per hit; ``missing`` hits are ``null``."""
    import numpy as np
    values, missing = col["values"], col.get("missing")
    if isinstance(values, np.ndarray):
        kind = values.dtype.kind
        if kind == "b":
            out = ["true" if v else "false" for v in values.tolist()]
        elif kind in "iu":
            out = list(map(int.__repr__, values.tolist()))
        elif kind == "f":
            out = _floats(values)
        else:                                                   # an object array: whatever json.dumps makes of it
            out = [json.dumps(v, separators=_SEP) for v in values.tolist()]
    else:                                                       # list[str] for a keyword field
        out = [_esc(v) if isinstance(v, str) else json.dumps(v, separators=_SEP) for v in values]
    if len(out) != h:
        raise ValueError(f"a sort column has {len(out)} values for {h} hits")
    if missing is not None:
        for i in np.flatnonzero(np.asarray(missing, dtype=bool)).tolist():
            out[i] = "null"
    return out


def id_texts(ids) -> list[str]:
    """Each ``_id`` as JSON text. ``ids`` is a ``list[str]`` (``ids="str"``) or one numpy ``S<w>`` array (``ids="bytes"``,
    INTERFACES §5: UTF-8, NUL-padded, never ending in NUL). When every byte of the array is printable ASCII other
    than ``"`` and ``\\`` (the padding NULs aside), the bytes are the JSON text already; otherwise each entry is decoded
    and escaped as ``json.dumps`` does."""
    import numpy as np
    if isinstance(ids, np.ndarray) and ids.dtype.kind == "S":
        b = np.frombuffer(ids.tobytes(), np.uint8)
        plain = not ((b >= 0x7F) | ((b < 0x20) & (b != 0)) | (b == 0x22) | (b == 0x5C)).any()
        if plain:
            return ['"' + x + '"' for x in ids.astype("U").tolist()]
        return [_esc(x.decode("utf-8")) for x in ids.tolist()]
    return [_esc(i) if isinstance(i, str) else json.dumps(i) for i in ids]


def id_list(ids) -> list:
    """``_id`` as the dict form's ``list[str]`` whichever form it came in."""
    import numpy as np
    if isinstance(ids, np.ndarray) and ids.dtype.kind == "S":
        return [x.decode("utf-8") for x in ids.tolist()]
    return list(ids)


def encode_hits(columns: dict) -> str:
    """The dict form's ``hits.hits`` list as JSON text: ``[{"_id":..,"_score":..,"sort":[..]},...]``."""
    ids = columns["_id"]
    h = len(ids)
    if h == 0:
        return "[]"
    id_txt = id_texts(ids)
    import numpy as np
    score_txt = _floats(np.asarray(columns["_score"], np.float32))
    if len(score_txt) != h:
        raise ValueError(f"{len(score_txt)} scores for {h} hits")
    s64 = np.asarray(columns["_score"], np.float32).astype(np.float64)

    def text(c):                                                # a _score sort column is the score again: reuse its text
        v = c.get("values")
        if c.get("missing") is None and isinstance(v, np.ndarray) and v.dtype == np.float64 and len(v) == h \
                and np.array_equal(v, s64):
            return score_txt
        return _column_text(c, h)
    cols = [text(c) for c in columns.get("sort") or []]
    if not cols:
        body = [f'{{"_id":{i},"_score":{s},"sort":[]}}' for i, s in zip(id_txt, score_txt)]
    elif len(cols) == 1:
        body = [f'{{"_id":{i},"_score":{s},"sort":[{a}]}}' for i, s, a in zip(id_txt, score_txt, cols[0])]
    elif len(cols) == 2:
        body = [f'{{"_id":{i},"_score":{s},"sort":[{a},{b}]}}'
                for i, s, a, b in zip(id_txt, score_txt, cols[0], cols[1])]
    else:
        body = [f'{{"_id":{i},"_score":{s},"sort":[{",".join(r)}]}}'
                for i, s, r in zip(id_txt, score_txt, zip(*cols))]
    return "[" + ",".join(body) + "]"


# ---- the fast path (S6-15s) ------------------------------------------------------------------------------------------
try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - the slow path below writes the same bytes
    _orjson = None
if os.environ.get("S3_SERVING_FAST_ENCODE", "1").strip().lower() in ("0", "false", "no", "off"):
    _orjson = None                                              # the pure-Python path only (for comparison)

_TEMPLATES: dict = {}


def _safe_floats(a) -> bool:
    """Whether orjson writes exactly what ``float.__repr__`` writes for every value: finite, and 0 or 1e-4 <= |x| <
    1e16 (outside that, repr switches to exponent notation and orjson does not; orjson writes NaN and inf as null)."""
    import numpy as np
    m = np.abs(a)
    return bool(np.isfinite(a).all() and ((m == 0) | ((m >= 1e-4) & (m < 1e16))).all())


def _orjson_items(values) -> list[bytes]:
    out = _orjson.dumps(values)
    return out[1:-1].split(b",") if len(out) > 2 else []


def _fast_hits(columns: dict) -> bytes | None:
    """``encode_hits``'s bytes without a Python string per value: orjson writes the numbers (only where it agrees with
    repr), the ids are quoted with one vectorised operation, and one bytes template per page shape joins them.
    None when a column falls outside what this path handles (the caller takes ``encode_hits``)."""
    import numpy as np
    if _orjson is None:
        return None
    ids = columns["_id"]
    h = len(ids)
    if h == 0:
        return b"[]"
    if isinstance(ids, np.ndarray) and ids.dtype.kind == "S":
        b = np.frombuffer(ids.tobytes(), np.uint8)
        if ((b >= 0x7F) | ((b < 0x20) & (b != 0)) | (b == 0x22) | (b == 0x5C)).any():
            return None
        id_b = np.char.add(np.char.add(b'"', ids), b'"').tolist()
    else:
        id_b = [_esc(i).encode() if isinstance(i, str) else None for i in ids]
        if any(x is None for x in id_b):
            return None
    s64 = np.asarray(columns["_score"], np.float32).astype(np.float64)
    if len(s64) != h or not _safe_floats(s64):
        return None
    score_b = _orjson_items(s64.tolist())
    cols = []
    for c in columns.get("sort") or []:
        v, miss = c.get("values"), c.get("missing")
        if miss is not None or not isinstance(v, np.ndarray) or len(v) != h:
            return None
        if v.dtype == np.float64 and np.array_equal(v, s64):
            cols.append(score_b)
        elif v.dtype.kind in "iu":
            cols.append(_orjson_items(v.astype(np.int64).tolist()))
        elif v.dtype.kind == "f" and _safe_floats(v.astype(np.float64)):
            cols.append(_orjson_items(v.astype(np.float64).tolist()))
        else:
            return None
    k = len(cols)
    tpl = _TEMPLATES.get((h, k))
    if tpl is None:
        one = b'{"_id":%b,"_score":%b,"sort":[' + b",".join([b"%b"] * k) + b"]}"
        tpl = b"[" + b",".join([one] * h) + b"]"
        if len(_TEMPLATES) > 256:
            _TEMPLATES.clear()
        _TEMPLATES[(h, k)] = tpl
    import itertools
    return tpl % tuple(itertools.chain.from_iterable(zip(id_b, score_b, *cols)))


def _shell(resp: dict, hits_value) -> dict:
    """The dict form's key order, as ``engine.api.expand_hits`` makes it: ``hits`` first, and inside it the other keys
    (``total``) and then ``hits``."""
    h = resp["hits"]
    inner = {k: v for k, v in h.items() if k != "columns"}
    inner["hits"] = hits_value
    return {"hits": inner, **{k: v for k, v in resp.items() if k != "hits"}}


def _encode_response(resp: dict, fast: bool = True) -> bytes:
    text = json.dumps(_shell(resp, _PLACEHOLDER), separators=_SEP).encode()
    marker = b'"hits":' + json.dumps(_PLACEHOLDER).encode()
    if text.count(marker) != 1:                                 # the placeholder must be ours alone
        return json.dumps(expand(resp), separators=_SEP).encode()
    hits = _fast_hits(resp["hits"]["columns"]) if fast else None
    if hits is None:
        hits = encode_hits(resp["hits"]["columns"]).encode()
    return text.replace(marker, b'"hits":' + hits, 1)


def encode(obj: Any) -> bytes:
    """A response, a list of responses (``_msearch``) or any JSON value, as the bytes ``json.dumps`` gives for its
    dict form (compact separators, ASCII)."""
    if _is_columnar(obj):
        return _encode_response(obj)
    if isinstance(obj, list) and any(_is_columnar(r) for r in obj):
        return b"[" + b",".join(_encode_response(r) if _is_columnar(r) else json.dumps(r, separators=_SEP).encode()
                                for r in obj) + b"]"
    return json.dumps(obj, separators=_SEP).encode()


# ---- the dict form ---------------------------------------------------------------------------------------------------
def _py(col: dict, h: int) -> list:
    import numpy as np
    values, missing = col["values"], col.get("missing")
    vals = values.tolist() if isinstance(values, np.ndarray) else list(values)
    if missing is not None:
        for i in np.flatnonzero(np.asarray(missing, dtype=bool)).tolist():
            vals[i] = None
    return vals


def expand_columns(resp: dict) -> dict:
    """The dict form of one columnar response (a new dict; the columns are not changed). Others are returned as is."""
    if not _is_columnar(resp):
        return resp
    import numpy as np
    cols = resp["hits"]["columns"]
    ids = id_list(cols["_id"])
    scores = np.asarray(cols["_score"], np.float32).astype(np.float64).tolist()
    sort = [_py(c, len(ids)) for c in cols.get("sort") or []]
    rows = list(zip(*sort)) if sort else [()] * len(ids)
    hits = [{"_id": i, "_score": s, "sort": list(r)} for i, s, r in zip(ids, scores, rows)]
    return _shell(resp, hits)


def expand(obj: Any) -> Any:
    if isinstance(obj, list):
        return [expand_columns(r) for r in obj]
    return expand_columns(obj)
