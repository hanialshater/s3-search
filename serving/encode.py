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
    cols = [_column_text(c, h) for c in columns.get("sort") or []]
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


def _shell(resp: dict, hits_value) -> dict:
    """The dict form's key order, as ``engine.api.expand_hits`` makes it: ``hits`` first, and inside it the other keys
    (``total``) and then ``hits``."""
    h = resp["hits"]
    inner = {k: v for k, v in h.items() if k != "columns"}
    inner["hits"] = hits_value
    return {"hits": inner, **{k: v for k, v in resp.items() if k != "hits"}}


def _encode_response(resp: dict) -> str:
    text = json.dumps(_shell(resp, _PLACEHOLDER), separators=_SEP)
    marker = '"hits":' + json.dumps(_PLACEHOLDER)
    if text.count(marker) != 1:                                 # the placeholder must be ours alone
        return json.dumps(expand(resp), separators=_SEP)
    return text.replace(marker, '"hits":' + encode_hits(resp["hits"]["columns"]), 1)


def encode(obj: Any) -> bytes:
    """A response, a list of responses (``_msearch``) or any JSON value, as the bytes ``json.dumps`` gives for its
    dict form (compact separators, ASCII)."""
    if _is_columnar(obj):
        return _encode_response(obj).encode()
    if isinstance(obj, list) and any(_is_columnar(r) for r in obj):
        return ("[" + ",".join(_encode_response(r) if _is_columnar(r) else json.dumps(r, separators=_SEP)
                               for r in obj) + "]").encode()
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
