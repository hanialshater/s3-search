"""The mapping (docs/INTERFACES.md §2): an OpenSearch mapping body with engine settings under ``_meta.engine``.

``parse_mapping`` checks a mapping dict once and returns a ``Mapping``; ``coerce`` turns one source value into the
value the store keeps, or raises ``BadValue``. Everything in the data plane takes either the dict or a ``Mapping``.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TYPES = ("text", "keyword", "float", "integer", "long", "boolean", "knn_vector", "geo_point", "nested")
NESTED_CHILD_TYPES = ("text", "keyword", "float", "integer", "long", "boolean")   # S9-5: no vectors, geo, nesting
MAX_NESTED = (1 << 16) - 1   # children per parent (their first_offset is parent << 16 | i)
NUMERIC = ("float", "integer", "long", "boolean")
ID_MAX_BYTES = 512
INT32 = (-(2 ** 31), 2 ** 31 - 1)
INT64 = (-(2 ** 63), 2 ** 63 - 1)
_F32_MAX = float(np.finfo(np.float32).max)


class InvalidMapping(ValueError):
    """A malformed or unsupported mapping. ``.path`` is the JSON path (for example ``properties.price.type``).

    The interfaces name ``engine.api.InvalidMapping``; the data plane may not import ``engine.api`` for it, so this
    class is the data plane's until the engine puts the error where both can import it (see the S2 report).
    """

    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


class BadValue(ValueError):
    """A value that does not fit its field. ``.reason`` is a rejection reason (``bad_type`` or ``dims_mismatch``)."""

    def __init__(self, reason: str, detail: str, field: str | None = None):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.field = field          # set for a nested child's sub-field ("path.sub"); else the caller's field


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    dim: int | None = None                       # knn_vector
    ignore_above: int | None = None              # keyword
    subfields: tuple[tuple[str, str], ...] = ()  # (name, type), e.g. ("keyword", "keyword")
    children: tuple[tuple[str, "Field"], ...] = ()  # nested: (sub-field name, its Field)

    @property
    def child(self) -> dict:
        return dict(self.children)


@dataclass
class Mapping:
    raw: dict
    id_field: str
    fields: dict[str, Field]
    hot: tuple[str, ...]
    dynamic: str                                  # "strict" rejects unknown fields; anything else keeps them unindexed
    embeddings: dict[str, dict] = field(default_factory=dict)   # vector field -> {"model", "inputs"}
    sha256: str = ""

    @property
    def vector_fields(self) -> list[str]:
        return sorted(n for n, f in self.fields.items() if f.type == "knn_vector")


def mapping_sha256(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def as_mapping(m: dict | Mapping) -> Mapping:
    return m if isinstance(m, Mapping) else parse_mapping(m)


def _parse_field(name: str, spec, path: str, in_nested: bool = False) -> Field:
    if not isinstance(spec, dict):
        raise InvalidMapping(path, "must be an object")
    t = spec.get("type")
    allowed = NESTED_CHILD_TYPES if in_nested else TYPES
    if t not in allowed:
        what = "nested properties are" if in_nested else "v1 supports"
        raise InvalidMapping(f"{path}.type", f"unsupported type {t!r}; {what} {', '.join(allowed)}")
    dim = None
    if t == "knn_vector":
        dim = spec.get("dimension")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise InvalidMapping(f"{path}.dimension", "knn_vector needs a positive integer dimension")
    if t == "geo_point":
        extra = sorted(set(spec) - {"type"})
        if extra:
            raise InvalidMapping(f"{path}.{extra[0]}", "geo_point options are not supported")
    children: tuple = ()
    if t == "nested":
        extra = sorted(set(spec) - {"type", "properties"})
        if extra:
            raise InvalidMapping(f"{path}.{extra[0]}", "nested options are not supported")
        sub = spec.get("properties")
        if not isinstance(sub, dict) or not sub:
            raise InvalidMapping(f"{path}.properties", "a nested field needs properties")
        children = tuple((k, _parse_field(k, ks, f"{path}.properties.{k}", in_nested=True)) for k, ks in sub.items())
    ignore_above = spec.get("ignore_above")
    if ignore_above is not None and (not isinstance(ignore_above, int) or ignore_above < 0):
        raise InvalidMapping(f"{path}.ignore_above", "must be a non-negative integer")
    subs = []
    for sub, sspec in (spec.get("fields") or {}).items():
        st = (sspec or {}).get("type")
        if st != "keyword" or in_nested:
            raise InvalidMapping(f"{path}.fields.{sub}.type", "v1 sub-fields are keyword only, and not inside nested")
        subs.append((sub, st))
    return Field(name, t, dim, ignore_above, tuple(subs), children)


def parse_mapping(raw: dict) -> Mapping:
    if not isinstance(raw, dict):
        raise InvalidMapping("$", "a mapping is a JSON object")
    eng = (raw.get("_meta") or {}).get("engine") or {}
    if not isinstance(eng, dict):
        raise InvalidMapping("_meta.engine", "must be an object")
    props = raw.get("properties")
    if not isinstance(props, dict) or not props:
        raise InvalidMapping("properties", "must be a non-empty object")
    fields = {name: _parse_field(name, spec, f"properties.{name}") for name, spec in props.items()}
    id_field = eng.get("id_field", "_id")
    if not isinstance(id_field, str) or not id_field:
        raise InvalidMapping("_meta.engine.id_field", "must be a non-empty string")
    if id_field in fields and fields[id_field].type not in ("keyword",):
        raise InvalidMapping(f"properties.{id_field}.type", "the id field, if mapped, must be a keyword")
    hot = eng.get("hot", [])
    if not isinstance(hot, list):
        raise InvalidMapping("_meta.engine.hot", "must be a list of field names")
    for i, h in enumerate(hot):
        if h not in fields:
            raise InvalidMapping(f"_meta.engine.hot[{i}]", f"{h!r} is not a mapped field")
        if fields[h].type not in NUMERIC:
            raise InvalidMapping(f"_meta.engine.hot[{i}]", f"hot fields are numeric or boolean; {h!r} is {fields[h].type}")
    embeddings = {}
    for vf, espec in (eng.get("embeddings") or {}).items():
        path = f"_meta.engine.embeddings.{vf}"
        if vf not in fields or fields[vf].type != "knn_vector":
            raise InvalidMapping(path, f"{vf!r} is not a knn_vector field")
        inputs = (espec or {}).get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise InvalidMapping(f"{path}.inputs", "must be a non-empty list of text or keyword fields")
        for i, inp in enumerate(inputs):
            if inp not in fields or fields[inp].type not in ("text", "keyword"):
                raise InvalidMapping(f"{path}.inputs[{i}]", f"{inp!r} is not a text or keyword field")
        model = (espec or {}).get("model")
        if not isinstance(model, str):
            raise InvalidMapping(f"{path}.model", "must name an encoder")
        embeddings[vf] = {"model": model, "inputs": list(inputs)}
    dynamic = str(raw.get("dynamic", "true")).lower()
    return Mapping(raw, id_field, fields, tuple(hot), dynamic, embeddings, mapping_sha256(raw))


# ---- values ---------------------------------------------------------------------------------------------------------

try:
    import pandas as _pd
    _PD_MISSING = (_pd.NA, _pd.NaT)
except ImportError:  # pragma: no cover
    _PD_MISSING = ()
_PLAIN = (str, int, bool)      # never missing: the common case returns at once


def is_missing(v: Any) -> bool:
    t = type(v)
    if t in _PLAIN:
        return False
    if v is None:
        return True
    if t is float:
        return v != v
    if isinstance(v, np.floating):
        return bool(np.isnan(v))
    return any(v is m for m in _PD_MISSING)


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def coerce(f: Field, v: Any) -> Any:
    """One non-missing source value to the stored value: str, float, int, bool or a float32 ``numpy`` vector."""
    t = f.type
    if t == "text":
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple, np.ndarray)) and all(isinstance(x, str) for x in v):
            return "\n".join(v)
        if isinstance(v, numbers.Number) and not _is_bool(v):
            return str(v)
        raise BadValue("bad_type", f"text field {f.name!r} got {type(v).__name__}")
    if t == "keyword":
        if isinstance(v, str):
            return v
        if isinstance(v, numbers.Number) and not _is_bool(v):
            return str(v)
        raise BadValue("bad_type", f"keyword field {f.name!r} (single-valued in v1) got {type(v).__name__}")
    if t == "float":
        if _is_bool(v):
            raise BadValue("bad_type", f"float field {f.name!r} got a boolean")
        try:
            x = float(v)
        except (TypeError, ValueError):
            raise BadValue("bad_type", f"float field {f.name!r} got {v!r:.80}") from None
        if not math.isfinite(x) or abs(x) > _F32_MAX:
            raise BadValue("bad_type", f"float field {f.name!r} got a non-finite or out-of-range value {x!r}")
        return x
    if t in ("integer", "long"):
        lo, hi = INT32 if t == "integer" else INT64
        if _is_bool(v):
            raise BadValue("bad_type", f"{t} field {f.name!r} got a boolean")
        if isinstance(v, numbers.Integral):
            x = int(v)
        elif isinstance(v, numbers.Real) and float(v).is_integer():
            x = int(v)
        elif isinstance(v, str):
            try:
                x = int(v.strip())
            except ValueError:
                raise BadValue("bad_type", f"{t} field {f.name!r} got {v!r:.80}") from None
        else:
            raise BadValue("bad_type", f"{t} field {f.name!r} got {v!r:.80}")
        if not lo <= x <= hi:
            raise BadValue("bad_type", f"{t} field {f.name!r} value {x} is out of range")
        return x
    if t == "boolean":
        if _is_bool(v):
            return bool(v)
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return v.lower() == "true"
        raise BadValue("bad_type", f"boolean field {f.name!r} got {v!r:.80}")
    if t == "knn_vector":
        if isinstance(v, (str, bytes, dict)) or not hasattr(v, "__len__"):
            raise BadValue("bad_type", f"knn_vector field {f.name!r} got {type(v).__name__}")
        try:
            a = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError):
            raise BadValue("bad_type", f"knn_vector field {f.name!r} holds non-numbers") from None
        if a.ndim != 1 or a.shape[0] != f.dim:
            raise BadValue("dims_mismatch", f"knn_vector field {f.name!r} wants {f.dim} dims, got shape {list(a.shape)}")
        if not np.all(np.isfinite(a)):
            raise BadValue("bad_type", f"knn_vector field {f.name!r} holds non-finite values")
        n = float(np.linalg.norm(a))
        if n == 0.0:
            raise BadValue("bad_type", f"knn_vector field {f.name!r} is the zero vector (cosine is undefined)")
        return (a / n).astype(np.float32)
    if t == "geo_point":
        return coerce_point(f, v)
    if t == "nested":
        return coerce_nested(f, v)
    raise AssertionError(t)  # pragma: no cover


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WKT = re.compile(rf"\s*POINT\s*\(\s*({_NUM})\s+({_NUM})\s*\)\s*", re.IGNORECASE)
_LATLON = re.compile(rf"\s*({_NUM})\s*,\s*({_NUM})\s*")


def _num(x, what: str, f: Field) -> float:
    if _is_bool(x) or isinstance(x, (dict, list, tuple)):
        raise BadValue("bad_geo", f"geo_point field {f.name!r}: {what} is not a number")
    try:
        y = float(x)
    except (TypeError, ValueError):
        raise BadValue("bad_geo", f"geo_point field {f.name!r}: {what} {x!r:.40} is not a number") from None
    if not math.isfinite(y):
        raise BadValue("bad_geo", f"geo_point field {f.name!r}: {what} is not finite")
    return y


def coerce_point(f: Field, v: Any) -> dict:
    """OpenSearch's point forms to ``{"lat", "lon"}``: an object ``{"lat", "lon"}``, a string ``"lat,lon"``, an array
    ``[lon, lat]``, or a WKT ``"POINT (lon lat)"``. Geohashes are refused (the engine does not decode them)."""
    if isinstance(v, dict):
        if set(v) != {"lat", "lon"}:
            raise BadValue("bad_geo", f"geo_point field {f.name!r}: an object point is {{lat, lon}}, got keys {sorted(v)}")
        lat, lon = _num(v["lat"], "lat", f), _num(v["lon"], "lon", f)
    elif isinstance(v, str):
        m = _WKT.fullmatch(v)
        if m:
            lon, lat = float(m.group(1)), float(m.group(2))
        else:
            m = _LATLON.fullmatch(v)
            if not m:
                why = ("geohashes are not supported" if re.fullmatch(r"\s*[0-9a-z]+\s*", v)
                       else "not 'lat,lon' or WKT 'POINT (lon lat)'")
                raise BadValue("bad_geo", f"geo_point field {f.name!r}: {v!r:.40}: {why}")
            lat, lon = float(m.group(1)), float(m.group(2))
    elif isinstance(v, (list, tuple, np.ndarray)):
        if len(v) != 2:
            raise BadValue("bad_geo", f"geo_point field {f.name!r}: an array point is [lon, lat], got {len(v)} values")
        lon, lat = _num(v[0], "lon", f), _num(v[1], "lat", f)
    else:
        raise BadValue("bad_geo", f"geo_point field {f.name!r}: a point is {{lat, lon}}, 'lat,lon', [lon, lat] "
                                  f"or 'POINT (lon lat)', got {type(v).__name__}")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise BadValue("geo_out_of_range", f"geo_point field {f.name!r}: ({lat}, {lon}) is outside "
                                           "lat [-90, 90], lon [-180, 180]")
    return {"lat": lat, "lon": lon}


def coerce_nested(f: Field, v: Any, strict: bool = False) -> list[dict]:
    """A nested value (an object or a list of objects) to a list of child objects, each sub-field coerced by the
    nested mapping; missing sub-values are left out; unknown sub-fields are kept in the source (the engine ignores
    them) unless ``strict``. An empty list is no children."""
    objs = [v] if isinstance(v, dict) else list(v) if isinstance(v, (list, tuple, np.ndarray)) else None
    if objs is None:
        raise BadValue("bad_nested", f"nested field {f.name!r}: a value is an object or a list of objects, "
                                     f"got {type(v).__name__}")
    if len(objs) > MAX_NESTED:
        raise BadValue("too_many_nested", f"nested field {f.name!r}: {len(objs)} objects; at most {MAX_NESTED}")
    kids = f.child
    out = []
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            raise BadValue("bad_nested", f"nested field {f.name!r}: item {i} is {type(o).__name__}, not an object")
        c = {}
        for k, x in o.items():
            sub = kids.get(k)
            if sub is None:
                if strict:
                    raise BadValue("unknown_field", f"{f.name}.{k} is not mapped and the mapping is strict",
                                   f"{f.name}.{k}")
                if not is_missing(x):
                    c[k] = json_safe(x)
                continue
            if is_missing(x):
                continue
            try:
                c[k] = coerce(sub, x)
            except BadValue as e:
                raise BadValue(e.reason, f"{f.name}[{i}]: {e.detail}", f"{f.name}.{k}") from None
        out.append(c)
    return out


def json_safe(v: Any) -> Any:
    """A source value (possibly numpy or pandas) as plain JSON data, for unmapped fields and rejected records."""
    if is_missing(v):
        return None
    if isinstance(v, (str, bool, int)) or v is None:
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, np.generic):
        return json_safe(v.item())
    if isinstance(v, np.ndarray):
        return [json_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
