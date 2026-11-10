"""The mapping (docs/INTERFACES.md §2): an OpenSearch mapping body, engine settings under ``_meta.engine``."""
from __future__ import annotations

from dataclasses import dataclass, field

from .analysis import ANALYZERS
from .errors import InvalidMapping

NUMERIC_TYPES = {"float": "float32", "integer": "int32", "long": "int64", "boolean": "bool"}
TOP_KEYS = {"_meta", "properties", "dynamic"}


@dataclass
class Mapping:
    raw: dict
    id_field: str
    text: list[str]                        # text fields, in BM25F order
    keyword: dict[str, dict]               # name (e.g. "product_brand.keyword") -> {"source": column, "ignore_above"}
    numeric: dict[str, str]                # name -> dtype
    vector: dict[str, dict]                # name -> {"dim", "space_type"}
    hot: list[str]
    k1: float = 1.2
    b: float = 0.75
    embeddings: dict[str, dict] = field(default_factory=dict)
    dynamic: str = "true"
    positions: bool = True                 # _meta.engine.positions (S7-1): index term positions for phrases

    def kind(self, name: str) -> str | None:
        if name in self.text:
            return "text"
        if name in self.keyword:
            return "keyword"
        if name in self.numeric:
            return "numeric"
        if name in self.vector:
            return "vector"
        return None


def parse_mapping(raw: dict) -> Mapping:
    if not isinstance(raw, dict):
        raise InvalidMapping("$", "the mapping is a dict (an OpenSearch mapping body)")
    if "mappings" in raw and "properties" not in raw:      # accept the index-creation body as well
        raw = raw["mappings"]
    for k in raw:
        if k not in TOP_KEYS:
            raise InvalidMapping(k, "unsupported top-level key")
    props = raw.get("properties")
    if not isinstance(props, dict) or not props:
        raise InvalidMapping("properties", "required: a dict of fields")
    eng = (raw.get("_meta") or {}).get("engine") or {}
    text, keyword, numeric, vector = [], {}, {}, {}
    for name, spec in props.items():
        path = f"properties.{name}"
        if not isinstance(spec, dict) or "type" not in spec:
            raise InvalidMapping(path, "each field needs a type")
        t = spec["type"]
        if t == "text":
            an = spec.get("analyzer", "s3_english")
            if an not in ANALYZERS:
                raise InvalidMapping(f"{path}.analyzer", f"supported analyzers: {ANALYZERS}")
            text.append(name)
            for sub, sspec in (spec.get("fields") or {}).items():
                if not isinstance(sspec, dict) or sspec.get("type") != "keyword":
                    raise InvalidMapping(f"{path}.fields.{sub}", "only keyword sub-fields are supported")
                keyword[f"{name}.{sub}"] = {"source": name, "ignore_above": int(sspec.get("ignore_above", 2 ** 31))}
        elif t == "keyword":
            keyword[name] = {"source": name, "ignore_above": int(spec.get("ignore_above", 2 ** 31))}
        elif t in NUMERIC_TYPES:
            numeric[name] = NUMERIC_TYPES[t]
        elif t == "knn_vector":
            dim = spec.get("dimension")
            if not isinstance(dim, int) or dim <= 0:
                raise InvalidMapping(f"{path}.dimension", "a positive int")
            space = (spec.get("method") or {}).get("space_type", spec.get("space_type", "cosinesimil"))
            if space not in ("cosinesimil", "innerproduct"):
                raise InvalidMapping(f"{path}.method.space_type", "cosinesimil or innerproduct")
            vector[name] = {"dim": dim, "space_type": space}
        else:
            raise InvalidMapping(f"{path}.type", f"unsupported type {t!r}")
    hot = list(eng.get("hot", []))
    for h in hot:
        if h not in numeric:
            raise InvalidMapping("_meta.engine.hot", f"{h!r}: hot fields are numeric or boolean fields")
    bm25 = eng.get("bm25", {})
    embeddings = dict(eng.get("embeddings", {}))
    for v, spec in embeddings.items():
        if v not in vector:
            raise InvalidMapping(f"_meta.engine.embeddings.{v}", "not a knn_vector field")
        for src in spec.get("inputs", []):
            if src not in props:
                raise InvalidMapping(f"_meta.engine.embeddings.{v}.inputs", f"{src!r} is not a field")
    return Mapping(raw=raw, id_field=eng.get("id_field", "_id"), text=text, keyword=keyword, numeric=numeric,
                   vector=vector, hot=hot, k1=float(bm25.get("k1", 1.2)), b=float(bm25.get("b", 0.75)),
                   embeddings=embeddings, dynamic=str(raw.get("dynamic", "true")).lower(),
                   positions=bool(eng.get("positions", True)))
