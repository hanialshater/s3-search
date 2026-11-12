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
    b: float = 0.75                        # the b of every text field not in field_b
    field_b: dict = field(default_factory=dict)    # text field -> its own b (BM25F per-field length normalisation)
    embeddings: dict[str, dict] = field(default_factory=dict)
    dynamic: str = "true"
    positions: bool = True                 # _meta.engine.positions (S7-1): index term positions for phrases
    source: bool = True                    # _meta.engine.source (S7-5): keep text fields' raw text, for highlight
    synonyms: dict = field(default_factory=dict)   # S7-6: analysed token -> tuple of alternatives (query time)

    geo: dict = field(default_factory=dict)        # S9-4: geo_point fields
    nested: dict = field(default_factory=dict)     # S9-5: path -> the children's mapping (a mapping of its own)

    def b_of(self, name: str) -> float:
        """The BM25 ``b`` of a text field: its own (``_meta.engine.bm25.b`` as a map), else the default."""
        return float(self.field_b.get(name, self.b))

    def kind(self, name: str) -> str | None:
        if name in self.text:
            return "text"
        if name in self.keyword:
            return "keyword"
        if name in self.numeric:
            return "numeric"
        if name in self.vector:
            return "vector"
        if name in self.geo:
            return "geo"
        if name in self.nested:
            return "nested"
        if any(name.startswith(p + ".") for p in self.nested):
            return "in_nested"
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
    geo_fields = {}
    nested_fields = {}
    eng_raw = (raw.get("_meta") or {}).get("engine") or {}
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
        elif t == "nested":
            sub = spec.get("properties")
            for k in spec:
                if k not in ("type", "properties"):
                    raise InvalidMapping(f"{path}.{k}", "nested options are not supported")
            if not isinstance(sub, dict) or not sub:
                raise InvalidMapping(f"{path}.properties", "a nested field needs properties")
            child_props = {}
            for k, ks in sub.items():
                if not isinstance(ks, dict) or ks.get("type") not in ("text", "keyword", *NUMERIC_TYPES):
                    raise InvalidMapping(f"{path}.properties.{k}",
                                         "nested properties are text, keyword or numeric (no nesting, vectors, geo)")
                child_props[f"{name}.{k}"] = ks
            child_props["_parent"] = {"type": "long"}
            nested_fields[name] = {"_meta": {"engine": {"id_field": "_child_id",
                                                        "positions": bool(eng_raw.get("positions", True)),
                                                        "source": bool(eng_raw.get("source", True))}},
                                   "properties": child_props}
        elif t == "geo_point":
            for k in spec:
                if k != "type":
                    raise InvalidMapping(f"{path}.{k}", "geo_point options are not supported")
            geo_fields[name] = {}
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
    k1, b_default, field_b = _bm25_params(bm25, text)
    embeddings = dict(eng.get("embeddings", {}))
    for v, spec in embeddings.items():
        if v not in vector:
            raise InvalidMapping(f"_meta.engine.embeddings.{v}", "not a knn_vector field")
        for src in spec.get("inputs", []):
            if src not in props:
                raise InvalidMapping(f"_meta.engine.embeddings.{v}.inputs", f"{src!r} is not a field")
    synonyms = parse_synonyms(eng.get("synonyms", []))
    return Mapping(raw=raw, id_field=eng.get("id_field", "_id"), text=text, keyword=keyword, numeric=numeric,
                   vector=vector, hot=hot, k1=k1, b=b_default, field_b=field_b,
                   embeddings=embeddings, dynamic=str(raw.get("dynamic", "true")).lower(),
                   positions=bool(eng.get("positions", True)), source=bool(eng.get("source", True)),
                   synonyms=synonyms, geo=geo_fields, nested=nested_fields)


def _bm25_params(bm25: dict, text: list) -> tuple[float, float, dict]:
    """``_meta.engine.bm25``: ``k1`` (>= 0, default 1.2) and ``b`` (in [0, 1], default 0.75), where ``b`` is a number
    for every text field or a map of text field -> b, with ``b_default`` for the fields it does not name."""
    if not isinstance(bm25, dict):
        raise InvalidMapping("_meta.engine.bm25", "an object with k1 and b")
    try:
        k1 = float(bm25.get("k1", 1.2))
    except (TypeError, ValueError):
        raise InvalidMapping("_meta.engine.bm25.k1", "a number") from None
    if not k1 >= 0:
        raise InvalidMapping("_meta.engine.bm25.k1", "a number >= 0")

    def one(path, v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise InvalidMapping(path, "a number in [0, 1]") from None
        if not 0.0 <= v <= 1.0:
            raise InvalidMapping(path, "a number in [0, 1]")
        return v

    raw_b = bm25.get("b", 0.75)
    if isinstance(raw_b, dict):
        b_default = one("_meta.engine.bm25.b_default", bm25.get("b_default", 0.75))
        field_b = {}
        for f, v in raw_b.items():
            if f not in text:
                raise InvalidMapping(f"_meta.engine.bm25.b.{f}", "not a text field")
            field_b[f] = one(f"_meta.engine.bm25.b.{f}", v)
        return k1, b_default, field_b
    if "b_default" in bm25:
        raise InvalidMapping("_meta.engine.bm25.b_default", "only with a per-field b map")
    return k1, one("_meta.engine.bm25.b", raw_b), {}


def parse_synonyms(lines) -> dict:
    """Solr-style synonym lines (S7-6), analysed with the index's analyzer: ``"a, b, c"`` makes the three
    equivalent (each expands to all); ``"a, b => c, d"`` replaces a or b by c or d. ``#`` starts a comment. Each
    entry must analyse to one token (multi-word synonyms are not supported). Returns token -> tuple of
    alternatives (sorted; for an equivalence the token itself is among them)."""
    from .analysis import analyze
    if isinstance(lines, str):
        lines = lines.splitlines()
    if not isinstance(lines, list):
        raise InvalidMapping("_meta.engine.synonyms", "a list of Solr-style lines")
    out: dict[str, set] = {}

    def one(word, i):
        toks = analyze(word)
        if len(toks) != 1:
            raise InvalidMapping(f"_meta.engine.synonyms[{i}]",
                                 f"{word.strip()!r} analyses to {len(toks)} tokens; only one-word synonyms")
        return toks[0]
    for i, line in enumerate(lines):
        if not isinstance(line, str):
            raise InvalidMapping(f"_meta.engine.synonyms[{i}]", "a string")
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        def entries(part, side):
            words = [w.strip() for w in part.split(",")]
            if not words or any(not w for w in words):
                raise InvalidMapping(f"_meta.engine.synonyms[{i}]",
                                     f"rule {line!r}: an empty {side} (as OpenSearch, the rule is refused)")
            return [one(w, i) for w in words]
        if "=>" in line:
            lhs, rhs = line.split("=>", 1)
            if "=>" in rhs:
                raise InvalidMapping(f"_meta.engine.synonyms[{i}]", f"rule {line!r}: more than one =>")
            sources, targets = entries(lhs, "left side"), set(entries(rhs, "right side"))
            for t in sources:
                out.setdefault(t, set()).update(targets)
        else:
            group = set(entries(line, "term"))
            for t in group:
                out.setdefault(t, set()).update(group)
    return {t: tuple(sorted(alts)) for t, alts in out.items()}
