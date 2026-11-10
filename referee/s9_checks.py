"""S9's CPU contract checks (owner: evaluation-steward, P6), run by `s7_checks.check_landed_correct` once the item is
marked landed in `referee/scope.py`.

S9-1, on CPU (no cuVS), as INTERFACES section 5 now says (6d1f7f4): `knn` and `neural` take `"method": "exact" |
"cagra" | "ivf_pq"` and `method_parameters` (cagra `itopk_size`, `search_width`; ivf_pq `n_probes`; positive
integers, anything else `InvalidRequest`). Without cuVS the method runs as exact, so every answer must equal exact
knn's (the same ids in order, scores within 1e-6), and the response names it: `knn.<field> = {"requested": method,
"ran": "exact"}`, a list in request order when several clauses use the field.
"""
from __future__ import annotations

import numpy as np

METHODS = ("cagra", "ivf_pq")
PARAMS = {"cagra": {"itopk_size": 64, "search_width": 2}, "ivf_pq": {"n_probes": 20}}


def method_reported(resp, field="embedding", clause=0):
    """(ran, requested) for `field`, as INTERFACES section 5 now says (architect, 2026-11-10, ratifying the steward's
    proposal): `knn.<field> = {"requested", "ran"}`, or a list of those objects in request order when several knn
    clauses use the field; `clause` picks one. None when absent."""
    k = resp.get("knn")
    v = k.get(field) if isinstance(k, dict) else None
    if isinstance(v, list):
        v = v[clause] if clause < len(v) else None
    if isinstance(v, dict):
        return v.get("ran"), v.get("requested")
    return None


def check_s9_1(index, ds, build=None):
    import engine.api as api
    problems = []
    emb = np.stack(ds.catalogue["embedding"].to_numpy())
    rng = np.random.default_rng(91)
    probes = [emb[3], emb[150], rng.normal(size=emb.shape[1])]
    filters = [None, {"term": {"category": "mug"}}, {"range": {"price": {"lt": 60}}}]
    close = lambda a, b: abs(a - b) <= 1e-6 * max(1.0, abs(b))  # noqa: E731
    for name in METHODS:
        for params in (None, PARAMS[name]):
            for v in probes:
                for flt in filters:
                    for k in (1, 10, 37):
                        base = {"vector": [float(x) for x in v], "k": k}
                        if flt is not None:
                            base["filter"] = flt
                        spec = {**base, "method": name, **({"method_parameters": params} if params else {})}
                        a = index.search({"query": {"knn": {"embedding": spec}}, "size": k})
                        e = index.search({"query": {"knn": {"embedding": base}}, "size": k})
                        if [h["_id"] for h in a["hits"]["hits"]] != [h["_id"] for h in e["hits"]["hits"]] or \
                                not all(close(x["_score"], y["_score"]) for x, y in zip(a["hits"]["hits"], e["hits"]["hits"])):
                            problems.append(f"knn {name} params={params} k={k} filter={flt}: not the exact-knn answer")
                        rep = method_reported(a)
                        if rep != ("exact", name):
                            problems.append(f"knn {name}: the response's knn.embedding is {a.get('knn')!r}, "
                                            f"want {{requested: {name!r}, ran: 'exact'}}")
    # the list form: two knn clauses on one field, in request order
    two = {"query": {"bool": {"should": [{"knn": {"embedding": {"vector": [float(x) for x in emb[3]], "k": 5, "method": "cagra"}}},
                                         {"knn": {"embedding": {"vector": [float(x) for x in emb[9]], "k": 5, "method": "ivf_pq"}}}]}},
           "size": 10}
    r = index.search(two)
    if method_reported(r, clause=0) != ("exact", "cagra") or method_reported(r, clause=1) != ("exact", "ivf_pq"):
        problems.append(f"two knn clauses on one field: knn.embedding is {r.get('knn')!r}, want a list in request order")
    # neural takes the method too (toy has given vectors and no encoder, so a neural request on it is refused alike)
    # bad method_parameters are refused by name
    for bad in ({"method": "cagra", "method_parameters": {"itopk_size": 0}},
                {"method": "ivf_pq", "method_parameters": {"n_probes": "many"}},
                {"method": "cagra", "method_parameters": {"n_probes": 3}},
                {"method": "hnsw"}):
        try:
            index.search({"query": {"knn": {"embedding": {"vector": [float(x) for x in emb[3]], "k": 5, **bad}}}})
            problems.append(f"knn {bad}: answered, want InvalidRequest or UnsupportedQuery")
        except api.UnsupportedQuery:
            pass
    return problems[:12]


from referee.s9_geo_nested_checks import check_geo, check_nested  # noqa: E402

CHECKS = {"S9-1": check_s9_1, "S9-4": check_geo, "S9-5": check_nested}
