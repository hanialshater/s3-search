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


def _reported_obj(resp, field="embedding", clause=0):
    k = resp.get("knn")
    v = k.get(field) if isinstance(k, dict) else None
    if isinstance(v, list):
        v = v[clause] if clause < len(v) else None
    return v if isinstance(v, dict) else None


def _filter_mask(ds, flt):
    c = ds.catalogue
    if flt is None:
        return np.ones(len(c), bool)
    (kind, body), = flt.items()
    (field, val), = body.items()
    if kind == "term":
        return (c[field] == val).to_numpy()
    lo, hi = val.get("gte", -np.inf), val.get("lt", np.inf)
    return ((c[field] >= lo) & (c[field] < hi)).to_numpy()


def check_s9_1(index, ds, build=None):
    """S9-1 on any device (INTERFACES section 5 after b0d79f5).

    On a cpu or torch-cpu index, and wherever the response says `ran: "exact"`: the answer equals exact knn (ids in
    order, scores 1e-6) and `knn.embedding` names {requested, ran: "exact"} (a `reason` may accompany it). That is
    prediction 1's CPU half.

    On a cuda index where cuVS ran (`ran` is the method or "<method>+postfilter"): `ran` names the method asked for,
    every hit's score is the exact similarity of its id ((1 + cos) / 2, 1e-5), every hit passes the filter, the ids
    are unique and at most k. Recall is the Colab judge's (referee/gpu_judges.py), not this check's.
    """
    import engine.api as api
    problems = []
    device = getattr(index, "device", "cpu")
    emb = np.stack(ds.catalogue["embedding"].to_numpy()).astype(np.float64)
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    row = {p: i for i, p in enumerate(ds.catalogue["product_id"])}
    rng = np.random.default_rng(91)
    probes = [emb[3], emb[150], rng.normal(size=emb.shape[1])]
    filters = [None, {"term": {"category": "mug"}}, {"range": {"price": {"lt": 60}}}]
    close = lambda a, b, t=1e-6: abs(a - b) <= t * max(1.0, abs(b))  # noqa: E731
    for name in METHODS:
        for params in (None, PARAMS[name]):
            for v in probes:
                q = np.asarray(v, dtype=np.float64)
                q = q / np.linalg.norm(q)
                for flt in filters:
                    mask = _filter_mask(ds, flt)
                    for k in (1, 10, 37):
                        base = {"vector": [float(x) for x in v], "k": k}
                        if flt is not None:
                            base["filter"] = flt
                        spec = {**base, "method": name, **({"method_parameters": params} if params else {})}
                        a = index.search({"query": {"knn": {"embedding": spec}}, "size": k})
                        obj = _reported_obj(a)
                        ran = (obj or {}).get("ran")
                        if obj is None or obj.get("requested") != name:
                            problems.append(f"knn {name}: the response's knn.embedding is {a.get('knn')!r}, "
                                            f"want requested {name!r}")
                            continue
                        if ran == "exact":
                            if device == "cuda" and not obj.get("reason"):
                                problems.append(f"knn {name} on cuda ran exact without a reason")
                            e = index.search({"query": {"knn": {"embedding": base}}, "size": k})
                            if [h["_id"] for h in a["hits"]["hits"]] != [h["_id"] for h in e["hits"]["hits"]] or \
                                    not all(close(x["_score"], y["_score"]) for x, y in zip(a["hits"]["hits"], e["hits"]["hits"])):
                                problems.append(f"knn {name} params={params} k={k} filter={flt}: ran exact but is not "
                                                "the exact-knn answer")
                        elif device in ("cpu", "torch-cpu"):
                            problems.append(f"knn {name} on {device}: ran {ran!r}; without cuVS it must run exact")
                        elif ran in (name, f"{name}+postfilter"):
                            ids = [h["_id"] for h in a["hits"]["hits"]]
                            if len(ids) != len(set(ids)) or len(ids) > k:
                                problems.append(f"knn {name} on cuda: {len(ids)} hits for k={k}, or duplicates")
                            for h in a["hits"]["hits"]:
                                i = row[h["_id"]]
                                if not mask[i]:
                                    problems.append(f"knn {name} on cuda: {h['_id']} breaks the filter {flt}")
                                    break
                                if not close(h["_score"], (1.0 + float(unit[i] @ q)) / 2.0, 1e-5):
                                    problems.append(f"knn {name} on cuda: {h['_id']} score {h['_score']} is not its "
                                                    "exact similarity")
                                    break
                        else:
                            problems.append(f"knn {name}: ran {ran!r}, want {name!r}, {name + '+postfilter'!r} or 'exact'")
    # the list form: two knn clauses on one field, in request order
    two = {"query": {"bool": {"should": [{"knn": {"embedding": {"vector": [float(x) for x in emb[3]], "k": 5, "method": "cagra"}}},
                                         {"knn": {"embedding": {"vector": [float(x) for x in emb[9]], "k": 5, "method": "ivf_pq"}}}]}},
           "size": 10}
    r = index.search(two)
    o0, o1 = _reported_obj(r, clause=0), _reported_obj(r, clause=1)
    if not (o0 and o1 and o0.get("requested") == "cagra" and o1.get("requested") == "ivf_pq"):
        problems.append(f"two knn clauses on one field: knn.embedding is {r.get('knn')!r}, want a list in request order")
    elif device in ("cpu", "torch-cpu") and (o0.get("ran"), o1.get("ran")) != ("exact", "exact"):
        problems.append(f"two knn clauses on {device}: ran {o0.get('ran')!r}, {o1.get('ran')!r}; want exact")
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
