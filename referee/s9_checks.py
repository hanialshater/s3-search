"""S9's CPU contract checks (owner: evaluation-steward, P6), run by `s7_checks.check_landed_correct` once the item is
marked landed in `referee/scope.py`.

S9-1, on CPU (no cuVS): a knn field whose mapping asks for an approximate method (`"method": {"name": "cagra"}` or
`"ivf_pq"`, OpenSearch's place for it) answers every knn request, filtered or not, exactly as exact knn does (the
same ids in the same order, scores within 1e-6), and the response names the method that ran.

**The response field is the steward's proposal** until INTERFACES section 5 names one (S9-1 is the architect's):
`response["knn"][field] == {"requested": "<method>", "ran": "exact", ...}` on CPU. When S9-1 lands, the check is
aligned to whatever section 5 says before the item is marked landed.
"""
from __future__ import annotations

import copy

import numpy as np

METHODS = ("cagra", "ivf_pq")


def _with_method(mapping, name):
    m = copy.deepcopy(mapping)
    m["properties"]["embedding"]["method"] = {"name": name, "space_type": "cosinesimil"}
    return m


def method_reported(resp, field="embedding"):
    """The method the response says ran for `field` (the proposed place), or None."""
    k = resp.get("knn")
    if isinstance(k, dict) and isinstance(k.get(field), dict):
        return k[field].get("ran"), k[field].get("requested")
    return None


def check_s9_1(index, ds, build=None):
    if build is None:
        import engine.api
        build = engine.api.build_index
    problems = []
    emb = np.stack(ds.catalogue["embedding"].to_numpy())
    rng = np.random.default_rng(91)
    probes = [emb[3], emb[150], rng.normal(size=emb.shape[1])]
    filters = [None, {"term": {"category": "mug"}}, {"range": {"price": {"lt": 60}}}]
    for name in METHODS:
        approx = build(ds.catalogue, _with_method(ds.mapping, name))
        for v in probes:
            for flt in filters:
                for k in (1, 10, 37):
                    spec = {"vector": [float(x) for x in v], "k": k}
                    if flt is not None:
                        spec["filter"] = flt
                    req = {"query": {"knn": {"embedding": spec}}, "size": k}
                    a, e = approx.search(req), index.search(req)
                    ai, ei = [h["_id"] for h in a["hits"]["hits"]], [h["_id"] for h in e["hits"]["hits"]]
                    if ai != ei or any(abs(x["_score"] - y["_score"]) > 1e-6 * max(1.0, abs(y["_score"]))
                                       for x, y in zip(a["hits"]["hits"], e["hits"]["hits"])):
                        problems.append(f"{name}: k={k} filter={flt}: not the exact-knn answer")
                    rep = method_reported(a)
                    if rep is None or rep[0] != "exact" or rep[1] != name:
                        problems.append(f"{name}: the response does not name the method that ran "
                                        f"(want knn.embedding = {{requested: {name!r}, ran: 'exact'}}, got {a.get('knn')!r})")
                        break
    return problems[:12]


CHECKS = {"S9-1": check_s9_1}
