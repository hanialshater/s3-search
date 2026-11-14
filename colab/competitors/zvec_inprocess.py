"""zvec in-process at depth 2,000, with no HTTP (S6-18s; the steward's ruling asks for it next to the engine's
in-process figure). Development only.

zvec's ``Collection.query`` takes one query, or several routes of one query fused by a re-ranker; it has no batch of
independent queries. So "batch sizes 1, 16 and 64" are measured as 1, 16 and 64 queries in flight on threads:
zvec releases the GIL during a query and runs its own query threads.

    python colab/competitors/zvec_inprocess.py --collection DIR --queries queries.json [--fields f^w,...]
        [--fusion weighted|rrf] [--k 2000] [--seconds 20] > result.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--queries", required=True, help="a JSON list of query texts")
    ap.add_argument("--fields", default="product_title^1.0")
    ap.add_argument("--fusion", default="weighted", choices=("weighted", "rrf"))
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--query-threads", type=int, default=0)
    a = ap.parse_args()
    import zvec
    zvec.init(query_threads=a.query_threads) if a.query_threads else zvec.init()
    col = zvec.open(a.collection, option=zvec.CollectionOption(read_only=True))
    texts = json.load(open(a.queries))
    fields = [(f.split("^")[0], float(f.split("^")[1]) if "^" in f else 1.0) for f in a.fields.split(",")]

    def one(text):
        routes = [zvec.Query(field_name=f, fts=zvec.Fts(match_string=text)) for f, _ in fields]
        if len(routes) == 1:
            docs = col.query(queries=routes[0], topk=a.k, output_fields=[])
        else:
            rr = zvec.RrfReRanker() if a.fusion == "rrf" else zvec.WeightedReRanker([w for _, w in fields])
            docs = col.query(queries=routes, topk=a.k, output_fields=[], reranker=rr)
        return [(d.id, d.score) for d in docs]

    for t in texts[:50]:                                       # warm-up
        one(t)
    out = {"kind": "zvec-inprocess", "label": "development only", "k": a.k, "fields": a.fields, "fusion": a.fusion,
           "zvec": getattr(zvec, "__version__", None), "runs": {}}
    for inflight in (1, 16, 64):
        lat, n, t_end, i = [], 0, time.time() + a.seconds, 0
        with ThreadPoolExecutor(inflight) as ex:
            t0 = time.time()
            while time.time() < t_end:
                chunk = texts[i % len(texts):i % len(texts) + inflight] or texts[:inflight]
                i += len(chunk)
                ts = time.perf_counter()
                list(ex.map(one, chunk))
                lat.append((time.perf_counter() - ts) * 1000.0)
                n += len(chunk)
            wall = time.time() - t0
        lat.sort()
        out["runs"][str(inflight)] = {"qps": n / wall, "group_ms_p50": lat[len(lat) // 2],
                                      "group_ms_p99": lat[min(len(lat) - 1, int(len(lat) * 0.99))], "queries": n}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
