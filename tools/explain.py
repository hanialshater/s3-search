"""S5-2, debug (explain): why one product ranked where it did, from the engine's own explain trace (§5 ``explain``).

The trace is Lucene-shaped: ``{value, description, details}`` per hit, with per-term, per-field and (under ``rescore``)
per-feature contributions. This tool adds nothing to it: it finds the product's hit (widening the page until the
product is in it), then **checks every number against its children** and the root against the hit's ``_score``:

- ``sum of ...`` nodes: value = sum of children; ``product of ...`` or ``... * ...`` nodes: product; ``max of``: max;
  ``result of:`` wrappers with one child: equal to it.
- Tolerance 1e-4, absolute or relative to the value, whichever is larger (scores are float32).
- A node with children whose combiner the tool does not know is **unchecked**, and is reported as such, never as passed.
  ``ok`` means nothing failed; ``complete`` means nothing was left unchecked.

    python -m tools.explain --api http://localhost:8080 --index esci-sample --query "red shoes" --product B0001
"""
from __future__ import annotations

import argparse
import json
import math
import random

from .common import SearchFn, text_request

TOL = 1e-4
WINDOWS = (10, 100, 1000, 10000)


class ExplainUnavailable(RuntimeError):
    """The response had hits but no ``explain`` list: the engine or serving does not implement it (yet)."""


def _close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def _text_combiner(desc: str, n_children: int):
    """The fallback for nodes without an ``op``: Lucene-style wording only. Anything else is unknown (unchecked)."""
    d = desc.lower().strip()
    if d.startswith("sum of"):
        return "sum", lambda v: sum(v)
    if d.startswith("product of") or (d.startswith("score(") and " * " in d):
        return "product", lambda v: math.prod(v)
    if d.startswith("max of"):
        return "max", lambda v: max(v)
    if d.startswith("result of") and n_children == 1:
        return "wrapper", lambda v: v[0]
    return None, None


def _combiner(node: dict, n_children: int):
    """(kind, fn) from the node's machine-readable ``op`` when it has one (the engine's explain nodes do), else the
    text rules. ops: sum, product, max, value (a leaf: nothing to check), ranker (base + contributions = sum)."""
    op = node.get("op")
    if op in ("sum", "ranker"):
        return op, lambda v: sum(v)
    if op == "product":
        return op, lambda v: math.prod(v)
    if op == "max":
        return op, lambda v: max(v)
    if op == "value":
        return "value", None
    if op is not None:
        return None, None  # an op this tool does not know: unchecked, not guessed from the text
    return _text_combiner(node.get("description", ""), n_children)


def check_tree(node: dict, path: str = "root") -> list[dict]:
    """Every node with children, in tree order: ``{path, description, value, expected, kind, ok}``; ``ok`` is None when
    the combiner is unknown (unchecked). An ``op: value`` node's children are explanatory and not summed."""
    out = []
    kids = node.get("details") or []
    if kids:
        kind, fn = _combiner(node, len(kids))
        if kind != "value":
            if fn is None:
                out.append({"path": path, "description": node["description"], "value": node["value"], "expected": None,
                            "kind": None, "ok": None})
            else:
                exp = float(fn([float(k["value"]) for k in kids]))
                out.append({"path": path, "description": node["description"], "value": node["value"], "expected": exp,
                            "kind": kind, "ok": _close(float(node["value"]), exp)})
        for i, k in enumerate(kids):
            out += check_tree(k, f"{path}/{i}")
    if isinstance(node.get("source"), dict):  # the tree a value came from: never summed into this node, but checked itself
        out += check_tree(node["source"], f"{path}/source")
    return out


def leaves(node: dict, path: str = "root") -> list[dict]:
    kids = node.get("details") or []
    if not kids or node.get("op") == "value":
        return [{"path": path, "description": node["description"], "value": node["value"]}]
    return [x for i, k in enumerate(kids) for x in leaves(k, f"{path}/{i}")]


def _filters(request: dict) -> list:
    q = request.get("query") or {}
    b = q.get("bool") if isinstance(q, dict) else None
    if not b:
        return []
    f = b.get("filter", [])
    return f if isinstance(f, list) else [f]


def explain_product(search: SearchFn, request: dict, product_id: str, windows=WINDOWS) -> dict:
    """Explain one product for one request. The request's own ``from``/``size``/``search_after``/``sort`` are replaced by
    the widening window (a product's explanation does not depend on the page it is on)."""
    product_id = str(product_id)
    base = {k: v for k, v in request.items() if k not in ("from", "size", "search_after", "aggs", "aggregations")}
    base["explain"] = True
    total = None
    for w in windows:
        resp = search({**base, "from": 0, "size": w})
        hits = resp["hits"]["hits"]
        total = resp["hits"]["total"]
        if hits and "explain" not in resp:
            raise ExplainUnavailable("the response has hits but no `explain` (INTERFACES §5): not implemented yet")
        for rank, h in enumerate(hits, 1):
            if str(h["_id"]) == product_id:
                tree = next(e for e in resp["explain"] if str(e["_id"]) == product_id)
                checks = check_tree(tree)
                failed = [c for c in checks if c["ok"] is False]
                unchecked = [c for c in checks if c["ok"] is None]
                root_ok = _close(float(tree["value"]), float(h["_score"]))
                return {"found": True, "product_id": product_id, "rank": rank, "total": total,
                        "score": h["_score"], "explain_value": tree["value"], "root_matches_score": root_ok,
                        "filters_passed": _filters(request), "stage_ms": resp.get("took_ms"),
                        "tree": tree, "leaves": leaves(tree), "checks": checks,
                        "n_checked": sum(c["ok"] is not None for c in checks), "n_failed": len(failed),
                        "n_unchecked": len(unchecked), "ok": root_ok and not failed, "complete": not unchecked}
        if len(hits) >= total:
            break
    return {"found": False, "product_id": product_id, "total": total,
            "reason": "not in the match set (the query does not match it, or a filter removed it)"
            if total is not None and len(hits) >= total else "not within the top %d" % windows[-1]}


def render_text(e: dict) -> str:
    if not e["found"]:
        return f"{e['product_id']}: not found: {e['reason']} (total {e['total']})"
    lines = [f"{e['product_id']}: rank {e['rank']} of {e['total']}, score {e['score']}",
             f"explain root {e['explain_value']} vs hit score {e['score']}: "
             f"{'match' if e['root_matches_score'] else 'MISMATCH'}"]
    if e["filters_passed"]:
        lines.append("filters passed: " + json.dumps(e["filters_passed"]))
    root = float(e["explain_value"]) or 1.0

    def walk(n, depth):
        pct = f"  ({100 * float(n['value']) / root:.1f}%)" if depth else ""
        lines.append(f"{'  ' * depth}{n['value']:>10.4f}  {n['description']}{pct}")
        for k in n.get("details") or []:
            walk(k, depth + 1)

    walk(e["tree"], 0)
    lines.append(f"checked {e['n_checked']} sums: {e['n_failed']} failed, {e['n_unchecked']} unchecked"
                 f" -> ok={e['ok']} complete={e['complete']}")
    return "\n".join(lines)


def sample_check(search: SearchFn, queries: list[str], n: int = 100, seed: int = 0, fields=None, top: int = 100) -> dict:
    """The S5 prediction's check: explain ``n`` sampled (query, product) pairs from top hits and count the failures."""
    rng = random.Random(seed)
    pairs = []
    for q in queries:
        hits = search(text_request(q, fields, size=top))["hits"]["hits"]
        pairs += [(q, str(h["_id"])) for h in hits]
    rng.shuffle(pairs)
    results = [explain_product(search, text_request(q, fields), pid) for q, pid in pairs[:n]]
    bad = [r for r in results if not (r["found"] and r["ok"])]
    return {"n": len(results), "n_ok": len(results) - len(bad), "n_complete": sum(r.get("complete", False) for r in results),
            "failures": [{"product_id": r["product_id"], "found": r["found"], "n_failed": r.get("n_failed")} for r in bad]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.explain", description=__doc__.split("\n\n")[0])
    ap.add_argument("--api", required=True, help="serving base URL, e.g. http://localhost:8080")
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", help="free text (multi_match), or use --request")
    ap.add_argument("--request", help="a §5 request as a JSON file (filters, rescore, ...)")
    ap.add_argument("--product", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    from .common import http_search
    req = json.load(open(args.request)) if args.request else text_request(args.query or "")
    e = explain_product(http_search(args.api, args.index), req, args.product)
    print(json.dumps(e, indent=1) if args.json else render_text(e))
    return 0 if e["found"] and e["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
