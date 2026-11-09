"""S5-1, side by side: two engines, two versions, or an engine and a recorded OpenSearch run, on the same queries.

Reports per query the result overlap, the rank changes and the nDCG@10 delta, and the means. **Fit-split judgments only**:
the judgments passed in are the only queries scored, and the CLI builds them from ``referee.loader.load_fit``; the test
judgments are never read here (the referee's rule). Every metric is ``referee.metrics`` (imported, not re-implemented).

A recorded run is a JSONL: one object per line, ``{"query_id": ..., "ids": [ranked product ids]}`` (``ranked_ids`` and
``ranked`` are accepted as the key too). ``save_run`` writes the same format, so a run recorded in Colab against
OpenSearch compares against the engine's.

    python -m tools.compare --a run:opensearch.jsonl --b http://localhost:8080/esci-sample [--limit 200] [--out out.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from referee import metrics  # the referee's own metrics: imported, never copied

from .common import SearchFn, hit_ids, text_request

ID_KEYS = ("ids", "ranked_ids", "ranked")


def load_run(path) -> dict[str, list[str]]:
    """Read a run file. Ids are normalised to str. A repeated query_id is an error (it would silently drop a run)."""
    run: dict[str, list[str]] = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = next((k for k in ID_KEYS if k in row), None)
        if "query_id" not in row or key is None:
            raise ValueError(f"{path}:{n}: need query_id and one of {ID_KEYS}")
        qid = str(row["query_id"])
        if qid in run:
            raise ValueError(f"{path}:{n}: query_id {qid} appears twice")
        run[qid] = [str(i) for i in row[key]]
    return run


def save_run(run: dict, path) -> None:
    with open(path, "w") as f:
        for qid, ids in run.items():
            f.write(json.dumps({"query_id": str(qid), "ids": [str(i) for i in ids]}) + "\n")


def record_run(search: SearchFn, queries: dict, fields=None, size: int = 100) -> dict[str, list[str]]:
    """Run ``{query_id: text}`` through a search function; the ranked ids per query. Same request as the demo's."""
    return {str(q): hit_ids(search(text_request(t, fields, size))) for q, t in queries.items()}


def fit_judgments(fit_df) -> dict[str, dict[str, str]]:
    """``referee.loader.load_fit`` rows -> {query_id: {product_id: label}}, ids as str."""
    out: dict[str, dict[str, str]] = {}
    for q, p, l in zip(fit_df["query_id"], fit_df["product_id"], fit_df["esci_label"]):
        out.setdefault(str(q), {})[str(p)] = l
    return out


def _ranks(ids, k):
    return {p: i + 1 for i, p in enumerate(ids[:k])}


def compare_runs(run_a: dict, run_b: dict, judgments: dict, queries: dict | None = None, k: int = 10,
                 label_a: str = "A", label_b: str = "B") -> dict:
    """Compare two runs on the queries that ran in either and have fit judgments.

    Queries outside ``judgments`` are not scored (counted in ``skipped_not_judged``): the comparison is fit-split only.
    A judged query one run lacks is listed in ``missing_a`` / ``missing_b`` and scored as empty, as the referee does.
    """
    queries = {str(q): t for q, t in (queries or {}).items()}
    judged = {str(q): {str(p): l for p, l in j.items()} for q, j in judgments.items()}
    qids = [q for q in dict.fromkeys([*run_a, *run_b]) if q in judged]
    skipped = len(set(run_a) | set(run_b)) - len(qids)
    nd = f"ndcg@{k}"
    per = []
    for q in qids:
        a, b = run_a.get(q, []), run_b.get(q, [])
        na = metrics.score_query(a, judged[q], k_ndcg=k)[nd]
        nb = metrics.score_query(b, judged[q], k_ndcg=k)[nd]
        ta, tb = _ranks(a, k), _ranks(b, k)
        ra, rb = _ranks(a, len(a)), _ranks(b, len(b))
        changes = [{"id": p, "rank_a": ra.get(p), "rank_b": rb.get(p),
                    "moved": (rb[p] - ra[p]) if p in ra and p in rb else None}
                   for p in [*ta, *[p for p in tb if p not in ta]]]
        per.append({"query_id": q, "query": queries.get(q), "overlap": len(set(ta) & set(tb)) / k,
                    "rank_changes": changes, "ndcg_a": na, "ndcg_b": nb,
                    "ndcg_delta": (nb - na) if na is not None and nb is not None else None})
    sub = {q: judged[q] for q in qids}
    sa = metrics.score_run({q: run_a.get(q, []) for q in qids}, sub, k_ndcg=k)["mean"]
    sb = metrics.score_run({q: run_b.get(q, []) for q in qids}, sub, k_ndcg=k)["mean"]
    delta = {m: (sb[m] - sa[m]) if sa[m] is not None and sb[m] is not None else None
             for m in sa if not m.startswith("n_")}
    deltas = [p["ndcg_delta"] for p in per if p["ndcg_delta"] is not None]
    return {"labels": [label_a, label_b], "k": k, "split": "fit", "n_queries": len(qids),
            "skipped_not_judged": skipped,
            "missing_a": [q for q in qids if q not in run_a], "missing_b": [q for q in qids if q not in run_b],
            "mean": {"a": sa, "b": sb, "delta": delta},
            "mean_overlap": sum(p["overlap"] for p in per) / len(per) if per else None,
            "n_better": sum(d > 1e-12 for d in deltas), "n_worse": sum(d < -1e-12 for d in deltas),
            "n_same": sum(abs(d) <= 1e-12 for d in deltas), "per_query": per}


def render_markdown(cmp: dict, top: int = 15) -> str:
    a, b = cmp["labels"]
    m, k = cmp["mean"], cmp["k"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    lines = [f"# {a} vs {b}", "",
             f"Fit-split judgments only; {cmp['n_queries']} queries ({cmp['skipped_not_judged']} not judged, skipped). "
             "Metrics are the referee's (`referee.metrics`). Development numbers, not a score of record.", "",
             f"| metric | {a} | {b} | delta |", "|---|---|---|---|"]
    for key in m["delta"]:
        lines.append(f"| {key} | {f(m['a'][key])} | {f(m['b'][key])} | {f(m['delta'][key])} |")
    lines += ["", f"Mean top-{k} overlap {f(cmp['mean_overlap'])}; nDCG better on {cmp['n_better']}, worse on "
              f"{cmp['n_worse']}, same on {cmp['n_same']} queries.", ""]
    if cmp["missing_a"] or cmp["missing_b"]:
        lines += [f"Missing (scored as empty): {a}: {len(cmp['missing_a'])}, {b}: {len(cmp['missing_b'])}.", ""]
    scored = [p for p in cmp["per_query"] if p["ndcg_delta"] is not None]
    for title, rows in (("Biggest drops", sorted(scored, key=lambda p: p["ndcg_delta"])[:top]),
                        ("Biggest gains", sorted(scored, key=lambda p: -p["ndcg_delta"])[:top])):
        lines += [f"## {title} (nDCG@{k})", "", f"| query | overlap | {a} | {b} | delta |", "|---|---|---|---|---|"]
        for p in rows:
            lines.append(f"| {p['query'] or p['query_id']} | {p['overlap']:.2f} | {f(p['ndcg_a'])} | {f(p['ndcg_b'])} | "
                         f"{p['ndcg_delta']:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def _source(spec: str, queries: dict, fields, size: int):
    """``run:FILE.jsonl`` (recorded) or ``http://host:port/INDEX`` (the serving API, §7)."""
    if spec.startswith("run:"):
        return load_run(spec[4:])
    if spec.startswith(("http://", "https://")):
        from .common import http_search
        base, _, index = spec.rstrip("/").rpartition("/")
        return record_run(http_search(base, index), queries, fields, size)
    raise SystemExit(f"unknown source {spec!r}: use run:FILE.jsonl or http://host:port/INDEX")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.compare", description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--data-dir", default=None, help="ESCI directory (default: referee.loader's)")
    ap.add_argument("--limit", type=int, default=200, help="fit queries to run against a live engine")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--fields", default=None, help="comma-separated multi_match fields")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None, help="also write the full comparison here")
    args = ap.parse_args(argv)

    from referee import loader  # the only way to read ESCI
    fit = loader.load_fit(**({"data_dir": args.data_dir} if args.data_dir else {}))
    qtext = dict(zip(fit["query_id"].astype(str), fit["query"]))
    queries = {q: qtext[q] for q in list(dict.fromkeys(qtext))[: args.limit]}
    fields = args.fields.split(",") if args.fields else None
    ra, rb = (_source(s, queries, fields, args.size) for s in (args.a, args.b))
    cmp = compare_runs(ra, rb, fit_judgments(fit), qtext, label_a=args.a, label_b=args.b)
    md = render_markdown(cmp)
    if args.out:
        Path(args.out).write_text(md)
    else:
        print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(cmp, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
