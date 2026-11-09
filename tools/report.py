"""S5-3, the test-drive report: a self-contained static HTML page from a referee ``results.json``.

    python -m tools.report results.json --out report.html [--milestones PATH/company/milestones.json --milestone S5]
                           [--recompute-run run.jsonl]

**The report computes no metric.** Every number is copied from ``results.json`` and tagged in the HTML
(``data-key`` = its dotted path, ``data-value`` = its exact value). ``check_report`` re-reads the HTML and fails when
any tagged number differs from the file, or any numeric field of the file is missing from the page: that is the
correctness check the floor's land gate runs (a mismatch fails the land). ``--recompute-run`` additionally recomputes
the quality metrics from a recorded run with ``referee.metrics`` (fit-split judgments, so only for a results.json
written on the fit split) and shows both, failing loudly on any difference.

Shape it reads (a proposal to the steward, the referee's ``full.py`` is not written yet: any extra numeric field is
still shown, under "Other"):

    {"meta": {"device": "cuda|cpu", "gpu": "...", "dataset": "...", "split": "...", "commit": "...", "date": "..."},
     "quality": {"ndcg@10": 0.3, "exact_recall@100": 0.9, "exact_recall@1000": 0.95, "n_queries": 8956},
     "latency": {"page1_p50_ms": 20.6, "page1_p99_ms": 33.8, "stages": {"retrieve": {"p50_ms": 5.0}}},
     "throughput": {"qps_at_p99_100ms": 275}, "cost": {"per_million_queries_usd": 1.2, "...": "assumptions"},
     "slices": {"query_length_1": {"ndcg@10": 0.2, "n": 100}, "negation": {...}, "filtered": {...}},
     "freshness": {...}, "correctness": {...},
     "worst_queries": [{"query_id": 7, "query": "...", "ndcg@10": 0.0, "explain": {value, description, details}}]}

A report is labelled "development only" unless ``meta.device`` names a GPU (cuda): CPU numbers are never the score of
record (the founder's Colab ``results.json`` is).
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from referee import metrics

DEV_BANNER = ("Development only: these are CPU numbers. They show the engine working, not how fast or how good it is. "
              "The score of record is the GPU referee's results.json from the founder's Colab.")
REC_BANNER = ("GPU run. It is the score of record only if this is the founder's Colab results.json for the commit "
              "shown.")


def flatten(obj, path=()) -> dict:
    """{path tuple: number} for every numeric (non-bool) leaf, lists indexed by position."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "explain":  # a trace is shown verbatim as text (see worst()), not as tagged numbers
                continue
            out.update(flatten(v, path + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, path + (str(i),)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[path] = obj
    return out


def key_of(path) -> str:
    return ".".join(path)


def is_development(results: dict) -> bool:
    meta = results.get("meta") or {}
    dev = str(meta.get("device", "")).lower()
    return not ("cuda" in dev or "gpu" in dev)


def fmt(v) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}"


def load_predictions(path, milestone: str) -> list[dict]:
    d = json.loads(Path(path).read_text())
    for m in d if isinstance(d, list) else d.get("milestones", d):
        if isinstance(m, dict) and m.get("id") == milestone:
            return m.get("predictions", [])
    raise KeyError(f"milestone {milestone} not in {path}")


def recompute(results: dict, runs: dict, judgments: dict) -> list[dict]:
    """Quality metrics recomputed from a run with the referee's metrics, against ``results['quality']``. Exact equality
    up to float noise (1e-9): a report that disagrees with the referee on the same run is wrong."""
    mean = metrics.score_run({str(q): v for q, v in runs.items()},
                             {str(q): {str(p): l for p, l in j.items()} for q, j in judgments.items()})["mean"]
    rows = []
    for k, v in mean.items():
        if k.startswith("n_"):
            continue
        rep = (results.get("quality") or {}).get(k)
        rows.append({"metric": k, "referee_results": rep, "recomputed": v,
                     "ok": rep is not None and v is not None and abs(rep - v) <= 1e-9})
    return rows


# -- rendering ------------------------------------------------------------------------------------------------------
class _Page:
    def __init__(self, results: dict):
        self.r, self.flat, self.seen = results, flatten(results), set()
        self.out = []

    def num(self, path) -> str:
        self.seen.add(path)
        v = self.flat[path]
        return f'<td class="n" data-key="{html.escape(key_of(path))}" data-value="{repr(v)}">{fmt(v)}</td>'

    def paths(self, prefix, contains=None):
        return [p for p in self.flat if p not in self.seen and p[:len(prefix)] == prefix
                and (contains is None or contains in key_of(p).lower())]

    def table(self, title, paths, strip=0, note=""):
        if not paths:
            return
        rows = "".join(f"<tr><th>{html.escape('.'.join(p[strip:]))}</th>{self.num(p)}</tr>" for p in paths)
        self.out.append(f"<section><h2>{html.escape(title)}</h2>{f'<p class=note>{note}</p>' if note else ''}"
                        f"<table>{rows}</table></section>")

    def slices(self):
        ps = self.paths(("slices",))
        if not ps:
            return
        names = list(dict.fromkeys(p[1] for p in ps))
        cols = list(dict.fromkeys(p[2] if len(p) > 2 else "value" for p in ps))
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = ""
        for n in names:
            cells = ""
            for c in cols:
                p = ("slices", n, c) if c != "value" else ("slices", n)
                cells += self.num(p) if p in self.flat and p not in self.seen else "<td></td>"
            body += f"<tr><th>{html.escape(n)}</th>{cells}</tr>"
        self.out.append(f"<section><h2>Quality by slice</h2><table><tr><th></th>{head}</tr>{body}</table></section>")

    def worst(self):
        items = self.r.get("worst_queries") or []
        if not items:
            return
        parts = []
        for i, w in enumerate(items):
            nums = "".join(f"<span>{html.escape(k)} <b data-key=\"worst_queries.{i}.{html.escape(k)}\" "
                           f"data-value=\"{repr(v)}\">{fmt(v)}</b></span> "
                           for k, v in w.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                           and (self.seen.add(("worst_queries", str(i), k)) or True))
            ex = w.get("explain")
            ex_html = ("<details><summary>why it ranked there</summary><pre>"
                       f"{html.escape(_tree(ex) if isinstance(ex, dict) else json.dumps(ex, indent=1))}</pre></details>"
                       if ex else "<p class=note>no explanation in results.json</p>")
            parts.append(f"<li><div class=q>{html.escape(str(w.get('query', w.get('query_id', ''))))}</div>"
                         f"<div class=nums>{nums}</div>{ex_html}</li>")
        self.out.append(f"<section><h2>Worst queries</h2><ol class=worst>{''.join(parts)}</ol></section>")


def _tree(node, depth=0) -> str:
    line = f"{'  ' * depth}{node.get('value')}  {node.get('description', '')}"
    return "\n".join([line] + [_tree(k, depth + 1) for k in node.get("details") or []])


CSS = """
:root{--bg:#fff;--fg:#1c1b19;--mut:#6b675e;--line:#dedbd2;--card:#f7f5f0;--warn-bg:#fdecea;--warn:#a1271b;--ok:#155d3b}
@media (prefers-color-scheme:dark){:root{--bg:#171613;--fg:#ece9e1;--mut:#a29d90;--line:#37342d;--card:#201f1b;
--warn-bg:#3a1f1c;--warn:#ff9d90;--ok:#7fd1a4}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.4em 0}h2{font-size:1.1rem;margin:1.4em 0 .4em}
.banner{padding:10px 14px;border-radius:6px;background:var(--warn-bg);color:var(--warn);font-weight:600}
.banner.rec{background:var(--card);color:var(--ok)}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:24px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
td.n{text-align:right;font-variant-numeric:tabular-nums}thead th{color:var(--mut)}
.note,.mut{color:var(--mut);font-size:.9em}
aside .pred{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:10px 12px;margin:0 0 10px}
.wi{color:var(--mut);font-size:.9em}
ol.worst{padding-left:1.2em}.q{font-weight:600}.nums span{margin-right:12px;color:var(--mut)}
pre{overflow:auto;background:var(--card);padding:8px;font-size:.85em}
"""


def render(results: dict, predictions=None, recomputed=None, title="Test-drive report") -> str:
    p = _Page(results)
    meta = results.get("meta") or {}
    dev = is_development(results)
    banner = f'<div class="banner{"" if dev else " rec"}">{DEV_BANNER if dev else REC_BANNER}</div>'
    meta_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in meta.items() if not isinstance(v, (dict, list)) and not isinstance(v, (int, float)))
    p.table("Recall", p.paths((), "recall"))
    p.table("Quality" + (" (development only)" if dev else ""), p.paths(("quality",)), 1)
    p.slices()
    p.table("Latency" + (" (development only: CPU)" if dev else ""), p.paths(("latency",)), 1)
    p.table("Throughput and cost" + (" (development only)" if dev else ""),
            p.paths(("throughput",)) + p.paths(("cost",)) + p.paths((), "qps") + p.paths((), "cost"))
    p.table("Freshness", p.paths(("freshness",)), 1)
    p.table("Correctness", p.paths(("correctness",)), 1)
    p.table("Run", p.paths(("meta",)), 1)
    p.worst()
    p.table("Other numbers in results.json", p.paths(()))
    rec = ""
    if recomputed is not None:
        rows = "".join(f"<tr><th>{html.escape(r['metric'])}</th><td class=n>{r['referee_results']}</td>"
                       f"<td class=n>{r['recomputed']}</td><td>{'same' if r['ok'] else 'DIFFERENT'}</td></tr>"
                       for r in recomputed)
        rec = ("<section><h2>Recomputed from the run</h2><p class=note>With <code>referee.metrics</code> on the "
               "recorded run and fit-split judgments.</p><table><tr><th></th><th>results.json</th><th>recomputed</th>"
               f"<th></th></tr>{rows}</table></section>")
    side = ""
    if predictions:
        cards = "".join(f"<div class=pred><div>{html.escape(x['text'])}</div>"
                        f"<div class=wi>Wrong if: {html.escape(x.get('wrongIf', ''))}</div>"
                        f"<div class=wi>By: {html.escape(x.get('by', ''))}</div></div>" for x in predictions)
        side = ("<aside><h2>Predictions</h2><p class=note>Written before the run. Judged by a person against the "
                f"numbers beside them, wrong ones included.</p>{cards}</aside>")
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content=\"width=device-width,initial-scale=1\"><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>{banner}"
            f"<table class=mut>{meta_rows}</table><div class=grid><div>{''.join(p.out)}{rec}</div>{side}</div>"
            "</main></body></html>")


def check_report(results: dict, page: str) -> list[str]:
    """Problems between the page and results.json (empty list = the page shows the referee's numbers exactly)."""
    flat = {key_of(k): v for k, v in flatten(results).items()}
    shown = {}
    for m in re.finditer(r'data-key="([^"]*)"\s+data-value="([^"]*)"', page):
        shown[html.unescape(m.group(1))] = m.group(2)
    problems = [f"{k}: page {v} != results {flat.get(k)!r}" for k, v in shown.items()
                if k not in flat or repr(flat[k]) != v]
    problems += [f"{k}: in results.json but not on the page" for k in flat if k not in shown]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.report", description=__doc__.split("\n\n")[0])
    ap.add_argument("results")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--milestones")
    ap.add_argument("--milestone", default="S5")
    ap.add_argument("--recompute-run", help="a run JSONL; recompute quality on the fit split (dev results.json only)")
    ap.add_argument("--data-dir")
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text())
    preds = load_predictions(args.milestones, args.milestone) if args.milestones else None
    rec = None
    if args.recompute_run:
        from referee import loader
        from .compare import fit_judgments, load_run
        fit = loader.load_fit(**({"data_dir": args.data_dir} if args.data_dir else {}))
        rec = recompute(results, load_run(args.recompute_run), fit_judgments(fit))
    page = render(results, preds, rec)
    problems = check_report(results, page) + [f"recompute {r['metric']}: {r['referee_results']} != {r['recomputed']}"
                                              for r in rec or [] if not r["ok"]]
    Path(args.out).write_text(page)
    for pr in problems:
        print("REPORT CHECK FAILED:", pr)
    print(f"wrote {args.out}" + ("" if problems else "; every number matches results.json"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
