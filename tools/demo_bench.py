"""Measure the demo page against a real serving replica over the whole of ESCI-sample (CPU, development only).

    python -m tools.demo_bench [--queries 100] [--out results/demo-bench.json] [--cores 0,1]

Starts ``python -m serving`` (tools.stack) and ``python -m tools.demo`` pinned to ``--cores``, then drives the page in
headless Chromium (``tools/demo_bench.js``, Playwright for node) with fit-split queries: autocomplete keystrokes, a search
with highlights, two multi-select facet clicks and a clear per query. Reports p50/p95/p99 on the page's own clock and as
Playwright sees it, and every error. The evaluation steward judges the S7 prediction; this is the maker's own check.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.demo_bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--out", default="results/demo-bench.json")
    ap.add_argument("--cores", default="0,1")
    ap.add_argument("--products", type=int, default=None, help="first N products (default: all of ESCI-sample)")
    ap.add_argument("--fields", action="store_true", help="add synthetic price, stock, category (not the dataset as served)")
    args = ap.parse_args(argv)

    import numpy as np
    from referee import loader
    from tools.stack import Stack, free_port

    ds = loader.load_dataset("esci-sample")
    q = ds.fit[["query_id", "query"]].drop_duplicates("query_id")
    rng = np.random.default_rng(20261110)
    texts = [str(t) for t in q["query"].iloc[np.sort(rng.choice(len(q), args.queries, replace=False))]]
    work = Path(tempfile.mkdtemp(prefix="demo-bench-"))
    (work / "queries.json").write_text(json.dumps(texts))
    t0 = time.time()
    st = Stack(args.products, args.fields, cores=args.cores)
    demo = None
    try:
        print(f"serving up with the index loaded in {time.time() - t0:.0f} s", flush=True)
        port = free_port()
        demo = subprocess.Popen(["taskset", "-c", args.cores, sys.executable, "-m", "tools.demo", "--api", st.base,
                                 "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
        out_js = work / "browser.json"
        p = subprocess.run(["taskset", "-c", args.cores, "node", str(ROOT / "tools" / "demo_bench.js"),
                            f"http://127.0.0.1:{port}/?index=esci-sample", str(work / "queries.json"), str(out_js)],
                           capture_output=True, text=True, env={**__import__("os").environ, "NODE_PATH": npm_root}, timeout=1800)
        if not out_js.exists():
            print(p.stdout[-800:], p.stderr[-800:])
            return 1
        res = json.loads(out_js.read_text())
        res.update({"dataset": "esci-sample", "n_products": args.products or int(len(ds.catalogue)), "cores": args.cores,
                    "synthetic_fields": args.fields})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        a, s = res["autocomplete"], res["page1"]
        print(f"autocomplete keystroke->list (page clock) p95 {a['page_keystroke_to_render_ms']['p95']:.0f} ms, "
              f"fetch p95 {a['page_fetch_rtt_ms']['p95']:.0f} ms, playwright p95 {a['playwright_action_ms']['p95']:.0f} ms "
              f"(n={a['page_keystroke_to_render_ms']['n']})")
        print(f"page 1 fetch p95 {s['page_fetch_rtt_ms']['p95']:.0f} ms, action-to-render p95 "
              f"{s['playwright_action_ms']['p95']:.0f} ms (n={s['playwright_action_ms']['n']}); errors {len(res['errors'])}")
        return 0 if not res["errors"] else 1
    finally:
        if demo:
            demo.terminate()
        st.close()


if __name__ == "__main__":
    raise SystemExit(main())
