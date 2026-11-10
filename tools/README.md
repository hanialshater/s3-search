# tools/

owner: tools-engineer · S5. Clients of `docs/INTERFACES.md` §5 (request, response, explain) and §7 (the HTTP API).
Every number shown is the referee's: metrics are `referee.metrics`, imported, and the report copies `results.json`.

| Tool | Command | What |
|---|---|---|
| Side by side (S5-1) | `python -m tools.compare --a run:os.jsonl --b http://host:8080/INDEX` | Per-query overlap, rank changes, nDCG@10 delta, means. **Fit-split judgments only.** A source is a live serving index or a recorded run file (`{"query_id", "ids"}` per line; `save_run` writes one, so an OpenSearch run recorded in Colab compares against the engine's). |
| Explain (S5-2) | `python -m tools.explain --api URL --index NAME --query "red shoes" --product ID` | Finds the product's hit, prints its trace, checks every sum and the root against `_score` (1e-4). Unknown combiners are reported unchecked, not passed. `sample_check` does it for N sampled products. |
| Report (S5-3) | `python -m tools.report results.json --out report.html --milestones PATH --milestone S5` | Static HTML: recall, quality, slices, latency, cost, worst queries with explanations, predictions beside the results. CPU runs say "development only". `check_report` fails on any difference from `results.json`. |
| Demo (S5-4, S7-8) | `python -m tools.demo --api URL` (or `--real [--fields]`) | Serves `demo/` on 127.0.0.1 and forwards `/indexes/*` to the API. Local only. |

`results.json`'s shape is not yet defined by the referee; `tools/report.py`'s docstring states the shape it reads (a
proposal to the evaluation steward). Any other numeric field is still shown, under "Other".

Tests: `python -m pytest -q tools/tests` (about 5 s). Fakes of the engine and the serving API are in
`tools/tests/fakes.py` and are used only by the compare, explain and report unit tests. The demo is tested against a
real serving replica over a slice of ESCI-sample (`tools/stack.py`, `tools/tests/browser_real.js`, Playwright; they
skip without it). `python -m tools.stack` starts such a replica.
