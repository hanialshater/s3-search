# S3 Search

A GPU-native search engine for product catalogues. One GPU holds the whole index, a key-value store feeds it live
updates, and every result is judged against a matched OpenSearch baseline on public data (Amazon ESCI).

It is the engine of **Floor 2 · Search engine** in Shakabuku Labs, a company of the System 3 city. The floor's records
live in the lab's repository, `hanialshater/shakabuku-labs`:
- the brief: `docs/search-floor/BRIEF.md`;
- the interfaces: now `docs/INTERFACES.md` here (the lab's `docs/search-floor/INTERFACES.md` was the skeleton);
- the milestones S1 to S4, with predictions: `company/milestones.json`;
- the owner's asks: `company/asks.json`.

**Status:** seeded 2026-11-09. S1 (the prototype reproduced under CI) is next.

## Layout

The design is the floor's design 2, run from Claude Code with the roles as subagents. See `docs/KICKOFF.md`; the
work items are in `docs/BOARD.md`.

| Path | What | Owner | Milestone |
|---|---|---|---|
| `baseline/` | The Colab prototype `esci_gpu_vs_opensearch.ipynb` (A100, 2026-11-05): the numbers S1 must reproduce | founder | |
| `datasets/`, `referee/` | The datasets (`toy`, `esci-sample`, `esci-us`), the loader, the metrics, the checks, the benchmark, the Colab referee | evaluation-steward | B1 |
| `engine/` | The GPU operators (BM25F, dense, filters, facets, pagination, fusion), ranking, the query DSL and planner, `engine/api.py` | architect | S1, S3, S6 |
| `colab/` | The makers' GPU development tools (`s9_gpu_check.py`): **development only, never the score of record**; S9 predictions 1 to 3 are judged only by the steward's S9 cell in `referee/colab_referee.ipynb`; data only through `referee/loader.py`, the fit split only | architect | S9 |
| `dataplane/` | Intake, the key-value store and its change log, the indexer and segments, the GPU updater | dataplane-engineer | S2 |
| `serving/` | HTTP and RPC, batching, caching, the admin API, observability | platform-engineer | S4 |
| `deploy/` | The Kubernetes operator (`SearchIndex`), tested on kind | platform-engineer | S4 |
| `tools/`, `demo/` | Side by side, explain, the test-drive report, the demo page | tools-engineer | S5 |

## The referee

```bash
pip install -r requirements.txt
python -m referee fast     # CPU: the referee's own tests, then the engine's contract checks once engine/api.py exists
```

The score of record comes from `referee/colab_referee.ipynb`, run on a GPU in Colab by the founder. It writes
`results.json`. See `referee/README.md`.
