# S3 Search

A GPU-native search engine for product catalogues. One GPU holds the whole index, a key-value store feeds it live
updates, and every result is judged against a matched OpenSearch baseline on public data (Amazon ESCI).

It is the engine of **Floor 2 · Search engine** in Shakabuku Labs, a company of the System 3 city. The floor's records
live in the lab's repository, `hanialshater/shakabuku-labs`:
- the brief: `docs/search-floor/BRIEF.md`;
- the interfaces: `docs/search-floor/INTERFACES.md`;
- the milestones S1 to S4, with predictions: `company/milestones.json`;
- the owner's asks: `company/asks.json`.

**Status:** seeded 2026-11-09. S1 (the prototype reproduced under CI) is next.

## Layout

| Path | What | Owner |
|---|---|---|
| `baseline/` | The Colab prototype `esci_gpu_vs_opensearch.ipynb` (A100, 2026-11-05): the numbers S1 must reproduce | founder |
| `engine/` | The query engine: analysis, BM25F, dense, filters, facets, fusion, features, ranker; `engine/api.py` | architect |
| `dataplane/` | The key-value store and change log, segments, the double-buffered swap, the GPU updater | dataplane-engineer |
| `tools/` | Side by side, debug (explain), the test-drive report | tools-engineer |
| `demo/` | The public demo (deploys only with the founder's approval) | tools-engineer |
| `referee/` | The loader, the metrics, the checks, the Colab referee: the only judge | evaluation-steward |

## The referee

```bash
pip install -r requirements.txt
python -m referee fast     # CPU: the referee's own tests, then the engine's contract checks once engine/api.py exists
```

The score of record comes from `referee/colab_referee.ipynb`, run on a GPU in Colab by the founder. It writes
`results.json`. See `referee/README.md`.
