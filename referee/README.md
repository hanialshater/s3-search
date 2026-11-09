# The referee

owner: evaluation-steward (the founder) · prepared by onboarding, 2026-11-09 · makers do not edit this folder (P6)

## What it measures

The five parts are from the floor's brief:

1. **Quality.** Whole-catalogue nDCG@10 and Exact recall@k on ESCI US (reduced set), computed as the prototype
   computes them (`metrics.py`):
   - unjudged products count as gain 0;
   - the ideal DCG is taken over all judged products;
   - the ESCI paper's gains are used.

   It is scored on all test queries, and separately on the founder's private draw of 2,000 of them (the lab's
   `lab/held-out/search-esci.md`).
2. **Speed and cost.**
   - Page-1 p50 and p99 at batch 1.
   - Throughput at a 100 ms p99.
   - Cost per million queries, on the stated GPU and price.
3. **Freshness.** 10,000 price and stock changes. It measures the time until they are visible at p99, and checks that
   the results equal a full rebuild.
4. **Correctness.** The prototype's built-in checks, run on CPU against a small synthetic catalogue: scorer vs
   reference, metrics, fusion and features, facets vs brute force, filters, pagination.
5. **Slices.** Query length, negation, filtered requests.

## How it runs

| Command | Where | What |
|---|---|---|
| `python -m referee fast` | CPU, CI on every pull request | `tests/` now: metrics against hand-computed values, the loader's boundary. `contract/` once `engine/api.py` exists: the engine's correctness checks |
| `python -m referee full` | GPU, `colab_referee.ipynb`, run by the founder | The score of record, written to `results.json`. Its scorer is written with the engine's API in S1 |

## The boundary

ESCI is public, so the hold-out is a rule:

- Members read data only through `loader.py`. It serves the catalogue, the fit split (train-split judgments) and the
  test queries' texts, with no judgments.
- Only the scorer reads the test judgments.
- The founder's draw is scored only in Colab, so no member knows which 2,000 queries it holds.
- A stronger hold-out, for later, is fresh queries labelled by an LLM judge.

## Changes awaiting the steward's approval

- **2026-11-09, onboarding (not the steward): `loader.py` falls back to GitHub's LFS host** when `github.com/raw` is
  blocked. It changes no metric and no split. It takes effect only if the founder, as evaluation steward, merges
  s3-search pull request #1.

## Status (2026-11-09)

- `metrics.py` and `loader.py` exist, and 14 tests are green.
- Waiting on the architect's API (INTERFACES §3):
  - `contract/`, the correctness checks;
  - `full.py`, the scorer for `referee full`.
- `colab_referee.ipynb` is a skeleton.
