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
| `python -m referee fast` | CPU, CI on every pull request | `tests/`: the metrics against hand-computed values, the analyzer, the datasets and the loader's boundary, the contract checks on the reference index (with planted errors they must catch), the bench, freshness, conformance and full harnesses on toy. `contract/`, once `engine/api.py` exists: the engine's contract on toy, on every device it reports (`contract/README.md`). About 2 s |
| `python -m referee bench --dataset=toy\|esci-sample [--device=cpu] [--engine=engine\|reference\|opensearch] [--request=t.json]` | CPU for development; any device | Quality on the **fit** split (nDCG@10, Exact recall@k), page-1 latency p50/p99 at batch 1, throughput, cost to `results.json`. CPU numbers are labelled "development only" |
| `python -m referee conformance [--against=reference\|opensearch]` | CPU | The DSL conformance suite: 200 fixed requests; the pass rate against S3-3's 98% |
| `python -m referee full` | GPU, `colab_referee.ipynb`, run by the founder | The score of record: the contract on the device, every ESCI-US test query and the held-out draw, speed, cost, freshness, to `results.json` |

The files:

| File | What |
|---|---|
| `loader.py` | The only way to read data: `load_dataset("toy" \| "esci-sample" \| "esci-us")`, `fetch()`, `build_esci_sample()`; test judgments only through `_load_dataset_test_judgments` (the scorer) |
| `toy.py` | The toy catalogue and its judgments |
| `analysis.py`, `porter.py` | The referee's own `s3_english` analyzer and Porter stemmer |
| `reference.py` | The reference index: the engine's API, plainly, in float64 (the contract's oracle; a stand-in engine for the harnesses) |
| `checks.py`, `contract/` | The contract checks, and their runner against `engine.api` |
| `bench.py`, `freshness.py`, `conformance.py`, `full.py` | The benchmark (B1-3), the freshness check (B1-4), the conformance suite (B1-5), the score of record (B1-6) |
| `opensearch.py`, `requests/` | OpenSearch behind the same API, for the matched baseline; request templates (the prototype's tuned `cross_fields`, its hand BM25F weights) |

## The boundary

ESCI is public, so the hold-out is a rule:

- Members read data only through `loader.py`. It serves the catalogue, the fit split (train-split judgments) and the
  test queries' texts, with no judgments.
- Only the scorer reads the test judgments.
- The founder's draw is scored only in Colab, so no member knows which 2,000 queries it holds.
- A stronger hold-out, for later, is fresh queries labelled by an LLM judge.

## Status (2026-11-09, B1)

- Done on CPU: the datasets (`toy`, `esci-sample`), the contract (waiting for `engine/api.py`), the bench, the
  freshness harness, the conformance skeleton, `full.py` and the notebook. `referee fast`: 124 tests, about 2 s.
- Not yet run: anything on the engine (S1), anything on a GPU, and OpenSearch through `opensearch.py` (no OpenSearch
  here; the adapter's request bodies are tested offline only).
- Freshness through the data plane (S2) and OpenSearch's recorded responses for the conformance suite are still to come.
