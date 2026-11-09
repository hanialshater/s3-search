# The board

The search floor's work items, by milestone. A member claims an item by putting its role and the date in the
Claimed column, in a commit, before touching the area (P4). It releases an item by clearing the column. Predictions
for each milestone are in the lab's `company/milestones.json`.

Status values: open · claimed · in review · done.

## B1 · Datasets and the benchmark (evaluation-steward)

| Id | Item | Status | Claimed |
|---|---|---|---|
| B1-1 | `toy` dataset: a synthetic catalogue of about 200 products with judgments, built in under 1 s, for tests | in review | evaluation-steward, 2026-11-09 |
| B1-2 | `esci-sample`: the judged products of 2,000 training and 500 test queries, cached as parquet under `data/` | in review | evaluation-steward, 2026-11-09 |
| B1-3 | `python -m referee bench --dataset=…`: quality, latency (p50 and p99), throughput and cost to `results.json` | in review | evaluation-steward, 2026-11-09 |
| B1-4 | The freshness check: 10,000 hot-field changes; time to visible at p99; results equal to a rebuild | in review | evaluation-steward, 2026-11-09 |
| B1-5 | The DSL conformance suite: 200 requests, replayed against OpenSearch and the engine (feeds S3) | in review | evaluation-steward, 2026-11-09 |
| B1-6 | `referee full` and the Colab notebook wired to the engine's API | in review | evaluation-steward, 2026-11-09 |

## S1 · GPU operators (architect)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S1-1 | Interfaces sections 4 to 6: segment format, request and response, plan | claimed | architect, 2026-11-09 |
| S1-2 | `engine/ops/`: analysis and BM25F (multi-field, query-time weights), checked against a reference implementation | open | |
| S1-3 | Exact dense search with a small CPU model; device-agnostic (CPU and CUDA) | open | |
| S1-4 | Filters as bitmaps, terms facets over the full match set, top-k with from/size and search_after | open | |
| S1-5 | RRF fusion and feature gather; the GBDT trained on judged items plus random negatives, not top unjudged | open | |
| S1-6 | `engine/api.py` and the referee's contract checks on `toy` | open | |
| S1-7 | Reproduce 0.304 nDCG@10 on ESCI-US in Colab, with title-only parity against OpenSearch | open | |

## S2 · Data plane (dataplane-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S2-1 | Interfaces sections 2 and 3: mappings, intake, the key-value store and its change log | claimed | architect, 2026-11-09 (drafted for the dataplane-engineer, who owns and may amend it) |
| S2-2 | Intake: batch parquet and JSONL, validation, rejected records, enrichment with embeddings | open | |
| S2-3 | The key-value store (embedded, for example RocksDB or SQLite) with an ordered change log | open | |
| S2-4 | The indexer: segments from the store; merges and deletes | open | |
| S2-5 | The GPU updater: segment loading, the double-buffered swap, hot columns patched in place | open | |

## S3 · Query DSL and planner (architect)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S3-1 | Parser for the OpenSearch query DSL subset; named errors for anything outside it | open | |
| S3-2 | Planner onto the S1 operators; batched executor | open | |
| S3-3 | Pass the B1-5 conformance suite at 98% or better | open | |

## S4 · Serving and platform (platform-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S4-1 | Interfaces section 7: the admin API and the `SearchIndex` resource | claimed | architect, 2026-11-09 (drafted for the platform-engineer, who owns and may amend it) |
| S4-2 | Serving: HTTP and RPC, the batching scheduler, caching, health, stage timings | open | |
| S4-3 | The admin API: create, reindex, list versions, roll back | open | |
| S4-4 | The Kubernetes operator: the `SearchIndex` resource, the indexer job, replicas, rolling swaps; on kind | open | |

## S5 · Tools and the demo page (tools-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S5-1 | Side by side: the engine against OpenSearch, or two versions | open | |
| S5-2 | Debug (explain): every score and feature contribution for one product | open | |
| S5-3 | The test-drive report from a referee run | open | |
| S5-4 | The demo page over ESCI-sample: search, facets, pagination | open | |

## S6 · Serving speed (architect)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S6-1 | A compiled query encoder, CUDA graphs, fused fusion and features; measured only in Colab | open | |
