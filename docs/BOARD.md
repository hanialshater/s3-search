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
| S1-1 | Interfaces sections 4 to 6: segment format, request and response, plan | in review | architect, 2026-11-09 |
| S1-2 | `engine/ops/`: analysis and BM25F (multi-field, query-time weights), checked against a reference implementation | in review | architect, 2026-11-09 |
| S1-3 | Exact dense search with a small CPU model; device-agnostic (CPU and CUDA) | in review | architect, 2026-11-09 |
| S1-4 | Filters as bitmaps, terms facets over the full match set, top-k with from/size and search_after | in review | architect, 2026-11-09 |
| S1-5 | RRF fusion and feature gather; the GBDT trained on judged items plus random negatives, not top unjudged | in review | architect, 2026-11-09 |
| S1-6 | `engine/api.py` and the referee's contract checks on `toy` | in review | architect, 2026-11-09 |
| S1-7 | Reproduce 0.304 nDCG@10 on ESCI-US in Colab, with title-only parity against OpenSearch | open | |

## S2 · Data plane (dataplane-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S2-1 | Interfaces sections 2 and 3: mappings, intake, the key-value store and its change log | in review | dataplane-engineer, 2026-11-09 (drafted by the architect) |
| S2-2 | Intake: batch parquet and JSONL, validation, rejected records, enrichment with embeddings | in review | dataplane-engineer, 2026-11-09 |
| S2-3 | The key-value store (embedded, for example RocksDB or SQLite) with an ordered change log | in review | dataplane-engineer, 2026-11-09 |
| S2-4 | The indexer: segments from the store; merges and deletes | in review | dataplane-engineer, 2026-11-09 |
| S2-5 | The GPU updater: segment loading, the double-buffered swap, hot columns patched in place | in review | dataplane-engineer, 2026-11-09 |

## S3 · Query DSL and planner (architect)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S3-1 | Parser for the OpenSearch query DSL subset; named errors for anything outside it | in review | architect, 2026-11-10 |
| S3-2 | Planner onto the S1 operators; batched executor | in review | architect, 2026-11-10 |
| S3-3 | Pass the B1-5 conformance suite at 98% or better | claimed | architect, 2026-11-10 |

## S4 · Serving and platform (platform-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S4-1 | Interfaces section 7: the admin API and the `SearchIndex` resource | in review | platform-engineer 2026-11-09 (the architect drafted it; the platform-engineer owns and amends it) |
| S4-2 | Serving: HTTP and RPC, the batching scheduler, caching, health, stage timings | in review | platform-engineer 2026-11-09 |
| S4-3 | The admin API: create, reindex, list versions, roll back | in review | platform-engineer 2026-11-09 |
| S4-4 | The Kubernetes operator: the `SearchIndex` resource, the indexer job, replicas, rolling swaps; on kind | in review | platform-engineer 2026-11-09 |

## S5 · Tools and the demo page (tools-engineer)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S5-1 | Side by side: the engine against OpenSearch, or two versions | in review | tools-engineer, 2026-11-09 |
| S5-2 | Debug (explain): every score and feature contribution for one product | in review | tools-engineer, 2026-11-09 |
| S5-3 | The test-drive report from a referee run | in review | tools-engineer, 2026-11-09 |
| S5-4 | The demo page over ESCI-sample: search, facets, pagination | in review | tools-engineer, 2026-11-09 |

## S6 · Serving speed (architect)

| Id | Item | Status | Claimed |
|---|---|---|---|
| S6-1 | A compiled query encoder, CUDA graphs, fused fusion and features; measured only in Colab | open | |

## S7 · Feature coverage (architect, platform-engineer, tools-engineer, evaluation-steward)

Added by the coordinator, 2026-11-10, on the founder's ask for good feature coverage. Predictions are in the lab's
`company/milestones.json` (S7), written before the work started.

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S7-1 | Positions in segments (additive to §4); `match_phrase` with `slop` | architect | claimed | architect, 2026-11-10 |
| S7-2 | Typo tolerance and autocomplete: `fuzzy`, `match` with `fuzziness`, `prefix`, `match_bool_prefix` | architect | claimed | architect, 2026-11-10 |
| S7-3 | `exists`, `ids`; `post_filter` (multi-select facets); `collapse` on a keyword field | architect | claimed | architect, 2026-11-10 |
| S7-4 | Metric and bucket aggregations: `min`, `max`, `avg`, `sum`, `stats`, `value_count`, `cardinality` (exact), `range`, `histogram`, one level of sub-aggregations under `terms` | architect | claimed | architect, 2026-11-10 |
| S7-5 | `highlight` (plain, on text fields) and `function_score` with `field_value_factor` and `weight` | architect | claimed | architect, 2026-11-10 |
| S7-6 | Synonyms in the analyzer, defined in the mapping (query-time) | architect | claimed | architect, 2026-11-10 |
| S7-7 | RPC serving next to HTTP; the operator on watches and a leader-election Lease | platform-engineer | in review | platform-engineer 2026-11-10 |
| S7-8 | Demo: autocomplete, highlights, multi-select facets with `post_filter`, sort control, collapse by brand | tools-engineer | open | |
| S7-9 | Conformance families for S7's features, recorded against OpenSearch | evaluation-steward | open | |

## S8 · Fast writes (dataplane-engineer, architect, platform-engineer, evaluation-steward)

Added by the coordinator, 2026-11-10, on the founder's ask that writes be fast. Baseline (2 cores, ESCI-sample):
lexical intake 14,439 docs/s; build from the store 18.1 s (10.7 s before S7-1's positions); a new or edited document
is searchable only after a full rebuild. Predictions are in the lab's `company/milestones.json` (S8).

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S8-1 | The multi-segment contract (§4, §5): a base plus delta segments, global corpus statistics, tombstones, merge; proposal by the dataplane-engineer, ruled by the architect | dataplane-engineer, architect | open | |
| S8-2 | Engine: search over base plus deltas with statistics equal to a rebuild's; `Index.add_segment`, `remove_segment` | architect | open | |
| S8-3 | Build speed: back under 10 s on ESCI-sample with positions | architect | open | |
| S8-4 | Data plane: delta segments from the change log every refresh interval, a tiered merge policy, the updater swapping deltas in and merges out | dataplane-engineer | open | |
| S8-5 | Intake speed: at least 20,000 docs/s lexical, the encoder fit under 10 s | dataplane-engineer | open | |
| S8-6 | `POST _bulk` (NDJSON), refresh interval and index-lag metrics | platform-engineer | open | |
| S8-7 | The write-stream judge: 200 upserts/s, visibility, equality with a rebuild, query p99 under writes | evaluation-steward | open | |
