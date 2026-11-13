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
| S1-8 | Per-field BM25F `b` in the mapping (`_meta.engine.bm25.b` as a map, `b_default`), on every scoring path (lexical, term groups, phrase, explain, segments and merges), so S1-7's tuned BM25F can be expressed; a segment with other parameters is refused. Built out of process by the coordinator (ba28176); reviewed and amended by the architect | in review | architect, 2026-11-13 |

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
| S6-2 | `colab/pipeline_profile.py`: the S6 pipeline's per-stage latency and throughput on a Colab GPU (development only). Built out of process by the coordinator (dd319a7); reviewed by the architect | in review | architect, 2026-11-13 |
| S6-3 | `colab/fair_bench.py`: a matched open-loop load test, the engine against OpenSearch, with GPU and server CPU sampled (development only). Built out of process by the coordinator (d0ae1a6, a702cc4); reviewed by the architect; free ports and a fail-fast start after the run of 2026-11-13 died on port 8080 (63731a2) | in review | architect, 2026-11-13 |
| S6-4 | Query encode reads only the components rows a query uses (the A100 profile at 87baeec: encode 150.8 ms of a 272.6 ms pipeline query, from copying all V x 128 components to float64 per call); bit-equal to before | in review | architect, 2026-11-13 |
| S6-5 | The rank stage (100.7 ms p50 at batch 1): the profile's ranker ran the engine's own tree walk (`add_ranker` defaults to `runtime="engine"`: one small kernel per tree and depth), not XGBoost; measure `runtime="auto"`/`"fil"`, and walk all trees at once | in review | architect, 2026-11-13 (e108d2b; A100 profile: rank 100.7 -> 3.62 ms, pipeline batch 1 p50 272.6 -> 23.4 ms) |
| S6-6 | Batching does not pay: lexical `search_batch` stays at about 200 QPS at batch 16, 64 and 128, and the pipeline at 4 QPS (per-query work inside the batch: encode, rescore, facets, page) | in review | architect, 2026-11-13 (3ea8950; A100 profile: lexical batch 64 201 -> 689 QPS, pipeline 58.7 -> 75 QPS) |
| S6-7 | At depth 2,000 over HTTP the engine's server is one saturated Python core with the GPU at 7% (load test of 2026-11-13: 50 QPS sustained, collapse at 100; OpenSearch the same on 8 cores): hit materialisation and JSON per hit in `Executor.run`, and `serving`'s single GIL-bound `ThreadingHTTPServer`. Engine side (vectorised hits) for the architect; the server (processes or a faster encoder) is the platform-engineer's area | in review (engine side, 043205e, 8119836) | architect, 2026-11-13 (engine side: columnar hits, INTERFACES §5); the server side is the platform-engineer's |
| S6-8 | Batched rescoring: the candidates' features and the ranker for every rescored plan of a batch at once (the A100 profile at 3ea8950: features 4.3 ms and rank 1.4 ms per query, pipeline 75 QPS at batch 64), equal to the per-plan path | in review | architect, 2026-11-13 (1b3256b; A100 profile: pipeline batch 64 75 -> 113 QPS) |
| S6-9 | Batched fusion: the dense top-k masks and each hybrid query's RRF windows and fusion for every plan of a batch at once (the A100 profile at 1b3256b: fuse 4.7 ms per query, pipeline 113 QPS at batch 64), equal to the per-plan path | in review | architect, 2026-11-13 (b75d708, dcdcadb; A100 profile: pipeline batch 64 113 -> 177 QPS) |
| S6-10 | One GPU process for every worker, engine side: INTERFACES §5 "The engine in one GPU process" (what search_batch promises, errors="return"); the process architecture and section 7 are the platform-engineer's | in review | architect, 2026-11-13 (contract 2da9b47, errors="return" 211a50c) |
| S6-11 | The per-plan work left in a batch: device syncs per plan and stage, the per-plan filter and mask kernels, the candidate stacking; measured with a kernel and sync count (the A100 profile at 211a50c: 179 QPS pipeline, 693 lexical at batch 64, per GPU process) | claimed | architect, 2026-11-13 |

## S7 · Feature coverage (architect, platform-engineer, tools-engineer, evaluation-steward)

Added by the coordinator, 2026-11-10, on the founder's ask for good feature coverage. Predictions are in the lab's
`company/milestones.json` (S7), written before the work started.

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S7-1 | Positions in segments (additive to §4); `match_phrase` with `slop` | architect | in review | architect, 2026-11-10 |
| S7-2 | Typo tolerance and autocomplete: `fuzzy`, `match` with `fuzziness`, `prefix`, `match_bool_prefix` | architect | in review | architect, 2026-11-10 |
| S7-3 | `exists`, `ids`; `post_filter` (multi-select facets); `collapse` on a keyword field | architect | in review | architect, 2026-11-10 |
| S7-4 | Metric and bucket aggregations: `min`, `max`, `avg`, `sum`, `stats`, `value_count`, `cardinality` (exact), `range`, `histogram`, one level of sub-aggregations under `terms` | architect | in review | architect, 2026-11-10 |
| S7-5 | `highlight` (plain, on text fields) and `function_score` with `field_value_factor` and `weight` | architect | in review | architect, 2026-11-10 |
| S7-6 | Synonyms in the analyzer, defined in the mapping (query-time) | architect | in review | architect, 2026-11-10 |
| S7-7 | RPC serving next to HTTP; the operator on watches and a leader-election Lease | platform-engineer | in review | platform-engineer 2026-11-10 |
| S7-8 | Demo: autocomplete, highlights, multi-select facets with `post_filter`, sort control, collapse by brand | tools-engineer | in review | tools-engineer, 2026-11-10 |
| S7-9 | Conformance families for S7's features, recorded against OpenSearch | evaluation-steward | open | |

## S8 · Fast writes (dataplane-engineer, architect, platform-engineer, evaluation-steward)

Added by the coordinator, 2026-11-10, on the founder's ask that writes be fast. Baseline (2 cores, ESCI-sample):
lexical intake 14,439 docs/s; build from the store 18.1 s (10.7 s before S7-1's positions); a new or edited document
is searchable only after a full rebuild. Predictions are in the lab's `company/milestones.json` (S8).

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S8-1 | The multi-segment contract (§4, §5): a base plus delta segments, global corpus statistics, tombstones, merge; proposal by the dataplane-engineer, ruled by the architect | dataplane-engineer, architect | claimed | dataplane-engineer, 2026-11-10 |
| S8-2 | Engine: search over base plus deltas with statistics equal to a rebuild's; `Index.add_segment`, `remove_segment` | architect | in review | architect, 2026-11-10 |
| S8-3 | Build speed: back under 10 s on ESCI-sample with positions | architect | in review | architect, 2026-11-10 (c98b038) |
| S8-2b | Incremental `add_segment` / `replace_segments`: no full `_relayout` per refresh; heavy work outside the `Index` lock (found by the dataplane-engineer's 2-core stream: add_segment 417-538 ms, query p99 +442%, visibility p99 2.0 s) | architect | open | |
| S8-2c | Searches never wait on writes: build each new `IndexState` outside the `Index` lock and swap the reference under it; searches read a snapshot (found by the dataplane-engineer's re-run after S8-2b: query p99 4-6x idle under writes, from lock waits in add_segment and replace_segments) | architect | open | |
| S8-4 | Data plane: delta segments from the change log every refresh interval, a tiered merge policy, the updater swapping deltas in and merges out | dataplane-engineer | claimed | dataplane-engineer, 2026-11-10 |
| S8-5 | Intake speed: at least 20,000 docs/s lexical, the encoder fit under 10 s | dataplane-engineer | claimed | dataplane-engineer, 2026-11-10 |
| S8-6 | `POST _bulk` (NDJSON), refresh interval and index-lag metrics | platform-engineer | in review | platform-engineer 2026-11-10 |
| S8-7 | The write-stream judge: 200 upserts/s, visibility, equality with a rebuild, query p99 under writes | evaluation-steward | open | |
| S8-8 | `Index.patch` with an id repeated in one call keeps the last value on every device (cuda does not order a scatter with repeated indices; found by `referee fast` on an L4). Built out of process by the coordinator (8053055); reviewed and amended by the architect (a deterministic CPU test) | architect | in review | architect, 2026-11-13 |

## S9 · GPU-friendly coverage (architect, evaluation-steward, tools-engineer)

Added by the coordinator, 2026-11-10. The founder: GPU-friendly features only, drop HNSW, use cuVS and cuML; budget
$400 (the lab's `search-floor-s9` line). Predictions are in the lab's `company/milestones.json` (S9), written before the
work started. cuVS and cuML are optional backends, testable only in the founder's Colab; on CPU, exact knn and the numpy
encoder stay the reference and the fallback.

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S9-1 | cuVS approximate knn: CAGRA and IVF-PQ, filtered; built per segment, rebuilt on merge; exact knn as the CPU fallback, the method named in the response | architect | in review | architect, 2026-11-10 |
| S9-2 | The tfidf-svd encoder fitted and applied with cuML on the GPU; the fitted encoder stored as now | architect | in review | architect, 2026-11-10 |
| S9-3 | GBDT inference on the GPU (cuML FIL or XGBoost), equal to the reference within 1e-5 | architect | in review | architect, 2026-11-10 |
| S9-3b | The XGBoost export in sibling order (each right child right after its left), as XGBoost's GPU predictor assumes; the depth-first export was off by up to 6.57 on cuda. Built out of process by the coordinator (87baeec); reviewed by the architect | architect | in review | architect, 2026-11-13 |
| S9-4 | Geo: the `geo_point` type, `geo_distance`, `geo_bounding_box`, `_geo_distance` sort | architect | in review | architect, 2026-11-10 |
| S9-5 | Nested documents: the `nested` type, the `nested` query with `score_mode`, the `nested` aggregation | architect | in review | architect, 2026-11-10 |
| S9-6 | Conformance families for geo and nested against OpenSearch 2.19; the Colab judges for S9-1 to S9-3 | evaluation-steward | open | |
| S9-7 | Demo: a geo filter and nested facets | tools-engineer | claimed | tools-engineer, 2026-11-10 |

## CI · GPU CI (architect)

Added by the architect, 2026-11-13. CI runs on CPU, so cuda-only bugs (S8-8, S9-3b) reached a GPU before a test saw
them. The round's landing rule (lab `company/decisions/2026-11-13-search-floor-round.md`) needs `colab/gpu_ci.py` green
for an engine change.

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| CI-1 | `colab/gpu_ci.py` (referee fast and the engine tests on a Colab GPU) and `engine/tests/test_write_paths_devices.py` (every write path on each device against cpu). Built out of process by the coordinator (ba28176); reviewed by the architect; a serving-tests suite and short shas (b75d708, from the platform-engineer) | architect | in review | architect, 2026-11-13 |
| CI-2 | A CPU Colab runner for the referee and the engine tests at `requirements.txt`'s versions, for machines whose local Python cannot run them | architect | open | |

## S10 · A PyTerrier-style pipeline DSL (architect)

Added by the architect, 2026-11-13, from the lab's milestone S10 (predictions in `company/milestones.json`, written
before any work). The JSON request stays the wire format; the DSL is a client of `engine/api.py`.

| Id | Item | Owner | Status | Claimed |
|---|---|---|---|---|
| S10-1 | Interfaces: the pipeline section (the transformers, their DataFrame columns, which chains compile into one request and which fall back) | architect | open | |
| S10-2 | The adapter: engine retrievers (lexical, dense, hybrid) and the GBDT re-ranker as PyTerrier transformers; `>>`, `%`, `+`, `**`; hits and scores equal to `Index.search` for the same request | architect | open | |
| S10-3 | `pt.Experiment` over at least three engine pipelines on ESCI-sample's fit queries, on CPU (judgments of the fit split only, through `referee/loader.py`) | architect | open | |
| S10-4 | The compiler: a chain of engine stages (`retrieve % k >> rerank`, filters, fusion) into one batched engine request, with results identical to the DataFrame path | architect | open | |
| S10-5 | Its speed on an A100 against the DataFrame path and the hand-written JSON request (needs a GPU run within the round's cost target) | architect | open | |
