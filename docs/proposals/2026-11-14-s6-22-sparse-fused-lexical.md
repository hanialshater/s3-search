# S6-22: a fused lexical path on the GPU (design; architect, 2026-11-14)

Status: design for review. No code yet. The Colab spend (up to 8 units, the founder's line, delegated to the
coordinator) is released when this design and its predictions are in the records. Development measurements only;
the score of record (0.3039127040512092) and the response contract must not change.

## 1. The roofline, checked against the engine

The founder's analysis of the depth-2,000 request assumed about 200k postings per query, about 5 MB of posting traffic
per query, and 3 to 5 passes over arrays of N = 1.2M. I measured the inputs on ESCI-US, and estimated the bytes from
the code and the A100 profiles.

### 1a. Counts

The counts come from the tuned BM25F request (5 fields), over the first 1,000 test-query texts, on the CPU at
s3-search 93c51d3. The run is `scratchpad/s3/s622_counts.py`; it reads only the catalogue and the query texts.

| per query | p10 | median | mean | p90 | p99 | max |
|---|---|---|---|---|---|---|
| terms | 2 | 4 | 4.6 | 7 | 12 | 18 |
| postings read | 23.7k | 275k | **357k** | 789k | 1.30M | 1.59M |
| matched documents (distinct, in a queried field) | 23.4k | 249k | **280k** | 571k | 729k | 803k |

- **The matched documents are 23% of the catalogue on average**, because multi_match is an OR of the terms. 97% of
  queries match more than 2,000 documents.
- "Touch only matched documents" therefore saves about 4 times on the per-document passes, not orders of magnitude.
  At this density, a dense tile beats a sparse candidate list: a list costs 8 or more bytes per candidate, against 4
  per document for a tile.

### 1b. Where the device time goes

These figures are for a batch of 64 on the A100 with both kernels and the plan cache off (707f5fa): 34.2 ms of
device time, 54.2 ms of wall.

- **The BM25F sum kernel itself (`_segsum_kernel`) is 2.97 ms of the 34.2.**
- **The rest is the work around it.**
  - On the postings, about 15 separate elementwise, gather and index kernels:
    - `csr_expand` twice: int64 arange, repeat_interleave and add;
    - the gathers `doc_ids[entry]`, `tf[entry]` (uint8 [P, 5] to float32), `rel_len[docs]` (float32 [P, 5]) and
      `idf[t_rep]`;
    - the norm, the per-field products, the left-to-right field sum, and the saturation, each a pass over [P] or
      [P, 5] float32.
  - On full-length [64, N] arrays, about 15 passes:
    - zeroing the scores, counts and the kernel's float64 accumulator;
    - float64 to float32;
    - two adds;
    - stacking S and NM;
    - the msm compare, the boost multiply and two selects;
    - the live mask;
    - count_nonzero for the totals;
    - the top-k's masked copy, radix select, count pass and write pass.
- **Bytes, estimated from the code:**
  - Postings side: about 560 B per posting, about 200 MB per query at 357k postings. Most of it is int64 index arrays
    (8 B each, made and read several times) and float32 [P, 5] intermediates.
  - Full-length side: about 130 B per document, about 160 MB per query.
  - Total: about 360 MB per query, about 23 GB per batch of 64. In 34 ms that is about 0.68 TB/s, roughly 45% of the
    A100-40GB's 1.55 TB/s.

### 1c. The correction

- **The engine is not running at a few percent of its bandwidth.** It runs at roughly half of it.
- **The waste is the bytes moved:** about 360 MB per query, where about 25 to 35 MB would do. The cause is
  materialising every intermediate: int64 index arrays, float32 [P, 5] temporaries, and about 15 full-length [Q, N]
  arrays. Random access is a secondary cause, and arithmetic is not one.
- **The founder's direction holds:** fuse the stages and keep per-query state on-chip. The two figures it rests on
  change:
  - the postings are 1.8 times the assumed 200k;
  - the matched set is dense (23% of N), not sparse.
- **Host time matters as well.** In one process the request is already host-bound at batch 64: 0.70 ms of host per
  request is 45 ms per batch, against 34 ms of device. So S6-22 raises the device ceiling, from about 1,900 QPS per GPU
  today to the figures in section 4. Serving reaches that ceiling only with host work spread over several GPU
  processes (K > 1 under MPS, already built), or with less of it (CUDA graphs, the planner).

## 2. The design: one fused pass per (query, document tile), then the existing top-k

### 2a. Kernel A: `lexical_tile` (Triton)

The grid is (query, tile of T docs), with T = 2048, so a tile's accumulators fit in on-chip memory: 2048 x (8 + 4) B.

1. **Score, in posting order.** For each of the query's terms, in order:
   - binary-search the term's postings for the tile's doc range, as `_segsum_kernel` does now;
   - read the postings once (doc id int32, tf uint8 x F);
   - gather `rel_len[doc, f]` and the term's idf;
   - compute the weight in float32 with **the torch path's exact operation order**: norm = (1 - b_f) + b_f x rel_len,
     clamped at 1e-6; ptf = sum over fields, left to right, of (w_f x tf_f) / norm_f; imp = idf x ptf x (k1 + 1) /
     (k1 + ptf);
   - add imp to the tile's float64 accumulator and 1 to its match count, in posting order, with a barrier between
     terms.

   The sum is the same as today's kernel's, so it is the same bits.
2. **Match.** match = count ≥ msm, AND live, AND the request's filter bitsets: term and range filters precomputed as
   bitsets of N bits, about 150 KB each.
3. **Write one full-length row:** the float32 score, or -inf where the doc does not match. This is the only [Q, N]
   write.
4. **Totals.** A per-tile match count, added into the query's total with one integer atomic. Integers give the same
   total in any order.
5. **Facets** (when asked). Per matched doc, an int32 atomic add into counts[q, code] in global memory. This is the
   facet kernel's method; it is exact in any order.

### 2b. Kernel B: the existing Triton top-k

Kernel B is the existing Triton top-k (S6-18) over the score row: a threshold, then compaction, then a key sort. The
order is score descending, then doc id ascending, the same as today. The page gathers its columns as now.

### 2c. Bytes per query in the fused design

- Postings: 357k x about 9 B of reads, plus the rel_len gathers (20 B per posting, partly cached), plus idf. About
  12 MB.
- Full-length: one float32 write, plus the top-k's reads (about 3 passes of 4 B). About 20 MB.
- Total: about 32 MB per query, against about 360 MB, roughly 11 times less.

### 2d. Scope

The fused path is used for:
- a single segment;
- a bare lexical root (match or multi_match) with any msm and boost;
- optional term and range filters, which are pre-made as bitsets;
- terms facets on keyword fields;
- sorting by score.

Everything else takes today's path, unchanged: bool trees, nested, function_score, hybrid, rescore, collapse,
search_after, explain, highlight and multi-segment indexes. The fallback is per plan, and its results are the same.

## 3. The founder's four techniques

| Technique | Adopt? | Bit-identity, totals, facets | Expected gain, against 1c |
|---|---|---|---|
| 1. Blocked bit-packed postings (128 per block) with doc-id reordering (BP) | **Not now.** Re-weigh after the fused pass. | Packing alone is exact (layout only). Reordering changes slot order, and today's tie-break is slot order. Slots would need to stay in first_offset order, or every top-k key would need first_offset. Totals and facets are unaffected. | Small at this density. The raw postings are about 3 MB of the fused 32 MB per query, and packing saves at most about 2 MB (about 6%). BP's locality gain is mostly in the rel_len gathers. The 23%-dense matched set already fills tiles. |
| 2. Safe pruning: block-max tile skipping, WAND or MaxScore | **Later, as an option.** Only for requests with `track_total_hits: false` and no aggregations, which the engine does not yet accept. | An exact top-k is possible: skip a tile only if its upper bound is below the running k-th score. Totals and facets need every match, so pruning cannot apply to them. | Weak at depth 2,000: the 2,000th score of a 280k-match OR query is low, so few tiles fall below it. My guess is under 2 times, not measured. The fused pass comes first. |
| 3a. Binary vectors (dense track) | Out of S6-22 (S9 and the dense benchmark) | n/a | n/a |
| 3b. Bitsets for filters and the live mask | **Adopt** for filters and live, inside kernel A | Exact: boolean | Removes the [Q, N] boolean passes, which are part of the about 130 B per doc. Match counts stay small int32s on-chip, because msm needs counts, not bits. |
| 4a. Doc-id tiles sized to L2, with per-tile score, filter and facet counts and an exact merge | **Adopt.** This is kernel A's grid. | Exact: the sum stays in posting order; totals and facets are integer adds. The per-tile partial top-k is not adopted: at k = 2,000 against 2,048-doc tiles it costs more to merge than kernel B costs to run. | It is the source of the about 11 times fewer bytes. |
| 4b. Sharding across GPUs with global statistics | Out of scope (one A100) | It would be exact with global idf and average lengths. | n/a |
| Tiered or popularity-first index | **No** | Approximate | n/a |

## 4. Expected gain, and how to verify it

**Device time per batch of 64, depth 2,000, no plan cache, A100.**
- Today: 34.2 ms.
- The fused design moves about 11 times fewer bytes. At today's roughly 45% bandwidth efficiency that is about 3 ms.
  Atomics, the binary searches, the top-k's radix select (torch.topk for the threshold) and launch overheads make
  **8 to 12 ms per batch** my estimate. That is a device ceiling of about 5,000 to 8,000 QPS per A100, against about
  1,900 now.

**In-process QPS** (one process, batch 64) will barely change, because the request is host-bound. Wall per batch is
about the host's 45 ms. A gain shows in:
- the device time;
- serving with K GPU processes (fair_bench);
- what CUDA graphs and the planner later take off the host. That ordering argues for doing CUDA graphs right after
  S6-22.

**Faceted lexical request** (two terms facets): today 49.3 ms of device per batch with the facet kernel. The facets
ride in kernel A's pass, so I expect **12 to 18 ms**.

**Bit-identity tests**, in GPU CI. The fused path against today's path:
- scores' bits, match counts, totals, facet buckets, and whole responses;
- batch against alone;
- run to run;
- the toy, ESCI-sample, and synthetic ties.

## 5. Risks

1. **Exact float32 weights inside a Triton kernel.** Today torch computes the weights in separate kernels, each
   rounding to nearest, with no fused multiply-adds. A Triton kernel may contract a x b + c into an FMA, and its
   float32 `/` may be approximate. Mitigation: compile with `enable_fp_fusion=False`, divide with
   `tl.math.div_rn`, and keep the exact order of the products and the left-to-right field sum. The bitwise tests are
   the bar. If any bit differs that cannot be fixed, fall back to computing the weights in one torch elementwise pass:
   about half the postings-side gain, with the sum, masks, totals and facets still fused.
2. **Facets.**
   - Codes must be single-valued per doc. Multi-valued keyword fields are checked; if they exist, kernel A loops
     over a doc's codes.
   - Code 0 (missing) is counted, as today.
   - `sum_other_doc_count` comes from the same counts.
   - The facet kernel's two open questions (one slow batch at size 128; a slower batch-1 p50) are checked first. My
     hypothesis is a Triton re-specialisation on the n_values argument: integer arguments are specialised by
     divisibility by 16, so a new value class compiles again. If so, pass it as a tensor or mark it
     `do_not_specialize`.
3. **minimum_should_match.** Counts are per (query, doc) and exact. msm = 0 queries (all docs, score 0) and boosted
   or negative msm stay on today's path unless tested.
4. **Ties and order.** Kernel B keeps score descending, then slot ascending, and slots stay in first_offset order
   (no reordering, see technique 1).
5. **Memory.** One [Q, N] float32 row per query (64 x 4.9 MB = 311 MB per batch), plus per-query totals and
   [Q, n_values] facet counts. This is less than today.
6. **Scope creep.** Only the plans in 2d go through the fused path. The fallback is per plan and proven by the
   existing tests.

## 6. Plan and cost

1. **CPU, free.** The kernel's reference in numpy, which mirrors the Triton kernel's loops, against today's
   path: bitwise on cpu.
2. **L4, about 0.2 units a run.** Kernel A in Triton, the bitwise tests, and GPU CI. The facet re-specialisation
   check goes here too.
3. **A100, about 0.8 units a run.**
   - A profile: `--deep=1 --facets=1 --plan-cache=0` with the fused path on and off, 256 responses compared, and the
     device time.
   - A GPU CI run on the A100.
   - One repeat for fixes.
4. **Estimate:** 3 to 4 units of the 8.
5. **Opt-in first** (`S3_TRITON_FUSED_LEXICAL=1`). It becomes the default only after the steward's score of record
   with it on reproduces 0.3039127040512092 exactly, and the founder or the coordinator decides.

## 7. Draft predictions (to file as S6-arch ids before the first A100 run)

- **P1.** The fused path gives bit-identical responses to today's path: 0 of 256 depth-2,000 responses, and 0 of 256
  faceted responses, differ on the A100. GPU CI passes on the L4 and the A100.
- **P2.** Device time per batch of 64 at depth 2,000, no plan cache: at most 12 ms, against 34.2.
- **P3.** At least 8 ms: a sanity bound, below which I have misread something.
- **P4.** The faceted lexical request's device time per batch of 64 is at most 18 ms, against 49.3 with the facet
  kernel.
- **P5.** In-process QPS at batch 64 (depth 2,000) is at least 1,450. The request is host-bound, so the gain there is
  small.
- **P6.** Host CPU per request is no higher than today's 0.70 ms.
- **P7.** All S6-22 runs cost at most 4 units.
