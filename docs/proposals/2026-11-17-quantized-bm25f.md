# Quantized BM25F: integer bounds first, the exact score second (CPU feasibility study; architect, 2026-11-17)

Status: a study, not a plan of record. Nothing in the engine changed. The numbers are CPU development measurements,
never scores. The founder's idea was a binary-first, exact-second lexical path, later refined to a RaBitQ-style path.
This note measures it on ESCI-US for the founder's workload: tuned BM25F multi_match at depth 2,000, through the
engine's own postings and operations.

## Data, rules and what was not done

- **Data.** ESCI was read only through `referee/loader.py`: the catalogue, the 8,956 test-query texts, and the fit
  split.
- **Judgments.** The quality pass used only fit-split (train) judgments, the `fit` frame `loader.load_dataset` serves, on
  a deterministic 2,000-query sample. No test judgments were loaded or searched for.
  - So the quality figures below cannot be set against the score of record (0.3039 / 0.5338 / 0.7416, test split).
  - They are compared with the exact ranking on the same fit queries.
- **Predictions.** None were filed. The run had already started on the Mac when the request for predictions arrived,
  and I had seen a 10-query smoke test. Predictions written then would have been made after the fact, so I filed
  none. This is a measurement study; its expectations, stated before the full run, were:
  - presence alone (b = 0) would not be enough, a guess based on the field weights below;
  - 8-bit codes would be close to K.
- **Where it ran.** ESCI-US was built once in one process on the Mac. It was not a multi-worker build, but it peaked
  at about 32 GB of footprint with swap nearly full: the kind of heavy local build the founder's rule is about. The
  segment is cached; a rerun should reuse it, or run on a Colab High-RAM CPU. No Colab units were spent.
- **Reproducing it.** `docs/proposals/assets/2026-11-17-quantized-bm25f/`:
  - `prep.py` builds and caches the segment and the plans;
  - `study.py` runs the measurement;
  - `study.json` holds the numbers.
- **Held to the engine.** On 50 checked queries, the study's exact scores give the engine's ids, order and totals for
  all 50.

## The method, per bit width b

- **Codes.** For each posting (t, d), take the engine's float32 contribution c = idf_t · g(ptf), with
  g(ptf) = ptf (k1+1)/(k1+ptf), k1 = 0.036, ptf = Σ_f w_f tf_f / norm_f.
  - The step is D_t = cmax_t / 2^b. The code is q = min(⌊c / D_t⌋, 2^b − 1), and the reconstruction is (q + ½) D_t.
  - b = 0 is presence only, with cmax_t = idf_t (k1+1), which is the founder's B(d).
- **Bounds.** Ŝ(d) = Σ reconstructions and E(d) = Σ D_t/2, both over d's present terms. Then L = Ŝ − E and U = Ŝ + E,
  with a 1e-6 relative margin for the float sum and the boost.
  - θ̂ = the 2,000th largest L. The candidates are {U ≥ θ̂}, and they are rescored exactly.
  - "Two-step" also raises θ̂ to the exact 2,000th score of the approximate top 2,000.
- **Totals, msm and facets** come from presence (ptf > 0) and integer counts, so they are exact at any b.

## Exactness

- **The bound.** Per posting, |c − (q+½)D_t| ≤ D_t/2, so L(d) ≤ Σ_t c ≤ U(d). It also holds after the float64-then-
  float32 sum and the boost, within the margin.
  - L ≤ s for every doc, so θ̂ ≤ θ, the exact 2,000th score.
  - Every doc in the exact top 2,000, ties at θ included, has U ≥ s ≥ θ ≥ θ̂, so it is a candidate.
  - Rescoring the candidates with the same float32 contributions, summed in float64 in query-term order (the engine's
    posting-order sum, which per doc is term order), gives the engine's bits.
  - A top-k over the candidates with the engine's tie-break (score, then slot) is the engine's top 2,000.
- **Measured over the 8,918 queries with terms** (38 analyse to nothing):
  - at every b the bounds held and the candidates covered the exact top 2,000;
  - the largest exact score relative to the presence bound (k1+1)·B was 0.998.
- **Field weights** must be ≥ 0, which the DSL already enforces.
  - Codes built for one weight vector are valid only for that vector, like today's impacts cache. The cache key is
    (w, idf mode).
  - For any other w only b = 0 holds, because g < k1 + 1 for every ptf.
- **Boost** must be > 0. Boost 0 (all ties) and negative boosts (the order reversed) go to today's path, as do the
  LexicalGroups roots (synonyms, fuzzy, const). The Rust host path already serves only plain Lexical roots.

## Numbers (8,918 test-query texts; postings per query p50 125,202, p90 485,000, p99 978,956; matched p50 118,238)

| b | candidates p50 | p90 | p99 | max | two-step p50 / p99 | share of candidates whose interval straddles θ̂ (p50) | approx. recall@2000 vs exact | approx. recall@100 vs exact |
|---|---|---|---|---|---|---|---|---|
| 0 (presence) | 118,238 | 413,706 | 645,329 | 811,198 | 88,267 / 396,659 | 0 (L = 0) | 0.533 | 0.400 |
| 1 | 28,286 | 62,918 | 160,492 | 445,668 | 24,907 / 120,197 | 0.84 | 0.628 | 0.511 |
| 2 | 8,717 | 23,031 | 95,366 | 366,593 | 7,666 / 64,867 | 0.63 | 0.781 | 0.691 |
| 4 | 2,984 | 8,759 | 39,423 | 165,126 | 2,849 / 38,179 | 0.13 | 0.852 | 0.771 |
| 8 | **2,088** | **3,124** | **6,545** | 15,662 | 2,049 / 4,054 | 0.04 | 0.982 | 0.967 |

Approximate-only quality, ranking by Ŝ (ties by slot). Development, fit split, 2,000 queries, exact recall:

| ranking | nDCG@10 | recall@100 | recall@1000 |
|---|---|---|---|
| exact (today's engine scores) | 0.3088 | 0.5308 | 0.7410 |
| b = 0 (presence: the founder's binary pass alone) | 0.1632 | 0.4017 | 0.6353 |
| b = 1 | 0.2012 | 0.4443 | 0.6727 |
| b = 2 | 0.2549 | 0.4955 | 0.7117 |
| b = 4 | 0.2687 | 0.5062 | 0.7210 |
| b = 8 | 0.3084 | 0.5309 | 0.7402 |

**Why presence alone fails.** Saturation is real for a title match: g ≥ 90% of its maximum once ptf ≥ 0.324. But the
tuned weights are small for the other fields: brand 0.126, colour 0.14, bullet points 0.06, description 0.036.
- A term found only in the description has ptf ≈ 0.036·tf/norm, far from saturation. Presence counts it as a full
  idf, so a description-only match ties a title match.
- Docs with the same matched-term set all tie in B, so the threshold falls inside huge tied groups.
- Hence 59× K candidates at the median, and nDCG@10 falling from 0.309 to 0.163.
- 8 bits are enough. 4 bits leave a long tail (p99 39,423 candidates).

**Tiles** (1,024 docs; 1,188 tiles):
- **Postings.** Every query's postings touch all 1,188 tiles at p50, p90 and p99.
- **Reordering.** A cheap locality reordering (docs sorted by two min-hashes of their term sets) gives 1,181 at the
  median, which is no help. Real BP was not tried; at a matched density of about 10% (median 118k of 1.2M) it cannot
  do much for the posting pass.
- **Candidates.** Even at b = 8 the candidates fall in 584 tiles at the median (673 after the reordering), which holds
  about 59% of the postings. So skipping tiles in the posting pass is ruled out for this workload. Stage 2 should
  look candidates up rather than rescan tiles.

## Bytes moved per query (p50 query; the A100 reads, by layout)

| path | postings | the [N] row and its top-k | stage 2 | total, p50 | total, p90 |
|---|---|---|---|---|---|
| fused, no impacts (29 B/posting: doc 4, tf 5, norm 20) | 3.6 MB | write 4.9 + read ≥ 4.9 MB | n/a | 13.4 MB | 23.8 MB |
| fused, impacts (8 B: doc 4, imp 4; plat-m40's setting) | 1.0 MB | 9.7 MB | n/a | 10.7 MB | 13.6 MB |
| quantized b = 8, two passes, doc ids as today (5 B) | 2 × 0.63 MB | none: per-tile integer histograms on-chip | 0.57 MB | **1.8 MB** | 5.7 MB |
| quantized b = 8, tile-local 10-bit doc offsets (2.25 B) | 2 × 0.28 MB | none | 0.57 MB | **1.1 MB** | 3.0 MB |
| quantized b = 4, doc ids as today (4.5 B) | 2 × 0.56 MB | none | 0.81 MB | 1.9 MB | 6.7 MB |

How the quantized rows are built:
- **The two passes.** Pass 1 does presence, counts, totals and facets, plus an on-chip histogram of L per tile. It
  gives a global θ̂ rounded down to a bin edge, which only adds candidates.
- **Pass 2** reads the codes again, recomputes Ŝ on-chip and emits {U ≥ θ̂}: about 2–3k (doc, U) pairs.
- **Stage 2** binary-searches each query term's sorted postings for each candidate: |C| × T × ⌈log2 P_t⌉ × 4 B, with
  T ≈ 4 and about 17 probes, and reads the exact impact. No forward index is needed.
  - A doc-major forward index would cost 105.7M postings × (4 B term + 4 B impact) ≈ 845 MB per weight vector, or about
    950 MB with tf instead.
- **Why it wins.** Today's dominant cost is not the postings but writing and reading the full [N] float32 row for the
  masked top-k. The quantized path never materialises it, and its top-k runs over about 2k candidates.

## What integer accumulation unlocks, and how it combines

- **Order-free deterministic adds.** Integer sums are associative, so Ŝ, E, counts and facets give the same bits in any
  order.
  - Today's float64 accumulator needs a barrier between terms, and posting order within a term, to stay bit-identical.
  - Exactness now lives only in stage 2, on about 2k docs.
- **Posting-parallel kernels.** Every posting of every query term can go to any thread: no per-term barrier, better
  occupancy for short-and-long term mixes, and plain atomics.
- **On-chip integer accumulators.** 1,024 docs × int16/int32 is 2–4 KB per (query, tile), against 12 KB for float64
  plus a count today. More queries per SM, and the per-tile histogram fits beside it.
- **The on-chip accumulator redesign.**
  - **Today's design:** the fused S6-22 kernel keeps per-tile float64 accumulators on-chip. It still writes a full
    float32 row per query for the top-k, and its adds are ordered.
  - **What the quantized path adds:** it is the same tile grid with integer accumulators, plus a θ̂ histogram instead of
    the row, plus a candidate stage. So it combines with that redesign rather than replacing it.
  - **Expected effect:** the cut in bytes (about 6–10× at the median) is mostly the row, and the integer accumulators
    are what make dropping the row cheap.
- **Tile skipping.** Measured as useless here: every query touches every tile, and candidates spread over half the
  tiles. Block-max skipping only pays for shallow k or very selective queries.
  - The θ̂ histogram is the depth-2,000 form of the same idea: it prunes docs, not tiles.
- **Weight vectors.** Codes (like impacts) are per weight vector. The tuned vector serves the workload. Other vectors
  fall back to b = 0 (always valid, but about 59× K candidates) or to today's path.
- **Approximate mode.** Ŝ at b = 8 ranks almost exactly (recall@2000 0.982; nDCG@10 0.3084 against 0.3088 exact on fit).
  But the exact path costs only about 2k lookups more, so an approximate mode buys little and loses the bit-for-bit
  rule. I'd not offer it.

## Dense and hybrid: RaBitQ on the 128-d tfidf-svd vectors (an opinion, no build)

- **Probably worth a later, small study,** for a different reason than the lexical one. Dense retrieval is a batched
  GEMM over 1.2M × 128 float32, about 622 MB per batch. That is already compute-efficient on an A100.
- **What RaBitQ would change.** 1-bit RaBitQ codes are about 16 B per vector plus two factors, about 25 MB in all, which
  fits in L2. That would make the scan nearly free and the k = 1,000 window cheap.
- **Exactness.** RaBitQ's own error bound is probabilistic. A deterministic one is easy: store each vector's
  quantization residual norm ‖x − x̂‖. Then |⟨q,x⟩ − ⟨q,x̂⟩| ≤ ‖q‖‖x − x̂‖ (Cauchy–Schwarz), and the same candidates-
  then-exact scheme keeps today's bits.
- **When.** Only once dense is on the critical path. Today the lexical [N] row and the host are.

## Recommendation

- **Next:** a GPU prototype of the two-pass, 8-bit, integer-accumulator path behind a switch, for the tuned weight
  vector only.
  - It reuses the impacts cache's build kernel for the codes.
  - Stage 2 binary-searches the exact impacts.
  - Its gate is the existing bit-for-bit tests and the 8,956-query byte A/B.
- **Expected effect:** about 6× fewer bytes per p50 query, and no [N] row.
- **Order:** after the R4c/R4d serving A/B, because the serving path is still host-bound at 5,200 QPS.
