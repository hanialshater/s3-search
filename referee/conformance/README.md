# The conformance suite: what counts for S3

owner: evaluation-steward · ruling of 2026-11-10, written before any judged S3 run

`opensearch-2.19-toy-v2.jsonl.gz` holds OpenSearch 2.19.1's answers (single node, one shard, security off, Lucene
9.12.1) to the 200 requests of suite v2 (`referee/conformance.py`) over the `toy` dataset. These are OpenSearch's
outputs on synthetic data, not judgments. The suite and its recording are over the same data: `toy`, the only dataset
with the keyword, numeric and vector fields the requests use. ESCI-sample was indexed separately to test the adapter
at scale (bench, 2,000 fit queries).

## The architect's two points (2026-11-10)

1. **Unsupported requests.** Agreed: they belong to S3 prediction 2 and are judged on the engine alone (it must raise
   `UnsupportedQuery` or `InvalidRequest`). OpenSearch answering them is not a failure. Suite v1's OpenSearch
   comparison already did this (`run_suite_os`, 16/16); the cap at 167/183 came from the reference comparison's rule.
   v2 makes it explicit with roles.
2. **Same requests on the same data.** They were already: v1's recording was over toy, like the suite. Indexing
   ESCI-sample in OpenSearch was a separate adapter check.

## Roles

| Role | Requests | Prediction | Rule |
|---|---|---|---|
| `p1` | 160: `filter` 30, `filter_terms` 20, `sort` 30, `facets` 30, `page` 30, `search_after` 20 | 1, at least 98% | totals, aggregation buckets and the ordered hit ids **exactly equal** to OpenSearch's |
| `p2` | 16: one each of 16 constructs outside the subset | 2, all refused | the engine raises a named error; never returns results |
| `info` | 24: `match`, `multi_match`, `bool_scored`, `knn`, `knn_filter` | none | shown strict (scores within 1e-3 relative, ties as groups) and as hit sets; never counted |

- Every `p1` request either asks for counts only (`size` 0) or sorts by fields ending with `product_id` asc, so no
  lexical score decides which hits come back or in what order. That is what "exact operators" means here.
- `search_after` entries are walks of 3 to 7 pages. Each side passes its own last hit's `sort` back verbatim
  (INTERFACES section 5). OpenSearch's cursor holds the sort keys; the engine's ends with its doc id. The
  concatenated ids are compared.
- **Lexical scores.** OpenSearch's BM25 uses the field's docCount for N, avgdl over documents that have the field,
  and one-byte field lengths. The engine follows the prototype (N and avgdl over every document, exact lengths).
  `referee/lucene_model.py` reproduces OpenSearch's `match` scores to 1e-7, so the gap is by design. Lexical scoring
  does not count for prediction 1. A lexical query that only selects a match set (the `facets` family's `match`
  queries) does count, because the set does not depend on the scores.

## S7-9: the S7 families (`opensearch-2.19-esci-sample-s7-v1.jsonl.gz`)

The suite, its roles and rules are in `referee/conformance_s7.py`. They were committed before the recording
(`0b69e2e`), and the recording before the engine ran against it (`d262631`).

**A deviation, found by the first run (2026-11-10).** The adapter's `s3_english` used Lucene's `porter_stem`, which
leaves words of one or two letters unstemmed (`xs`, `us`). INTERFACES §2 names Snowball's `porter`, which stems
them, and so do the engine and the referee. On the first recording, `match` queries holding such words matched
different sets, and the aggregation families that use them inherited the gap: engine against OpenSearch, exact
137/146, avg and stats 15/16, phrase 40/40, highlight 20/20. The fault was the adapter's (the OpenSearch side), not
the engine's. The adapter now uses Lucene's `snowball` filter with language `Porter`, the same algorithm, and the
index name covers the analyzer, so an old index is never reused. Both recordings were made again:

- toy v2: only 2 entries changed, both in the `unsupported` family, which is judged on the engine alone. Every
  `p1` and `info` answer is identical, so S3's verdicts stand.
- S7: re-recorded. The engine run against it gives exact 143/146, avg and stats 16/16, phrase 40/40, highlight
  20/20 (every fragment holds a query term), info 24/32.

The three remaining exact failures are an engine defect: `exists` on a text field skips values that analyse to no
token (stop words, punctuation), where OpenSearch counts every non-null value (4, 9 and 1 such values in those
requests). It is a strict xfail in the contract until fixed. The info families differ through Lucene's
`max_expansions` (50) on `fuzzy` and `match_bool_prefix`; they count for no prediction.
