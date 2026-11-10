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
