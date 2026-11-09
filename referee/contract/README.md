# The engine's contract

owner: evaluation-steward · 2026-11-09 · written against INTERFACES v1 (architect, 2026-11-09)

`python -m referee fast` runs this folder once `engine/api.py` exists, on every device in
`engine.api.available_devices()`. The checks are in `referee/checks.py`. They also pass on the referee's own reference
index (`referee/reference.py`), which `referee/tests/` checks, along with planted errors the checks must catch.

## What is checked, on `toy`

| Check | Compared with | Exactness |
|---|---|---|
| `match`, `multi_match` (all types), `bool` scores and order: 158 requests over 36 query texts | the reference index (float64) | `abs(engine - ref) <= 1e-4 * max(1, abs(ref))`; ids exact except inside tie groups |
| One BM25 score worked by hand in the check | the interfaces' formula | the same tolerance |
| Filters: `term`, `terms`, `range` (float, integer), keyword sub-fields, `must_not`, nested `should`; alone and under a query | a pandas mask over the catalogue | exact set and total; filter-only scores are 0; a filter never changes a score |
| `term` alone, `match_all` | OpenSearch's constant score | exactly 1.0 |
| `terms` aggregations (keyword, sub-field, integer) under 4 queries | brute-force counts over the full match set | exact buckets (count desc, key asc), `sum_other_doc_count` |
| Field sorts (asc, desc, `missing` first and last, two keys) | a brute-force sort, ties by doc id | exact order |
| Ties | exact duplicates in toy | equal scores, doc-id order |
| Pagination: from/size (1, 7, 50) and a `search_after` walk (7, 64) | the implementation's own single page | exact, no duplicates, empty page past `total` |
| `knn`, with and without a pre-filter | brute-force cosine | `_score = (1 + cos) / 2` within tolerance; ids exact except inside tie groups |
| Errors | named classes | `UnsupportedQuery` with `.path` naming the key (fuzzy, wildcard, query_string, match_phrase, highlight, histogram, collapse); `InvalidRequest` for `from + size > 10,000` and a non-int size |
| Shape and determinism | itself | the same response twice; `search_batch == [search]`; `took_ms.total`; no `aggregations` unless asked |
| Devices agree (when more than one) | the `cpu` device | rtol 1e-5 (INTERFACES section 0) |
| Freshness (`test_freshness.py`) | a rebuild of the updated catalogue | 10,000 changes through `Index.patch`, all visible; probes equal a rebuild |

## The referee's reading of the interfaces (architect: please confirm or tell the steward)

1. **The analyzer's word pattern.** The referee follows the prototype: words are `[a-z0-9]+(?:'[a-z0-9]+)?`, then a
   trailing `'s` is cut, then `'` is removed. INTERFACES section 2 says `[a-z0-9']+`. The two differ only on stray
   apostrophes (a lone `'s` token, a word ending in `'`, `''`). The toy texts avoid them, so the contract does not
   depend on the choice, but the prototype's reproduction of 0.304 used its own pattern.
2. **The stemmer.** The referee carries its own Porter (1980) stemmer, as Snowball's `porter` defines it, checked
   against the paper's 72 examples. It has not been compared with `snowballstemmer` (running that package here was
   not allowed). If a lexical check fails, compare `engine.analysis.analyze` with `referee.analysis.analyze` on the
   toy texts first. A mismatch is the steward's to fix if it is in the referee.
3. **idf.** `match` uses its field's df. `multi_match` uses the global df (a document has the term in any text field
   of the mapping, the segment's `df`), **except a `multi_match` that names one field, which uses that field's df**
   (the prototype's `single_field` rule). Field weights do not change the df.
4. **Constant scores.** `term`, `terms`, `range` and `match_all` score `1.0 * boost` in a scoring context (alone, or
   in `must` or `should`), and 0 inside `filter` and `must_not`. A bool with only filters scores 0.
5. **`minimum_should_match`.** An int, a negative int (all but n), or a percentage rounded down. `operator: and`
   needs every analysed query term, including terms not in the vocabulary (no match).
6. **`product_id` is a keyword property** in both mappings (`toy` and ESCI), so a request can filter by id. The
   freshness probe needs it.
7. **`sort`.** Default `_score` desc; field sorts default to asc; `missing` defaults to `_last`. Every hit carries
   `sort`; the contract treats it as opaque and passes it back verbatim as `search_after`.
8. **knn.** At the top level, the hits are the top `k` among live documents passing the pre-filter. `total` is not
   checked for knn.
9. **Facets on integer fields** return int keys.

## The toy dataset

`loader.load_dataset("toy")`: 200 products with ESCI's column names (`product_id`, `product_title`, `product_brand`,
`product_color`, `product_bullet_point`, `product_description`), plus `category` (keyword), `price` (float), `stock`
(integer, hot; 57 are 0) and `embedding` (16 float32 values, so no encoder is needed). Six products duplicate earlier
ones (same text, same price), so ties occur. The mapping is `referee/toy.py: MAPPING` (INTERFACES section 2 format).
