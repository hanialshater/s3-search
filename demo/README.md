# demo/

owner: tools-engineer · S5-4, S7-8. A single static page (`index.html`, `app.js`, `style.css`) over the serving HTTP
API. It assumes no fields: `GET /indexes/{name}/_mapping` (`s3_fields`) says which are searchable, facetable and
sortable, and the page renders only those.

- **Autocomplete:** `match_bool_prefix` on the title, 40 ms debounce, six suggestions with highlights, arrow keys and
  Enter. Metrics in `window.__suggest` (fetch round trip and keystroke-to-rendered-list).
- **Highlights:** `highlight` on the title with tags that cannot occur in text; the page builds `<mark>` nodes itself, and
  nothing from the API is ever set as HTML.
- **Multi-select facets:** keyword facets as checkboxes; selections go in `post_filter`, so counts ignore them. Where
  one facet is selected and another is not, the page asks again (`_msearch`, aggregations only) so each facet's counts
  ignore only its own selection.
- **Numeric fields** (when the mapping has them): min/max inputs, a `range` facet, a `histogram`, and a sort control
  (relevance, low to high, high to low). ESCI-sample as served has none, and the page then shows none.
- **Geo** (S9-7; only when `s3_fields` has a `geo_point` field): a place, lat/lon and a radius in km; "only within this
  distance" adds a `geo_distance` clause to `post_filter`; "Nearest to the place" sorts by `_geo_distance` and shows each
  hit's distance.
- **Nested facets** (S9-7; only when the mapping has a `nested` field): one nested aggregation per path gives the keyword
  sub-fields as checkboxes and numeric ones as range chips (counts are offers, not products); the selection is one
  `nested` query per path in `post_filter`, so the chosen seller and price must hold in the same offer.
- **Collapse by brand** (`collapse` on `product_brand.keyword`; pages by from/size, as `search_after` is refused).
- **Pagination:** Next sends `search_after` (the last hit's `sort`, verbatim), Previous sends `from`/`size`. Explain
  toggle sends `explain: true`. The page shows server and browser times and the page-1 p95.

Run it locally: `python -m tools.demo --api http://127.0.0.1:8080` then open `http://127.0.0.1:8000/?index=esci-sample`,
or `python -m tools.demo --real [--fields]` to start a real replica over ESCI-sample first. **Local only.** It deploys
only with the founder's approval of that deploy, and no commit here carries `[deploy]`.

## The Colab notebook (S5-5): `demo/colab_demo.ipynb`

owner: tools-engineer · the founder's ask of 2026-11-19. One notebook, opened in Colab (GPU runtime, A100 or T4; it also runs on CPU), run top to bottom. **Development only, never the score of record** (that is `referee/colab_referee.ipynb`; the rigorous load test is `colab/fair_bench.py`).

1. **Setup.** Clones `REF` (default `main`; the repository is private, so a Colab secret `GITHUB_TOKEN` is needed, as in the referee notebook), installs `requirements.txt` (not `requirements-gpu.txt`: its cu12 wheels would overwrite Colab's torch and RAPIDS, as the referee notebook notes), picks `cuda` or `cpu`, loads `DATASET` (`esci-sample`, the default, or `esci-us`) only through `referee/loader.py`.
2. **Serving.** `python -m serving` in the background, query cache off, waits on `/healthz`.
3. **Indexing through the API.** `PUT /indexes/esci` (mapping + Parquet source; timed), `GET` the index and its `_mapping`, `_bulk` NDJSON into a small `esci-live` index.
4. **Querying.** The tuned BM25F request, facets (`terms`, `range`, `stats`), a `bool` filter, multi-select `post_filter`, `from` and `search_after`, `_msearch`, `explain` (through `tools/explain.py`), and a small `tfidf-svd-64` dense and hybrid (RRF) example.
5. **Live updates** on `esci-live`: patch, delete, upsert, `_refresh`, `versions`, `_reindex`, `_rollback`.
6. **Benchmark against OpenSearch 2.19.1** (started as in `colab/fair_bench.py`, indexed by `referee/opensearch.py`): indexing time, nDCG@10 and exact recall@100 on **fit-split judgments only**, batch-1 latency p50/p95/p99 on test query **texts**, and a short closed-loop concurrency run. A table, a chart, and `/content/demo-bench.json`. The predictions, with sources and a wrong-if each, are in a cell **before** the benchmark; the cell also says what is and is not matched (a closed loop of Python threads hides queueing; `fair_bench` is the rigorous version).

If OpenSearch cannot be downloaded or started, that section prints why and the benchmark runs for the engine alone. Both servers are stopped by the last cell (and by `atexit`).

Run it unattended (a test, not Colab): `S3_DEMO_OVERRIDES='{"N_QUALITY":150,"N_LATENCY":60,"SECONDS_PER_LEVEL":3}'` with `nbclient`; the `IN_COLAB` guard then uses the local checkout and skips the clone and `pip`. `tools/tests/test_demo_notebook.py` checks the file is valid, has no outputs, and compiles.
