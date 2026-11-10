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
