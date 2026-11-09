# demo/

owner: tools-engineer · S5-4. A single static page (`index.html`, `app.js`, `style.css`) over the serving HTTP API:
`POST /indexes/{name}/_search` with a `multi_match`, `bool.filter` (terms on `category` and `product_brand.keyword`,
a `range` on `price`) and terms `aggs` as the facets. Pagination: "Next" sends `search_after` (the last hit's `sort`,
verbatim), "Previous" sends `from`/`size`. The explain toggle sends `explain: true`. The page shows the server's
`took_ms.total`, the browser round trip and the page-1 p95 (the S5 prediction's number).

Run it locally: `python -m tools.demo --api http://127.0.0.1:8080` then open `http://127.0.0.1:8000/?index=esci-sample`
(`--fake` runs it on a fake engine). **Local only.** It deploys only with the founder's approval of that deploy, and
no commit here carries `[deploy]`.

Known limits: facet counts follow the filters (no post-filter facets); it needs `_source` on hits for titles (§7 says
serving adds it) and falls back to the id; the mapping's field names are the §2 example's.
