// The demo page: one static page over the serving HTTP API (docs/INTERFACES.md §5 request/response, §7 routes).
// Local only. Query string: ?api=http://host:port (default: same origin) and ?index=NAME (default esci-sample).
//
// The page does not assume fields. On load it discovers which fields the index has (see discoverSchema): text fields
// to search, keyword fields to facet and filter on, numeric fields to range-filter and sort on. It renders only those.
"use strict";

const PARAMS = new URLSearchParams(location.search);
const API = (PARAMS.get("api") || "").replace(/\/$/, "");
const INDEX = PARAMS.get("index") || "esci-sample";
const SIZE = 10;

// Candidates probed when the index info carries no mapping (GET /indexes/{name} has none in §7 yet).
const TEXT_CANDIDATES = ["product_title^3", "product_brand^2", "product_color", "product_bullet_point", "product_description"];
const KEYWORD_CANDIDATES = ["category", "product_brand.keyword", "product_color.keyword"];
const NUMERIC_CANDIDATES = ["price", "stock"];
const NUMERIC_TYPES = new Set(["float", "double", "half_float", "scaled_float", "integer", "long", "short", "byte"]);

const state = { q: "", filters: {}, ranges: {}, sort: "", page: 1, explain: false, last: null, pendingAfter: null };
let schema = { text: [], facets: [], numerics: [] };
const latencies = []; // browser round trips of page-1 searches (ms), for the p95 the S5 prediction names
let seq = 0;

/** The §5 request for the current state. `after` is the previous page's last hit `sort` (next), else `from` is used. */
function buildRequest(s, page, after, sch = schema) {
  const filter = [];
  for (const f of sch.facets) {
    const sel = s.filters[f.field];
    if (sel && sel.size) filter.push({ terms: { [f.field]: [...sel] } });
  }
  for (const n of sch.numerics) {
    const r = s.ranges[n.field] || {};
    const range = {};
    if (r.min !== "" && r.min != null && !isNaN(+r.min)) range.gte = +r.min;
    if (r.max !== "" && r.max != null && !isNaN(+r.max)) range.lte = +r.max;
    if (Object.keys(range).length) filter.push({ range: { [n.field]: range } });
  }
  const text = s.q.trim();
  const must = text && sch.text.length ? [{ multi_match: { query: text, fields: sch.text, type: "cross_fields" } }] : [{ match_all: {} }];
  const req = { query: { bool: { must, filter } }, size: SIZE, _source: true };
  if (sch.facets.length) req.aggs = Object.fromEntries(sch.facets.map((f) => [f.field, { terms: { field: f.field, size: 10 } }]));
  if (s.sort) {
    const [field, order] = s.sort.split(":");
    req.sort = [{ [field]: { order } }, "_score"];
  }
  if (after) { req.search_after = after; req.from = 0; } else { req.from = (page - 1) * SIZE; }
  if (s.explain) req.explain = true;
  return req;
}

async function post(req, index = INDEX) {
  const t0 = performance.now();
  const res = await fetch(`${API}/indexes/${encodeURIComponent(index)}/_search`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body, ms: performance.now() - t0 };
}

/** What the index supports, from GET /indexes/{name}/_mapping (§7): {name: {mappings, s3_fields: {field: {type,
 *  searchable, filterable, facetable, sortable}}}}. Sub-fields are `f.keyword`. The id field is not offered as a facet. */
function schemaFromCapabilities(caps, mapping) {
  const idField = (((mapping || {})._meta || {}).engine || {}).id_field;
  const boosts = { product_title: 3, product_brand: 2 };
  const out = { text: [], facets: [], numerics: [] };
  const label = (f) => f.replace(/^product_/, "").replace(/\.keyword$/, "");
  for (const [field, c] of Object.entries(caps || {})) {
    if (field === idField) continue;
    if (c.searchable) out.text.push(boosts[field] ? `${field}^${boosts[field]}` : field);
    if (c.facetable && c.filterable) out.facets.push({ field, label: label(field) });
    if (NUMERIC_TYPES.has(c.type) && c.filterable && c.sortable) out.numerics.push({ field, label: label(field) });
  }
  return out;
}

/** Find what the index supports: its `_mapping` route, else probe the candidate fields one by one. */
async function discoverSchema() {
  try {
    const res = await fetch(`${API}/indexes/${encodeURIComponent(INDEX)}/_mapping`);
    if (res.ok) {
      const body = await res.json();
      const m = body[INDEX] || Object.values(body)[0];
      if (m && m.s3_fields) return schemaFromCapabilities(m.s3_fields, m.mappings);
    }
  } catch (e) { /* fall through to probing */ }
  const ok = async (req) => { try { return (await post({ size: 1, ...req })).ok; } catch (e) { return false; } };
  const [text, facets, numerics] = await Promise.all([
    Promise.all(TEXT_CANDIDATES.map(async (f) => (await ok({ query: { multi_match: { query: "a", fields: [f] } } })) ? f : null)),
    Promise.all(KEYWORD_CANDIDATES.map(async (f) => (await ok({ query: { match_all: {} }, aggs: { x: { terms: { field: f, size: 1 } } } }))
      ? { field: f, label: f.replace(/^product_/, "").replace(/\.keyword$/, "") } : null)),
    Promise.all(NUMERIC_CANDIDATES.map(async (f) => (await ok({ query: { bool: { filter: [{ range: { [f]: { gte: 0 } } }] } }, sort: [{ [f]: { order: "asc" } }, "_score"] }))
      ? { field: f, label: f } : null)),
  ]);
  return { text: text.filter(Boolean), facets: facets.filter(Boolean), numerics: numerics.filter(Boolean) };
}

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) n.append(k);
  return n;
};

function p95(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)];
}

function buildControls() {
  const box = $("dynamic-filters");
  box.replaceChildren();
  for (const f of schema.facets) {
    state.filters[f.field] = new Set();
    const fs = el("fieldset", {}, el("legend", { textContent: f.label }));
    const opts = el("div", { className: "opts" });
    opts.dataset.field = f.field;
    fs.append(opts);
    box.append(fs);
  }
  for (const n of schema.numerics) {
    state.ranges[n.field] = { min: "", max: "" };
    const mk = (kind) => {
      const inp = el("input", { type: "number", min: "0", step: "any", id: `${n.field}-${kind}` });
      inp.inputMode = "decimal";
      inp.addEventListener("change", () => { state.ranges[n.field][kind] = inp.value; resetAndSearch(); });
      return el("label", {}, `${kind === "min" ? "Min" : "Max"} `, inp);
    };
    box.append(el("fieldset", {}, el("legend", { textContent: n.label }), mk("min"), mk("max")));
  }
  const sel = $("sort");
  sel.replaceChildren(el("option", { value: "", textContent: "Relevance" }));
  for (const n of schema.numerics) {
    sel.append(el("option", { value: `${n.field}:asc`, textContent: `${n.label}, low to high` }),
               el("option", { value: `${n.field}:desc`, textContent: `${n.label}, high to low` }));
  }
  sel.parentElement.hidden = schema.numerics.length === 0;
}

function renderFacet(f, agg) {
  const box = document.querySelector(`.opts[data-field="${CSS.escape(f.field)}"]`);
  if (!box) return;
  box.replaceChildren();
  const chosen = state.filters[f.field];
  const buckets = new Map(((agg && agg.buckets) || []).map((b) => [String(b.key), b.doc_count]));
  for (const k of chosen) if (!buckets.has(k)) buckets.set(k, 0); // a chosen value stays visible, so it can be unchecked
  for (const [key, count] of buckets) {
    const cb = el("input", { type: "checkbox", checked: chosen.has(key) });
    cb.dataset.facet = f.field; cb.dataset.value = key;
    cb.addEventListener("change", () => {
      cb.checked ? chosen.add(key) : chosen.delete(key);
      resetAndSearch();
    });
    box.append(el("label", {}, cb, el("span", { textContent: key }), el("span", { className: "count", textContent: String(count) })));
  }
}

function explainText(n, depth = 0) {
  const line = `${"  ".repeat(depth)}${n.value}  ${n.description || ""}`;
  return [line, ...(n.details || []).map((k) => explainText(k, depth + 1))].join("\n");
}

function renderHits(body) {
  const list = $("results");
  list.replaceChildren();
  const ex = new Map((body.explain || []).map((e) => [String(e._id), e]));
  for (const h of body.hits.hits) {
    const src = h._source || {};
    const li = el("li");
    li.dataset.id = h._id;
    li.append(el("div", { className: "title", textContent: src.product_title || h._id }));
    const bits = [src.product_brand, src.product_color, src.category, src.price != null ? "$" + src.price : null].filter(Boolean);
    li.append(el("div", { className: "meta", textContent: bits.join(" · ") }));
    li.append(el("div", { className: "score", textContent: `${h._id} · score ${Number(h._score).toFixed(4)}` }));
    if (state.explain && ex.has(String(h._id))) {
      li.append(el("details", {}, el("summary", { textContent: "Why this rank" }),
        el("pre", { className: "explain", textContent: explainText(ex.get(String(h._id))) })));
    }
    list.append(li);
  }
}

async function search() {
  const my = ++seq;
  const after = state.pendingAfter; state.pendingAfter = null;
  const req = buildRequest(state, state.page, after);
  $("error").hidden = true;
  let r;
  try { r = await post(req); } catch (e) { r = { ok: false, status: 0, body: { error: { reason: String(e) } }, ms: 0 }; }
  if (my !== seq) return; // a newer search is in flight; drop this answer
  if (!r.ok) {
    const err = (r.body && r.body.error) || {};
    $("error").textContent = `Search failed (${r.status}): ${err.type || ""} ${err.reason || ""}`.trim();
    $("error").hidden = false;
    return;
  }
  const body = r.body;
  state.last = body;
  const total = body.hits.total;
  const from = (state.page - 1) * SIZE;
  renderHits(body);
  for (const f of schema.facets) renderFacet(f, (body.aggregations || {})[f.field]);
  $("summary").textContent = total === 0 ? "No results" : `${total.toLocaleString()} results`;
  $("page-label").textContent = total ? `Page ${state.page} of ${Math.max(1, Math.ceil(Math.min(total, 10000) / SIZE))}` : "";
  $("prev").disabled = state.page <= 1;
  $("next").disabled = !(body.hits.hits.length === SIZE && from + SIZE < Math.min(total, 10000));
  if (state.page === 1) latencies.push(r.ms);
  const took = body.took_ms && body.took_ms.total;
  $("timing").textContent = `server ${took != null ? took.toFixed(1) : "?"} ms · browser ${r.ms.toFixed(0)} ms` +
    (latencies.length ? ` · page-1 p95 ${p95(latencies).toFixed(0)} ms (n=${latencies.length})` : "");
  window.__demo = { latencies, p95: p95(latencies), state: { q: state.q, page: state.page }, lastRequest: req, schema };
}

function resetAndSearch() { state.page = 1; state.pendingAfter = null; search(); }

let typing;
$("search-form").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(typing); state.q = $("q").value; resetAndSearch(); });
$("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => { state.q = $("q").value; resetAndSearch(); }, 250); });
$("explain").addEventListener("change", (e) => { state.explain = e.target.checked; search(); });
$("sort").addEventListener("change", (e) => { state.sort = e.target.value; resetAndSearch(); });
$("clear").addEventListener("click", () => {
  for (const s of Object.values(state.filters)) s.clear();
  for (const r of Object.values(state.ranges)) { r.min = ""; r.max = ""; }
  for (const i of document.querySelectorAll("#dynamic-filters input[type=number]")) i.value = "";
  resetAndSearch();
});
// Next uses search_after (the last hit's `sort`, passed back verbatim); previous uses from/size.
$("next").addEventListener("click", () => {
  const hits = state.last.hits.hits;
  state.pendingAfter = hits[hits.length - 1].sort;
  state.page += 1; search();
});
$("prev").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); state.pendingAfter = null; search(); });

window.DemoCore = { buildRequest, p95, schemaFromCapabilities };
(async () => {
  schema = await discoverSchema();
  buildControls();
  search();
})();
