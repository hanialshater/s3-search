// The demo page: one static page over the serving HTTP API (docs/INTERFACES.md §5 request/response, §7 routes).
// Local only. Query string: ?api=http://host:port (default: same origin) and ?index=NAME (default esci-sample).
//
// The page assumes no fields. It reads GET /indexes/{name}/_mapping (`s3_fields`) and renders only what the index has:
// text fields to search and highlight, keyword fields as multi-select facets, numeric fields as range inputs, a range
// facet, a histogram and a sort control. Features: autocomplete (match_bool_prefix), highlighted titles, multi-select
// facets (post_filter), sort, collapse by brand, explain, from/size and search_after pagination.
"use strict";

const PARAMS = new URLSearchParams(location.search);
const API = (PARAMS.get("api") || "").replace(/\/$/, "");
const INDEX = PARAMS.get("index") || "esci-sample";
const SIZE = 10;
const SUGGEST_SIZE = 6;
const SUGGEST_DEBOUNCE_MS = 40;
// Highlight tags: the engine does not escape the stored text, so the page asks for tags that cannot occur in titles and
// builds <mark> nodes itself. Nothing from the API is ever set as HTML.
const OPEN = "⟦", CLOSE = "⟧";
const NUMERIC_TYPES = new Set(["float", "double", "half_float", "scaled_float", "integer", "long", "short", "byte"]);
const TITLE = "product_title";

const state = { q: "", sel: {}, ranges: {}, sort: "", collapse: false, page: 1, explain: false, last: null, pendingAfter: null };
let schema = { text: [], title: false, facets: [], numerics: [], brand: null };
const latencies = [];            // browser round trips of page-1 searches (ms)
const suggestRtt = [], suggestKey = []; // autocomplete: fetch round trip, and keystroke to rendered list (ms)
let seq = 0, sseq = 0;

// ---------------------------------------------------------------- the schema ------------------------------------
const niceStep = (span, parts) => {
  const raw = span / parts;
  if (!(raw > 0)) return 1;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 5, 10].map((m) => m * pow).find((x) => x >= raw);
};

/** From `s3_fields` (§7 _mapping): text to search, facetable keyword fields, sortable numeric fields. */
function schemaFromCapabilities(caps, mapping) {
  const idField = (((mapping || {})._meta || {}).engine || {}).id_field;
  const boosts = { product_title: 3, product_brand: 2 };
  const out = { text: [], title: false, facets: [], numerics: [], brand: null };
  const label = (f) => f.replace(/^product_/, "").replace(/\.keyword$/, "");
  for (const [field, c] of Object.entries(caps || {})) {
    if (field === idField) continue;
    if (c.searchable) { out.text.push(boosts[field] ? `${field}^${boosts[field]}` : field); if (field === TITLE) out.title = true; }
    if (c.facetable && c.filterable) out.facets.push({ field, label: label(field) });
    if (NUMERIC_TYPES.has(c.type) && c.filterable && c.sortable) out.numerics.push({ field, label: label(field), min: null, max: null, interval: 1, ranges: [] });
  }
  const brand = out.facets.find((f) => f.field === "product_brand.keyword") || out.facets.find((f) => f.label === "brand");
  out.brand = brand ? brand.field : null;
  return out;
}

async function discoverSchema() {
  const res = await fetch(`${API}/indexes/${encodeURIComponent(INDEX)}/_mapping`);
  if (!res.ok) throw new Error(`mapping route answered ${res.status}`);
  const body = await res.json();
  const m = body[INDEX] || Object.values(body)[0];
  const sch = schemaFromCapabilities(m.s3_fields, m.mappings);
  if (sch.numerics.length) { // one stats aggregation per numeric field fixes the histogram interval and the range buckets
    const r = await post({ size: 0, query: { match_all: {} }, aggs: Object.fromEntries(sch.numerics.map((n) => [n.field, { stats: { field: n.field } }])) });
    for (const n of sch.numerics) {
      const st = r.ok && r.body.aggregations && r.body.aggregations[n.field];
      if (!st || st.min == null) continue;
      n.min = st.min; n.max = st.max;
      n.interval = niceStep(st.max - st.min, 8);
      const e = niceStep(st.max - st.min, 4);
      const lo = Math.floor(st.min / e) * e;
      n.ranges = [{ to: lo + e }, { from: lo + e, to: lo + 2 * e }, { from: lo + 2 * e, to: lo + 3 * e }, { from: lo + 3 * e }];
    }
    sch.numerics = sch.numerics.filter((n) => n.min != null);
  }
  return sch;
}

// ---------------------------------------------------------------- the request -----------------------------------
const textQuery = (s, sch) => {
  const text = s.q.trim();
  return text && sch.text.length ? { multi_match: { query: text, fields: sch.text, type: "cross_fields" } } : { match_all: {} };
};

/** One clause per facet group that has a selection: terms for keyword facets, range for numeric fields. */
function groupClauses(s, sch, except = null) {
  const out = [];
  for (const f of sch.facets) {
    const sel = s.sel[f.field];
    if (f.field !== except && sel && sel.size) out.push({ terms: { [f.field]: [...sel] } });
  }
  for (const n of sch.numerics) {
    const r = s.ranges[n.field] || {};
    const range = {};
    if (r.min !== "" && r.min != null && !isNaN(+r.min)) range.gte = +r.min;
    if (r.max !== "" && r.max != null && !isNaN(+r.max)) range.lte = +r.max;
    if (n.field !== except && Object.keys(range).length) out.push({ range: { [n.field]: range } });
  }
  return out;
}

const groupAggs = (sch, only = null) => {
  const aggs = {};
  for (const f of sch.facets) if (!only || only === f.field) aggs[f.field] = { terms: { field: f.field, size: 10 } };
  for (const n of sch.numerics) {
    if (only && only !== n.field) continue;
    aggs[`${n.field}:histogram`] = { histogram: { field: n.field, interval: n.interval, min_doc_count: 0 } };
    aggs[`${n.field}:range`] = { range: { field: n.field, ranges: n.ranges } };
  }
  return aggs;
};

/** The §5 request. Facet selections go in `post_filter`, so the aggregations (the facets' counts) ignore them.
 *  `after` is the previous page's last hit `sort` (Next); collapse refuses search_after, so it pages by from/size. */
function buildRequest(s, page, after, sch = schema) {
  const post = groupClauses(s, sch);
  const req = { query: { match_all: {} }, size: SIZE, _source: true, aggs: groupAggs(sch) };
  req.query = textQuery(s, sch);
  if (post.length) req.post_filter = { bool: { filter: post } };
  if (s.sort) { const [field, order] = s.sort.split(":"); req.sort = [{ [field]: { order } }, "_score"]; }
  if (s.collapse && sch.brand) req.collapse = { field: sch.brand };
  if (s.q.trim() && sch.title) {
    req.highlight = { fields: { [TITLE]: { number_of_fragments: 0 } }, pre_tags: [OPEN], post_tags: [CLOSE] };
  }
  if (after && !req.collapse) { req.search_after = after; req.from = 0; } else { req.from = (page - 1) * SIZE; }
  if (s.explain) req.explain = true;
  return req;
}

/** post_filter makes every facet ignore all selections. To make a facet ignore only its *own* (the OpenSearch
 *  dashboards behaviour) the page asks again, aggregations only, for each group when some other group has a selection. */
function correctionRequests(s, sch = schema) {
  const groups = [...sch.facets.map((f) => f.field), ...sch.numerics.map((n) => n.field)];
  const out = [];
  for (const g of groups) {
    const others = groupClauses(s, sch, g);
    if (!others.length) continue;
    out.push({ group: g, req: { query: { bool: { must: [textQuery(s, sch)], filter: others } }, size: 0, aggs: groupAggs(sch, g) } });
  }
  return out;
}

async function post(req, path = "_search") {
  const t0 = performance.now();
  const res = await fetch(`${API}/indexes/${encodeURIComponent(INDEX)}/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body, ms: performance.now() - t0 };
}

// ---------------------------------------------------------------- the view --------------------------------------
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

/** Text with OPEN..CLOSE markers -> a node list of text and <mark>. Everything else stays text. */
function highlighted(text) {
  const frag = document.createDocumentFragment();
  let mark = false;
  for (const part of String(text).split(new RegExp(`([${OPEN}${CLOSE}])`))) {
    if (part === OPEN) mark = true;
    else if (part === CLOSE) mark = false;
    else if (part) frag.append(mark ? el("mark", { textContent: part }) : document.createTextNode(part));
  }
  return frag;
}

function buildControls() {
  const box = $("dynamic-filters");
  box.replaceChildren();
  for (const f of schema.facets) {
    state.sel[f.field] = new Set();
    const opts = el("div", { className: "opts" });
    opts.dataset.field = f.field;
    box.append(el("fieldset", {}, el("legend", { textContent: f.label }), opts));
  }
  for (const n of schema.numerics) {
    state.ranges[n.field] = { min: "", max: "" };
    const mk = (kind) => {
      const inp = el("input", { type: "number", min: "0", step: "any", id: `${n.field}-${kind}` });
      inp.inputMode = "decimal";
      inp.addEventListener("change", () => { state.ranges[n.field][kind] = inp.value; resetAndSearch(); });
      return el("label", {}, `${kind === "min" ? "Min" : "Max"} `, inp);
    };
    const fs = el("fieldset", { className: "numeric" }, el("legend", { textContent: n.label }), mk("min"), mk("max"),
      el("div", { className: "ranges" }), el("div", { className: "histogram", ariaLabel: `${n.label} histogram` }));
    fs.dataset.field = n.field;
    box.append(fs);
  }
  const sel = $("sort");
  sel.replaceChildren(el("option", { value: "", textContent: "Relevance" }));
  for (const n of schema.numerics) {
    sel.append(el("option", { value: `${n.field}:asc`, textContent: `${n.label}, low to high` }),
               el("option", { value: `${n.field}:desc`, textContent: `${n.label}, high to low` }));
  }
  $("sort-box").hidden = schema.numerics.length === 0;
  $("collapse-box").hidden = !schema.brand;
}

const fmtNum = (x) => (Number.isInteger(x) ? String(x) : String(Math.round(x * 100) / 100));

function setRange(field, min, max) {
  state.ranges[field] = { min: min == null ? "" : String(min), max: max == null ? "" : String(max) };
  $(`${field}-min`).value = state.ranges[field].min;
  $(`${field}-max`).value = state.ranges[field].max;
  resetAndSearch();
}

function renderTerms(f, agg) {
  const box = document.querySelector(`.opts[data-field="${CSS.escape(f.field)}"]`);
  if (!box) return;
  box.replaceChildren();
  const chosen = state.sel[f.field];
  const buckets = new Map(((agg && agg.buckets) || []).map((b) => [String(b.key), b.doc_count]));
  for (const k of chosen) if (!buckets.has(k)) buckets.set(k, 0); // a chosen value stays visible, so it can be unchecked
  for (const [key, count] of buckets) {
    const cb = el("input", { type: "checkbox", checked: chosen.has(key) });
    cb.dataset.facet = f.field; cb.dataset.value = key;
    cb.addEventListener("change", () => { cb.checked ? chosen.add(key) : chosen.delete(key); resetAndSearch(); });
    box.append(el("label", {}, cb, el("span", { textContent: key }), el("span", { className: "count", textContent: String(count) })));
  }
}

function renderNumeric(n, aggs) {
  const fs = document.querySelector(`fieldset.numeric[data-field="${CSS.escape(n.field)}"]`);
  if (!fs) return;
  const rng = fs.querySelector(".ranges"), hist = fs.querySelector(".histogram");
  const rb = (aggs[`${n.field}:range`] || {}).buckets || [];
  rng.replaceChildren(...rb.map((b) => {
    const label = b.from == null ? `Under ${fmtNum(b.to)}` : b.to == null ? `${fmtNum(b.from)} and up` : `${fmtNum(b.from)} to ${fmtNum(b.to)}`;
    const btn = el("button", { type: "button", className: "chip", textContent: `${label} (${b.doc_count})` });
    btn.dataset.from = b.from ?? ""; btn.dataset.to = b.to ?? "";
    btn.addEventListener("click", () => setRange(n.field, b.from, b.to == null ? null : b.to));
    return btn;
  }));
  const hb = (aggs[`${n.field}:histogram`] || {}).buckets || [];
  const top = Math.max(1, ...hb.map((b) => b.doc_count));
  hist.replaceChildren(...hb.map((b) => {
    const bar = el("button", { type: "button", className: "bar", title: `${fmtNum(b.key)} to ${fmtNum(b.key + n.interval)}: ${b.doc_count}` });
    bar.style.setProperty("--h", `${Math.round((100 * b.doc_count) / top)}%`);
    bar.dataset.key = b.key; bar.dataset.count = b.doc_count;
    bar.addEventListener("click", () => setRange(n.field, b.key, b.key + n.interval));
    return bar;
  }));
}

function renderAggs(aggs, only = null) {
  for (const f of schema.facets) if (!only || only === f.field) renderTerms(f, aggs[f.field]);
  for (const n of schema.numerics) if (!only || only === n.field) renderNumeric(n, aggs);
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
    const frag = ((h.highlight || {})[TITLE] || [])[0];
    const title = el("div", { className: "title" });
    title.append(frag ? highlighted(frag) : document.createTextNode(src[TITLE] || h._id));
    li.append(title);
    const bits = [src.product_brand, src.product_color, src.category, src.price != null ? "$" + src.price : null].filter(Boolean);
    li.append(el("div", { className: "meta", textContent: bits.join(" · ") }));
    li.append(el("div", { className: "score", textContent: `${h._id} · score ${Number(h._score).toFixed(4)}` +
      (h.fields && state.collapse ? " · top hit for this brand" : "") }));
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
  renderAggs(body.aggregations || {});
  $("summary").textContent = total === 0 ? "No results" : `${total.toLocaleString()} ${state.collapse && schema.brand ? "matches, one per brand shown" : "results"}`;
  const pages = req.collapse ? null : Math.max(1, Math.ceil(Math.min(total, 10000) / SIZE));
  $("page-label").textContent = total ? `Page ${state.page}${pages ? ` of ${pages}` : ""}` : "";
  $("prev").disabled = state.page <= 1;
  $("next").disabled = req.collapse ? body.hits.hits.length < SIZE : !(body.hits.hits.length === SIZE && from + SIZE < Math.min(total, 10000));
  if (state.page === 1) latencies.push(r.ms);
  const took = body.took_ms && body.took_ms.total;
  $("timing").textContent = `server ${took != null ? took.toFixed(1) : "?"} ms · browser ${r.ms.toFixed(0)} ms` +
    (latencies.length ? ` · page-1 p95 ${p95(latencies).toFixed(0)} ms (n=${latencies.length})` : "");
  window.__demo = { latencies, p95: p95(latencies), state: { q: state.q, page: state.page }, lastRequest: req, schema };
  // Each facet's counts should ignore its own selection only: refresh the groups whose counts post_filter got wrong.
  const fixes = correctionRequests(state);
  if (fixes.length) {
    try {
      const m = await post(fixes.map((x) => x.req), "_msearch");
      if (m.ok && my === seq) fixes.forEach((x, i) => renderAggs((m.body[i] || {}).aggregations || {}, x.group));
    } catch (e) { /* the counts stay as post_filter left them */ }
  }
}

function resetAndSearch() { state.page = 1; state.pendingAfter = null; search(); }

// ---------------------------------------------------------------- autocomplete ----------------------------------
const sug = { items: [], active: -1 };

function closeSuggest() { sseq++; sug.items = []; sug.active = -1; $("suggest").hidden = true; $("suggest").replaceChildren(); $("q").setAttribute("aria-expanded", "false"); }

function pick(title) { $("q").value = title; closeSuggest(); state.q = title; resetAndSearch(); }

function setActive(i) {
  sug.active = i;
  [...$("suggest").children].forEach((li, j) => { li.setAttribute("aria-selected", String(j === i)); li.classList.toggle("active", j === i); });
}

async function runSuggest(text, t0) {
  const my = ++sseq;
  const req = { query: { match_bool_prefix: { [TITLE]: text } }, size: SUGGEST_SIZE, _source: [TITLE],
    highlight: { fields: { [TITLE]: { number_of_fragments: 0 } }, pre_tags: [OPEN], post_tags: [CLOSE] } };
  let r;
  try { r = await post(req); } catch (e) { return; }
  if (my !== sseq) return; // typed on, or closed
  if (!r.ok) {
    const err = (r.body && r.body.error) || {};
    $("error").textContent = `Autocomplete failed (${r.status}): ${err.type || ""} ${err.reason || ""}`.trim();
    $("error").hidden = false;
    return;
  }
  const list = $("suggest");
  list.replaceChildren();
  sug.items = [];
  for (const h of r.body.hits.hits) {
    const title = (h._source || {})[TITLE] || h._id;
    if (sug.items.includes(title)) continue;
    sug.items.push(title);
    const li = el("li", { role: "option" });
    li.append(highlighted(((h.highlight || {})[TITLE] || [])[0] || title));
    const idx = sug.items.length - 1;
    li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(title); });
    li.addEventListener("mouseenter", () => setActive(idx));
    list.append(li);
  }
  sug.active = -1;
  list.hidden = sug.items.length === 0;
  $("q").setAttribute("aria-expanded", String(sug.items.length > 0));
  suggestRtt.push(r.ms);
  suggestKey.push(performance.now() - t0);
  window.__suggest = { rtt: suggestRtt, keystroke: suggestKey, p95: p95(suggestKey), shown: sug.items.length, text };
}

let sTimer;
$("q").addEventListener("input", () => {
  const t0 = performance.now(), v = $("q").value.trim();
  clearTimeout(sTimer);
  if (!v || !schema.title) { closeSuggest(); return; }
  sTimer = setTimeout(() => runSuggest(v, t0), SUGGEST_DEBOUNCE_MS);
});
$("q").addEventListener("keydown", (e) => {
  const n = sug.items.length;
  if (e.key === "ArrowDown" && n) { e.preventDefault(); setActive((sug.active + 1) % n); }
  else if (e.key === "ArrowUp" && n) { e.preventDefault(); setActive((sug.active - 1 + n) % n); }
  else if (e.key === "Escape") closeSuggest();
  else if (e.key === "Enter" && sug.active >= 0) { e.preventDefault(); pick(sug.items[sug.active]); }
});
$("q").addEventListener("blur", () => setTimeout(closeSuggest, 120));

// ---------------------------------------------------------------- controls --------------------------------------
$("search-form").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(sTimer); closeSuggest(); state.q = $("q").value; resetAndSearch(); });
$("explain").addEventListener("change", (e) => { state.explain = e.target.checked; search(); });
$("sort").addEventListener("change", (e) => { state.sort = e.target.value; resetAndSearch(); });
$("collapse").addEventListener("change", (e) => { state.collapse = e.target.checked; resetAndSearch(); });
$("clear").addEventListener("click", () => {
  for (const s of Object.values(state.sel)) s.clear();
  for (const r of Object.values(state.ranges)) { r.min = ""; r.max = ""; }
  for (const i of document.querySelectorAll("#dynamic-filters input[type=number]")) i.value = "";
  resetAndSearch();
});
// Next uses search_after (the last hit's `sort`, passed back verbatim); collapse refuses it, so it pages by from/size.
$("next").addEventListener("click", () => {
  const hits = state.last.hits.hits;
  state.pendingAfter = state.collapse ? null : hits[hits.length - 1].sort;
  state.page += 1; search();
});
$("prev").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); state.pendingAfter = null; search(); });

window.DemoCore = { buildRequest, correctionRequests, schemaFromCapabilities, p95, highlighted, niceStep };
(async () => {
  try {
    schema = await discoverSchema();
  } catch (e) {
    $("error").textContent = `Could not read the index's fields: ${e}`;
    $("error").hidden = false;
    return;
  }
  buildControls();
  search();
})();
