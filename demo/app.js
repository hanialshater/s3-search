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

const state = { q: "", sel: {}, ranges: {}, nsel: {}, nrange: {}, geo: { on: false, lat: "", lon: "", km: "" }, sort: "",
                collapse: false, page: 1, explain: false, last: null, pendingAfter: null };
let schema = { text: [], title: false, facets: [], numerics: [], brand: null, geo: null, nested: [] };
// A few places to measure from (the synthetic catalogue's locations cluster around cities like these).
const PLACES = [["New York", 40.71, -74.01], ["Los Angeles", 34.05, -118.24], ["London", 51.51, -0.13], ["Paris", 48.86, 2.35],
                ["Tokyo", 35.68, 139.69], ["Sydney", -33.87, 151.21]];
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
  const out = { text: [], title: false, facets: [], numerics: [], brand: null, geo: null, nested: [] };
  const label = (f) => f.replace(/^product_/, "").replace(/\.keyword$/, "");
  for (const [field, c] of Object.entries(caps || {})) {
    if (field === idField) continue;
    if (c.type === "geo_point" && !out.geo) { out.geo = { field, label: field }; continue; }
    if (c.searchable) { out.text.push(boosts[field] ? `${field}^${boosts[field]}` : field); if (field === TITLE) out.title = true; }
    if (c.facetable && c.filterable) out.facets.push({ field, label: label(field) });
    if (NUMERIC_TYPES.has(c.type) && c.filterable && c.sortable) out.numerics.push({ field, label: label(field), min: null, max: null, interval: 1, ranges: [] });
  }
  // nested paths: their sub-fields are in the mapping, not in s3_fields. keyword -> a terms facet, numeric -> a range.
  for (const [path, prop] of Object.entries((mapping || {}).properties || {})) {
    if (prop.type !== "nested") continue;
    const subs = [];
    for (const [sub, sp] of Object.entries(prop.properties || {})) {
      if (sp.type === "keyword") subs.push({ sub, field: `${path}.${sub}`, kind: "keyword" });
      else if (NUMERIC_TYPES.has(sp.type)) subs.push({ sub, field: `${path}.${sub}`, kind: "number", min: null, max: null, ranges: [] });
    }
    if (subs.length) out.nested.push({ path, subs });
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
  for (const nd of sch.nested) { // the same for numeric sub-fields of a nested path, through a nested aggregation
    const nums = nd.subs.filter((x) => x.kind === "number");
    if (!nums.length) continue;
    const r = await post({ size: 0, query: { match_all: {} }, aggs: { n: { nested: { path: nd.path },
      aggs: Object.fromEntries(nums.map((x) => [x.sub, { stats: { field: x.field } }])) } } });
    for (const x of nums) {
      const st = r.ok && r.body.aggregations && r.body.aggregations.n && r.body.aggregations.n[x.sub];
      if (!st || st.min == null) continue;
      x.min = st.min; x.max = st.max;
      const e = niceStep(st.max - st.min, 4), lo = Math.floor(st.min / e) * e;
      x.ranges = [{ to: lo + e }, { from: lo + e, to: lo + 2 * e }, { from: lo + 2 * e, to: lo + 3 * e }, { from: lo + 3 * e }];
    }
    nd.subs = nd.subs.filter((x) => x.kind === "keyword" || x.min != null);
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
  const g = s.geo;
  if (sch.geo && except !== "geo" && g.on && g.lat !== "" && g.lon !== "" && +g.km > 0) {
    out.push({ geo_distance: { distance: `${+g.km}km`, [sch.geo.field]: { lat: +g.lat, lon: +g.lon } } });
  }
  for (const nd of sch.nested) { // one nested query per path, so the chosen seller and price must hold in the same offer
    if (nd.path === except) continue;
    const conds = [];
    for (const x of nd.subs) {
      if (x.kind === "keyword") {
        const sel = s.nsel[x.field];
        if (sel && sel.size) conds.push({ terms: { [x.field]: [...sel] } });
      } else {
        const r = s.nrange[x.field] || {}, range = {};
        if (r.min !== "" && r.min != null && !isNaN(+r.min)) range.gte = +r.min;
        if (r.max !== "" && r.max != null && !isNaN(+r.max)) range.lte = +r.max;
        if (Object.keys(range).length) conds.push({ range: { [x.field]: range } });
      }
    }
    if (conds.length) out.push({ nested: { path: nd.path, query: { bool: { filter: conds } }, score_mode: "none" } });
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
  for (const nd of sch.nested) {
    if (only && only !== nd.path) continue;
    const sub = {};
    for (const x of nd.subs) {
      if (x.kind === "keyword") sub[x.sub] = { terms: { field: x.field, size: 10 } };
      else sub[`${x.sub}:range`] = { range: { field: x.field, ranges: x.ranges } };
    }
    aggs[`nested:${nd.path}`] = { nested: { path: nd.path }, aggs: sub };
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
  if (s.sort === "geo" && sch.geo && s.geo.lat !== "" && s.geo.lon !== "") {
    req.sort = [{ _geo_distance: { [sch.geo.field]: { lat: +s.geo.lat, lon: +s.geo.lon }, order: "asc", unit: "km" } }, "_score"];
  } else if (s.sort && s.sort !== "geo") { const [field, order] = s.sort.split(":"); req.sort = [{ [field]: { order } }, "_score"]; }
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
  const groups = [...sch.facets.map((f) => f.field), ...sch.numerics.map((n) => n.field), ...sch.nested.map((n) => n.path)];
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
  for (const nd of schema.nested) { // facets over a nested path: choices are offers, counted as offers
    const fs = el("fieldset", { className: "nested" }, el("legend", { textContent: `${nd.path} (counts are ${nd.path})` }));
    fs.dataset.path = nd.path;
    for (const x of nd.subs) {
      if (x.kind === "keyword") {
        state.nsel[x.field] = new Set();
        const opts = el("div", { className: "opts" });
        opts.dataset.field = x.field;
        fs.append(el("div", { className: "sub" }, el("b", { textContent: x.sub }), opts));
      } else {
        state.nrange[x.field] = { min: "", max: "" };
        const mk = (kind) => {
          const inp = el("input", { type: "number", min: "0", step: "any", id: `${x.field}-${kind}` });
          inp.inputMode = "decimal";
          inp.addEventListener("change", () => { state.nrange[x.field][kind] = inp.value; resetAndSearch(); });
          return el("label", {}, `${x.sub} ${kind === "min" ? "min" : "max"} `, inp);
        };
        const ch = el("div", { className: "ranges" });
        ch.dataset.field = x.field;
        fs.append(mk("min"), mk("max"), ch);
      }
    }
    box.append(fs);
  }
  if (schema.geo) {
    const g = state.geo;
    [g.lat, g.lon, g.km] = [String(PLACES[0][1]), String(PLACES[0][2]), "50"];
    const num = (id, key, label, step) => {
      const inp = el("input", { type: "number", id, step, value: g[key] });
      inp.addEventListener("change", () => { g[key] = inp.value; if (g.on || state.sort === "geo") resetAndSearch(); });
      return el("label", {}, `${label} `, inp);
    };
    const place = el("select", { id: "geo-place" }, ...PLACES.map(([n, la, lo], i) => el("option", { value: String(i), textContent: n })));
    place.addEventListener("change", () => {
      const [, la, lo] = PLACES[+place.value];
      g.lat = String(la); g.lon = String(lo); $("geo-lat").value = g.lat; $("geo-lon").value = g.lon;
      if (g.on || state.sort === "geo") resetAndSearch();
    });
    const on = el("input", { type: "checkbox", id: "geo-on" });
    on.addEventListener("change", () => { g.on = on.checked; resetAndSearch(); });
    box.append(el("fieldset", { id: "geo-box" }, el("legend", { textContent: "Near a place" }),
      el("label", {}, "From ", place), num("geo-lat", "lat", "Lat", "any"), num("geo-lon", "lon", "Lon", "any"),
      num("geo-km", "km", "Within (km)", "any"), el("label", {}, on, " Only within this distance")));
  }
  const sel = $("sort");
  sel.replaceChildren(el("option", { value: "", textContent: "Relevance" }));
  for (const n of schema.numerics) {
    sel.append(el("option", { value: `${n.field}:asc`, textContent: `${n.label}, low to high` }),
               el("option", { value: `${n.field}:desc`, textContent: `${n.label}, high to low` }));
  }
  if (schema.geo) sel.append(el("option", { value: "geo", textContent: "Nearest to the place" }));
  $("sort-box").hidden = schema.numerics.length === 0 && !schema.geo;
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

function renderNested(nd, agg) {
  agg = agg || {};
  for (const x of nd.subs) {
    if (x.kind === "keyword") {
      const box = document.querySelector(`.opts[data-field="${CSS.escape(x.field)}"]`);
      if (!box) continue;
      box.replaceChildren();
      const chosen = state.nsel[x.field];
      const buckets = new Map((((agg[x.sub] || {}).buckets) || []).map((b) => [String(b.key), b.doc_count]));
      for (const k of chosen) if (!buckets.has(k)) buckets.set(k, 0);
      for (const [key, count] of buckets) {
        const cb = el("input", { type: "checkbox", checked: chosen.has(key) });
        cb.dataset.facet = x.field; cb.dataset.value = key;
        cb.addEventListener("change", () => { cb.checked ? chosen.add(key) : chosen.delete(key); resetAndSearch(); });
        box.append(el("label", {}, cb, el("span", { textContent: key }), el("span", { className: "count", textContent: String(count) })));
      }
    } else {
      const ch = document.querySelector(`.ranges[data-field="${CSS.escape(x.field)}"]`);
      if (!ch) continue;
      ch.replaceChildren(...(((agg[`${x.sub}:range`] || {}).buckets) || []).map((b) => {
        const label = b.from == null ? `Under ${fmtNum(b.to)}` : b.to == null ? `${fmtNum(b.from)} and up` : `${fmtNum(b.from)} to ${fmtNum(b.to)}`;
        const btn = el("button", { type: "button", className: "chip", textContent: `${label} (${b.doc_count})` });
        btn.addEventListener("click", () => {
          state.nrange[x.field] = { min: b.from == null ? "" : String(b.from), max: b.to == null ? "" : String(b.to) };
          document.getElementById(`${x.field}-min`).value = state.nrange[x.field].min;
          document.getElementById(`${x.field}-max`).value = state.nrange[x.field].max;
          resetAndSearch();
        });
        return btn;
      }));
    }
  }
}

function renderAggs(aggs, only = null) {
  for (const f of schema.facets) if (!only || only === f.field) renderTerms(f, aggs[f.field]);
  for (const n of schema.numerics) if (!only || only === n.field) renderNumeric(n, aggs);
  for (const nd of schema.nested) if (!only || only === nd.path) renderNested(nd, aggs[`nested:${nd.path}`]);
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
    if (state.sort === "geo" && schema.geo && typeof (h.sort || [])[0] === "number") bits.push(`${h.sort[0].toFixed(1)} km away`);
    if (Array.isArray(src.offers)) bits.push(`${src.offers.length} offers` + (src.offers.length ? ` from $${Math.min(...src.offers.map((o) => o.price))}` : ""));
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
let sTimer;

function closeSuggest() { clearTimeout(sTimer); sseq++; sug.items = []; sug.active = -1; $("suggest").hidden = true; $("suggest").replaceChildren(); $("q").setAttribute("aria-expanded", "false"); }

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
    li.addEventListener("mousemove", () => { if (sug.active !== idx) setActive(idx); }); // not mouseenter: a list appearing under a resting pointer must not take Enter
    list.append(li);
  }
  sug.active = -1;
  list.hidden = sug.items.length === 0;
  $("q").setAttribute("aria-expanded", String(sug.items.length > 0));
  suggestRtt.push(r.ms);
  suggestKey.push(performance.now() - t0);
  window.__suggest = { rtt: suggestRtt, keystroke: suggestKey, p95: p95(suggestKey), shown: sug.items.length, text };
}

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
$("q").addEventListener("blur", () => setTimeout(() => { if (document.activeElement !== $("q")) closeSuggest(); }, 120)); // refocused since: keep it

// ---------------------------------------------------------------- controls --------------------------------------
$("search-form").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(sTimer); closeSuggest(); state.q = $("q").value; resetAndSearch(); });
$("explain").addEventListener("change", (e) => { state.explain = e.target.checked; search(); });
$("sort").addEventListener("change", (e) => { state.sort = e.target.value; resetAndSearch(); });
$("collapse").addEventListener("change", (e) => { state.collapse = e.target.checked; resetAndSearch(); });
$("clear").addEventListener("click", () => {
  for (const s of Object.values(state.sel)) s.clear();
  for (const r of [...Object.values(state.ranges), ...Object.values(state.nrange)]) { r.min = ""; r.max = ""; }
  for (const s of Object.values(state.nsel)) s.clear();
  state.geo.on = false;
  if ($("geo-on")) $("geo-on").checked = false;
  for (const i of document.querySelectorAll("#dynamic-filters fieldset:not(#geo-box) input[type=number]")) i.value = "";
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
