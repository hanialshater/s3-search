// The demo page: one static page over the serving HTTP API (docs/INTERFACES.md §5 request/response, §7 routes).
// Local only. Query string: ?api=http://host:port (default: same origin) and ?index=NAME (default esci-sample).
"use strict";

const PARAMS = new URLSearchParams(location.search);
const API = (PARAMS.get("api") || "").replace(/\/$/, "");
const INDEX = PARAMS.get("index") || "esci-sample";
const SIZE = 10;
const FIELDS = ["product_title^3", "product_brand^2", "product_color", "product_bullet_point", "product_description"];
const FACETS = { category: "category", brand: "product_brand.keyword" };

const state = { q: "", category: new Set(), brand: new Set(), min: "", max: "", page: 1, explain: false, last: null };
const latencies = []; // browser round trips of page-1 searches (ms), for the p95 the S5 prediction names
let seq = 0;

/** The §5 request for the current state. `after` is the previous page's last hit `sort` (next), else `from` is used. */
function buildRequest(s, page, after) {
  const filter = [];
  for (const [name, field] of Object.entries(FACETS)) {
    if (s[name].size) filter.push({ terms: { [field]: [...s[name]] } });
  }
  const range = {};
  if (s.min !== "" && !isNaN(+s.min)) range.gte = +s.min;
  if (s.max !== "" && !isNaN(+s.max)) range.lte = +s.max;
  if (Object.keys(range).length) filter.push({ range: { price: range } });
  const text = s.q.trim();
  const must = text ? [{ multi_match: { query: text, fields: FIELDS, type: "cross_fields" } }] : [{ match_all: {} }];
  const req = {
    query: { bool: { must, filter } },
    aggs: Object.fromEntries(Object.entries(FACETS).map(([n, f]) => [n, { terms: { field: f, size: 10 } }])),
    size: SIZE,
  };
  if (after) { req.search_after = after; req.from = 0; } else { req.from = (page - 1) * SIZE; }
  if (s.explain) req.explain = true;
  return req;
}

async function post(req) {
  const t0 = performance.now();
  const res = await fetch(`${API}/indexes/${encodeURIComponent(INDEX)}/_search`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(req),
  });
  const body = await res.json().catch(() => ({}));
  return { ok: res.ok, status: res.status, body, ms: performance.now() - t0 };
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

function renderFacet(name, agg) {
  const box = $("facet-" + name).querySelector(".opts");
  box.replaceChildren();
  const buckets = new Map(((agg && agg.buckets) || []).map((b) => [String(b.key), b.doc_count]));
  for (const k of state[name]) if (!buckets.has(k)) buckets.set(k, 0); // a chosen value stays visible, so it can be unchecked
  for (const [key, count] of buckets) {
    const cb = el("input", { type: "checkbox", checked: state[name].has(key) });
    cb.dataset.facet = name; cb.dataset.value = key;
    cb.addEventListener("change", () => {
      cb.checked ? state[name].add(key) : state[name].delete(key);
      state.page = 1; search();
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
    const bits = [src.product_brand, src.category, src.price != null ? "$" + src.price : null].filter(Boolean);
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
  for (const name of Object.keys(FACETS)) renderFacet(name, (body.aggregations || {})[name]);
  $("summary").textContent = total === 0 ? "No results" : `${total.toLocaleString()} results`;
  $("page-label").textContent = total ? `Page ${state.page} of ${Math.max(1, Math.ceil(Math.min(total, 10000) / SIZE))}` : "";
  $("prev").disabled = state.page <= 1;
  $("next").disabled = !(body.hits.hits.length === SIZE && from + SIZE < Math.min(total, 10000));
  if (state.page === 1) latencies.push(r.ms);
  const took = body.took_ms && body.took_ms.total;
  $("timing").textContent = `server ${took != null ? took.toFixed(1) : "?"} ms · browser ${r.ms.toFixed(0)} ms` +
    (latencies.length ? ` · page-1 p95 ${p95(latencies).toFixed(0)} ms (n=${latencies.length})` : "");
  window.__demo = { latencies, p95: p95(latencies), state: { q: state.q, page: state.page }, lastRequest: req };
}

function resetAndSearch() { state.page = 1; state.pendingAfter = null; search(); }

let typing;
$("search-form").addEventListener("submit", (e) => { e.preventDefault(); clearTimeout(typing); state.q = $("q").value; resetAndSearch(); });
$("q").addEventListener("input", () => { clearTimeout(typing); typing = setTimeout(() => { state.q = $("q").value; resetAndSearch(); }, 250); });
$("explain").addEventListener("change", (e) => { state.explain = e.target.checked; search(); });
for (const [id, key] of [["price-min", "min"], ["price-max", "max"]]) {
  $(id).addEventListener("change", (e) => { state[key] = e.target.value; resetAndSearch(); });
}
$("clear").addEventListener("click", () => {
  state.category.clear(); state.brand.clear(); state.min = state.max = "";
  $("price-min").value = $("price-max").value = ""; resetAndSearch();
});
// Next uses search_after (the last hit's `sort`, passed back verbatim); previous uses from/size.
$("next").addEventListener("click", () => {
  const hits = state.last.hits.hits;
  state.pendingAfter = hits[hits.length - 1].sort;
  state.page += 1; search();
});
$("prev").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); state.pendingAfter = null; search(); });

window.DemoCore = { buildRequest, p95 };
search();
