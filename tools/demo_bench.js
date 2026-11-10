// Measure the demo page in headless Chromium: autocomplete keystrokes and page-1 searches (with highlights and
// multi-select facets). Usage: NODE_PATH=$(npm root -g) node tools/demo_bench.js <page url> <queries.json> <out.json>
// Two clocks for each: the page's own (input event to rendered list; fetch round trip) and the action-to-render time
// seen from outside (Playwright). CPU, development only. The evaluation steward judges the S7 prediction, not this.
"use strict";
const fs = require("fs");
const { chromium } = require("playwright");

const [url, queriesPath, outPath] = process.argv.slice(2);
const queries = JSON.parse(fs.readFileSync(queriesPath, "utf8"));
const pct = (xs, q) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)]; };
const stats = (xs) => ({ n: xs.length, p50: pct(xs, 0.5), p95: pct(xs, 0.95), p99: pct(xs, 0.99), max: xs.length ? Math.max(...xs) : null });

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror " + e));
  page.on("response", async (r) => { if (r.request().method() === "POST" && r.status() >= 400) errors.push(r.status() + " " + (await r.text()).slice(0, 160)); });
  await page.goto(url);
  await page.waitForFunction(() => window.__demo || !document.getElementById("error").hidden, null, { timeout: 60000 });

  const search = { pageRtt: [], action: [] }, auto = { page: [], rtt: [], action: [] };
  const settle = async (fn, kind) => {
    await page.evaluate(() => { window.__mark = window.__demo; document.getElementById("error").hidden = true; });
    const t0 = Date.now();
    await fn();
    await page.waitForFunction(() => window.__demo !== window.__mark || !document.getElementById("error").hidden, null, { timeout: 30000, polling: "raf" });
    const ms = Date.now() - t0;
    const err = await page.evaluate(() => document.getElementById("error").hidden ? null : document.getElementById("error").textContent);
    if (err) errors.push(`${kind}: ${err}`);
    const rtt = await page.evaluate(() => window.__demo.latencies[window.__demo.latencies.length - 1]);
    search.action.push(ms); if (rtt != null) search.pageRtt.push(rtt);
  };

  let searches = 0, facetClicks = 0, keystrokes = 0, withMarks = 0;
  for (const q of queries) {
    try {
    // autocomplete: type the first characters one at a time, waiting for each suggestion list
    await page.fill("#q", "");
    const head = q.slice(0, 6);
    for (const ch of head) {
      const n0 = await page.evaluate(() => (window.__suggest ? window.__suggest.keystroke.length : 0));
      const t0 = Date.now();
      await page.keyboard.type(ch);
      try {
        await page.waitForFunction((n) => window.__suggest && window.__suggest.keystroke.length > n, n0, { timeout: 5000, polling: "raf" });
      } catch (e) { errors.push(`autocomplete: no answer for "${head}"`); }
      auto.action.push(Date.now() - t0);
      keystrokes++;
    }
    // the search, with highlights
    await settle(async () => { await page.fill("#q", q); await page.press("#q", "Enter"); }, "search");
    searches++;
    withMarks += (await page.$$eval("#results .title mark", (l) => l.length)) > 0 ? 1 : 0;
    // multi-select facets (post_filter): two brands
    const nBoxes = await page.locator("#filters input[type=checkbox]").count();
    for (let i = 0; i < Math.min(2, nBoxes); i++) {  // re-find each box: the facets re-render after every answer
      await settle(() => page.locator("#filters input[type=checkbox]").nth(i).click(), "facet"); facetClicks++;
    }
    if (nBoxes) { await settle(() => page.click("#clear"), "clear"); }
    } catch (e) {
      const st = await page.evaluate(() => [document.getElementById("suggest").hidden, document.getElementById("suggest").children.length, document.activeElement.id, document.getElementById("q").value]).catch(() => null);
      errors.push(`driver: query ${JSON.stringify(q)}: ${String(e).split("\n")[0]} state ${JSON.stringify(st)}`);
      await page.keyboard.press("Escape").catch(() => {});
    }
  }
  const sug = await page.evaluate(() => ({ key: window.__suggest.keystroke, rtt: window.__suggest.rtt }));
  const out = {
    kind: "demo-bench", label: "CPU, development only", url,
    n_queries: queries.length, searches, facetClicks, keystrokes, searchesWithMarks: withMarks,
    autocomplete: { page_keystroke_to_render_ms: stats(sug.key), page_fetch_rtt_ms: stats(sug.rtt), playwright_action_ms: stats(auto.action) },
    page1: { page_fetch_rtt_ms: stats(search.pageRtt), playwright_action_ms: stats(search.action) },
    errors,
  };
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1));
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ fail: String(e) })); process.exit(1); });
