// S7 prediction 5, the steward's own browser run (owner: evaluation-steward, P6). CPU, development only.
//   NODE_PATH=$(npm root -g) node referee/judge_s7p5.js <page url> <queries.json> <out.json>
// Written before reading the tools-engineer's own bench or its numbers. Clocks (all in Node, Date-free: performance.now):
// - autocomplete: each of the first 12 characters of a query is pressed one at a time; a keystroke is timed from the
//   press until the suggestion list is rendered for exactly that prefix (a new render, with the prefix as its text).
//   The next key waits for that render (a user pausing, so answers never pile up); the page's 40 ms debounce is in it.
//   The page's own clock (window.__suggest.keystroke: input event to rendered list) is recorded alongside.
// - page 1: from the action (Enter on the full query; a facet click) until the page rendered the answer
//   (window.__demo replaced) or showed an error. The page's own fetch round trip (window.__demo.latencies) alongside.
// - multi-select: two values of one facet are checked in turn; the request must carry post_filter.
// - highlights: every search request must ask for them (lastRequest.highlight) and results show <mark>ed words.
// - errors: the page's error box, page errors and console errors, over every action.
"use strict";
const fs = require("fs");
const { performance } = require("perf_hooks");
const { chromium } = require("playwright");

const [url, queriesPath, outPath] = process.argv.slice(2);
const queries = JSON.parse(fs.readFileSync(queriesPath, "utf8"));
const pct = (xs, q) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)]; };

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const pageErrors = [], consoleErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  await page.goto(url);
  await page.waitForFunction(() => window.__demo || !document.getElementById("error").hidden, null, { timeout: 60000 });
  const errorText = () => page.evaluate(() => document.getElementById("error").hidden ? null : document.getElementById("error").textContent);
  const first = await errorText();

  const settle = async (fn) => {
    await page.evaluate(() => { window.__mark = window.__demo; document.getElementById("error").hidden = true; });
    const t0 = performance.now();
    await fn();
    await page.waitForFunction(() => window.__demo !== window.__mark || !document.getElementById("error").hidden,
                               null, { timeout: 30000, polling: "raf" });
    return { ms: performance.now() - t0, err: await errorText() };
  };

  const ac = [], p1 = [], turns = [], errors = [], facetRows = [];
  let highlightReq = 0, searches = 0, markedResults = 0, multiSelect = 0, multiSelectWithPostFilter = 0;
  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    // autocomplete, keystroke by keystroke
    await page.click("#clear");
    await page.fill("#q", "");
    await page.focus("#q");
    for (const ch of q.slice(0, 12)) {
      const before = await page.evaluate(() => (window.__suggest ? window.__suggest.rtt.length : 0));
      const t0 = performance.now();
      await page.keyboard.type(ch);
      const want = await page.evaluate(() => document.getElementById("q").value.trim());
      if (!want) continue;   // only spaces typed: the page sends nothing
      try {
        await page.waitForFunction(([b, w]) => (window.__suggest && window.__suggest.rtt.length > b && window.__suggest.text === w)
                                     || !document.getElementById("error").hidden, [before, want], { timeout: 10000, polling: "raf" });
        const err = await errorText();
        if (err) errors.push({ kind: "autocomplete", q, err });
        else ac.push(performance.now() - t0);
      } catch (e) { errors.push({ kind: "autocomplete", q, err: "no answer within 10 s for " + JSON.stringify(want) }); }
    }
    // the search, page 1, with highlights
    await page.fill("#q", q);
    let r = await settle(() => page.press("#q", "Enter"));
    if (r.err) { errors.push({ kind: "search", q, err: r.err }); continue; }
    p1.push(r.ms); searches++;
    const info = await page.evaluate(() => ({ hl: !!(window.__demo.lastRequest || {}).highlight,
                                              marked: document.querySelectorAll("#results li mark").length,
                                              n: document.querySelectorAll("#results li").length }));
    highlightReq += info.hl ? 1 : 0;
    markedResults += info.marked > 0 ? 1 : 0;
    // multi-select: two values of the first facet group that has two
    const groups = await page.$$eval(".opts[data-field]", (gs) => gs.map((g) => [g.dataset.field, g.querySelectorAll("input[type=checkbox]").length]));
    const g = groups.find(([, n]) => n >= 2);
    if (g) {
      const sel = `.opts[data-field="${g[0]}"] input[type=checkbox]`;
      r = await settle(() => page.locator(sel).nth(0).click());
      if (r.err) errors.push({ kind: "facet", q, err: r.err }); else p1.push(r.ms);
      r = await settle(() => page.locator(sel).nth(1).click());
      if (r.err) errors.push({ kind: "facet", q, err: r.err }); else p1.push(r.ms);
      const pf = await page.evaluate(() => (window.__demo.lastRequest || {}).post_filter);
      multiSelect++;
      if (pf) multiSelectWithPostFilter++;
      facetRows.push({ field: g[0], post_filter: !!pf });
    }
    // a page turn, every second query (not part of the page-1 measure)
    if (qi % 2 === 0 && !(await page.isDisabled("#next"))) {
      r = await settle(() => page.click("#next"));
      if (r.err) errors.push({ kind: "next", q, err: r.err }); else turns.push(r.ms);
    }
  }
  const own = await page.evaluate(() => ({ p1: window.__demo ? window.__demo.latencies : [],
                                           ac: window.__suggest ? window.__suggest.keystroke : [] }));
  const out = {
    first_load_error: first, queries: queries.length, searches, errors: errors.length, error_examples: errors.slice(0, 5),
    page_errors: pageErrors.slice(0, 5), console_errors: consoleErrors.slice(0, 5),
    autocomplete_ms_judge: { n: ac.length, p50: pct(ac, 0.5), p95: pct(ac, 0.95), max: pct(ac, 1) },
    autocomplete_ms_page_clock: { n: own.ac.length, p50: pct(own.ac, 0.5), p95: pct(own.ac, 0.95) },
    page1_ms_judge: { n: p1.length, p50: pct(p1, 0.5), p95: pct(p1, 0.95), max: pct(p1, 1) },
    page1_ms_page_clock: { n: own.p1.length, p50: pct(own.p1, 0.5), p95: pct(own.p1, 0.95) },
    page_turn_ms_judge: { n: turns.length, p50: pct(turns, 0.5), p95: pct(turns, 0.95) },
    highlights: { searches_requesting: highlightReq, searches_with_marked_results: markedResults, searches },
    multi_select: { done: multiSelect, with_post_filter: multiSelectWithPostFilter, fields: [...new Set(facetRows.map((f) => f.field))] },
  };
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + "\n");
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
