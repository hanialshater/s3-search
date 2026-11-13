// The steward's S5-b judging run: headless Chrome (Playwright) drives the demo page against a real serving replica.
// Every number is CPU, development only. Usage (from referee/judge.py s5b):
//   NODE_PATH=$(npm root -g) node referee/judge_s5b.js <page url> <queries.json> <out.json>
// Page-1 actions are searches (typed and submitted) and facet clicks; page turns use Next (search_after).
// Two clocks per action: the page's own fetch round trip (window.__demo.latencies, what the page reports) and the
// judge's action-to-render time (from the click or Enter until the page has rendered the answer).
"use strict";
const fs = require("fs");
const { chromium } = require("playwright");

const [url, queriesPath, outPath] = process.argv.slice(2);
const queries = JSON.parse(fs.readFileSync(queriesPath, "utf8"));

function pct(xs, q) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)];
}

(async () => {
  const browser = await chromium.launch({ headless: true });   // Playwright's bundled Chromium
  const page = await browser.newPage();
  const consoleErrors = [];
  page.on("pageerror", (e) => consoleErrors.push(String(e)));
  await page.goto(url);
  const settle = async (fn) => {   // run an action, wait until the page rendered a new answer (or an error)
    await page.evaluate(() => { window.__mark = window.__demo; document.getElementById("error").hidden = true; });
    const t0 = Date.now();
    await fn();
    await page.waitForFunction(() => window.__demo !== window.__mark || !document.getElementById("error").hidden,
                               null, { timeout: 30000, polling: "raf" });
    const ms = Date.now() - t0;
    const err = await page.evaluate(() => document.getElementById("error").hidden ? null
                                     : document.getElementById("error").textContent);
    return { ms, err };
  };
  // the page's first search (match_all) on load
  await page.waitForFunction(() => window.__demo || !document.getElementById("error").hidden, null, { timeout: 30000 });
  const first = await page.evaluate(() => document.getElementById("error").hidden ? null
                                     : document.getElementById("error").textContent);
  const actions = [];
  for (let i = 0; i < queries.length; i++) {
    const q = queries[i];
    let r = await settle(async () => { await page.fill("#q", q); await page.press("#q", "Enter"); });
    actions.push({ kind: "search", page1: true, q, ...r });
    if (i % 3 === 0) {   // a facet click (page 1)
      const box = await page.$("#filters input[type=checkbox]");
      if (box) {
        r = await settle(() => box.click());
        actions.push({ kind: "facet", page1: true, q, ...r });
        r = await settle(() => page.click("#clear"));
        actions.push({ kind: "facet-clear", page1: true, q, ...r });
      } else {
        actions.push({ kind: "facet", page1: true, q, ms: null, err: "no facet value on the page" });
      }
    }
    if (i % 2 === 0 && !(await page.isDisabled("#next"))) {   // page turns
      r = await settle(() => page.click("#next"));
      actions.push({ kind: "next", page1: false, q, ...r });
      r = await settle(() => page.click("#prev"));
      actions.push({ kind: "prev", page1: true, q, ...r });
    }
  }
  const demo = await page.evaluate(() => window.__demo ? { latencies: window.__demo.latencies, p95: window.__demo.p95 }
                                                        : null);
  const ok = actions.filter((a) => a.ms != null && !a.err);
  const p1 = ok.filter((a) => a.page1).map((a) => a.ms);
  const turns = ok.filter((a) => !a.page1).map((a) => a.ms);
  const out = {
    first_load_error: first, console_errors: consoleErrors.slice(0, 5),
    n_actions: actions.length, n_errors: actions.filter((a) => a.err).length,
    errors: actions.filter((a) => a.err).slice(0, 5),
    by_kind: Object.fromEntries(["search", "facet", "facet-clear", "next", "prev"].map((k) =>
      [k, actions.filter((a) => a.kind === k && a.ms != null && !a.err).length])),
    judge_page1_ms: { n: p1.length, p50: pct(p1, 0.5), p95: pct(p1, 0.95), max: pct(p1, 1) },
    judge_page_turn_ms: { n: turns.length, p50: pct(turns, 0.5), p95: pct(turns, 0.95) },
    page_reported: demo && { n: demo.latencies.length, p95: demo.p95, p50: pct(demo.latencies, 0.5) },
  };
  fs.writeFileSync(outPath, JSON.stringify(out, null, 1) + "\n");
  console.log(JSON.stringify({ ...out, errors: undefined }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
