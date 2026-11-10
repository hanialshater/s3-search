// The demo page against an index that lacks category and price (like ESCI-sample as served): the page must discover
// that, render only what exists, and make no request the index rejects. Usage: node browser_check_esci.js URL
const { chromium } = require("playwright");

(async () => {
  const url = process.argv[2];
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.setDefaultTimeout(5000);
  const bad = [];
  const sent = [];
  page.on("pageerror", (e) => bad.push(String(e)));
  page.on("response", async (r) => { if (r.request().method() === "POST" && r.status() >= 400 && r.request().postData().includes("_source")) bad.push(r.status() + " " + (await r.text())); });
  page.on("request", (r) => { if (r.method() === "POST") sent.push(r.postData()); });
  const fail = (msg) => { console.log(JSON.stringify({ fail: msg, bad })); process.exit(1); };
  const settle = async (fn) => {
    await page.evaluate(() => { window.__mark = window.__demo; });
    await fn();
    await page.waitForFunction(() => window.__demo !== window.__mark || !document.getElementById("error").hidden);
    if (await page.$("#error:not([hidden])")) fail("error shown: " + await page.textContent("#error"));
  };
  await page.goto(url);
  await page.waitForFunction(() => window.__demo || !document.getElementById("error").hidden);
  if (await page.$("#error:not([hidden])")) fail("first search failed: " + await page.textContent("#error"));
  const schema = await page.evaluate(() => window.__demo.schema);
  const facets = schema.facets.map((f) => f.field);
  if (facets.includes("category") || schema.numerics.length) fail("assumed fields: " + JSON.stringify(schema));
  if (!facets.includes("product_brand.keyword")) fail("brand facet not discovered: " + JSON.stringify(schema));
  if (await page.$("#price-max")) fail("price input rendered");
  const titles = await page.$$eval("#results .title", (l) => l.map((x) => x.textContent));
  if (!titles.length || titles.some((t) => /^P\d+$/.test(t))) fail("titles not shown: " + titles.slice(0, 2));
  if (!sent.some((b) => b.includes('"_source":true'))) fail("_source not requested");

  let actions = 0;
  for (const q of ["running shoe", "winter jacket", "desk lamp", "smart phone"]) {
    await settle(async () => { await page.fill("#q", q); await page.press("#q", "Enter"); }); actions++;
    await settle(() => page.click('input[data-facet="product_brand.keyword"]')); actions++;
    if (await page.isEnabled("#next")) { await settle(() => page.click("#next")); actions++; }
    await settle(() => page.click("#clear")); actions++;
  }
  if (bad.length) fail("errors during actions");
  console.log(JSON.stringify({ ok: true, actions, facets }));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ fail: String(e).split("\n").slice(0, 3).join(" ") })); process.exit(1); });
