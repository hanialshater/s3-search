// Headless check of the demo page against a fake server. Usage: node browser_check.js http://127.0.0.1:PORT/
// Prints one JSON object of results; exits 1 on the first failed expectation.
const { chromium } = require("playwright");

let PAGE;
(async () => {
  const url = process.argv[2];
  const browser = await chromium.launch();
  const page = PAGE = await browser.newPage();
  const reqs = [];
  page.on("request", (r) => { if (r.method() === "POST") reqs.push(JSON.parse(r.postData())); });
  page.setDefaultTimeout(5000);
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const check = (ok, msg) => { if (!ok) { console.log(JSON.stringify({ fail: msg, errors })); process.exit(1); } };
  const wf = page.waitForFunction.bind(page);
  let n = 0;
  page.waitForFunction = async (...a) => { n++; try { return await wf(...a); } catch (e) { throw new Error("wait #" + n + " " + String(a[0]).slice(0, 90)); } };
  const ids = () => page.$$eval("#results > li", (l) => l.map((x) => x.dataset.id));

  await page.goto(url);
  await page.waitForSelector("#results > li");
  check((await ids()).length === 10, "10 hits on load");
  const total0 = await page.textContent("#summary");

  // search
  await page.fill("#q", "running shoe");
  await page.press("#q", "Enter");
  await page.waitForFunction(() => /results/.test(document.getElementById("summary").textContent) &&
    window.__demo && window.__demo.state.q === "running shoe");
  const shoes = await page.$$eval("#results .meta", (l) => l.map((x) => x.textContent));
  check(shoes.length > 0 && shoes.every((t) => t.includes("Shoes")), "search returns shoes: " + shoes.slice(0, 2));

  // facet checkbox (category) narrows and sends a terms filter
  await page.fill("#q", "");
  await page.press("#q", "Enter");
  await page.waitForFunction(() => window.__demo.state.q === "");
  await page.check('input[data-facet="category"][data-value="Lamps"]');
  await page.waitForFunction(() => window.__demo.lastRequest.query.bool.filter.length === 1);
  await page.waitForFunction(() => [...document.querySelectorAll("#results .meta")].every((x) => x.textContent.includes("Lamps")));
  const lamps = await ids();
  check(lamps.length > 0, "lamps shown");
  const brandBoxes = await page.$$eval(`input[data-facet="product_brand.keyword"]`, (l) => l.length);
  check(brandBoxes > 0, "brand facet rendered");

  // price range
  await page.fill("#price-max", "40");
  await page.dispatchEvent("#price-max", "change");
  await page.waitForFunction(() => JSON.stringify(window.__demo.lastRequest).includes('"lte":40'));
  const prices = await page.$$eval("#results .meta", (l) => l.map((x) => +x.textContent.match(/\$(\d+(\.\d+)?)/)[1]));
  check(prices.every((p) => p <= 40), "price <= 40: " + prices);

  // clear, then paginate: next uses search_after, previous uses from/size
  await page.click("#clear");
  await page.waitForFunction(() => window.__demo.lastRequest.query.bool.filter.length === 0);
  const p1 = await ids();
  await page.click("#next");
  await page.waitForFunction(() => window.__demo.state.page === 2);
  const nextReq = reqs[reqs.length - 1];
  check(Array.isArray(nextReq.search_after) && nextReq.from === 0, "next uses search_after: " + JSON.stringify(nextReq.search_after));
  const p2 = await ids();
  check(p2.length === 10 && !p2.some((i) => p1.includes(i)), "page 2 is disjoint from page 1");
  await page.click("#prev");
  await page.waitForFunction(() => window.__demo.state.page === 1);
  const prevReq = reqs[reqs.length - 1];
  check(prevReq.from === 0 && !("search_after" in prevReq), "previous uses from/size");
  check(JSON.stringify(await ids()) === JSON.stringify(p1), "back on page 1 shows the same hits");
  // from/size: page 3 by two nexts equals a direct from=20 request
  await page.click("#next"); await page.waitForFunction(() => window.__demo.state.page === 2);
  await page.click("#next"); await page.waitForFunction(() => window.__demo.state.page === 3);
  const p3 = await ids();
  const direct = await page.evaluate(async () => {
    const r = await fetch("/indexes/esci-sample/_search", { method: "POST", body: JSON.stringify({ size: 10, from: 20 }) });
    return (await r.json()).hits.hits.map((h) => h._id);
  });
  check(JSON.stringify(p3) === JSON.stringify(direct), "search_after page 3 equals from=20");

  // explain toggle
  check((await page.$$("pre.explain")).length === 0, "no explain before the toggle");
  await page.check("#explain");
  await page.waitForSelector("pre.explain", { state: "attached" });
  check(reqs[reqs.length - 1].explain === true, "explain:true sent");
  const text = await page.textContent("pre.explain");
  check(/sum of|match_all/.test(text), "explain tree shown: " + text.slice(0, 60));

  // errors from the API are shown, not swallowed
  await page.goto(url + "?index=nope");
  await page.waitForSelector("#error:not([hidden])");
  check(/404/.test(await page.textContent("#error")), "404 shown");

  const lat = await (async () => { await page.goto(url); await page.waitForSelector("#results > li");
    return page.evaluate(() => window.__demo.latencies.length); })();
  console.log(JSON.stringify({ ok: true, total0, latencySamples: lat, errors }));
  await browser.close();
  process.exit(errors.length ? 1 : 0);
})().catch(async (e) => {
  const st = PAGE ? await PAGE.evaluate(() => [document.getElementById("error").textContent, document.getElementById("summary").textContent, document.getElementById("next").disabled, JSON.stringify(window.__demo && window.__demo.state)]).catch(() => null) : null;
  console.log(JSON.stringify({ fail: String(e).split("\n").slice(0, 3).join(" "), state: st })); process.exit(1); });
