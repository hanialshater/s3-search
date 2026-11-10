// Headless check of the demo page against a REAL serving replica (tools/stack.py), no fakes.
// Usage: node browser_real.js URL plain|fields      plain: ESCI-sample as served (no price, no category)
//                                                   fields: with synthetic price, stock and category
// Prints one JSON object; exits 1 on the first failed expectation.
const { chromium } = require("playwright");

let PAGE;
(async () => {
  const [url, mode] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = PAGE = await browser.newPage();
  page.setDefaultTimeout(8000);
  const bad = [], reqs = [];
  page.on("pageerror", (e) => bad.push(String(e)));
  page.on("response", async (r) => { if (r.request().method() === "POST" && r.status() >= 400) bad.push(r.status() + " " + (await r.text()).slice(0, 200)); });
  page.on("request", (r) => { if (r.method() === "POST" && r.url().endsWith("_search")) reqs.push(JSON.parse(r.postData())); });
  const check = (ok, msg) => { if (!ok) { console.log(JSON.stringify({ fail: msg, bad })); process.exit(1); } };
  const settle = async (fn) => {
    await page.evaluate(() => { window.__mark = window.__demo; });
    await fn();
    await page.waitForFunction(() => window.__demo !== window.__mark || !document.getElementById("error").hidden);
    check(!(await page.$("#error:not([hidden])")), "error shown: " + (await page.textContent("#error")));
  };
  const lastReq = () => reqs[reqs.length - 1];
  const metas = () => page.$$eval("#results .meta", (l) => l.map((x) => x.textContent));
  const prices = async () => (await metas()).map((t) => +t.match(/\$(\d+(\.\d+)?)/)[1]);
  const summaryTotal = async () => +(await page.textContent("#summary")).replace(/,/g, "").match(/^(\d+)/)[1];

  await page.goto(url);
  await page.waitForSelector("#results > li");
  const schema = await page.evaluate(() => window.__demo.schema);
  const facets = schema.facets.map((f) => f.field);
  check(facets.includes("product_brand.keyword"), "brand facet: " + JSON.stringify(facets));
  if (mode === "plain") {
    check(!facets.includes("category") && schema.numerics.length === 0, "assumed fields: " + JSON.stringify(schema));
    check(await page.$eval("#sort-box", (e) => e.hidden), "sort control hidden without numeric fields");
  } else {
    check(schema.numerics.map((n) => n.field).includes("price") && facets.includes("category"), "fields discovered: " + JSON.stringify(schema));
    check(!(await page.$eval("#sort-box", (e) => e.hidden)), "sort control shown");
  }

  // ---- autocomplete: match_bool_prefix on the title, highlighted, keyboard-selectable
  await page.click("#q");
  await page.type("#q", "run", { delay: 25 });
  await page.waitForSelector("#suggest li");
  const sugg = await page.$$eval("#suggest li", (l) => l.map((x) => ({ t: x.textContent, marks: x.querySelectorAll("mark").length })));
  check(sugg.length > 0 && sugg.length <= 6 && sugg.every((s) => s.marks > 0), "suggestions with highlights: " + JSON.stringify(sugg.slice(0, 2)));
  check(sugg.every((s) => !/[⟦⟧]/.test(s.t)), "raw highlight markers leaked into suggestions");
  const sreq = await page.evaluate(() => window.__suggest.shown);
  check(sreq > 0, "suggest metrics recorded");
  await page.keyboard.press("ArrowDown");
  const pickTitle = sugg[0].t;
  await settle(() => page.keyboard.press("Enter"));
  check((await page.inputValue("#q")) === pickTitle, "picking a suggestion fills the box");
  check(await page.$eval("#suggest", (e) => e.hidden), "suggestions closed after picking");

  // ---- a search with highlights
  await settle(async () => { await page.fill("#q", "red shoes"); await page.press("#q", "Enter"); });
  check("highlight" in lastReq() && lastReq().highlight.fields.product_title, "highlight requested");
  const marked = await page.$$eval("#results .title", (l) => l.map((x) => ({ marks: x.querySelectorAll("mark").length, t: x.textContent })));
  check(marked.length > 0 && marked.filter((m) => m.marks > 0).length >= marked.length / 2, "most result titles highlighted: " + JSON.stringify(marked.map((m) => m.marks)));
  check(marked.every((m) => !/[⟦⟧]/.test(m.t)), "raw markers leaked into titles");
  const markText = await page.$$eval("#results mark", (l) => l.map((x) => x.textContent.toLowerCase()));
  check(markText.every((t) => /red|shoe/.test(t)), "marks are query terms: " + markText.slice(0, 5));
  const total0 = await summaryTotal();

  // ---- multi-select facets via post_filter; a facet's counts ignore its own selection
  const brandBoxes = async () => page.$$eval('input[data-facet="product_brand.keyword"]', (l) => l.map((x) => x.dataset.value));
  const before = await brandBoxes();
  check(before.length >= 3, "enough brands: " + before);
  const beforeCounts = await page.$$eval('.opts[data-field="product_brand.keyword"] .count', (l) => l.map((x) => +x.textContent));
  await settle(() => page.check(`input[data-facet="product_brand.keyword"][data-value="${before[0]}"]`));
  await settle(() => page.check(`input[data-facet="product_brand.keyword"][data-value="${before[1]}"]`));
  const pf = lastReq().post_filter;
  check(pf && JSON.stringify(pf).includes(before[0]) && JSON.stringify(pf).includes(before[1]), "post_filter has both brands: " + JSON.stringify(pf));
  check(!JSON.stringify(lastReq().query).includes(before[0]), "selection is not in the query");
  const metaBrands = (await metas()).map((t) => t.split(" · ")[0]);
  check(metaBrands.length > 0 && metaBrands.every((b) => before.slice(0, 2).includes(b)), "results limited to the two brands: " + metaBrands);
  const totalSel = await summaryTotal();
  check(totalSel <= total0 && totalSel === beforeCounts[0] + beforeCounts[1], `total ${totalSel} = counts ${beforeCounts[0]} + ${beforeCounts[1]}`);
  const after = await brandBoxes();
  check(before.every((b) => after.includes(b)), "the brand facet still lists the other brands (counts ignore its own selection)");
  await settle(() => page.click("#clear"));

  // ---- collapse by brand: one hit per brand, from/size paging
  await settle(() => page.check("#collapse"));
  check(lastReq().collapse && lastReq().collapse.field === "product_brand.keyword", "collapse requested");
  const brands1 = (await metas()).map((t) => t.split(" · ")[0]);
  check(new Set(brands1).size === brands1.length, "one result per brand: " + brands1);
  if (await page.isEnabled("#next")) {
    await settle(() => page.click("#next"));
    check(!("search_after" in lastReq()) && lastReq().from === 10, "collapsed paging uses from/size");
    const brands2 = (await metas()).map((t) => t.split(" · ")[0]);
    check(brands2.every((b) => !brands1.includes(b)), "page 2 has other brands");
    await settle(() => page.click("#prev"));
  }
  await settle(() => page.uncheck("#collapse"));
  check(!("collapse" in lastReq()), "collapse off");

  // ---- pagination: Next is search_after, Previous is from/size
  const p1 = await page.$$eval("#results > li", (l) => l.map((x) => x.dataset.id));
  await settle(() => page.click("#next"));
  check(Array.isArray(lastReq().search_after) && lastReq().from === 0, "next uses search_after");
  const p2 = await page.$$eval("#results > li", (l) => l.map((x) => x.dataset.id));
  check(p2.length === 10 && !p2.some((i) => p1.includes(i)), "page 2 disjoint");
  await settle(() => page.click("#prev"));
  check(JSON.stringify(await page.$$eval("#results > li", (l) => l.map((x) => x.dataset.id))) === JSON.stringify(p1), "previous returns page 1");

  if (mode === "fields" || mode === "geo") {
    // ---- sort control from the mapping's sortable fields
    await settle(() => page.selectOption("#sort", "price:asc"));
    let ps = await prices();
    check(ps.length > 1 && ps.every((p, i) => i === 0 || p >= ps[i - 1]), "price ascending: " + ps);
    await settle(() => page.selectOption("#sort", "price:desc"));
    ps = await prices();
    check(ps.every((p, i) => i === 0 || p <= ps[i - 1]), "price descending: " + ps);
    await settle(() => page.selectOption("#sort", ""));

    // ---- range and histogram facets: clicking sets the range; post_filter keeps the facet whole
    const barSum = () => page.$$eval("fieldset[data-field=price] .histogram .bar", (l) => l.reduce((a, x) => a + +x.dataset.count, 0));
    const rangeSum = () => page.$$eval("fieldset[data-field=price] .ranges .chip", (l) => l.reduce((a, x) => a + +x.textContent.match(/\((\d+)\)$/)[1], 0));
    const all = await summaryTotal();
    check((await barSum()) === all && (await rangeSum()) === all, `histogram ${await barSum()} and range ${await rangeSum()} cover all ${all}`);
    await settle(() => page.click("fieldset[data-field=price] .ranges .chip >> nth=1"));
    const lo = +(await page.inputValue("#price-min")), hi = +(await page.inputValue("#price-max"));
    check(lastReq().post_filter && JSON.stringify(lastReq().post_filter).includes('"range"'), "range selection is a post_filter");
    ps = await prices();
    check(ps.length > 0 && ps.every((p) => p >= lo && p <= hi), `prices within ${lo}..${hi}: ${ps}`);
    check((await barSum()) === all, "histogram still covers the whole match set");
    await settle(() => page.click("fieldset[data-field=price] .histogram .bar >> nth=0"));
    check((await page.inputValue("#price-max")) !== "", "histogram bar sets the range");
    await settle(() => page.click("#clear"));

    // ---- cross-facet counts: with a category chosen, the price histogram counts that category only
    await settle(() => page.check('input[data-facet="category"][data-value="toys"]'));
    const catTotal = await summaryTotal();
    await page.waitForFunction((n) => [...document.querySelectorAll("fieldset[data-field=price] .histogram .bar")].reduce((a, x) => a + +x.dataset.count, 0) === n, catTotal);
    const catCounts = await page.$$eval('.opts[data-field="category"] .count', (l) => l.map((x) => +x.textContent));
    check(catCounts.length >= 3, "the category facet lists the other categories");
  }

  if (mode === "geo") {
    // ---- S9-7: geo filter, sort by distance, nested facets (needs serving to accept geo_point and nested)
    check(!!(await page.$("#geo-box")) && !!(await page.$("fieldset.nested")), "geo box and nested facets shown");
    await settle(() => page.click("#clear"));
    await page.selectOption("#geo-place", "2");   // London
    await settle(() => page.check("#geo-on"));
    check(JSON.stringify(lastReq().post_filter).includes("geo_distance"), "geo_distance in post_filter");
    const near = await summaryTotal();
    await settle(() => page.selectOption("#sort", "geo"));
    check(lastReq().sort[0]._geo_distance, "sorted by distance");
    const km = (await metas()).map((t) => +t.match(/(\d+(\.\d+)?) km away/)[1]);
    check(km.length > 0 && km.every((d, i) => i === 0 || d >= km[i - 1]) && km.every((d) => d <= 50.0001), "nearest first, within 50 km: " + km);
    await settle(() => page.check('fieldset.nested input[type=checkbox] >> nth=0'));
    check(JSON.stringify(lastReq().post_filter).includes('"nested"'), "nested query in post_filter");
    check((await summaryTotal()) <= near, "the nested selection narrows the geo results");
    await settle(() => page.click("#clear"));
  }

  check(!(await page.$("#error:not([hidden])")), "an error is showing");
  check(bad.length === 0, "errors during the run");
  const sug = await page.evaluate(() => window.__suggest.p95);
  console.log(JSON.stringify({ ok: true, mode, suggestP95: sug, requests: reqs.length }));
  await browser.close();
})().catch(async (e) => {
  const st = PAGE ? await PAGE.evaluate(() => [document.getElementById("error").textContent, document.getElementById("summary").textContent]).catch(() => null) : null;
  console.log(JSON.stringify({ fail: String(e).split("\n").slice(0, 3).join(" "), state: st }));
  process.exit(1);
});
