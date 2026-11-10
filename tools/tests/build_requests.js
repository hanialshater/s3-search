// Print the requests the demo page builds for a geo + nested state, as JSON, so a Python test can run them on the real
// engine. Usage: node build_requests.js <page url>. The page is loaded (against any real index) only to get its code.
const { chromium } = require("playwright");

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto(process.argv[2]);
  await page.waitForFunction(() => window.DemoCore);
  const out = await page.evaluate(() => {
    const sch = {
      text: ["product_title^3", "product_brand^2"], title: true,
      facets: [{ field: "product_brand.keyword", label: "brand" }], numerics: [], brand: "product_brand.keyword",
      geo: { field: "location", label: "location" },
      nested: [{ path: "offers", subs: [
        { sub: "seller", field: "offers.seller", kind: "keyword" },
        { sub: "price", field: "offers.price", kind: "number", min: 1, max: 500,
          ranges: [{ to: 100 }, { from: 100, to: 200 }, { from: 200 }] }] }],
    };
    const state = (q, geoOn, sort) => ({
      q, sel: { "product_brand.keyword": new Set() }, ranges: {},
      nsel: { "offers.seller": new Set(["seller-03", "seller-07"]) }, nrange: { "offers.price": { min: "", max: "200" } },
      geo: { on: geoOn, lat: "40.71", lon: "-74.01", km: "400" }, sort, collapse: false, explain: false,
    });
    const s = state("", true, "geo");
    const s2 = state("shoes", false, "");
    return { schema: sch, main: DemoCore.buildRequest(s, 1, null, sch), fixes: DemoCore.correctionRequests(s, sch),
             text: DemoCore.buildRequest(s2, 1, null, sch), textFixes: DemoCore.correctionRequests(s2, sch) };
  });
  console.log(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.log(JSON.stringify({ fail: String(e) })); process.exit(1); });
