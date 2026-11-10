"""S9-7: the demo's geo filter, sort by distance and nested facets, on the real engine.

Part 1 runs the requests the page builds (the page's own JS, in headless Chromium) on the real `engine.api` over a slice
of ESCI-sample with the synthetic locations and offers (`tools/stack.py`, the kind the steward's s9-v1 suite uses), and
checks the answers against a brute-force computation on the catalogue.

Part 2 drives the page against a real serving replica holding the same index.
"""
import json
import math
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("serving.app", reason="serving not importable")
api = pytest.importorskip("engine.api", reason="engine not importable")

from tools import demo  # noqa: E402
from tools.stack import Stack, catalogue  # noqa: E402

HERE = Path(__file__).parent
N = 1500
EARTH_M = 6371008.7714


def _node(*args, timeout=90):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    if subprocess.run([node, "-e", "require('playwright')"], capture_output=True, cwd=str(HERE)).returncode:
        pytest.skip("playwright for node not installed")
    r = subprocess.run([node, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    if "Executable doesn't exist" in r.stdout + r.stderr:
        pytest.skip("no playwright browser installed")
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    assert r.returncode == 0, out
    return json.loads(out)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_M * math.asin(math.sqrt(a)) / 1000.0


@pytest.fixture(scope="module")
def page_url():
    try:
        st = Stack(300)  # any real index: the page is loaded only for its code
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no real stack: {e}")
    srv = demo.make_server(st.base, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    st.close()


@pytest.fixture(scope="module")
def world(page_url):
    cat, mapping = catalogue(N, fields=True, geo=True)
    index = api.build_index(cat, mapping)
    reqs = _node(HERE / "build_requests.js", page_url)
    return cat, index, reqs


def _geo_ok(row, km=400, lat=40.71, lon=-74.01):
    return haversine_km(lat, lon, row["location"]["lat"], row["location"]["lon"]) <= km


def _offer_ok(row, sellers=("seller-03", "seller-07"), max_price=200.0):
    # one offer must satisfy the seller and the price together (a nested query, not two independent filters)
    return any(o["seller"] in sellers and o["price"] <= max_price for o in row["offers"])


def test_page_requests_run_on_the_real_engine_and_match_brute_force(world):
    cat, index, reqs = world
    rows = cat.to_dict("records")
    ids = {str(r["product_id"]): r for r in rows}
    main = reqs["main"]
    assert "geo_distance" in json.dumps(main["post_filter"]) and "nested" in json.dumps(main["post_filter"])
    assert main["sort"][0]["_geo_distance"]["location"] == {"lat": 40.71, "lon": -74.01}
    resp = index.search({**main, "size": 100})
    want = [r for r in rows if _geo_ok(r) and _offer_ok(r)]
    assert resp["hits"]["total"] == len(want) > 0
    hits = resp["hits"]["hits"]
    assert all(_geo_ok(ids[h["_id"]]) and _offer_ok(ids[h["_id"]]) for h in hits)
    dists = [h["sort"][0] for h in hits]
    assert dists == sorted(dists) and all(abs(d - haversine_km(40.71, -74.01, ids[h["_id"]]["location"]["lat"], ids[h["_id"]]["location"]["lon"])) < 1e-3
                                          for d, h in zip(dists, hits))
    # post_filter: the aggregations in the main answer ignore the selections (the whole match set)
    assert resp["aggregations"]["nested:offers"]["doc_count"] == sum(len(r["offers"]) for r in rows)
    # the nested facet's corrections ignore its own selection but keep the geo filter
    fix = {f["group"]: f["req"] for f in reqs["fixes"]}
    assert set(fix) == {"product_brand.keyword", "offers"}
    near = [r for r in rows if _geo_ok(r)]
    agg = index.search(fix["offers"])["aggregations"]["nested:offers"]
    assert agg["doc_count"] == sum(len(r["offers"]) for r in near)
    seller_counts = {}
    for r in near:
        for o in r["offers"]:
            seller_counts[o["seller"]] = seller_counts.get(o["seller"], 0) + 1
    got = {b["key"]: b["doc_count"] for b in agg["seller"]["buckets"]}
    top = sorted(seller_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    assert got == dict(top)
    assert sum(b["doc_count"] for b in agg["price:range"]["buckets"]) == agg["doc_count"]
    # the brand facet's correction keeps both the geo filter and the nested selection
    both = [r for r in rows if _geo_ok(r) and _offer_ok(r)]
    brand = index.search(fix["product_brand.keyword"])["aggregations"]["product_brand.keyword"]
    counts = {}
    for r in both:
        counts[r["product_brand"]] = counts.get(r["product_brand"], 0) + 1
    assert {b["key"]: b["doc_count"] for b in brand["buckets"]} == dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10])


def test_text_request_with_nested_selection_runs(world):
    cat, index, reqs = world
    resp = index.search({k: v for k, v in reqs["text"].items() if k != "highlight"})
    rows = {str(r["product_id"]): r for r in cat.to_dict("records")}
    assert all(_offer_ok(rows[h["_id"]]) for h in resp["hits"]["hits"])
    assert "geo_distance" not in json.dumps(reqs["text"])  # the box is off: no geo filter is sent
    for f in reqs["textFixes"]:
        assert "aggregations" in index.search(f["req"])


def test_page_against_serving_with_geo_and_nested():
    st = Stack(N, fields=True, geo=True)  # a failure to start is a failure
    srv = demo.make_server(st.base, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = _node(HERE / "browser_real.js", f"http://127.0.0.1:{srv.server_address[1]}/", "geo")
        assert out["ok"]
    finally:
        srv.shutdown()
        st.close()
