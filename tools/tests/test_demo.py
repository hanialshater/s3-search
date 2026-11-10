"""The demo page against a REAL serving replica over a slice of ESCI-sample (tools/stack.py): no fakes."""
import json
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("serving.app", reason="serving not importable")
pytest.importorskip("engine.api", reason="engine not importable")

from tools import demo  # noqa: E402
from tools.stack import Stack  # noqa: E402

HERE = Path(__file__).parent
N_PRODUCTS = 1500


def _serve(fields):
    try:
        st = Stack(N_PRODUCTS, fields=fields)
    except Exception as e:  # noqa: BLE001 - esci-sample not built here, serving not startable
        pytest.skip(f"no real stack: {e}")
    srv = demo.make_server(st.base, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield st, f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    st.close()


@pytest.fixture(scope="module")
def plain():  # ESCI-sample as served: no price, no category
    yield from _serve(False)


@pytest.fixture(scope="module")
def fields():  # with synthetic price, stock, category
    yield from _serve(True)


def test_server_serves_page_and_proxies_the_real_api(plain):
    _, url = plain
    page = urllib.request.urlopen(url).read().decode()
    assert "<title>S3 Search demo</title>" in page and "app.js" in page
    assert b"buildRequest" in urllib.request.urlopen(url + "app.js").read()
    req = urllib.request.Request(url + "indexes/esci-sample/_search", data=json.dumps({"size": 3}).encode(), method="POST")
    assert len(json.loads(urllib.request.urlopen(req).read())["hits"]["hits"]) == 3
    m = json.loads(urllib.request.urlopen(url + "indexes/esci-sample/_mapping").read())
    assert m["esci-sample"]["s3_fields"]["product_brand.keyword"]["facetable"] is True
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "indexes/nope/_mapping")
    assert e.value.code == 404


def test_server_never_leaves_demo_dir(plain):
    _, url = plain
    for path in ("..%2fREADME.md", "%2e%2e/README.md", "../README.md"):
        try:
            urllib.request.urlopen(url + path)
            raise AssertionError("served a file outside demo/: " + path)
        except urllib.error.HTTPError as e:
            assert e.code in (400, 404)


def _node(url, mode):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    if subprocess.run([node, "-e", "require('playwright')"], capture_output=True, cwd=str(HERE)).returncode:
        pytest.skip("playwright for node not installed")
    r = subprocess.run([node, str(HERE / "browser_real.js"), url, mode], capture_output=True, text=True, timeout=90)
    if "Executable doesn't exist" in r.stdout + r.stderr:
        pytest.skip("no playwright browser installed")
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    assert r.returncode == 0, out
    return json.loads(out)


def test_page_on_esci_sample_as_served(plain):
    out = _node(plain[1], "plain")
    assert out["ok"] and out["suggestP95"] is not None


def test_page_with_numeric_fields_sort_range_histogram(fields):
    assert _node(fields[1], "fields")["ok"]
