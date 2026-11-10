import json
import shutil
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from tools import demo
from tools.tests.fakes import FakeIndex, serve_fake

HERE = Path(__file__).parent


ESCI_FIELDS = ["product_title", "product_brand", "product_color", "product_bullet_point", "product_description"]


@pytest.fixture()
def stack():
    yield from _stack(FakeIndex())


@pytest.fixture()
def esci_stack():  # no category, no price: ESCI-sample as served
    yield from _stack(FakeIndex(fields=ESCI_FIELDS))


@pytest.fixture()
def mapped_stack():  # the index info carries its mapping
    yield from _stack(FakeIndex(fields=ESCI_FIELDS, with_mapping=True))


def _stack(index):
    fake, api = serve_fake(index)
    srv = demo.make_server(api, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    fake.shutdown()


def test_server_serves_page_and_proxies_search(stack):
    page = urllib.request.urlopen(stack).read().decode()
    assert "<title>S3 Search demo</title>" in page and "app.js" in page
    assert b"buildRequest" in urllib.request.urlopen(stack + "app.js").read()
    req = urllib.request.Request(stack + "indexes/esci-sample/_search", data=json.dumps({"size": 3}).encode(), method="POST")
    body = json.loads(urllib.request.urlopen(req).read())
    assert len(body["hits"]["hits"]) == 3
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(stack + "../README.md")
    assert e.value.code == 404


def test_server_never_leaves_demo_dir(stack):
    for path in ("..%2fREADME.md", "%2e%2e/README.md"):
        try:
            urllib.request.urlopen(stack + path)
            raise AssertionError("served a file outside demo/: " + path)
        except urllib.error.HTTPError as e:
            assert e.code in (404, 400)


def test_demo_page_in_a_headless_browser(stack):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    probe = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, cwd=str(HERE))
    if probe.returncode:
        pytest.skip("playwright for node not installed")
    r = subprocess.run([node, str(HERE / "browser_check.js"), stack], capture_output=True, text=True, timeout=60)
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    if "Executable doesn't exist" in r.stdout + r.stderr:
        pytest.skip("no playwright browser installed")
    assert r.returncode == 0, out
    assert json.loads(out)["ok"] is True


def _node(script, url):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    if subprocess.run([node, "-e", "require('playwright')"], capture_output=True, cwd=str(HERE)).returncode:
        pytest.skip("playwright for node not installed")
    r = subprocess.run([node, str(HERE / script), url], capture_output=True, text=True, timeout=60)
    if "Executable doesn't exist" in r.stdout + r.stderr:
        pytest.skip("no playwright browser installed")
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    assert r.returncode == 0, out
    return json.loads(out)


def test_page_discovers_fields_by_probing_when_index_has_no_mapping(esci_stack):
    out = _node("browser_check_esci.js", esci_stack)
    assert out["ok"] and out["actions"] >= 12


def test_page_uses_the_mapping_when_index_info_has_one(mapped_stack):
    out = _node("browser_check_esci.js", mapped_stack)
    assert out["ok"] and "product_color.keyword" in out["facets"]
