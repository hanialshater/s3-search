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


@pytest.fixture()
def stack():
    fake, api = serve_fake(FakeIndex())
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
