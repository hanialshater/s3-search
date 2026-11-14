"""Vespa 8.781.11 on a Colab VM without Docker, for the matched load test (S6-18s). Development only until the
steward approves (lab/benchmarks/vespa-config.md; fairness-rulings.md V1 to V6).

The official image, pinned by its amd64 digest (V4), is exported with crane and run under chroot as the image's vespa
user. The host's /proc, /sys and /dev are bind-mounted, the VM's /etc/hosts is copied in, VESPA_HOSTNAME=localhost, and
the container is on port 8090 (the Colab VM holds 8080). These are the feasibility probe's steps (S6-plat-22/23): the
image's own binaries, run natively, with no emulation. The whole chroot runs under taskset on the server CPUs.

    from colab.competitors import vespa_run as V
    V.setup(); proc = V.start(server_cpus); V.deploy(); V.feed(catalogue); V.check_proton()
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "8.781.11"
DIGEST_AMD64 = "sha256:aa9e7347e8fb7e06bae439a99febdf5c30fe220f150205572dab0d00deaf3f20"   # V4
IMAGE = f"vespaengine/vespa@{DIGEST_AMD64}"
ROOT = "/content/vroot"
PORT = 8090
CONFIG = "http://localhost:19071"
CRANE = ("https://github.com/google/go-containerregistry/releases/download/v0.20.2/"
         "go-containerregistry_Linux_x86_64.tar.gz")
APP = Path(__file__).resolve().parent / "vespa_app"
FIELDS = ("product_title", "product_brand", "product_color", "product_bullet_point", "product_description")


def _sh(cmd, check=True):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, capture_output=True, text=True)


def _http(method, url, body=None, ctype="application/json", timeout=60):
    data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url}: HTTP {e.code}: {e.read()[:1500]!r}") from None


def _wait(url, what, seconds):
    t_end = time.time() + seconds
    while time.time() < t_end:
        try:
            if _http("GET", url, timeout=5)[0] == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{what} not up within {seconds} s")


def setup() -> dict:
    """Pull the image by digest and prepare the chroot. Returns what was set up (for the run's record)."""
    t0 = time.time()
    if not os.path.exists(f"{ROOT}/opt/vespa/bin"):
        _sh(f"curl -sL {CRANE} | tar -xz -C /usr/local/bin crane")
        os.makedirs(ROOT, exist_ok=True)
        _sh(f"crane export {IMAGE} - | tar -x -C {ROOT}")
    for d in ("proc", "sys", "dev"):
        os.makedirs(f"{ROOT}/{d}", exist_ok=True)
        if not os.path.ismount(f"{ROOT}/{d}"):
            _sh(["mount", "--rbind", f"/{d}", f"{ROOT}/{d}"])
    _sh(f"cp /etc/resolv.conf /etc/hosts {ROOT}/etc/")
    java = _sh(f"ls -d {ROOT}/usr/lib/jvm/*/bin/java | head -1", check=False).stdout.strip()
    return {"image": IMAGE, "version": VERSION, "java_home": java[len(ROOT):].rsplit("/bin/java", 1)[0],
            "vm.max_map_count": open("/proc/sys/vm/max_map_count").read().strip(), "setup_s": round(time.time() - t0, 1)}


def _uid():
    for line in open(f"{ROOT}/etc/passwd"):
        f = line.split(":")
        if f[0] == "vespa":
            return f[2], f[3]
    raise RuntimeError("no vespa user in the image")


def start(server_cpus, log_path="/content/vespa-start.log") -> subprocess.Popen:
    """Start the config server and services on the server CPUs; returns the process (its group is the server)."""
    uid, gid = _uid()
    java = _sh(f"ls -d {ROOT}/usr/lib/jvm/*/bin/java | head -1").stdout.strip()[len(ROOT):].rsplit("/bin/java", 1)[0]
    env = (f"VESPA_CONFIGSERVERS=localhost VESPA_HOSTNAME=localhost VESPA_HOME=/opt/vespa JAVA_HOME={java} "
           f"PATH={java}/bin:/opt/vespa/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    cmd = ["taskset", "-c", ",".join(map(str, server_cpus)), "chroot", "--userspec", f"{uid}:{gid}", ROOT,
           "/bin/bash", "-c", f"export {env}; exec /usr/local/bin/start-container.sh"]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    _wait(f"{CONFIG}/state/v1/health", "Vespa's config server", 900)
    return proc


def deploy(app_dir: Path = APP) -> dict:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in Path(app_dir).rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(app_dir).as_posix())
    st, body = _http("POST", f"{CONFIG}/application/v2/tenant/default/prepareandactivate", buf.getvalue(),
                     ctype="application/zip", timeout=600)
    _wait(f"http://localhost:{PORT}/state/v1/health", "Vespa's container", 900)
    return {"deploy_status": st, "deploy": json.loads(body).get("message")}


def feed(catalogue, threads: int = 32) -> dict:
    """Every product through /document/v1 with ``threads`` concurrent POSTs (the build is reported, not compared)."""
    ids = catalogue["product_id"].astype(str).tolist()
    cols = {f: catalogue[f].fillna("").astype(str).tolist() for f in FIELDS}
    ok, failed, lock = [0], [], threading.Lock()

    def one(i):
        doc = {"fields": {"product_id": ids[i], **{f: cols[f][i] for f in FIELDS}}}
        url = f"http://localhost:{PORT}/document/v1/esci/product/docid/{urllib.request.quote(ids[i], safe='')}"
        for attempt in range(5):
            try:
                st, _ = _http("POST", url, doc, timeout=60)
                with lock:
                    ok[0] += st == 200
                return
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(1 + attempt)
        with lock:
            failed.append((ids[i], err[-200:]))

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(len(ids))))
    return {"fed": ok[0], "failed": len(failed), "failed_first": failed[:5], "feed_s": round(time.time() - t0, 1),
            "docs_per_s": round(len(ids) / max(1e-9, time.time() - t0))}


def check_proton(log="/opt/vespa/logs/vespa/vespa.log") -> dict:
    """V5: proton up, and no mmap errors in Vespa's log."""
    text = open(f"{ROOT}{log}", errors="replace").read() if os.path.exists(f"{ROOT}{log}") else ""
    mmap = [l[-300:] for l in text.splitlines() if "mmap" in l.lower() and ("error" in l.lower() or "fail" in l.lower())]
    procs = _sh("ps -eo comm", check=False).stdout.split()
    return {"proton_running": any(p.startswith("vespa-proton") or p == "proton" for p in procs),
            "mmap_errors": len(mmap), "mmap_first": mmap[:3]}


def request(query: str, k: int, mode: str = "any", profile: str = "title", fieldset: str | None = None) -> dict:
    """The /search/ body. ``mode="any"``: grammar any, a plain OR over the query's terms, so the first phase ranks every
    match and the top k is exact (the headline). ``mode="weakAnd"``: grammar weakAnd with targetHits = k (WAND),
    declared approximate. ``profile="title"`` searches product_title with BM25 k1 1.2, b 0.75 (the parity run)."""
    index = "product_title" if profile == "title" and fieldset is None else (fieldset or "default")
    ann = f'defaultIndex: "{index}", grammar: "{mode}"' + (f", targetHits: {k}" if mode == "weakAnd" else "")
    return {"yql": f"select * from product where ({{{ann}}}userInput(@query))", "query": query, "hits": k,
            "ranking": profile, "presentation.summary": "ids", "timeout": "20s"}
