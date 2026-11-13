"""S7 prediction 4, the steward's own run (owner: evaluation-steward, P6). CPU, development only.

    python -m referee.judge_s7_p4 [--out=results/judge-s7-p4.json]

- **The operator**: the platform-engineer's envtest tests (`deploy/test-envtest.sh`, a real kube-apiserver and
  etcd), run here: `test_leader.py` (each change reconciled by exactly one replica, the Lease holder, within 1 s; the
  standby takes over within 20 s of the holder's SIGKILL) and `test_watches.py`. Their printed measurements are
  recorded. Timing-sensitive, so behind the quiet gate (`referee/loadguard.py`).
- **RPC parity**: a real `python -m serving` with an RPC port; the 200 v1 conformance requests (`conformance_v1`,
  frozen) sent over HTTP and over RPC to the same toy index; status and body must be equal (`took_ms` removed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _strip(body):
    if isinstance(body, dict):
        return {k: _strip(v) for k, v in body.items() if k != "took_ms"}
    if isinstance(body, list):
        return [_strip(x) for x in body]
    return body


def rpc_parity(work: Path) -> dict:
    from referee import conformance_v1, loader
    from serving.rpc import RpcClient
    ds = loader.load_dataset("toy")
    src = work / "toy.parquet"
    ds.catalogue.to_parquet(src, index=False)
    hp, rp = _free_port(), _free_port()
    (work / "serving").mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "serving", "--data-dir", str(work / "serving"), "--port", str(hp),
                             "--rpc-port", str(rp), "--device", "cpu"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=open(work / "serving.log", "w"))
    base = f"http://127.0.0.1:{hp}"

    def http(method, path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")
    try:
        for _ in range(200):
            try:
                http("GET", "/healthz")
                break
            except OSError:
                time.sleep(0.1)
        http("PUT", "/indexes/toy", {"mapping": ds.mapping, "source": {"path": str(src)}})
        rpc = RpcClient("127.0.0.1", rp)
        suite = conformance_v1.build_suite(ds)
        differ, examples = 0, []
        for s in suite:
            hs, hb = http("POST", "/indexes/toy/_search", s["request"])
            rs, rb, _ = rpc.call("POST", "/indexes/toy/_search", s["request"])
            if hs != rs or _strip(hb) != _strip(rb):
                differ += 1
                if len(examples) < 3:
                    examples.append({"id": s["id"], "family": s["family"], "http": hs, "rpc": rs})
        return {"requests": len(suite), "suite_sha256": conformance_v1.suite_sha256(suite), "differ": differ,
                "examples": examples}
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:  # noqa: BLE001
            proc.kill()


def operator_envtest() -> dict:
    t = time.perf_counter()
    p = subprocess.run(["bash", "deploy/test-envtest.sh", "-s", "-v", "deploy/tests/test_leader.py",
                        "deploy/tests/test_watches.py"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    tests = re.findall(r"(deploy/tests/\S+::\S+) (PASSED|FAILED|ERROR|SKIPPED)", out)
    return {"exit": p.returncode, "s": time.perf_counter() - t, "tests": tests,
            "measured": [l.strip() for l in out.splitlines() if re.match(r"\s*(LEADER LATENCY|TAKEOVER|HANDOVER|WATCH)", l)],
            "tail": out.strip().splitlines()[-3:]}


def main(argv=None) -> int:
    from referee import loadguard
    ap = argparse.ArgumentParser(prog="python -m referee.judge_s7_p4")
    ap.add_argument("--out", default="results/judge-s7-p4.json")
    a = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="judge-s7-p4-"))
    res = {"kind": "judge-s7-p4", "label": "CPU, development only"}
    res["rpc_parity"] = rpc_parity(work)
    try:
        res["load_at_start"] = loadguard.wait_for_quiet()
        with loadguard.Sampler() as s:
            res["operator"] = operator_envtest()
        res["load_during_run"] = s.summary()
        if res["load_during_run"]["overloaded"]:
            res["timings_not_judged"] = "busy during the run"
    except loadguard.Busy as e:
        res["timings_not_judged"] = str(e)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({"rpc": res["rpc_parity"], "operator": res.get("operator", {}).get("measured"),
                      "tests": res.get("operator", {}).get("tests"), "not_judged": res.get("timings_not_judged")},
                     indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
