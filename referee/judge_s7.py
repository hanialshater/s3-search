"""Judging runs for S7's predictions 3 and 6 (owner: evaluation-steward, P6). CPU, development only.

    taskset -c 0,1 python -m referee.judge_s7 [--pre-s7=DIR] [--rounds=3] [--out=results/judge-s7.json]

- p3, as written: on ESCI-sample fit, bench nDCG@10 0.4644 and Exact recall@100 0.7551 to 4 decimals; "the 200 v1
  conformance requests still pass against the reference" (v1 frozen in `referee/conformance_v1.py`, the engine
  against the reference index, v1's own rules); CPU p99 at batch 1 rises at most 20% (the engine before S7, `--pre-s7`
  a directory holding that commit's `engine/`, against the current one, benched alternately `--rounds` times on the
  same machine; the medians of the p99s compared); referee fast under 60 s on 2 cores.
- p6: the 184 v1 requests outside the `unsupported` family pass against the reference; every construct outside S7's
  scope (`referee/scope.py`, item None) is refused with a named error.

Timings wait for a quiet machine (`referee/loadguard.py`) and are refused when it was busy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bench(engine_dir, out):
    env = dict(os.environ)
    path = [str(engine_dir)] if engine_dir else []
    env["PYTHONPATH"] = os.pathsep.join(path + [str(ROOT)])
    cwd = str(Path(engine_dir).parent if engine_dir else ROOT)  # the engine dir first on sys.path
    code = ("import sys; sys.path.insert(0, %r); from referee import bench; "
            "sys.exit(bench.main(['--dataset=esci-sample', '--latency-queries=2000', '--out=%s']))"
            % (str(engine_dir) if engine_dir else str(ROOT), out))
    p = subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-800:])
    r = json.loads(Path(out).read_text())
    check = subprocess.run([sys.executable, "-c", code.split(";")[0] + "; import engine.api; print(engine.api.__file__)"],
                           env=env, cwd=cwd, capture_output=True, text=True).stdout.strip()
    r["engine_api_file"] = check
    return r


def p3_latency_and_quality(pre_s7, rounds, work):
    runs = {"pre": [], "post": []}
    for i in range(rounds):
        for tag, d in (("pre", pre_s7), ("post", None)):
            r = _bench(d, work / f"bench-{tag}-{i}.json")
            runs[tag].append(r)
    med = lambda tag, k: statistics.median(r["latency_ms"][k] for r in runs[tag])  # noqa: E731
    post_q = runs["post"][0]["quality"]
    return {
        "rounds": rounds,
        "engine_files": {t: runs[t][0]["engine_api_file"] for t in runs},
        "p99_ms": {t: [r["latency_ms"]["p99"] for r in runs[t]] for t in runs},
        "p50_ms": {t: [r["latency_ms"]["p50"] for r in runs[t]] for t in runs},
        "p99_median": {t: med(t, "p99") for t in runs},
        "p99_rise": med("post", "p99") / med("pre", "p99") - 1.0,
        "quality_post": post_q, "quality_pre": runs["pre"][0]["quality"],
        "ndcg_4dp": round(post_q["ndcg@10"], 4), "recall100_4dp": round(post_q["exact_recall@100"], 4),
    }


def v1_against_reference():
    import engine.api as api
    from referee import checks, conformance_v1 as v1, loader
    ds = loader.load_dataset("toy")
    suite = v1.build_suite(ds)
    rep = v1.run_suite(api.build_index(ds.catalogue, ds.mapping), checks.reference_for(ds), suite, "reference")
    non_unsup = [s for s in suite if s["family"] != "unsupported"]
    rep_184 = v1.run_suite(api.build_index(ds.catalogue, ds.mapping), checks.reference_for(ds), non_unsup, "reference")
    return {"suite_sha256": v1.suite_sha256(suite), "all_200": {k: rep[k] for k in ("n", "passed", "by_family")},
            "failures_200": [{"id": f["id"], "family": f["family"], "diffs": f["diffs"][:1]} for f in rep["failures"]],
            "outside_unsupported_184": {k: rep_184[k] for k in ("n", "passed", "by_family")},
            "failures_184": rep_184["failures"][:10]}


def outside_s7_refused():
    import engine.api as api
    from referee import loader, scope
    ds = loader.load_dataset("toy")
    idx = api.build_index(ds.catalogue, ds.mapping)
    rows = []
    for name, item, _, frag, req in scope.SCOPE:
        if item is not None:
            continue
        try:
            idx.search(req)
            rows.append({"construct": name, "refused": False})
        except api.UnsupportedQuery as e:
            rows.append({"construct": name, "refused": True, "path": e.path, "names_it": frag in str(e.path)})
    return {"n": len(rows), "refused_with_named_path": sum(r["refused"] and r.get("names_it") for r in rows),
            "answered": [r["construct"] for r in rows if not r["refused"]], "rows": rows}


def fast_time():
    t = time.perf_counter()
    p = subprocess.run(["taskset", "-c", "0,1", sys.executable, "-m", "referee", "fast"], cwd=ROOT,
                       capture_output=True, text=True)
    return {"exit": p.returncode, "s": time.perf_counter() - t, "last_line": p.stdout.strip().splitlines()[-1]}


def main(argv=None) -> int:
    import tempfile
    from referee import loadguard
    ap = argparse.ArgumentParser(prog="python -m referee.judge_s7")
    ap.add_argument("--pre-s7", required=True, help="a directory holding the pre-S7 commit's engine/ package")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="results/judge-s7.json")
    a = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="judge-s7-"))
    res = {"kind": "judge-s7", "label": "CPU, development only"}
    try:
        res["load_at_start"] = loadguard.wait_for_quiet()
    except loadguard.Busy as e:
        res["timings_not_judged"] = str(e)
    with loadguard.Sampler() as s:
        if "timings_not_judged" not in res:
            res["p3_bench"] = p3_latency_and_quality(Path(a.pre_s7), a.rounds, work)
        res["v1"] = v1_against_reference()
        res["outside_s7"] = outside_s7_refused()
        res["referee_fast"] = fast_time()
    res["load_during_run"] = s.summary()
    if res["load_during_run"]["overloaded"]:
        res["timings_not_judged"] = f"busy during the run: {res['load_during_run']}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    b = res.get("p3_bench", {})
    print(json.dumps({"p99_median": b.get("p99_median"), "p99_rise": b.get("p99_rise"), "ndcg": b.get("ndcg_4dp"),
                      "recall": b.get("recall100_4dp"), "engines": b.get("engine_files"),
                      "v1_200": res["v1"]["all_200"]["passed"], "v1_184": res["v1"]["outside_unsupported_184"]["passed"],
                      "outside_s7_refused": f"{res['outside_s7']['refused_with_named_path']}/{res['outside_s7']['n']}",
                      "fast": res["referee_fast"], "load": res["load_during_run"],
                      "not_judged": res.get("timings_not_judged")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
