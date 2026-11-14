#!/usr/bin/env -S colab run --gpu A100 --timeout 900
"""S6-30: the fused lexical kernel's tile, block and warp counts swept on ESCI-US. Development only, never a score.

    colab run --gpu A100 --timeout 900 colab/fused_sweep.py [branch-or-commit]

It builds a lexical-only ESCI-US index, then takes the depth-2,000 request of the first 64 and 32 test-query texts. For
each (TILE, BLOCK, num_warps) it reports:
- the fused kernel's device time per batch (CUDA events around the kernel's call, median of 10, after a compile);
- that the score rows, totals and facets equal the default's bit for bit.
Writes /content/fused-sweep.json.
"""
from __future__ import annotations

import copy
import json
import os
import statistics
import subprocess
import sys
import time

REPO = "/content/s3-search"
REF = sys.argv[1] if len(sys.argv) > 1 else "main"
CONFIGS = [(2048, 256, 4), (1024, 256, 4), (4096, 256, 4), (2048, 128, 4), (2048, 512, 4), (2048, 256, 8),
           (1024, 128, 4), (4096, 512, 8), (8192, 256, 8)]
t0 = time.time()


def say(m):
    print(f"[sweep {time.time() - t0:6.1f} s] {m}", flush=True)


def main():
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--quiet", "https://github.com/hanialshater/s3-search", REPO], check=True)
    if subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", REF]).returncode == 0:
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git", "-C", REPO, "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"],
                       check=True)
        subprocess.run(["git", "-C", REPO, "checkout", "--quiet", REF], check=True)
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    os.environ.setdefault("S3_SEARCH_DATA", os.path.join(REPO, "data", "esci"))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "snowballstemmer"], check=True)
    import numpy as np
    import torch
    from engine import api
    from engine.backend import triton_fused as TF
    from referee import loader
    loader.fetch()
    ds = loader.load_dataset("esci-us")
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["bm25"] = json.loads(open("referee/requests/bm25f-tuned.bm25.json").read())
    fields = json.loads(open("referee/requests/bm25f-tuned.json").read())["query"]["multi_match"]["fields"]
    say("build")
    ix = api.build_index(ds.catalogue, m, device="cuda")
    st = ix._st
    part, b = st.parts[0], st.b
    from engine.plan.plan import plan as make_plan
    texts = [str(t) for t in ds.test_queries["query"].iloc[:64]]
    plans = [make_plan({"query": {"multi_match": {"query": t, "fields": fields}}, "size": 2000}, st) for t in texts]
    F = len(st.text_fields)
    root = plans[0].root
    w = np.zeros(F, np.float32)
    for f, wt in root.fields:
        w[st.field_index[f]] = wt
    bb = np.array([st.mapping.b_of(f) for f in st.text_fields], np.float32)
    terms = [[part.vocab[t] for t in p.root.terms if t in part.vocab] for p in plans]
    codes = [st.keywords[f].codes for f in ("product_brand.keyword", "product_color.keyword") if f in st.keywords]
    sizes = [len(st.keywords[f].values) for f in ("product_brand.keyword", "product_color.keyword") if f in st.keywords]
    out = {"gpu": torch.cuda.get_device_name(), "ref": REF, "configs": []}
    ref = {}
    for bs in (64, 32):
        for facets in (False, True):
            T = terms[:bs]
            args = (b, part, T, w, bb, st.mapping.k1, part.idf, [p.root.msm for p in plans[:bs]],
                    [p.root.boost for p in plans[:bs]], st.live, codes if facets else [], sizes if facets else [])
            for cfg in CONFIGS:
                os.environ["S3_TRITON_FUSED_TILE"], os.environ["S3_TRITON_FUSED_BLOCK"], os.environ[
                    "S3_TRITON_FUSED_WARPS"] = map(str, cfg)
                try:
                    r = TF._run_kernel(*args)                       # compile and warm
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(10):
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        r = TF._run_kernel(*args)
                        e1.record()
                        torch.cuda.synchronize()
                        times.append(e0.elapsed_time(e1))
                    key = (bs, facets)
                    bits = (r[0].view(torch.int32), r[1], [x for x in r[2]])
                    if key not in ref:
                        ref[key] = bits
                        same = True
                    else:
                        a = ref[key]
                        same = (torch.equal(a[0], bits[0]) and torch.equal(a[1], bits[1])
                                and all(torch.equal(x, y) for x, y in zip(a[2], bits[2])))
                    rec = {"batch": bs, "facets": facets, "tile": cfg[0], "block": cfg[1], "warps": cfg[2],
                           "ms_median": statistics.median(times), "ms_min": min(times), "bit_identical": same}
                except Exception as e:  # noqa: BLE001
                    rec = {"batch": bs, "facets": facets, "tile": cfg[0], "block": cfg[1], "warps": cfg[2],
                           "error": repr(e)[:300]}
                out["configs"].append(rec)
                say(json.dumps(rec))
    json.dump(out, open("/content/fused-sweep.json", "w"), indent=1)
    say("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
