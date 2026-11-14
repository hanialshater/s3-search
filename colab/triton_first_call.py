"""S6-18: each Triton kernel's first-call cost (its compile) in a fresh process with an empty Triton cache.

    python colab/triton_first_call.py [out.json]

Run by colab/gpu_ci.py on cuda. Builds the toy index, warms the torch path, then for each kernel alone: the first
batch with it on (it compiles), and the next three (cached). The first call's extra time over the cached median is the
compile. TRITON_CACHE_DIR is a new empty directory, so nothing compiled earlier is reused. Development only.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time

os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-")
os.environ["S3_TRITON_CAPABILITIES"] = "any"       # measure on whatever GPU this is
for f in ("S3_TRITON_BM25", "S3_TRITON_TOPK", "S3_TRITON_FACETS"):
    os.environ[f] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import triton  # noqa: E402

from engine import api  # noqa: E402
from engine.backend import triton_bm25, triton_facets, triton_topk  # noqa: E402
from referee import checks, loader  # noqa: E402


def timed(ix, reqs):
    torch.cuda.synchronize()
    t = time.perf_counter()
    ix.search_batch(reqs, hits="columns", ids="bytes")
    torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ds = loader.load_dataset("toy")
    triton_topk.MIN_K = 1                           # the toy has 200 docs, fewer than MIN_K; the compile is the same
    ix = api.build_index(ds.catalogue, ds.mapping, device="cuda")
    aggs = {"b": {"terms": {"field": "product_brand.keyword", "size": 5}}}
    lex = checks.lexical_requests(ds)[:32]
    reqs = [{**r, "size": 300, "aggs": aggs} for r in lex]
    for _ in range(3):
        timed(ix, reqs)                             # torch path warm
    out = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
           "torch": torch.__version__, "triton": triton.__version__, "kernels": {}}
    calls = {}
    for flag, mod, fn in (("S3_TRITON_BM25", triton_bm25, "segment_sum_and_count"),
                          ("S3_TRITON_TOPK", triton_topk, "topk"), ("S3_TRITON_FACETS", triton_facets, "bincount_rows")):
        real = getattr(mod, fn)
        setattr(mod, fn, lambda *a, _r=real, _f=flag, **k: (calls.__setitem__(_f, calls.get(_f, 0) + 1), _r(*a, **k))[1])
    for flag in ("S3_TRITON_BM25", "S3_TRITON_TOPK", "S3_TRITON_FACETS"):
        os.environ[flag] = "1"
        first = timed(ix, reqs)
        cached = statistics.median(timed(ix, reqs) for _ in range(3))
        os.environ[flag] = "0"
        out["kernels"][flag] = {"first_s": round(first, 3), "cached_s": round(cached, 4),
                                "compile_s": round(first - cached, 3), "kernel_calls": calls.get(flag, 0)}
        if not calls.get(flag):
            print(f"[first-call] {flag}: the kernel never ran", flush=True)
            return 1
        print(f"[first-call] {flag}: first batch {first:.3f} s, cached {cached * 1000:.1f} ms, "
              f"compile about {first - cached:.2f} s", flush=True)
    print("[first-call] " + json.dumps(out), flush=True)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
