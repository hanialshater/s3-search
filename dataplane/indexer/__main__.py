"""``python -m dataplane.indexer``: the indexer job (§7 runs it in Kubernetes).

    python -m dataplane.indexer --store STORE.sqlite --mapping MAPPING.json --out SEGMENTS/ \
        [--ingest BATCH.parquet|.jsonl --rejected REJECTED.jsonl] [--keep 3]

Optionally ingests a batch file first, then builds a segment of the store at its head and prints one JSON line:
``{"segment": ..., "segment_id": ..., "version": ..., "n_docs": ..., "seconds": ...}``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dataplane.indexer import build, gc, read_manifest
from dataplane.intake import ingest
from dataplane.kv import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dataplane.indexer", description="Build a segment from the store.")
    ap.add_argument("--store", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ingest")
    ap.add_argument("--rejected")
    ap.add_argument("--keep", type=int, default=3)
    a = ap.parse_args(argv)
    mapping = json.loads(Path(a.mapping).read_text())
    t0 = time.perf_counter()
    with Store(a.store, mapping) as store:
        report = None
        if a.ingest:
            report = ingest(a.ingest, mapping, store, a.rejected or Path(a.out) / "rejected.jsonl")
        seg = build(store, mapping, a.out)
        gc(a.out, keep=a.keep, protect=(seg,))
    man = read_manifest(seg)
    out = {"segment": str(seg), "segment_id": man.get("segment_id", seg.name), "version": man.get("version"),
           "n_docs": man.get("n_docs"), "seconds": round(time.perf_counter() - t0, 3)}
    if report is not None:
        out["intake"] = {"accepted": report.accepted, "rejected": report.rejected, "reasons": report.reasons}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
