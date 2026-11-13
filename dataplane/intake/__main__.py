"""``python -m dataplane.intake --store S --mapping M.json --rejected R.jsonl FILE...``: ingest batch files."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dataplane.intake import ingest
from dataplane.kv import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dataplane.intake")
    ap.add_argument("--store", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--rejected", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    mapping = json.loads(Path(a.mapping).read_text())
    with Store(a.store, mapping) as store:
        for f in a.files:
            print(json.dumps({"file": f, **asdict(ingest(f, mapping, store, a.rejected))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
