"""Writes serving/tests/golden/binproto_frames.json: fixed GPU-process reply messages and the binproto frames the
Python encoder writes for them (S6-22s; the oracle s3_hostpath's reply must reproduce, took_ms masked).

    python -m serving.tests.golden.make_binproto_frames

Each case is a JSON spec (``build`` turns it into the reply message) and the frame as hex. Regenerate only when the
protocol itself changes, and say so in the commit."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "binproto_frames.json"


def build(spec: dict):
    """A spec -> the reply message tuple the GPU process hands its writer."""
    k = spec["kind"]
    if k == "ok":
        c = spec["columns"]
        ids = np.array([s.encode("utf-8") for s in c["_id"]], f"S{c['width']}")
        sc = np.array(c["_score"], np.float32)
        sort = []
        for col in c.get("sort", []):
            if col == "score":
                sort.append({"values": sc.astype(np.float64), "missing": None})
            elif col["dtype"] == "f8":
                sort.append({"values": np.array(col["values"], np.float64), "missing": col.get("missing")})
            else:
                sort.append({"values": np.array(col["values"], np.int64), "missing": col.get("missing")})
        resp = {"took_ms": spec["took_ms"], "hits": {"total": spec["total"],
                                                     "columns": {"_id": ids, "_score": sc, "sort": sort}}}
        resp.update(spec.get("extra", {}))
        return ("search", spec["id"], "ok", resp, tuple(spec["version"]), spec["queue_ms"], spec["batch"])
    if k == "error":
        return ("search", spec["id"], "error", spec["status"], spec["body"])
    if k == "call":
        return ("call", spec["id"], spec["status"], spec["body"].encode(), spec["type"], {})
    if k == "multi":
        return ("multi", [build(s) for s in spec["parts"]])
    raise ValueError(k)


def _ok(id_, ids, scores, sort=(), width=None, total=None, extra=None, took=None):
    return {"kind": "ok", "id": id_, "took_ms": took or {"parse": 0.1, "retrieve": 2.0, "page": 0.3},
            "total": total if total is not None else len(ids), "version": ["seg-1215854-59952b69", 1],
            "queue_ms": 1.234567, "batch": 3, "extra": extra or {},
            "columns": {"_id": list(ids), "_score": list(scores), "sort": list(sort),
                        "width": width or max([len(s.encode("utf-8")) for s in ids] + [1])}}


def cases():
    rng = np.random.default_rng(31)
    asins = ["B%09d" % int(x) for x in rng.integers(0, 10**9, 50)]
    sc = sorted((float(np.float32(x)) for x in rng.gamma(2, 2, 50)), reverse=True)
    yield "columnar, score sort widened (code 0) and an int64 column (code 2)", _ok(
        7, asins, sc, ["score", {"dtype": "i8", "values": [int(x) for x in rng.integers(0, 1_215_854, 50)]}])
    yield "columnar, a float64 sort column (code 1), aggs in the shell", _ok(
        8, asins[:5], sc[:5], [{"dtype": "f8", "values": [19.99, 5.0, 120.5, 0.0, 3.25]}],
        extra={"aggregations": {"c": {"buckets": [{"key": "lamps", "doc_count": 3}]}}})
    yield "columnar, no sort columns, mixed id widths (NUL padding)", _ok(9, ["A", "BB", "CCCCCCCCCC"], [3.0, 2.5, 1e-3])
    yield "whole: float32 1e-4 widens to 9.99999974e-05, under the positional range", _ok(23, ["A"], [1e-4])
    yield "columnar, total above the page", _ok(10, asins[:3], sc[:3], ["score"], total=185504)
    yield "whole: an id with a quote", _ok(11, ['B0"X'], [1.0])
    yield "whole: an id with a backslash", _ok(12, ["B0\\X"], [1.0])
    yield "whole: an id with a byte >= 0x7f", _ok(13, ["B0\xe9X"], [1.0])
    yield "whole: a score repr writes in exponent form", _ok(14, ["A", "B"], [3.0e16, 1.0])
    yield "whole: a tiny score", _ok(15, ["A"], [5e-5])
    yield "whole: a sort column with missing values", _ok(16, ["A", "B"], [2.0, 1.0],
                                                          [{"dtype": "f8", "values": [1.0, 0.0], "missing": [False, True]}])
    yield "whole: zero hits", _ok(17, [], [], total=0)
    yield "error", {"kind": "error", "id": 18, "status": 400,
                    "body": {"error": {"type": "invalid_request", "path": "", "reason": "body is not JSON: x"}}}
    yield "call", {"kind": "call", "id": 19, "status": 200, "body": '{"status":"ok"}', "type": "application/json"}
    yield "multi: three replies in one write", {"kind": "multi", "parts": [
        _ok(20, asins[:2], sc[:2], ["score"]), {"kind": "error", "id": 21, "status": 404,
                                               "body": {"error": {"type": "index_not_found", "path": "", "reason": "no"}}},
        _ok(22, ["A"], [1.5])]}


def main():
    from serving.hostpath import reply_frame
    out = [{"name": n, "spec": s, "frame_hex": reply_frame(build(s)).hex()} for n, s in cases()]
    OUT.write_text(json.dumps({"note": "S6-22s golden binproto reply frames; took_ms masked when compared",
                               "cases": out}, indent=1) + "\n")
    print(f"wrote {len(out)} cases to {OUT}")


if __name__ == "__main__":
    main()
