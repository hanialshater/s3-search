"""S6-22s: compare two fair_bench --ab-dump files (the stdlib front's and the native front's) byte for byte.

    python colab/front_ab.py A.bin B.bin [OUT.json]

Each file holds the same request bodies' responses, in order, as length-prefixed records with only the took_ms value
blanked. Prints and writes a JSON summary: records in each, how many are byte-identical, and the first differences
(index, lengths, the first differing byte offset and 80 bytes around it on each side). Development only."""
import json
import struct
import sys


def records(path):
    out, data, o = [], open(path, "rb").read(), 0
    while o < len(data):
        (n,) = struct.unpack_from("<I", data, o)
        out.append(data[o + 4:o + 4 + n])
        o += 4 + n
    return out


def compare(a, b, keep=10):
    ra, rb = records(a), records(b)
    diffs, same = [], 0
    for i, (x, y) in enumerate(zip(ra, rb)):
        if x == y:
            same += 1
            continue
        k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        if len(diffs) < keep:
            diffs.append({"i": i, "len_a": len(x), "len_b": len(y), "first_diff": k,
                          "a": x[max(0, k - 40):k + 40].decode("utf-8", "replace"),
                          "b": y[max(0, k - 40):k + 40].decode("utf-8", "replace")})
    return {"a": a, "b": b, "records_a": len(ra), "records_b": len(rb), "compared": min(len(ra), len(rb)),
            "identical": same, "differ": min(len(ra), len(rb)) - same, "first_differences": diffs}


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        json.dump(r, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["differ"] == 0 and r["records_a"] == r["records_b"] > 0 else 1)
