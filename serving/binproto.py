"""A binary protocol between a GPU process and a native front (S6-22s), next to the pickle frames.

A native front (``serving/native_front/``, Rust) cannot read pickles, so on a connection that opens with the 4 bytes
``S3B1`` the GPU process speaks this instead. Every integer is little-endian. Each frame is a u32 length and then
the payload.

Front → GPU process:

    u8 type = 1 (search)  u64 id  u16 len  index name  u32 len  request JSON (the HTTP body as it came)
    u8 type = 2 (call)    u64 id  u16 len  method  u16 len  path (with its query)  u32 len  body

GPU process → front:

    u8 kind = 1 (columnar hits, the front writes the hits)  u64 id  u32 len  meta JSON  hits block
    u8 kind = 2 (a whole response)                          u64 id  u32 len  meta JSON  u16 status  u32 len  body
    u8 kind = 3 (S3B2 only: kind 1's block, unchecked)      u64 id  u32 len  meta JSON  hits block

A connection that opens with ``S3B2`` (opt-in, S6-22s) gets kind 3 instead of kind 1: the same block, but the ids may
be any UTF-8 and the floats any float64; the front escapes and formats them as Python's encoder does.

meta is ``{"pre", "post", "version", "queue_ms", "batch", "type"}``. ``pre`` and ``post`` are the response JSON
before and after the hits list (``json.dumps`` of the response with the hits left out, split where they go). With kind
1 the body is ``pre + hits + post``, byte for byte what ``serving.encode.encode`` writes for the dict form.

The hits block:

    u32 h  u16 w  h*w bytes: the ids as numpy S<w> (UTF-8, NUL-padded, printable ASCII other than " and \\)
    h*4 bytes: the scores, float32
    u8 n_sort, then per column: u8 code, with 0 = the score widened to float64 (no values follow),
                                1 = float64 (h*8 bytes), 2 = int64 (h*8 bytes)

Everything kind 1 needs (plain ASCII ids, sort columns of those three kinds with nothing missing, finite floats in
the range where repr writes them positionally) is checked in the GPU process. Any other response, an error, or a
call's answer goes as kind 2, encoded in Python. So the front never has to decide on JSON edge cases.
"""
from __future__ import annotations

import json
import struct

import numpy as np

MAGIC = b"S3B1"
MAGIC2 = b"S3B2"                     # opt-in: kind 3 replies (S6-22s)
_U32 = struct.Struct("<I")
_HDR = struct.Struct("<BQ")          # kind or type, id
_PLACEHOLDER = "\x00s3-hits\x00"


def frame(payload: bytes) -> bytes:
    return _U32.pack(len(payload)) + payload


def search_head(payload) -> tuple[int, str]:
    """A type-1 (search) payload's ``(id, index name)``, without its JSON (``serving/hostpath.py``)."""
    _, id_ = _HDR.unpack_from(payload, 0)
    (n,) = struct.unpack_from("<H", payload, _HDR.size)
    return id_, bytes(payload[_HDR.size + 2:_HDR.size + 2 + n]).decode()


def search_body(payload) -> bytes:
    """A type-1 payload's request JSON bytes, as the HTTP body came."""
    o = _HDR.size
    (n,) = struct.unpack_from("<H", payload, o); o += 2 + n
    (m,) = _U32.unpack_from(payload, o); o += 4
    return bytes(payload[o:o + m])


def parse_request(payload: bytes):
    """-> ("search", id, name, request_dict or BadJSON) | ("call", id, method, path, body)"""
    typ, id_ = _HDR.unpack_from(payload, 0)
    o = _HDR.size
    if typ == 1:
        (n,) = struct.unpack_from("<H", payload, o); o += 2
        name = payload[o:o + n].decode(); o += n
        (n,) = _U32.unpack_from(payload, o); o += 4
        from .hostpath import parse_body                        # the Python front's BadJSON, not a closed connection
        from .errors import BadJSON
        try:
            return ("search", id_, name, parse_body(payload[o:o + n]))
        except BadJSON as exc:
            return ("search", id_, name, exc)
    (n,) = struct.unpack_from("<H", payload, o); o += 2
    method = payload[o:o + n].decode(); o += n
    (n,) = struct.unpack_from("<H", payload, o); o += 2
    path = payload[o:o + n].decode(); o += n
    (n,) = _U32.unpack_from(payload, o); o += 4
    return ("call", id_, method, path, payload[o:o + n])


def _meta(m: dict) -> bytes:
    b = json.dumps(m, separators=(",", ":")).encode()
    return _U32.pack(len(b)) + b


def whole(id_: int, status: int, body: bytes, meta: dict | None = None) -> bytes:
    return frame(_HDR.pack(2, id_) + _meta(meta or {}) + struct.pack("<H", status) + _U32.pack(len(body)) + body)


def _plain_ids(ids) -> bool:
    """NUL (numpy's padding) and printable ASCII other than " and \\ only. The numpy form stays: a bytes.translate
    pass measured slower on a 2,000-hit reply (S6-plat-100: 36 against 30 us)."""
    if not (isinstance(ids, np.ndarray) and ids.dtype.kind == "S"):
        return False
    b = np.frombuffer(ids.tobytes(), np.uint8)
    return not ((b >= 0x7F) | ((b < 0x20) & (b != 0)) | (b == 0x22) | (b == 0x5C)).any()


def _safe(a) -> bool:
    """Every value finite and either 0 or in [1e-4, 1e16), where repr writes it positionally (9.5 against 16 us for
    the earlier five-pass form on 2,000 values, S6-plat-100's breakdown). A NaN makes the max NaN
    and an infinity makes it inf, so ``max < 1e16`` is the finiteness check too; then only the values under 1e-4 (in
    practice none) must be 0."""
    m = np.abs(np.asarray(a, np.float64))
    if m.size == 0:
        return True
    if not m.max() < 1e16:
        return False
    return not m[m < 1e-4].any()


def columnar(id_: int, resp: dict, meta: dict) -> bytes | None:
    """Kind 1 for a columnar response the front can encode exactly, else None (send it whole)."""
    return _columnar(id_, resp, meta, kind=1)


def columnar3(id_: int, resp: dict, meta: dict) -> bytes | None:
    """Kind 3 (S3B2 connections, opt-in): the same block as kind 1 without the per-reply checks of the ids and the
    floats (plat-m25: _plain_ids 29% and _safe about 8% of the GPU process's writer); the front escapes each id as
    json.encoder.encode_basestring_ascii does and writes each float as float.__repr__ does (NaN, Infinity and -Infinity as
    json.dumps writes them). Still None (send it whole) for anything else: not columnar, ids not one S<w> array,
    missing sort values, or a sort column that is neither float64 nor an integer type."""
    return _columnar(id_, resp, meta, kind=3)


def _columnar(id_: int, resp: dict, meta: dict, kind: int) -> bytes | None:
    from .encode import _shell, _is_columnar
    if not _is_columnar(resp):
        return None
    c = resp["hits"]["columns"]
    ids = c["_id"]
    checked = kind == 1
    if not (isinstance(ids, np.ndarray) and ids.dtype.kind == "S") or len(ids) == 0:
        return None
    if checked and not _plain_ids(ids):
        return None
    h, w = len(ids), ids.dtype.itemsize
    s32 = np.asarray(c["_score"], np.float32)
    s64 = s32.astype(np.float64)
    if len(s32) != h or (checked and not _safe(s64)):
        return None
    cols = []
    for col in c.get("sort") or []:
        v, miss = col.get("values"), col.get("missing")
        if miss is not None or not isinstance(v, np.ndarray) or len(v) != h:
            return None
        if v.dtype == np.float64 and np.array_equal(v, s64):
            cols.append(b"\x00")
        elif v.dtype == np.float64 and (not checked or _safe(v)):
            cols.append(b"\x01" + v.astype("<f8").tobytes())
        elif v.dtype.kind in "iu":
            cols.append(b"\x02" + v.astype("<i8").tobytes())
        else:
            return None
    if len(cols) > 255:
        return None
    text = json.dumps(_shell(resp, _PLACEHOLDER), separators=(",", ":"))
    marker = '"hits":' + json.dumps(_PLACEHOLDER)
    if text.count(marker) != 1:
        return None
    pre, post = text.split(marker)
    m = {**meta, "pre": pre + '"hits":', "post": post}
    block = (_U32.pack(h) + struct.pack("<H", w) + ids.tobytes() + s32.astype("<f4").tobytes()
             + bytes([len(cols)]) + b"".join(cols))
    return frame(_HDR.pack(kind, id_) + _meta(m) + block)


def decode_reply(payload: bytes):
    """For tests and the Python reference front: -> (id, status, body_bytes, meta). Kind 1 is encoded here with
    ``serving.encode.encode_hits`` from the decoded columns, which is what the native front must match."""
    from .encode import encode_hits
    kind, id_ = _HDR.unpack_from(payload, 0)
    o = _HDR.size
    (n,) = _U32.unpack_from(payload, o); o += 4
    meta = json.loads(payload[o:o + n]); o += n
    if kind == 2:
        (st,) = struct.unpack_from("<H", payload, o); o += 2
        (n,) = _U32.unpack_from(payload, o); o += 4
        return id_, st, payload[o:o + n], meta
    (h,) = _U32.unpack_from(payload, o); o += 4
    (w,) = struct.unpack_from("<H", payload, o); o += 2
    ids = np.frombuffer(payload, dtype=f"S{w}", count=h, offset=o); o += h * w
    sc = np.frombuffer(payload, dtype="<f4", count=h, offset=o); o += 4 * h
    k = payload[o]; o += 1
    sort = []
    for _ in range(k):
        code = payload[o]; o += 1
        if code == 0:
            sort.append({"values": sc.astype(np.float64), "missing": None})
        elif code == 1:
            sort.append({"values": np.frombuffer(payload, "<f8", h, o).astype(np.float64), "missing": None}); o += 8 * h
        else:
            sort.append({"values": np.frombuffer(payload, "<i8", h, o).astype(np.int64), "missing": None}); o += 8 * h
    hits = encode_hits({"_id": ids, "_score": sc, "sort": sort})
    return id_, 200, (meta["pre"] + hits + meta["post"]).encode(), meta


# ---- S3B3: one frame per batch (kind 4), agreed with the architect through the coordinator, 2026-11-14 -------------
#
# A connection that opens with ``S3B3`` may get kind 4 frames, one per batch per connection, besides kinds 2 and 3.
# Payload (all integers and floats little-endian; ``pad8`` is zero bytes up to the next multiple of 8 of the payload
# offset, so every numeric array starts aligned):
#
#     u8 kind=4  u64 0  u32 n_replies  u16 w  u32 h_total  ids[h_total * w] (S<w>: UTF-8, NUL-padded)
#     pad8  scores f32[h_total]
#     u8 n_cols, per column:  u8 type (1 float64, 2 int64)  u8 flags (bit 0: a missing bitmap follows)
#                             pad8  values[h_total] (8 bytes each)  [bitmap: ceil(h_total / 8) bytes, bit i of byte i//8
#                             (LSB first) set when hit i's value is missing]
#     n_replies records:  u64 id  u8 rkind
#         rkind 3:  u32 offset  u32 h  u8 n_sort  n_sort * u8 code (0: the score widened; 1 + c: column c)
#                   u32 meta_len  meta JSON (pre, post, version, queue_ms, batch, ... as kind 3)
#         rkind 2:  u32 meta_len  meta JSON  u16 status  u32 len  body (a whole response, as kind 2)
#
# Rule (the architect's amendment 3): a reply whose ids include one with a NUL before its end, or that is not UTF-8,
# goes as rkind 2 (encoded in Python). The front renders an rkind 3 record exactly as kind 3: ids escaped as
# encode_basestring_ascii, floats as json.dumps, a missing value as null. took_ms stays in the meta's pre.
MAGIC3 = b"S3B3"                     # opt-in: kind 4 batch frames (S6-22s)


def _ids_ok(ids) -> bool:
    """Every id UTF-8 with no NUL before its end (its trailing NUL padding aside)."""
    for x in ids.tolist():                                  # numpy drops the trailing NULs
        if b"\x00" in x:
            return False
        try:
            x.decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def _batch_columns(resp):
    """``(ids, s32, cols)`` for a reply rkind 3 can carry, cols as ``[(kind, values, missing or None)]`` with kind 0
    (the score widened), 1 (float64) or 2 (int64); else None."""
    from .encode import _is_columnar
    if not _is_columnar(resp):
        return None
    c = resp["hits"]["columns"]
    ids = c["_id"]
    if not (isinstance(ids, np.ndarray) and ids.dtype.kind == "S") or len(ids) == 0 or not _ids_ok(ids):
        return None
    h = len(ids)
    s32 = np.asarray(c["_score"], np.float32)
    if len(s32) != h:
        return None
    s64 = s32.astype(np.float64)
    cols = []
    for col in c.get("sort") or []:
        v, miss = col.get("values"), col.get("missing")
        if not isinstance(v, np.ndarray) or len(v) != h:
            return None
        m = None if miss is None else np.asarray(miss, dtype=bool)
        if m is not None and len(m) != h:
            return None
        if m is not None and not m.any():
            m = None
        if v.dtype == np.float64 and m is None and np.array_equal(v, s64):
            cols.append((0, None, None))
        elif v.dtype == np.float64:
            cols.append((1, v, m))
        elif v.dtype.kind in "iu":
            cols.append((2, v.astype(np.int64), m))
        else:
            return None
    return ids, s32, cols


def batch4(msgs) -> bytes:
    """The reference writer of a kind 4 frame for one batch's reply messages to one front (what ``hostpath.reply_frame``
    writes one by one), in Python; s3_hostpath's Rust reply() writes the same bytes from the engine's page."""
    from .encode import encode, _shell
    import struct as _s
    recs, rows = [], []                                     # rows: (ids, s32, cols) of each rkind 3 record
    for msg in msgs:
        if msg[0] == "search" and msg[2] == "ok":
            _, id_, _, resp, version, queue_ms, *rest = msg
            meta = {"version": f"{version[0]}:{version[1]}", "queue_ms": round(float(queue_ms), 3),
                    "batch": rest[0] if rest else 1}
            if len(rest) > 1 and rest[1] is not None:
                meta["batch_ms"] = round(float(rest[1]), 3)
            got = _batch_columns(resp)
            text = json.dumps(_shell(resp, _PLACEHOLDER), separators=(",", ":")) if got is not None else ""
            marker = '"hits":' + json.dumps(_PLACEHOLDER)
            if got is not None and text.count(marker) == 1:
                pre, post = text.split(marker)
                recs.append(("hits", id_, {**meta, "pre": pre + '"hits":', "post": post}, len(rows)))
                rows.append(got)
            else:
                recs.append(("whole", id_, meta, 200, encode(resp)))
        elif msg[0] == "search":
            recs.append(("whole", msg[1], {}, msg[3], json.dumps(msg[4], separators=(",", ":")).encode()))
        else:
            _, id_, status, body, ctype, headers = msg
            recs.append(("whole", id_, {"type": ctype}, status, body))
    w = max([r[0].dtype.itemsize for r in rows] + [1])
    h_total = sum(len(r[0]) for r in rows)
    out = bytearray(_HDR.pack(4, 0) + _s.pack("<IHI", len(recs), w, h_total))
    for ids, _, _ in rows:
        out += ids.astype(f"S{w}").tobytes()

    def pad8():
        out.extend(b"\x00" * (-len(out) % 8))
    pad8()
    for _, s32, _ in rows:
        out += s32.astype("<f4").tobytes()
    # shared columns: one per (sort position, kind) used by any reply
    keys = []
    for _, _, cols in rows:
        for k, (kind, _, _) in enumerate(cols):
            if kind and (k, kind) not in keys:
                keys.append((k, kind))
    if len(keys) > 254:
        raise ValueError("too many shared columns")
    out.append(len(keys))
    offs = np.cumsum([0] + [len(r[0]) for r in rows])
    codes = []                                              # per row: its sort codes
    for r, (_, _, cols) in enumerate(rows):
        codes.append([0 if kind == 0 else 1 + keys.index((k, kind)) for k, (kind, _, _) in enumerate(cols)])
    for c, (k, kind) in enumerate(keys):
        vals = np.zeros(h_total, "<f8" if kind == 1 else "<i8")
        miss = np.zeros(h_total, bool)
        for r, (_, _, cols) in enumerate(rows):
            if k < len(cols) and cols[k][0] == kind:
                vals[offs[r]:offs[r + 1]] = cols[k][1]
                if cols[k][2] is not None:
                    miss[offs[r]:offs[r + 1]] = cols[k][2]
        flags = 1 if miss.any() else 0
        out += bytes([kind, flags])
        pad8()
        out += vals.tobytes()
        if flags:
            out += np.packbits(miss, bitorder="little").tobytes()
    for rec in recs:
        if rec[0] == "hits":
            _, id_, meta, r = rec
            mb = json.dumps(meta, separators=(",", ":")).encode()
            out += (_s.pack("<QB", id_, 3) + _s.pack("<IIB", int(offs[r]), int(offs[r + 1] - offs[r]), len(codes[r]))
                    + bytes(codes[r]) + _U32.pack(len(mb)) + mb)
        else:
            _, id_, meta, status, body = rec
            mb = json.dumps(meta, separators=(",", ":")).encode()
            out += _s.pack("<QB", id_, 2) + _U32.pack(len(mb)) + mb + _s.pack("<HI", status, len(body)) + body
    return frame(bytes(out))


def split_batch4(payload: bytes):
    """For tests and the Python reference: a kind 4 payload -> ``[(id, status, body, meta)]``, each rkind 3 record
    encoded with ``serving.encode.encode_hits`` from its slices of the shared columns (what the native front must match)."""
    from .encode import encode_hits
    import struct as _s
    o = _HDR.size
    n, w, ht = _s.unpack_from("<IHI", payload, o); o += 10
    ids_all = np.frombuffer(payload, dtype=f"S{w}", count=ht, offset=o) if ht else np.array([], "S1"); o += ht * w
    o += -o % 8
    sc_all = np.frombuffer(payload, "<f4", ht, o); o += 4 * ht
    nc = payload[o]; o += 1
    cols = []
    for _ in range(nc):
        kind, flags = payload[o], payload[o + 1]; o += 2
        o += -o % 8
        vals = np.frombuffer(payload, "<f8" if kind == 1 else "<i8", ht, o); o += 8 * ht
        miss = None
        if flags & 1:
            nb = (ht + 7) // 8
            miss = np.unpackbits(np.frombuffer(payload, np.uint8, nb, o), bitorder="little")[:ht].astype(bool); o += nb
        cols.append((kind, vals, miss))
    out = []
    for _ in range(n):
        id_, rk = _s.unpack_from("<QB", payload, o); o += 9
        if rk == 3:
            off, h, ns = _s.unpack_from("<IIB", payload, o); o += 9
            cs = list(payload[o:o + ns]); o += ns
            (ml,) = _U32.unpack_from(payload, o); o += 4
            meta = json.loads(payload[o:o + ml]); o += ml
            sc = sc_all[off:off + h]
            sort = []
            for code in cs:
                if code == 0:
                    sort.append({"values": sc.astype(np.float64), "missing": None})
                else:
                    kind, vals, miss = cols[code - 1]
                    v = vals[off:off + h].astype(np.float64 if kind == 1 else np.int64)
                    m = None if miss is None else miss[off:off + h]
                    sort.append({"values": v, "missing": m if m is not None and m.any() else None})
            hits = encode_hits({"_id": ids_all[off:off + h], "_score": sc, "sort": sort})
            out.append((id_, 200, (meta["pre"] + hits + meta["post"]).encode(), meta))
        else:
            (ml,) = _U32.unpack_from(payload, o); o += 4
            meta = json.loads(payload[o:o + ml]); o += ml
            st, bl = _s.unpack_from("<HI", payload, o); o += 6
            out.append((id_, st, payload[o:o + bl], meta)); o += bl
    return out
