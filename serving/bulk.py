"""``POST /indexes/{name}/_bulk``: OpenSearch NDJSON (index, create, update, delete) as store changes.

The body is streamed: bytes are read in 1 MB pieces, split into lines, grouped into chunks of ``bulk_chunk`` actions, and
each chunk is planned, applied to the store as one atomic batch (``dataplane.intake.apply_changes``, which rejects bad
records one by one) and replayed onto the live index before the next chunk is read. Memory is one chunk plus the item
list for the response, whatever the body size.

Every action is accounted for: ``items[i]`` answers action ``i`` with an HTTP-style ``status`` and, for an error, an
OpenSearch-style ``error.type`` plus ``error.s3_reason`` (the data plane's named rejection reason) and ``field``.

Semantics, as OpenSearch: ``index`` stores the whole document (201 created / 200 updated); ``create`` fails with 409
``version_conflict_engine_exception`` if the id exists; ``update`` merges a partial ``doc`` (404
``document_missing_exception`` if absent, unless ``doc_as_upsert`` or ``upsert``; hot fields only become an in-place
patch); ``delete`` is 200 ``deleted`` or 404 ``not_found``. An unparseable action line ends the request with 400 and the
items applied so far (they are already stored). Not supported: ``script``, versioning, routing, pipelines.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

OPS = ("index", "create", "update", "delete")
MAX_LINE = 16 * 1024 * 1024

# the data plane's rejection reasons -> (OpenSearch-style error type, HTTP status)
REASON_TYPES = {
    "missing_id": ("action_request_validation_exception", 400),
    "malformed": ("action_request_validation_exception", 400),
    "bad_op": ("action_request_validation_exception", 400),
    "bad_type": ("mapper_parsing_exception", 400),
    "dims_mismatch": ("mapper_parsing_exception", 400),
    "too_long": ("mapper_parsing_exception", 400),
    "unknown_field": ("strict_dynamic_mapping_exception", 400),
    "not_hot": ("illegal_argument_exception", 400),
    "no_such_key": ("document_missing_exception", 404),
}


class BulkFormatError(ValueError):
    """The body cannot be read as bulk NDJSON from here on (a bad action line)."""

    def __init__(self, reason: str, line: int):
        super().__init__(f"line {line}: {reason}")
        self.reason, self.line = reason, line


@dataclass
class Action:
    pos: int
    op: str
    meta: dict
    source: dict | None = None
    error: tuple[str, str, int] | None = None        # (type, reason, status) found while parsing


# ---- reading ----------------------------------------------------------------------------------------------------------
def iter_lines(read: Callable[[int], bytes], piece: int = 1 << 20) -> Iterator[bytes]:
    """Lines (without the newline) from a ``read(n)`` that returns b"" at the end."""
    carry = b""
    while True:
        chunk = read(piece)
        if not chunk:
            break
        parts = (carry + chunk).split(b"\n")
        carry = parts.pop()
        if len(carry) > MAX_LINE:
            raise BulkFormatError(f"a line is over {MAX_LINE} bytes", 0)
        yield from parts
    if carry:
        yield carry


def length_reader(rfile, length: int) -> Callable[[int], bytes]:
    left = [length]

    def read(n: int) -> bytes:
        if left[0] <= 0:
            return b""
        data = (rfile.read1 if hasattr(rfile, "read1") else rfile.read)(min(n, left[0]))
        left[0] -= len(data)
        return data

    return read


def chunked_reader(rfile) -> Callable[[int], bytes]:
    """Transfer-Encoding: chunked, decoded."""
    state = {"left": 0, "done": False}

    def read(n: int) -> bytes:
        if state["done"]:
            return b""
        if state["left"] == 0:
            size_line = rfile.readline().strip()
            if not size_line:
                size_line = rfile.readline().strip()
            size = int(size_line.split(b";")[0] or b"0", 16)
            if size == 0:
                rfile.readline()
                state["done"] = True
                return b""
            state["left"] = size
        data = rfile.read(min(n, state["left"]))
        state["left"] -= len(data)
        if state["left"] == 0:
            rfile.readline()  # the CRLF after the chunk
        return data

    return read


def parse_actions(lines: Iterator[bytes]) -> Iterator[Action]:
    """Pair action lines with their source lines. Blank lines are skipped."""
    it = ((i + 1, ln) for i, ln in enumerate(lines) if ln.strip())
    pos = 0
    for lineno, raw in it:
        try:
            head = json.loads(raw)
        except ValueError as exc:
            raise BulkFormatError(f"action is not JSON: {exc}", lineno) from None
        if not isinstance(head, dict) or len(head) != 1 or next(iter(head)) not in OPS \
                or not isinstance(next(iter(head.values())), dict):
            raise BulkFormatError('an action line is {"index"|"create"|"update"|"delete": {...}}', lineno)
        op, meta = next(iter(head.items()))
        act = Action(pos, op, meta)
        pos += 1
        if op != "delete":
            nxt = next(it, None)
            if nxt is None:
                act.error = ("action_request_validation_exception", f"line {lineno}: {op} has no source line", 400)
            else:
                try:
                    src = json.loads(nxt[1])
                    if not isinstance(src, dict):
                        raise ValueError("the source is not an object")
                    act.source = src
                except ValueError as exc:
                    act.error = ("mapper_parsing_exception", f"line {nxt[0]}: source is not a JSON object: {exc}", 400)
        yield act


# ---- planning one chunk -----------------------------------------------------------------------------------------------
@dataclass
class Planned:
    action: Action
    id: str | None
    change: dict | None = None
    result: str | None = None
    status: int = 200
    error: dict | None = None
    seq: int | None = None
    extra: dict = field(default_factory=dict)


def _err(type_: str, reason: str, **more) -> dict:
    return {"type": type_, "reason": reason, **more}


def action_id(a: Action, id_field: str) -> str | None:
    _id = a.meta.get("_id")
    if _id is None and a.source is not None and a.op in ("index", "create"):
        _id = a.source.get(id_field)
    return None if _id is None else str(_id)


def chunk_ids(actions: list[Action], id_field: str) -> list[str]:
    return list(dict.fromkeys(i for a in actions if a.error is None and (i := action_id(a, id_field))))


def plan_chunk(actions: list[Action], index_name: str, hot: set[str], id_field: str,
               existing: dict[str, dict | None]) -> list[Planned]:
    """Turn actions into planned results and change records. ``existing`` maps ids to their stored doc (None: absent);
    it is updated as the chunk is planned, so a later action sees an earlier one's effect."""
    out: list[Planned] = []
    for a in actions:
        pl = Planned(a, action_id(a, id_field))
        out.append(pl)
        if a.error is not None:
            pl.status, pl.error = a.error[2], _err(a.error[0], a.error[1])
            continue
        want = a.meta.get("_index")
        if want is not None and want != index_name:
            pl.status, pl.error = 404, _err("index_not_found_exception", f"no such index [{want}]; this is [{index_name}]")
            continue
        if pl.id is None:
            if a.op in ("index", "create"):
                pl.id = uuid.uuid4().hex                      # OpenSearch autogenerates ids
            else:
                pl.status, pl.error = 400, _err("action_request_validation_exception", f"{a.op} requires an _id",
                                                s3_reason="missing_id")
                continue
        cur = existing.get(pl.id)
        if a.op in ("index", "create"):
            if a.op == "create" and cur is not None:
                pl.status, pl.error = 409, _err("version_conflict_engine_exception",
                                                f"[{pl.id}]: version conflict, document already exists")
                continue
            pl.change = {"op": "upsert", "_id": pl.id, "doc": a.source}
            pl.result, pl.status = ("updated", 200) if cur is not None else ("created", 201)
            existing[pl.id] = dict(a.source)
        elif a.op == "delete":
            if cur is None:
                pl.result, pl.status = "not_found", 404
                continue
            pl.change = {"op": "delete", "_id": pl.id}
            pl.result = "deleted"
            existing[pl.id] = None
        else:  # update
            src = a.source or {}
            if "script" in src:
                pl.status, pl.error = 400, _err("illegal_argument_exception", "scripted updates are not supported")
                continue
            part = src.get("doc")
            if part is not None and not isinstance(part, dict):
                pl.status, pl.error = 400, _err("action_request_validation_exception", '"doc" must be an object')
                continue
            if cur is None:
                up = part if src.get("doc_as_upsert") else src.get("upsert")
                if isinstance(up, dict):
                    pl.change = {"op": "upsert", "_id": pl.id, "doc": up}
                    pl.result, pl.status = "created", 201
                    existing[pl.id] = dict(up)
                else:
                    pl.status, pl.error = 404, _err("document_missing_exception", f"[{pl.id}]: document missing")
                continue
            if not part:
                pl.result = "noop"
                continue
            if set(part) <= hot:
                pl.change = {"op": "patch", "_id": pl.id, "fields": dict(part)}
            else:  # a non-hot field: merge into the stored document and store it whole
                pl.change = {"op": "upsert", "_id": pl.id, "doc": {**cur, **part}}
            pl.result = "updated"
            existing[pl.id] = {**cur, **part}
    return out


def finish(plan: list[Planned], report: Any, index_name: str) -> list[dict]:
    """Items in action order. ``report`` is the data plane's ChangeReport (or None if nothing was applied)."""
    rejected: dict[int, Any] = {}
    first = 0
    if report is not None:
        rejected = {id(ch): rej for ch, rej in report.rejected}
        first = report.first_offset
    items: list[dict] = []
    seq = first
    for pl in plan:
        body: dict[str, Any] = {"_index": index_name, "_id": pl.id}
        if pl.change is not None:
            rej = rejected.get(id(pl.change))
            if rej is not None:
                type_, status = REASON_TYPES.get(rej.reason, ("illegal_argument_exception", 400))
                pl.status, pl.error, pl.result = status, _err(type_, rej.detail, s3_reason=rej.reason,
                                                              field=rej.field), None
            else:
                pl.seq = seq
                seq += 1
        if pl.error is not None:
            body.update(status=pl.status, error=pl.error)
        else:
            body.update(result=pl.result, status=pl.status)
            if pl.seq is not None:
                body["_seq_no"] = pl.seq
        items.append({pl.action.op: body})
    return items
