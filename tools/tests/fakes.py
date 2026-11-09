"""A fake engine and a fake serving HTTP server, written against INTERFACES §5 and §7, for the tools' tests and for
``python -m tools.demo --fake``. Not the engine: it scores by weighted term counts. It implements just what the tools
use: ``multi_match`` / ``match`` / ``match_all``, ``bool`` with ``filter`` (``term``, ``terms``, ``range``), terms
``aggs`` over the full match set, ``sort`` by score then doc id, ``from``/``size``, ``search_after`` and ``explain``.
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRANDS = ["Acme", "Zenith", "Nova", "Orbit", "Pixel"]
CATEGORIES = ["Shoes", "Jackets", "Lamps", "Phones"]
NOUNS = {"Shoes": "running shoe", "Jackets": "winter jacket", "Lamps": "desk lamp", "Phones": "smart phone"}
COLORS = ["red", "blue", "black", "green"]


def make_catalogue(n: int = 60) -> list[dict]:
    docs = []
    for i in range(n):
        cat, brand, color = CATEGORIES[i % 4], BRANDS[i % 5], COLORS[(i // 3) % 4]
        docs.append({"product_id": f"P{i:03d}", "product_title": f"{brand} {color} {NOUNS[cat]} model {i}",
                     "product_brand": brand, "product_color": color,
                     "product_bullet_point": f"{NOUNS[cat]} for everyday use" if i % 2 else "",
                     "product_description": "", "category": cat, "price": float(10 + (i * 7) % 90), "stock": i % 5})
    return docs


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


class FakeIndex:
    """``search(request) -> response`` per §5. Doc id (the tie-breaker) is the catalogue position."""

    def __init__(self, docs=None, explain_ok: bool = True):
        self.docs = list(docs if docs is not None else make_catalogue())
        self.explain_ok = explain_ok  # False: explain values that do NOT sum (to test the check)
        self.calls = 0

    # -- matching and scoring ------------------------------------------------------------------------------------
    def _score(self, doc, clause):
        """-> (matched, score, [(description, value)]) for one scoring clause."""
        if not clause or "match_all" in clause:
            return True, 1.0, [("match_all", 1.0)]
        if "multi_match" in clause:
            mm = clause["multi_match"]
            fields = [(f.split("^")[0], float(f.split("^")[1]) if "^" in f else 1.0) for f in mm["fields"]]
            terms = list(dict.fromkeys(_tokens(mm["query"])))
        else:
            ((field, spec),) = clause["match"].items()
            fields, terms = [(field, 1.0)], list(dict.fromkeys(_tokens(spec if isinstance(spec, str) else spec["query"])))
        parts = []
        for t in terms:
            for f, w in fields:
                tf = _tokens(str(doc.get(f, ""))).count(t)
                if tf:
                    parts.append((f"weight({f}:{t}) [BM25F], w={w}", round(w * tf * 1.5, 6)))
        return bool(parts), sum(v for _, v in parts), parts

    def _filter_ok(self, doc, f):
        if "term" in f:
            ((k, v),) = f["term"].items()
            return doc.get(k.removesuffix(".keyword")) == (v["value"] if isinstance(v, dict) else v)
        if "terms" in f:
            ((k, vs),) = f["terms"].items()
            return doc.get(k.removesuffix(".keyword")) in vs
        if "range" in f:
            ((k, r),) = f["range"].items()
            x = doc.get(k)
            return x is not None and all({"gte": x >= v, "gt": x > v, "lte": x <= v, "lt": x < v}[op]
                                         for op, v in r.items())
        if "bool" in f:
            return self._bool_ok(doc, f["bool"])
        if "match_all" in f:
            return True
        raise ValueError(f"fake: unsupported filter {f}")

    def _bool_ok(self, doc, b):
        for key in ("filter", "must_not"):
            fl = b.get(key, [])
            fl = fl if isinstance(fl, list) else [fl]
            oks = [self._filter_ok(doc, f) for f in fl]
            if key == "filter" and not all(oks) or key == "must_not" and any(oks):
                return False
        return True

    def _eval(self, doc, query):
        """-> (matched, score, explain parts) for the whole query."""
        query = query or {"match_all": {}}
        if "bool" in query:
            b = query["bool"]
            if not self._bool_ok(doc, b):
                return False, 0.0, []
            musts = b.get("must", [])
            musts = musts if isinstance(musts, list) else [musts]
            tot, parts = 0.0, []
            for m in musts:
                ok, s, p = self._score(doc, m)
                if not ok:
                    return False, 0.0, []
                tot += s
                parts += p
            if not musts:
                return True, 0.0, [("filter-only", 0.0)]
            return True, tot, parts
        return self._score(doc, query)

    # -- the API --------------------------------------------------------------------------------------------------
    def search(self, request: dict) -> dict:
        self.calls += 1
        q = request.get("query")
        size, frm = int(request.get("size", 10)), int(request.get("from", 0))
        if frm + size > 10000:
            raise ValueError("from + size over 10,000")
        matched = []
        for i, d in enumerate(self.docs):
            ok, s, parts = self._eval(d, q)
            if ok:
                matched.append((i, d, round(s, 6), parts))
        matched.sort(key=lambda m: (-m[2], m[0]))
        after = request.get("search_after")
        if after:
            s0, i0 = after[0], after[-1]
            matched_page = [m for m in matched if m[2] < s0 or (m[2] == s0 and m[0] > i0)]
        else:
            matched_page = matched
        page = matched_page[frm:frm + size]
        hits = [{"_id": d["product_id"], "_score": s, "sort": [s, i], "_source": d} for i, d, s, _ in page]
        resp = {"hits": {"total": len(matched), "hits": hits},
                "took_ms": {"parse": 0.1, "retrieve": 0.5, "total": 0.8}}
        aggs = request.get("aggs") or request.get("aggregations")
        if aggs:
            resp["aggregations"] = {}
            for name, a in aggs.items():
                field = a["terms"]["field"].removesuffix(".keyword")
                counts: dict = {}
                for _, d, _, _ in matched:
                    counts[d[field]] = counts.get(d[field], 0) + 1
                top = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[: a["terms"].get("size", 10)]
                resp["aggregations"][name] = {"buckets": [{"key": k, "doc_count": c} for k, c in top],
                                              "sum_other_doc_count": len(matched) - sum(c for _, c in top)}
        if request.get("explain"):
            resp["explain"] = []
            for i, d, s, parts in page:
                details = [{"value": v, "description": desc, "details": []} for desc, v in parts]
                value = s if self.explain_ok else s + 0.25
                resp["explain"].append({"_id": d["product_id"], "value": value, "description": "sum of",
                                        "details": details})
        return resp


# -- a fake of the serving HTTP API (§7), for the demo page and the HTTP client ------------------------------------
def serve_fake(index: FakeIndex | None = None, name: str = "esci-sample", host: str = "127.0.0.1", port: int = 0,
               extra_routes=None):
    """Start ``POST /indexes/{name}/_search`` + ``/healthz`` on a thread. Returns (server, url). ``server.shutdown()``
    to stop. Errors map to the §7 400 body."""
    index = index or FakeIndex()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "content-type"),
                         ("Access-Control-Allow-Methods", "POST, GET, OPTIONS")):
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            self._send(200, {"status": "ok"}) if self.path in ("/healthz", "/readyz") else self._send(404, {})

        def do_POST(self):
            if self.path != f"/indexes/{name}/_search":
                return self._send(404, {"error": {"type": "not_found", "path": self.path, "reason": "unknown index"}})
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                self._send(200, index.search(req))
            except ValueError as e:
                self._send(400, {"error": {"type": "invalid_request", "path": "from", "reason": str(e)}})

    srv = ThreadingHTTPServer((host, port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://{host}:{srv.server_address[1]}"
