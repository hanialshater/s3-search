"""Shared plumbing for the tools: a "search function" is any callable ``request: dict -> response: dict`` (INTERFACES §5).

Two real sources (the serving HTTP API of §7, or ``engine.api`` in process) and recorded run files. Tests use fakes.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable

SearchFn = Callable[[dict], dict]

# The text fields of the §2 example mapping.
DEFAULT_FIELDS = ["product_title^3", "product_brand^2", "product_color", "product_bullet_point", "product_description"]


class ApiError(RuntimeError):
    def __init__(self, status: int, body):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body


def http_search(base_url: str, index: str, timeout: float = 30.0) -> SearchFn:
    """``POST {base_url}/indexes/{index}/_search`` (§7). Errors come back as ApiError carrying the §7 error body."""
    url = f"{base_url.rstrip('/')}/indexes/{index}/_search"

    def search(request: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(request).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                body = json.loads(raw)
            except ValueError:
                body = raw
            raise ApiError(e.code, body) from None

    return search


def engine_search(index) -> SearchFn:
    """In-process: an ``engine.api.Index`` (or anything with ``.search``)."""
    return index.search


def text_request(text: str, fields=None, size: int = 10, **extra) -> dict:
    """The plain-text query the tools and the demo send: ``multi_match`` over the text fields (§5)."""
    req = {"query": {"multi_match": {"query": text, "fields": list(fields or DEFAULT_FIELDS), "type": "cross_fields"}},
           "size": size}
    req.update(extra)
    return req


def hit_ids(response: dict) -> list[str]:
    return [str(h["_id"]) for h in response["hits"]["hits"]]
