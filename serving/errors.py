"""Named errors for the HTTP front (docs/INTERFACES.md §7).

The engine's exceptions are recognised by class name along the MRO, so serving never needs the engine to import and a
fake index in a test can raise a stand-in with the same name.
"""
from __future__ import annotations


class ServingError(Exception):
    """Base of the errors serving raises itself. ``status`` is the HTTP status, ``type`` the named error."""

    status = 500
    type = "internal_error"

    def __init__(self, reason: str = "", path: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.path = path


class BadJSON(ServingError):
    status, type = 400, "invalid_request"


class InvalidRequestBody(ServingError):
    status, type = 400, "invalid_request"


class NotFound(ServingError):
    status, type = 404, "not_found"


class UnknownIndex(NotFound):
    type = "index_not_found"


class UnknownSegment(NotFound):
    type = "segment_not_found"


class IndexExists(ServingError):
    status, type = 409, "index_exists"


class OverBudget(ServingError):
    status, type = 409, "over_budget"


class NotReady(ServingError):
    status, type = 503, "not_ready"


class Overloaded(ServingError):
    status, type = 503, "overloaded"


class MethodNotAllowed(ServingError):
    status, type = 405, "method_not_allowed"


# engine / dataplane exception class name -> (status, type). Checked along the MRO, most specific class first.
_BY_NAME = {
    "InvalidRequest": (400, "invalid_request"),
    "UnsupportedQuery": (400, "unsupported_query"),
    "InvalidMapping": (400, "invalid_mapping"),
    "OverBudget": (409, "over_budget"),
    "DeviceUnavailable": (409, "device_unavailable"),
    "NotHot": (400, "not_hot"),
    "NoSuchKey": (404, "no_such_key"),
    "CorruptSegment": (500, "corrupt_segment"),
}


def classify(exc: BaseException) -> tuple[int, dict]:
    """Map any exception to ``(http_status, body)`` with ``{"error": {"type", "path", "reason"}}``."""
    if isinstance(exc, ServingError):
        return exc.status, {"error": {"type": exc.type, "path": exc.path, "reason": exc.reason}}
    for cls in type(exc).__mro__:
        hit = _BY_NAME.get(cls.__name__)
        if hit:
            status, type_ = hit
            return status, {"error": {"type": type_, "path": getattr(exc, "path", "") or "",
                                      "reason": getattr(exc, "reason", "") or str(exc)}}
    return 500, {"error": {"type": "internal_error", "path": "", "reason": f"{type(exc).__name__}: {exc}"}}
