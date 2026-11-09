"""The planner (docs/INTERFACES.md §6). In S1 parsing and planning are one pass (``engine.dsl.parse``), because the
parser needs the index's field kinds; S3 may split them."""
from __future__ import annotations

from .nodes import Plan


def plan(request: dict, state) -> Plan:
    from ..dsl.parse import parse
    return parse(request, state)
