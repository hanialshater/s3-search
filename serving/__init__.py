"""Serving, the admin API and observability (owner: platform-engineer). Contract: docs/INTERFACES.md §5 (the request
and response it carries) and §7 (the HTTP API, the admin API and the ``SearchIndex`` resource; the operator is in
``deploy/``).

It calls the engine only through ``engine.api`` and the data plane through ``dataplane``.
"""
