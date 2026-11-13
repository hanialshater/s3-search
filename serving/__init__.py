"""Serving: the HTTP front, the batching scheduler, the query cache, the admin API and metrics (S4).

Owner: platform-engineer. Imports ``engine.api`` and ``dataplane`` lazily, through ``serving.ports``.
"""
from .app import App, create_app, make_server
from .config import ServingConfig

__all__ = ["App", "ServingConfig", "create_app", "make_server"]
