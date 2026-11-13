"""The data plane (owner: dataplane-engineer). Contract: docs/INTERFACES.md §2 (mappings and intake) and §3 (the
key-value store, the change log, the indexer and the updater).

Packages: ``dataplane.intake``, ``dataplane.kv``, ``dataplane.indexer``, ``dataplane.updater``. It writes segments in
the format of §4 through ``engine.segment`` and ``engine.build``, and may import ``engine`` but not ``serving``.
"""
