"""The query engine (owner: architect). Contract: docs/INTERFACES.md §0 (devices), §4 (segment format),
§5 (request and response), §6 (plan).

Modules: ``engine.backend`` (the array backend, §0), ``engine.analysis``, ``engine.segment`` and ``engine.build`` (§4),
``engine.dsl`` and ``engine.plan`` (§6), ``engine.ops`` (the operators), ``engine.rank``, ``engine.encode``, and
``engine.api`` (§5; added by S1-6, which switches on the referee's contract tests). ``engine`` never imports
``dataplane``, ``serving`` or ``tools``.
"""
