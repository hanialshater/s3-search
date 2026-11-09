# dataplane/

owner: dataplane-engineer · milestone S2 · contract: `docs/INTERFACES.md` §2 (mappings, intake) and §3 (store, log,
indexer, updater)

The path of a document: a batch file or a change record → **intake** (validate, enrich) → the **store** (SQLite, the
source of truth, with a gapless change log) → the **indexer** (a segment of the store at its head, in the §4 format)
→ the **updater** (a replica's live `Index`: hot fields patched in place, deletes as tombstones, a double-buffered swap
onto each new segment).

| Module | What |
|---|---|
| `dataplane.mapping` | `parse_mapping`, `coerce`; `InvalidMapping` with a JSON path |
| `dataplane.intake` | `read_batch`, `validate`, `enrich`, `ingest`, `apply_changes`; `encoders` (the hook, `HashingEncoder`) |
| `dataplane.kv` | `Store`: `apply`, `head`, `get`, `changes`, `snapshot` |
| `dataplane.indexer` | `build`, `list_segments`, `latest_segment`, `gc`, `MergePolicy`, `reindex`; `python -m dataplane.indexer` |
| `dataplane.updater` | `Replica`: `poll`, `swap`, `start`/`stop`, `stats`; `memory_budget` |
| `dataplane.freshness` | `run`: N hot-field changes, apply-to-visible latency, equality with a rebuild (for B1-4) |
| `dataplane._engine` | the seam to `engine.build`, `engine.segment`, `engine.api` (tests install a fake) |

Tests: `python -m pytest -q dataplane/tests` (about 1 s). Until `engine.api` lands they run against
`dataplane/tests/fake_engine.py`, and `test_integration.py` skips.
