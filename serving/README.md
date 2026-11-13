# serving/

owner: platform-engineer · S4-2, S4-3 · contract: `docs/INTERFACES.md` §7 · decision: `docs/decisions/2026-11-09-serving-and-operator.md`

The HTTP front over `engine.api`, on the standard library: routes (`_search`, `_msearch`, create, `_docs`, `_reindex`,
`versions`, `_rollback`, `healthz`, `readyz`, `metrics`), the batching scheduler, the LRU query cache, stage timings and
counters, and named errors.

| File | What |
|---|---|
| `app.py` | `create_app`, `App.handle` (the whole API), `make_server`, SIGTERM drain |
| `manager.py` | indexes on disk, swaps, replay of the change log, `_source`, segment pruning |
| `scheduler.py` | `BatchScheduler`: groups concurrent requests into one `Index.search_batch` (`max_batch`, `max_wait_ms`) |
| `cache.py`, `metrics.py`, `errors.py`, `config.py` | LRU; Prometheus text; named errors; configuration |
| `ports.py` | the five functions serving needs from the engine and data plane; defaults import them lazily |
| `__main__.py` | `python -m serving` (serve) and `python -m serving build` (the indexer Job) |

```
python -m serving --data-dir data/serving --preload myindex          # serve (S3_SEARCH_DEVICE=cpu|torch-cpu|cuda)
python -m pytest serving/tests -q                                    # about 2 s; a fake Index that follows §5
```

Tests live in `serving/tests/` against `fakes.py` (an index and a data plane that follow §3 to §5). The test that runs the
real engine skips until `engine.api` imports. Not part of `python -m referee fast`.
