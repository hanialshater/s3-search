# R4: the GPU process's reader and batch loop in Rust (design note, CPU-only; architect, 2026-11-15)

Status: proposal. Nothing is built. The numbers are development measurements, never scores. py-spy from the next
serving session decides between R4a and R4.

## Why

- **The limit:** plat-m34 (K4, 4 fronts, server CPUs 4–11, A100) is limited by host CPU, not by the GPU.
  - Server CPU is 5.4–5.7 cores on average (p95 6.7–7.3) of 8, with the GPU about 45% busy.
- **The Rust line (R1–R3c) helped latency far more than throughput.**
  - It served 99.99% of requests.
  - At 4,800 QPS, p99 fell from 102 ms to 37 ms.
  - It moved the ladder only one rung (5,200 at p99 59.5 ms).
- **Where a GPU process's CPU goes:**
  - Each GPU process still uses about 0.8 core: about 0.67 ms of CPU per request at about 1,200 QPS.
  - The engine's host part is about 0.29 ms of that (in process, batch 16).
  - The remaining **~0.38 ms per request is the Python serving loop** in `serving/gpu.py`.
  - That loop runs on one GIL, so a GPU process tops out near one core.
- **The fronts:** they use about 2.2 cores (0.46 ms per request). Most of it is rendering 2,000 hits per deep
  response (`front_render_ms` p50 0.43 ms). That is the platform engineer's, and second on the list.

## What the loop does per request today (S3_HOSTPATH=1, S3B3)

| step | where | per request |
|---|---|---|
| read frame head and body, `bytes(payload)`, `search_head` | reader thread per front | Python, about 6 calls |
| `_Item(...)`, `queue.Queue.put`, `_account`, `_publish_load` (a LoadTable write) | reader thread | lock and condition traffic |
| `_collect`: `q.get` / `get_nowait` per item | batch loop | lock and condition traffic |
| `_prepare_raw`: group by index, `hp.prepare` (Rust), map positions to items | batch loop | Python loops |
| `_run_prepared` → `search_prepared` (engine, about 0.29 ms) | batch loop | the engine |
| `_answer`: `flush_response` (10–25 µs), metrics `inc`, tuple building, `_reply` | batch loop | Python |
| `_flush`, `conn.put`, the writer's `out.get`, `batch4` (Rust, about 5 µs), `sendall` | writer thread per front | Python and the GIL |

Every arrow between those threads costs a lock or condition variable and a GIL hand-off. With 4 fronts there are
4 reader threads, 4 writer threads and the batch loop, all contending for the same GIL.

## Two options

### R4a: trim the Python loop (about 1 day; no new Rust)

1. **Fewer locks.** Batch the queue: readers append to a `collections.deque` under one `Condition`, and the batch
   loop takes the whole deque at once. This replaces per-item `queue.Queue` traffic.
2. **Fewer per-request writes.** `_publish_load` becomes once per batch and once per reader wakeup. `_account` and
   `metrics.inc` count once per batch, adding n.
3. **A cheaper flush.** `flush_response` becomes a two-operation bit check with module-level constants (9–12 µs
   against 17–27 µs measured). Better still, the engine flushes the returned page on the device, at no host cost.
   This is the platform engineer's code: agree it with them.
4. **No item objects on the prepared path.** `_prepare_raw` keeps `(conn, id, raw)` tuples, and the eligible
   positions are answered straight from `search_prepared`'s list.

Expected: perhaps a third of the 0.38 ms. Low risk, and the existing serving tests cover it unchanged.

### R4: the loop in Rust, Python touched once per batch (about 5–6 days)

The `s3_hostpath` extension owns the front connections. Python keeps the engine and only sees batches.

- **Reader threads in Rust (no GIL).** `GpuServer` accepts each front's socket as today and passes its fd and
  protocol to `s3_hostpath.Loop.add_conn(fd, proto)`. A Rust thread per connection:
  - reads frames;
  - classifies them: search, call, or not JSON (400 as today);
  - for search frames, runs `HostIndex.prepare` as frames arrive, per index state, with the same eligibility rule.
- **Batch collection in Rust.** `Loop.next_batch(max, wait_ms)` blocks with the GIL released (`py.allow_threads`).
  It returns one object holding:
  - the prepared layout (the ints and floats blocks and the chunks, as `prepare` gives today);
  - the fallback positions' raw bodies, for Python's `search_batch`;
  - the call frames, for the existing calls pool;
  - per position, the connection, the id, the enqueue time and the parse time.
- **Python per batch:**
  - `search_prepared(layout)` once, plus `search_batch` for the fallback bodies;
  - `_source`, explain and highlight stay on the Python path (they are never eligible);
  - then `Loop.reply(batch, responses, version, batch_ms)` once.
- **Replies in Rust.** `reply` walks each response once (today's `batch4` code, about 5 µs a reply), flushes
  subnormals on the column bytes (the one-flush rule, moved here for this path), and queues one kind 4 frame per
  front. A Rust writer thread per connection sends it, off the GIL.
- **The rest moves to Rust or once per batch:**
  - the LoadTable writes (the same file layout);
  - the `X-S3-*` timers (queue_ms, batch_ms, post_ms, parse_ms);
  - the stats counters (`hostpath stats`, batches, requests), as one `Loop.stats()` dict read by `_stats`.
- **Fallbacks:** any Rust error on a frame goes to Python as a raw body, counted as today. A failed batch call
  answers that batch's items with the error mapping `serving/errors.py` gives. `S3_SERVING_R4=0` (the default)
  keeps today's loop untouched.

What Python does per request under R4: only what the engine itself does for a response, plus about 5 µs of reply
walking. The GPU process's host CPU should drop from about 0.67 ms to about 0.35 ms per request, roughly doubling
each process's headroom. The fronts' about 2.2 cores then become the largest consumer.

Rough serving expectation, to be checked against predictions: the 8 server cores then hold perhaps 6,500–7,500 QPS
before the CPU saturates. The device ceiling is about 7,000 (fused) to 8,200 (impacts) QPS per A100 at batch 256
(arch-prof39). A GPU-side win (item 2, the on-chip accumulator) would then matter again.

## Effort for R4 (architect; Rust built on Colab CPU per the founder's rule)

| part | days |
|---|---|
| Rust: connections, framing, readers and writers, batch collector (max and wait), LoadTable, stats | 2 |
| Python: an R4 mode of `GpuServer` (accept, batch thread, fallbacks, calls, versions, errors, shutdown) | 1.5 |
| Tests: byte equality with today's loop on the same frame streams (`serving/tests/test_gpu_process.py` under `S3_SERVING_R4=1`), concurrency, fallbacks, error mapping, slow-front back-pressure | 1.5 |
| CPU CI sessions, then one serving measurement on a GPU (about 1.5 units), with predictions filed first | 0.5 |
| **total** | **about 5.5** |

## Risks and what needs agreeing

- `serving/gpu.py` and the protocol are the platform engineer's. R4 changes who reads the sockets, not the wire
  format: the native front needs no change.
- **GIL correctness:** Rust may hold no Python object across `allow_threads`. Responses are walked under the GIL,
  then sent without it.
- **Back-pressure:** a slow front must not stall the batch loop. Today an unbounded writer queue per front absorbs
  that. R4 keeps a per-connection queue and counts its depth.
- Calls (`_mapping`, bulk, patches) keep their Python path. Their ordering relative to searches on one connection
  must stay as today: per connection, in arrival order.
- **Measuring the effect:** today's per-request timers stay, but their source changes. A same-session A/B
  (`S3_SERVING_R4=0` against `=1`) is the only fair comparison.

## Recommendation

Take py-spy of GPU process 0 and front 0 at 4,800 in the next serving session, about 20 s at 100 Hz. If the
queue, lock and per-request bookkeeping frames dominate the 0.38 ms, do R4a first (1 day) and measure. Go to R4
when R4a leaves most of it, or when the target is past about 6,000 QPS on this box.
