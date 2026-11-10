# RPC is length-prefixed JSON over TCP, and the operator moves to watches with a Lease

owner: platform-engineer · date: 2026-11-10 · status: decided · class: engineering

Board item S7-7. Adds no dependency.

## RPC serving

**Decision.** `serving/rpc.py`: a frame is a 4-byte big-endian length and a UTF-8 JSON object. A request names a method
and path exactly as the HTTP route does (`{"id", "method", "path", "body", "headers"}`), the response is
`{"id", "status", "body", "headers"}`, and both transports call the same `App.handle`, so the responses are the same by
construction. Listener: `S3_SERVING_RPC_PORT` / `--rpc-port` (off by default; the operator sets 9090 and exposes it on
the Service as `rpc`). Python client: `serving.rpc.RpcClient` (one connection shared by threads, one redial, typed
methods, `RpcError` with the named error) and a connection per thread for concurrency.

**Why not gRPC.** grpcio and protobuf are two new dependencies with generated code and a build step, to carry a body
that is already a JSON query DSL; a protobuf schema would wrap one string field. Its real advantages (HTTP/2
multiplexing, cross-language stubs) are not needed by the S7 prediction, which is parity of responses. **Why not
msgpack:** a dependency for a speed-up nobody has measured; JSON keeps the RPC debuggable with `nc`. If a non-Python
client or multiplexing is wanted, the transport is one file and `App.handle` does not change; gRPC would then get its own
record.

**Limits.** One request at a time per connection (no multiplexing, no TLS, no auth): in-cluster use, like the HTTP port.
Frames over 64 MB are refused with a named error.

## The operator: watches and a Lease

**Decision.** `deploy/s3operator/controller.py`: watches on `SearchIndex` and on the Deployments and Jobs the operator
owns (label `managed-by=s3-operator`, mapped to their index by `search.s3.dev/index`) feed a de-duplicating queue; one
worker reconciles; a resync (default 30 s) enqueues everything as the backstop, which also catches ConfigMap edits (not
watched). Each watcher lists, then watches from the list's resourceVersion, and relists on error or 410.
`deploy/s3operator/leader.py`: a `coordination.k8s.io/v1` Lease (`s3-operator`), client-go semantics: a candidate takes
over after seeing the Lease unchanged for `lease_duration` by its own clock; the holder renews every `retry_period` and
stops leading if it cannot renew for `renew_deadline`; flags default to 15 s, 10 s, 2 s; a clean stop releases the Lease.
Only the leader runs a Controller (a new one per term). RBAC adds `watch` and the `leases` rule; the manifest runs 2 replicas.

**Why.** Polling cost a few seconds of latency per change and listed everything every pass; watches make the reaction
event-time (status changes of an owned Deployment or Job reconcile its index at once). Two replicas without election
would both write; the Lease makes the standby idle. It is hand-written rather than a library because the stdlib client
is the decision already taken (`2026-11-09-serving-and-operator.md`).

**Measured (envtest, a real kube-apiserver, default timings; `deploy/tests/test_watches.py`, `test_leader.py`).**
Spec change to status: p50 60 ms, max 150 ms (resync at 3600 s, so it is the watch). Job finish to status: 109 ms.
Two replicas: every change reconciled by the leader only (18 reconciles on the leader, 0 on the standby), max 60 ms.
Kill -9 of the leader: the standby held the Lease after 15.0 s and reconciled the next change 45 ms later. SIGTERM of the
leader: handover in 1.1 s.

## _bulk and refresh (S8-6)

**Decision.** `POST _bulk` streams NDJSON through `serving/bulk.py` in chunks (default 2000 actions) into
`dataplane.intake.apply_changes`; the refresh is a hook (`Ports.refresh`) that is a full rebuild until S8-4 provides
deltas; the background refresher keeps at least 4x the last refresh's cost between refreshes in rebuild mode, so a stream
of writes cannot keep the index rebuilding more than 20% of the time. **Why a chunk and not the whole body:** a 66 MB body
is not held in memory, the store's batches stay small, and visibility of patches and deletes follows each chunk.
**Measured** (2 cores via `taskset`, over HTTP, `python -m serving.bench_bulk`): 48,439 documents (plus a 200-document seed
index = ESCI-sample's 48,639) in 2.9 to 5.9 s over 6 runs (8,200 to 16,600 docs/s; the 5.9 s run was the first, cold), 0 errors; a following full-rebuild
refresh 18.7 s.
