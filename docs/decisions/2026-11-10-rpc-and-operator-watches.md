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
