# Serving runs on the standard library, and the operator is a standard-library Python reconciler

owner: platform-engineer · date: 2026-11-09 · status: decided · class: engineering

Decided by the platform-engineer under the founder's brief for this session (build the whole floor; GPU-portable;
develop Kubernetes here). The contract is `docs/INTERFACES.md` §7. Adds no dependency to `requirements.txt`.

**Decision.**
1. **Serving is stdlib.** `http.server.ThreadingHTTPServer` (HTTP/1.1 keep-alive, one thread per connection) in front of
   a routing `App.handle(method, path, body, headers) -> Response`. The app is not ASGI: `create_app(config) -> App`
   (§7 amended). Concurrent connections are what the **batching scheduler** groups into one `Index.search_batch`
   (a worker thread: first request, then more until `max_batch` or `max_wait_ms` since the first).
2. **Where the work is.** The engine does the compute, in numpy or torch, and both release the GIL in their kernels;
   the HTTP front only parses JSON, so a thread-per-connection server is not the bottleneck at the S4 target (50 QPS
   on kind; the latency gate is measured by the referee, not asserted here). If the benchmark shows the front costs
   more than the engine, the swap is to `uvicorn` behind the same `App.handle`, with its own decision record.
3. **The operator is Python, standard library only**: a small REST client (`urllib`, server-side apply with field
   manager `s3-operator`) and a level-triggered loop that lists every `SearchIndex` each few seconds and reconciles it.
   It lives in `deploy/s3operator/` and ships in the same image as serving.
4. **The indexer Job is `python -m serving build`** (§7 amended; the draft said `dataplane.indexer`). It is the same
   ingest-then-build the `PUT /indexes/{name}` route runs, as a command, and it reports `{segment_id, version}` in
   the container's termination message, so the pod needs no API access and the operator needs no volume access.
5. **The segment id in the pod template is the whole rollout mechanism.** A new segment, or a pinned `segmentVersion`
   (a roll back), changes one annotation and one env var, and the Deployment's rolling update
   (`maxUnavailable: 0`, `maxSurge: 1`, readiness on `/readyz`, SIGTERM that fails readiness and drains
   `S3_SERVING_DRAIN_SECONDS` before stopping) does the rest. The operator never deletes a serving pod.
6. **GPU only for `device: cuda`**, and only on the replicas: `nvidia.com/gpu` limits (from `spec.resources`, else 1),
   `nodeSelector nvidia.com/gpu.present: "true"` and the `nvidia.com/gpu` toleration. The indexer builds with the numpy
   reference and never asks for a GPU. A GPU quantity in `spec.resources` with another device is refused
   (`Valid: False`, phase `Failed`), not silently dropped.
7. **One Dockerfile**, `ARG BASE=cpu|cuda`: `python:3.12-slim` with `requirements.txt`, or
   `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` with `requirements-gpu.txt`.
8. **Tests.** Unit (serving against a fake index that follows §5, kept in `serving/tests/`; render and validation)
   under 3 s. The operator's behaviour is tested against a real kube-apiserver and etcd started from the envtest
   binaries (`deploy/test-envtest.sh`, marker `envtest`, outside `referee fast`), with a simulator that plays the Job
   controller, the Deployment controller and the kubelet, which do not run in the dev container. The full kind run
   (`.github/workflows/kind-e2e.yml`, `deploy/e2e/run.sh`) is a CI test only.

**Why.**
- *No framework yet.* `requirements.txt` is numpy, pandas, pyarrow and pytest, and the rule is a decision record per
  dependency. The stdlib server is enough for the contract, keeps CI light, and the `App.handle` seam makes a later
  swap a small change.
- *Why not kopf or the official client.* kopf pulls an async stack (aiohttp, and the client or pykube) and decorators
  that hide the loop; the official client is large. The operator needs six verbs (get, list, apply, create, patch
  status, delete) on seven kinds. Server-side apply makes each write idempotent, so a polling loop is correct without
  informers or a work queue, and testing it against a real apiserver needs no mocking of a client library.
- *Why not Go and controller-runtime.* It is the production default for operators and Go is installed here, but it adds
  a second language, a second toolchain and a second image build to a repository that is otherwise Python, for a
  controller that is about 400 lines. If the operator grows (webhooks, leader election, many CRDs), port it; the CRD
  and the rendered objects do not change.
- *Termination message for the result.* The alternatives were a ConfigMap written by the Job (needs RBAC in the Job's
  pod and a second source of truth) or the operator reading the volume (needs it mounted). The termination message is
  one Kubernetes field, already part of the pod's status, and survives as long as the pod does; the operator copies it
  into `status.segments`, so it only needs it once.

**Alternatives.** FastAPI and uvicorn for serving (rejected for now, see 2); an event-driven operator with watches
(rejected: a few seconds of reconcile latency is invisible next to an index build, and polling survives a restart
with no state); a Helm chart instead of an operator (cannot compare a build result with the live segment, or roll
without a template change).

## Objections

- architect (anticipated): "`http.server` will not hold the p99." Answer: the front does JSON and queueing; the
  referee measures end to end on CPU and Colab. A measured regression is the trigger to swap, not a guess.
- dataplane-engineer (anticipated): "The indexer Job should be `python -m dataplane.indexer`." Answer: `dataplane` owns
  `build(store, mapping, out_dir)`; the Job needs ingest, build and a result report, which is serving's admin
  create flow as a command. The Job calls your functions, through `serving/ports.py`; nothing of yours moves.
- founder (anticipated): "Polling is not how operators are done." Answer: it is level-triggered like any reconciler;
  only the trigger differs. A watch is an optimisation behind `reconcile_all`.

## Known limits (v1alpha1)

- One operator replica, no leader election. Reconciling is idempotent, so two would only repeat work.
- The segment volume is one PVC shared by the Job and the replicas: `ReadWriteOnce` works on one node (kind), a
  multi-node cluster needs `storage.accessModes: [ReadWriteMany]`.
- Pruned segments (`keep_segments`, default 5) cannot be rolled back to; `status.segments` lists the last 10 built.
- `PUT /indexes/{name}` and the admin routes of a replica pod write to its own volume view; in the cluster the
  operator, not the admin API, is the path for create, reindex and roll back.
