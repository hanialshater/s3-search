# deploy/

owner: platform-engineer · S4-4 · contract: `docs/INTERFACES.md` §7 · decision: `docs/decisions/2026-11-09-serving-and-operator.md`

The Kubernetes operator: one `SearchIndex` (`search.s3.dev/v1alpha1`) becomes an indexer Job, a replica Deployment, a
Service and a PVC, with rolling segment swaps and roll back. Standard library only. **Nothing here deploys anywhere:**
it is tested against a local API server and, in CI, a kind cluster inside the runner.

| Path | What |
|---|---|
| `crd/searchindex.yaml` | the CRD (defaults, validation, status and scale subresources, printer columns) |
| `manifests/operator.yaml` | namespace, service account, RBAC, the operator Deployment |
| `s3operator/` | `render.py` (pure: spec to objects), `reconcile.py` (one resource), `controller.py` (watches, queue, resync), `leader.py` (Lease election), `kube.py` (REST client), `__main__.py` |
| `Dockerfile` | one image, `ARG BASE=cpu\|cuda`; build from the repo root |
| `samples/` | the toy index (60 synthetic products, mapping, `SearchIndex`) and a GPU example |
| `tests/` | `test_render.py`, `test_replay.py` (unit, under 1 s); `test_envtest.py` (a real kube-apiserver) |
| `test-envtest.sh` | runs `test_envtest.py` against `etcd` and `kube-apiserver` binaries (`S3_ENVTEST_BIN`) |
| `e2e/` | `run.sh` (the kind scenario), `replay.py` (steady QPS, counts failures) |

```
python -m pytest deploy/tests -q                       # unit; the envtest tests skip
deploy/test-envtest.sh                                  # 19 tests, about 45 s (two include a 15 s lease expiry), needs the envtest binaries and openssl
docker build -f deploy/Dockerfile -t s3-search:cpu .    # --build-arg BASE=cuda for the GPU image
```

How a roll works: the indexer Job is named by a hash of the mapping, source, image and `spec.reindexToken`; it builds a
segment on the shared volume and reports its id in its termination message. The operator puts that id in the Deployment's
pod template, which rolls with `maxUnavailable: 0` and readiness on `/readyz`. `spec.segmentVersion` pins another id: the
same roll in the other direction. A GPU (`nvidia.com/gpu` limits, nodeSelector, toleration) is added only for
`device: cuda`, and only to the replicas.

CI: `.github/workflows/kind-e2e.yml` runs `e2e/run.sh` on a kind cluster. It has not run yet (kind cannot run pods in
the dev container), and it needs the real engine and data plane (S1-6, S2-4).
