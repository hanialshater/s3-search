# The engine runs on any device through one array backend, with numpy on CPU as the reference

owner: architect · date: 2026-11-09 · status: decided · class: engineering

Decided by the coordinator for the founder, who delegated the floor's decisions for this session and asked for a
GPU-portable design ("the founder's requirement"; design 2 approved). The contract is `docs/INTERFACES.md` §0 and §7.

**Decision.**
1. **One backend.** Every operator calls one array backend, `engine/backend/` (`Backend` protocol), and never numpy,
   torch, cupy or CUDA directly. Its surface is the few ops the operators need: CSR range expansion and gather,
   deterministic segment sums (BM25F), matmul and top-k (dense), bitmap pack/unpack and boolean and/or/not (filters),
   bincount (facets), stable lexicographic sort and top-k with ties by doc id (pagination).
2. **Device is configuration**: `"cpu"` (numpy; the reference, always available), `"torch-cpu"` and `"cuda"` (torch,
   imported only when asked for; Colab has torch with CUDA). The same contract tests run on every available device.
3. **Deterministic order across devices**: scores compared in float32, ties broken by internal doc id ascending; doc ids
   follow `first_offset` order, so rebuilds keep it. On one device, responses are bit-identical run to run
   (sort-based reductions, no float atomics). Across devices, scores within `rtol=1e-5` and identical ids outside tie
   groups.
4. **GPU fast paths** (S6: CUDA graphs, fused kernels, a compiled encoder) are optional replacements behind the same
   backend or operator signature, switched by `S3_SEARCH_FAST`, and checked against the reference path.
5. **Memory is planned**: a segment declares `device_nbytes`; loading checks it plus the batch working set fits
   `Backend.memory_budget()`, and the updater checks twice that (double buffer) before a swap, on every device.
6. **Kubernetes never assumes a GPU**: the `SearchIndex` spec carries `device` and `resources`; GPU limits
   (`nvidia.com/gpu`), a nodeSelector and a toleration are added only when `device` is `cuda`; one Dockerfile with a
   build arg picks the CPU or CUDA base image. In this dev container kind cannot run pods (nested runc fails), so the
   operator is tested against a real kube-apiserver and etcd (envtest binaries at
   `/tmp/claude-0/envtest/controller-tools/envtest/`), and the full kind end-to-end runs in GitHub Actions.
7. **torch is not in `requirements.txt`.** The numpy reference keeps CI light, and `download.pytorch.org` is blocked
   here. `requirements-gpu.txt` lists torch for Colab and GPU nodes.

**Why.**
- The prototype (`baseline/esci_gpu_vs_opensearch.ipynb`) already ran its operators as torch code on either device,
  with cuVS and cupy behind a fallback; what it lacked was a reference that CI can run without torch and a rule that
  keeps device code from leaking into every operator.
- The referee's correctness half runs on CPU in CI on 2 cores in under a minute (`CLAUDE.md`); GPU numbers come only
  from Colab. A numpy reference makes the CI path the definition of correct, and the GPU path a claimed equivalent
  that the same tests check.
- The floor's freshness check demands results equal to a rebuild; a tie-break on a stable doc id is what makes that
  checkable at all.
- A swap that runs out of device memory halfway is the worst failure a live index has; checking 2x up front is cheap.

**Alternatives.**
- *torch everywhere (the prototype's way).* One code path, but CI must install torch (hundreds of MB, and its index is
  blocked here) and nothing is a reference independent of the device under test. Rejected; torch stays the GPU
  backend.
- *The Python array API standard (`array_api_compat`) instead of our own protocol.* Close to what we want, but sparse
  CSR expansion, segment sums, bitmaps and deterministic top-k are outside the standard, so we would need our own layer
  anyway, plus a new dependency. Rejected for v1; the protocol keeps array-API names where they exist (`asarray`,
  `zeros`, `argsort`), so moving later is cheap.
- *cupy as the GPU backend.* Good kernels, but a second GPU stack next to torch, which the encoder needs. Kept only
  inside optional fast paths.
- *JAX.* Compiles well, but it is a third stack beside torch (which the encoder needs), and its immutable arrays fight
  in-place hot-column patches. Rejected.

## Objections

- dataplane-engineer (anticipated by the architect): "A numpy reference will be far slower on ESCI-US, so the CPU
  benchmark will not fit 5 minutes." Answer: the CPU benchmark runs on `esci-sample` (about 40,000 products), where a
  dense `Q x N` buffer is small; ESCI-US runs only in Colab, on `cuda`.
- platform-engineer (anticipated): "Tie groups across devices make contract tests fuzzy." Answer: within one device the
  order is exact; only the cross-device comparison uses tie groups, and only for scores within `rtol=1e-5`. Exact
  cross-device equality would forbid every GPU summation order, and the prototype's 0.0002 title-only parity shows the
  differences do not move quality.
- evaluation-steward (anticipated): "Fast paths might drift from the reference unseen." Answer: they are switched by
  configuration and the referee runs both settings on every available device; a fast path that fails the reference
  tests does not land.
- Founder: none raised (delegated).
