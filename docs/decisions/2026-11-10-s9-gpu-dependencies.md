# S9's GPU libraries (CuPy, XGBoost, cuML, cuVS) are optional, in requirements-gpu.txt only

owner: architect · date: 2026-11-10 · status: decided · class: engineering

Decided by the architect within the founder's S9 rules, relayed by the coordinator: "GPU-friendly features only; no
HNSW; use cuVS and cuML where they fit", with the budget line search-floor-s9 ($400, pause at $800).

**Decision.**
1. **`requirements-gpu.txt` gains `cupy-cuda12x`, `xgboost`, `cuml-cu12` and `cuvs-cu12`** (with NVIDIA's index for
   the RAPIDS wheels). `requirements.txt` gains nothing: CI and the CPU container stay numpy-only, and every one of
   these is imported only inside the code path that needs it, when it is present.
2. **S9-2, the encoder fit on the GPU** (`TfidfSvdEncoder(fit="gpu"|"cuml"|"auto")`, or `"fit"` in the mapping's
   embeddings): the TF-IDF weights are computed as on CPU, and the randomized SVD (the same random start, the same
   three power iterations) runs on the GPU with CuPy's sparse products and QR. cuML's `TruncatedSVD` takes dense
   input only, and a dense TF-IDF matrix of ESCI-US (1.2M x ~200k) does not fit any GPU, so it is not used; cuML's
   `TfidfVectorizer` is not used either, because its tokenizer is not the index's analyzer (stemming, stopwords),
   and query encoding must see the same terms as the fit. The fitted arrays are the same format as the numpy fit's,
   stored in the segment, so every vector comes from one fitted encoder. The default stays numpy; on CPU nothing
   changes.
3. **S9-3, GPU trees**: the engine's GBDT is exported exactly to XGBoost's JSON model (`engine/rank/gpu_trees.py`;
   bins become float32 thresholds with no float32 between, NaN goes left) and run by XGBoost (`device=cuda`) or by
   cuML's FIL from that JSON, chosen with `Index.add_ranker(name, model, runtime="xgboost"|"fil"|"auto")`. The engine's
   own device walk stays the default and the reference. XGBoost accumulates leaves in float32, so its predictions
   differ from the reference by up to a few 1e-6 (measured 2.4e-6 on 10,000 rows here, on CPU XGBoost 3.4.1).
4. **S9-1, cuVS** for approximate knn (CAGRA, IVF-PQ), built per segment and filtered with cuVS's bitset prefilter;
   exact knn is the fallback without cuVS or a GPU, and the response names the method that ran.
5. **The GPU numbers of record** come only from the evaluation steward's S9 cell in `referee/colab_referee.ipynb`
   (a maker never supplies the numbers its claims are judged on). `python -m colab.s9_gpu_check` is the architect's
   development tool for the same measurements (cuVS recall against exact, filtered and not; the GPU fit's time and
   nDCG@10; the trees' differences and rescored orders), labelled development only.

**Why.** The founder asked for GPU-friendly features with cuVS and cuML where they fit, and no HNSW; CI must stay
light and CPU-only; and every GPU path must be checkable against the CPU reference.

**Alternatives.**
- *cuML TfidfVectorizer + TruncatedSVD as the fit.* Rejected for the reasons in 2 (dense-only SVD; another tokenizer).
- *FIL only, no XGBoost.* XGBoost's GPU predictor needs no RAPIDS install and reads the same JSON; FIL is offered too.
- *torch for the GPU SVD.* Possible, but CuPy's sparse CSR products are the natural fit and cuVS needs CuPy anyway.

## Objections

- evaluation-steward (anticipated): "XGBoost's float32 sums may reorder near-ties." Answer: S9's prediction 3 allows
  1e-5 per prediction; an order that differs only between candidates whose reference scores are within 1e-5 is the
  case to watch in the Colab judge, which compares orders on the bench's fit queries.
- Founder: none raised (delegated).
