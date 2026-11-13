# Search floor: the interfaces

owner: architect · status: **v1, decided** (architect, 2026-11-09; design 2) · moved here from the lab's
`docs/search-floor/INTERFACES.md` (a skeleton), which now points here.

The parts, their boundaries and what crosses them (P5: contract first). A change that crosses a boundary updates this
file in the same commit. Sections 2 and 3 were drafted by the architect for the dataplane-engineer and section 7 for
the platform-engineer; each owner may amend its sections, recording why at the end of this file.

## Parts and owners

One owner per area. Nobody edits another area's paths except by a claimed board item (`docs/BOARD.md`).

| Part | Owner | Paths | Sections it owns |
|---|---|---|---|
| Query engine: analysis, operators, DSL, planner, ranking, the API, the segment format | architect | `engine/` | 0, 4, 5, 6 |
| Data plane: intake, key-value store and change log, indexer, GPU updater | dataplane-engineer | `dataplane/` | 2, 3 |
| Serving, admin API, observability, Kubernetes operator | platform-engineer | `serving/`, `deploy/` | 7 |
| Tools and the demo page | tools-engineer | `tools/`, `demo/` | (a client of 5 and 7) |
| Referee: datasets, loader, metrics, contract tests, benchmark, Colab | evaluation-steward | `referee/`, `datasets/` | 1 |

Import direction (enforced by review, and by a test S1 adds): `referee` imports `engine.api` only;
`dataplane` imports `engine.analysis`, `engine.segment`, `engine.build`, `engine.backend`, `engine.errors` and
`engine.api` (for `Index.from_segment`; ruling of 2026-11-09); `serving` imports `engine.api`, `engine.errors`
and `dataplane`; `tools` imports `serving`'s HTTP API or `engine.api`. **`engine` never imports `dataplane`, `serving`
or `tools`.** Nobody imports `referee` except `referee.loader` (the data rule) and the referee itself.

## 0. Devices

Decision record: `docs/decisions/2026-11-09-gpu-portable-backend.md`.

- **One array backend.** Every operator gets a `Backend` (`engine/backend/__init__.py`) and calls only its methods
  plus the array's own arithmetic operators (`+ - * / < <= & | ~`, indexing with an integer or boolean array). No
  operator imports numpy, torch, cupy or CUDA directly.
- **Device is configuration**, one string:

  | device | arrays | availability |
  |---|---|---|
  | `"cpu"` | numpy | always; **the reference** |
  | `"torch-cpu"` | torch on CPU | when `import torch` works |
  | `"cuda"` | torch on CUDA (cupy only inside optional fast paths) | when `torch.cuda.is_available()` |

  `engine.backend.get_backend(device: str | None = None) -> Backend`; `None` reads `S3_SEARCH_DEVICE`, else `"cpu"`.
  `engine.backend.available_devices() -> list[str]` (always starts with `"cpu"`); `engine.api.available_devices()` is
  the same function. An unavailable device raises `engine.backend.DeviceUnavailable`.
- **The same contract tests run on every available device** (the referee parametrises on `available_devices()`).
- **Determinism.** Scores are float32. Ranked order is by the sort keys, then **internal doc id ascending** (§4). On one
  device a request gives the same response every time: reductions are deterministic (sort-based segment sums, no
  float atomics). Across devices scores agree to `rtol=1e-5`; ids agree exactly except inside groups whose scores are
  within that tolerance, which the contract compares as tie groups.
- **Fast paths** (S6: CUDA graphs, fused kernels, a compiled encoder) replace a backend method or an operator behind
  the same signature, are switched by `S3_SEARCH_FAST=0|1`, and are checked against the `"cpu"` path by the same tests.
- **Memory is planned, not discovered.** A segment declares `device_nbytes` (§4). `Backend.memory_budget() -> int`
  is the bytes the index may use on that device (`S3_SEARCH_DEVICE_BUDGET` overrides; default 80% of free CUDA memory,
  50% of available RAM on CPU). Loading checks `device_nbytes + work_nbytes <= budget`; the updater checks
  `2 * device_nbytes + work_nbytes <= budget` (double buffer) before it builds the second copy, on every device, and
  refuses with `engine.backend.OverBudget` otherwise.
- **What crosses to the host per search** (checked on every device by `engine/tests/test_gpu_paths.py`): the page
  (doc ids, sort values and scores stacked into one transfer) and, per terms aggregation, its buckets (one transfer);
  explain may read more. Rescoring stays on the device (`GBDT.predict_backend`: binning and tree walks as gathers, one
  step per depth level, trees added in the same order as the numpy `predict`, so both give equal float32 values).
  Numeric fields' value codes (facets) are computed on the device once per segment and field, and again only after a
  patch to that field. Query-side setup (small query vectors, bounds) may be built on the host.
- **Accumulation** (`get_backend(device, accumulate=)`, `Index(..., accumulate=)`, or `S3_SEARCH_ACCUM`):
  `"float64"` is the reference default. `"float32"` is an option for GPUs whose FP64 is slow (T4, L4: 1/32 rate):
  BM25F's segment sums stay deterministic (the same sort-based kernel) and **batch-invariant** (a doc's addends keep
  their order whatever else is in the batch); the dense matmul becomes a float32 GEMM, which is **not** promised to
  be batch-invariant (cuBLAS may choose kernels by shape), so with float32 the `search_batch == [search]` rule holds
  for lexical requests and holds within `rtol=1e-5` for dense ones. Scores differ from float64 by at most `rtol=1e-5`
  on the tests. The default does not change until S6 measures the gain in Colab.
- **torch is optional.** `requirements.txt` stays numpy-only; `requirements-gpu.txt` adds torch for Colab and GPU
  nodes.

## 1. Datasets (evaluation-steward → everyone)

Served only through `referee/loader.py`; nobody reads ESCI files any other way, and nobody loads test judgments.

| Name | Size | Built by | Use |
|---|---|---|---|
| `toy` | about 200 products, judgments, under 1 s | B1-1 | contract tests, every device |
| `esci-sample` | judged products of 2,000 train and 500 test queries (about 40,000 products) | B1-2 | CPU development and the CPU benchmark |
| `esci-us` | 1.22M products, 8,956 test queries | `fetch()` | Colab referee only |

Existing functions: `load_catalogue(data_dir, locale) -> DataFrame[product_id, product_title, product_brand,
product_color, product_bullet_point, product_description]`, `load_fit(...) -> DataFrame[query_id, query, product_id,
esci_label]`, `load_test_queries(...) -> DataFrame[query_id, query]`.

What the engine needs from every dataset (the steward names the functions in B1-1/B1-2, for example
`load_dataset(name) -> (catalogue, fit, test_queries)`):
- a catalogue `pandas.DataFrame` with one row per product, a unique string id column (`product_id`) and the text
  columns above;
- for filter, facet and sort tests (`toy`, optionally `esci-sample`): extra columns `price` (float), `stock` (int,
  hot), `category` (string keyword);
- the mapping (§2) that goes with it, as a dict, so the referee and the makers build the same index.

## 2. Mappings and intake (outside → data plane) · owner dataplane-engineer

**The mapping is an OpenSearch mapping body**, so the conformance suite (B1-5) sends the same dict to OpenSearch.
Engine-only settings live under `_meta.engine`, which OpenSearch stores and ignores. Field names are catalogue column
names.

```json
{
  "_meta": {"engine": {
    "id_field": "product_id",
    "hot": ["price", "stock"],
    "bm25": {"k1": 1.2, "b": 0.75},
    "embeddings": {"embedding": {"model": "tfidf-svd-128", "inputs": ["product_title", "product_brand", "product_color"]}}
  }},
  "properties": {
    "product_title":        {"type": "text", "analyzer": "s3_english"},
    "product_brand":        {"type": "text", "analyzer": "s3_english",
                             "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
    "product_color":        {"type": "text", "analyzer": "s3_english",
                             "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
    "product_bullet_point": {"type": "text", "analyzer": "s3_english"},
    "product_description":  {"type": "text", "analyzer": "s3_english"},
    "category":             {"type": "keyword"},
    "price":                {"type": "float"},
    "stock":                {"type": "integer"},
    "embedding":            {"type": "knn_vector", "dimension": 128,
                             "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}}
  }
}
```

- **BM25F parameters (`_meta.engine.bm25`):** `k1` (>= 0, default 1.2) and `b` (in [0, 1], default 0.75). `b` is
  one number for every text field, or a map of text field -> b with `b_default` (default 0.75) for the fields it does
  not name: the prototype's tuned BM25F has its own b per field (`referee/requests/bm25f-tuned.bm25.json`). Anything
  else is an `InvalidMapping` naming the path. `referee full --bm25=<file>` sets them for a run and records them.
  Text fields inside a `nested` field (S9-5) are scored with the defaults (k1 1.2, b 0.75) whatever the parent sets,
  and the map cannot name them (v1; noted 2026-11-13 by the architect).
- **Types (v1):** `text` (searchable; BM25F field), `keyword` (term/terms filters, terms aggs, sort; single-valued
  in v1), `float`, `integer`, `long`, `boolean` (term/range filters, sort), `knn_vector` (exact dense; the `method`
  is accepted for OpenSearch and ignored: the engine is exact). Sub-fields are addressed as `product_brand.keyword`.
- **Analyzer:** one, `s3_english` (the prototype's: lowercase, `[a-z0-9]+(?:'[a-z0-9]+)?` words, strip a trailing `'s`, then `'`, English
  stopwords, Porter stemmer via `snowballstemmer`, which S1 adds to `requirements.txt`). Code:
  `engine.analysis.analyze(text: str, analyzer: str = "s3_english") -> list[str]`. The conformance suite defines the
  same analyzer in the OpenSearch index settings.
- **nested** (S9-5): `{"type": "nested", "properties": {sub: text | keyword | numeric}}` (no nesting inside, no
  vectors or geo). A catalogue value is a list of objects (or one object). Each object is a child document of its
  own, in a child segment stored under `nested/<path>/` (its fields named `path.sub`, plus `_parent`, the parent's
  row; its `first_offset` is `parent_first_offset << 16 | i`, so children keep their parents' order; at most 65,535
  per parent). The children of each path form an index of their own (statistics over live children), kept in step
  with the parents: a child dies with its parent copy, and deltas, merges, saves and commits carry the children.
  A leaf query on `path.sub` outside a `nested` query matches no docs (as OpenSearch); a sort or aggregation on one
  is refused.
- **geo_point** (S9-4): `{"type": "geo_point"}` (no options). Values `{"lat", "lon"}`, `[lon, lat]` or `"lat,lon"`
  (a geohash or an out-of-range point is `InvalidMapping`). Stored with Lucene's encoding as two int32 columns
  (`num/<field>#lat.npy`, `num/<field>#lon.npy`, hidden from requests), and compared by their decoded values, as
  OpenSearch does.
- **Synonyms** (`_meta.engine.synonyms`, S7-6): Solr-style lines, applied at query time: `"a, b, c"` makes them
  equivalent, `"a, b => c, d"` replaces a or b by c or d, `#` starts a comment; entries are analysed with the
  index's analyzer and must be one token each (multi-word synonyms are `InvalidMapping`). In `match`,
  `multi_match` and `match_bool_prefix` a term with synonyms is one clause of all its alternatives, scored as
  Lucene's SynonymQuery (term frequencies add before saturation; the lowest idf of the alternatives); in
  `match_phrase` a slot matches any alternative; highlight marks the alternatives. `fuzzy`, `prefix` and terms with
  fuzziness are not expanded by synonyms.
- **Positions** (`_meta.engine.positions`, default `true`; S7-1): index each term's positions, for phrases. Every
  word takes a position, stopwords included (dropped, leaving a gap, as Lucene's stop filter does), so
  `"red the lamp"` is red@0, lamp@2 (`engine.analysis.analyze_with_positions`).
- **Hot fields** (`_meta.engine.hot`): numeric or boolean fields that may be patched in place (§3, §4). Only these
  (a keyword patch could need a new dictionary code, so keywords change by upsert and rebuild in v1).
- **Vectors:** if the catalogue has a column named like the `knn_vector` field holding float arrays, they are used
  (L2-normalised at intake). Else `_meta.engine.embeddings` names an encoder (`engine.encode.get_encoder(name)`; v1:
  `tfidf-svd-128`, fitted on the catalogue at build, and `bge-small-en-v1.5` where the network allows) and intake
  enriches. Neither: the field is absent and `knn` on it raises `UnsupportedQuery`.
- Malformed or unsupported mappings raise `engine.api.InvalidMapping(ValueError)` with the JSON path.

**Intake** (`dataplane/intake/`):

```python
def read_batch(path: str | Path, mapping: dict) -> Iterator[dict | Rejection]   # .parquet or .jsonl; one source doc
                                                                   # per row; a malformed JSONL line is a Rejection
def validate(doc: dict, mapping: dict) -> dict | Rejection         # coerced doc, or why not
def enrich(docs: list[dict], mapping: dict, encoder: Encoder | dict[str, Encoder] | None = None) -> list[dict]
def ingest(path, mapping, store: Store, rejected: Path, encoder=None) -> IntakeReport  # read → validate → enrich → apply
def apply_changes(changes: Iterable[dict], mapping, store, encoder=None, rejected=None) -> ChangeReport  # the feed
@dataclass
class Rejection: _id: str | None; reason: str; field: str | None; detail: str
@dataclass
class IntakeReport: accepted: int; rejected: int; first_offset: int; last_offset: int; reasons: dict[str, int]
@dataclass
class ChangeReport: accepted: int; first_offset: int; last_offset: int; rejected: list[tuple[dict, Rejection]]
class Encoder(Protocol): name: str; dim: int; def encode(self, texts: list[str]) -> float32[n, dim]
```

Rejection reasons (closed set, additive): `missing_id`, `duplicate_id_in_batch`, `bad_type`, `dims_mismatch`,
`unknown_field` (the mapping's `dynamic` is `strict`), `too_long` (keyword over `ignore_above` is *kept* but not indexed
as keyword, as OpenSearch does; this reason is for ids over 512 bytes), `malformed` (a JSONL line or change record that
is not an object, a patch without `fields`), `bad_op`, `not_hot` and `no_such_key` (the change feed), and for S9
`bad_geo`, `geo_out_of_range`, `bad_nested` and `too_many_nested`. Rejected records go to `rejected.jsonl`, one
`{"record": ..., "rejection": ...}` per line.

- **geo_point at intake (S9-4):** `{"lat", "lon"}` (numbers or numeric strings), `"lat,lon"`, `[lon, lat]` and WKT
  `"POINT (lon lat)"` (any case) are stored as `{"lat": float, "lon": float}`, the form `build_segment` reads. A
  geohash, another shape or a non-finite number is `bad_geo`; a latitude outside [-90, 90] or a longitude outside
  [-180, 180] is `geo_out_of_range`. Not hot: a location changes by upsert.
- **nested at intake (S9-5):** an object or a list of objects is stored as a list of child objects, each sub-field
  coerced by the nested mapping (`bad_type` and the other reasons name the field `path.sub`); missing sub-values are
  left out; an empty list is no children; a non-object item is `bad_nested`; more than 65,535 objects is
  `too_many_nested`; unknown sub-fields are kept in the source unless the mapping is `strict` (`unknown_field`). An
  upsert (and so `_bulk`'s `update` with a partial doc naming the path, which serving merges at the top level)
  replaces the whole array, as OpenSearch does. The snapshot's columns hold the point dicts and the lists of child
  objects; deltas, merges and commits carry them through `build_segment` and `merge_segments`.

- **Coercion.** `text`: strings, numbers (as text), lists of strings (joined by a newline). `keyword`: strings or
  numbers. `float`: finite numbers or numeric strings, within float32 range. `integer`/`long`: integers, integral
  floats or integer strings, within int32/int64. `boolean`: booleans or `"true"`/`"false"`. `knn_vector`: a list or
  array of `dimension` finite numbers, not all zero, L2-normalised to float32. Booleans are never numbers. `null` and
  NaN mean missing: the field is left out.
- **Unknown fields** are kept in the stored source but not indexed (OpenSearch's `dynamic: false`) unless the mapping
  is `strict`.
- **A batch file is one atomic batch** (one `store.apply`, consecutive offsets). Within it, the first valid record of
  an id wins; later ones are `duplicate_id_in_batch`. The change feed (`apply_changes`, §7's `_docs`) rejects bad
  records one by one and commits the rest as one batch; ids repeat freely there, in order.
- **Encoders.** `encoder=None` resolves each vector field's `_meta.engine.embeddings.<field>.model`: a factory
  registered with `dataplane.intake.encoders.register(name, factory)`, else **the engine's encoder**
  (`engine.encode.get_encoder`, v1 `tfidf-svd-<d>`), else the deterministic `HashingEncoder` (`hash-<dim>`; `knn`
  only). The encoder reads the engine's text form (`engine.encode.doc_texts`: inputs joined by `" | "`). A fittable
  encoder is fitted **once per store**, by the first batch that needs it, on every accepted doc of that batch (as
  `build_index` fits on its catalogue), and its arrays are kept in the store (`store.blob("encoder.<field>")`); later
  batches and the change feed reuse it (terms outside its vocabulary do not count, as for any fitted encoder). A store
  records each vector field's encoder name (`store.meta("encoder.<field>")`) and refuses to mix in another one.
  The indexer passes that name as `build_segment(..., vector_models=)` and puts the fitted arrays in the segment
  (`SegmentArrays.encoders`, `vector_fields.<f>.encoder = true`), so `neural` queries work on a data-plane segment.

## 3. Key-value store and change log (intake → indexer and updater) · owner dataplane-engineer

The store is the source of truth. **SQLite (stdlib), one file, one writer.** Module `dataplane/kv/`.

- **Key:** the document's `_id` (UTF-8 string, at most 512 bytes).
- **Value:** the validated, enriched document as JSON (vectors as lists of float32), plus `first_offset` (the offset
  of the upsert that first created this key; kept across later upserts and patches, reset by delete then re-insert)
  and `last_offset`.
- **Change record** (one JSON object; the log is the ordered list of them):

```json
{"offset": 1042, "op": "upsert", "_id": "B0001", "doc": {"product_title": "...", "price": 9.99}, "ts": "2026-11-09T12:00:00Z"}
{"offset": 1043, "op": "patch",  "_id": "B0001", "fields": {"price": 8.49, "stock": 0}, "ts": "..."}
{"offset": 1044, "op": "delete", "_id": "B0002", "ts": "..."}
```

- **Offsets:** int64, assigned by the store at commit, start at 1, strictly increasing by 1 with no gaps. A batch of
  changes commits atomically and gets consecutive offsets. `0` means "before any change".
- **Rules:** `upsert` carries the whole document. `patch` touches only hot fields of an existing live key; anything else
  is refused (`NotHot`, `NoSuchKey`) and not logged. `delete` of a missing key is logged and is a no-op.

```python
class Store:
    def __init__(self, path: str | Path, mapping: dict): ...          # refuses a store created with another mapping
    def apply(self, changes: Iterable[dict]) -> tuple[int, int]        # (first, last) offset committed; (0, 0) if empty
    def head(self) -> int                                              # last committed offset
    def get(self, ids: Sequence[str]) -> list[dict | None]
    def changes(self, after: int, limit: int = 10_000, docs: bool = True) -> list[dict]  # offset > after, ascending;
                                                                       # docs=False leaves upserts' doc out
    def snapshot(self, at: int | None = None) -> pandas.DataFrame     # live docs as of offset `at`, ordered by first_offset
    def meta(self, key) -> str | None; def set_meta(self, key, value); def count(self) -> int; def close(self)
class StoreError(ValueError); class NotHot(StoreError); class NoSuchKey(StoreError, KeyError); class BadChange(StoreError)
```

- **Atomic batches.** A refused change (`NotHot`, `NoSuchKey`, `BadChange`: unknown op, missing or over-long id, a
  patch value that does not coerce, a vector of the wrong shape) refuses its whole batch: nothing is logged, so offsets
  stay gapless. The store trusts intake to validate upserts; `dataplane.intake.apply_changes` is the checked entry.
  Every thread gets its own SQLite connection (WAL); writers serialise on `BEGIN IMMEDIATE`.
- **The snapshot frame.** Columns: the id field; every mapped top-level field (`text`, `keyword`: str or None;
  `float`: float64, NaN when missing; `integer`, `long`, `boolean`: int32, int64, bool, or pandas' nullable `Int32`,
  `Int64`, `boolean` when any value is missing; `knn_vector`: float32 arrays or None); then `_first_offset` and
  `_last_offset` (int64). Unmapped fields are not columns. `df.attrs["version"]` is the offset it is at. At head it
  reads the docs table; at an earlier offset it replays the log.

- **The indexer** (`dataplane/indexer/`): `build(store, mapping, out_dir, device_budget=None) -> Path` takes
  `snapshot(head)`, calls `engine.build.build_segment(snapshot, mapping)`, sets `seg.manifest["version"]` to that head
  offset, and writes it with `engine.segment.write_segment` into `out_dir/.tmp-*`, renamed to
  `out_dir/<segment_id>` when complete (it accepts a writer that treats the path as the segment directory or as its
  parent). Doc ids are row order of the snapshot, i.e. ascending `first_offset`, so a rebuild keeps the relative order
  of surviving documents (ties stay stable). `device_budget` refuses a segment whose `device_nbytes` exceeds it
  (`OverBudget`). v1 has one segment per index, so **every build is a full merge**: deletes are dropped and statistics
  recomputed. Also `list_segments`, `latest_segment`, `gc(out_dir, keep=3, protect=())`, `MergePolicy(max_deleted_ratio,
  max_pending, max_age_s).due(replica.stats())`, `reindex(store, out_dir, replica)` (§7 `_reindex`), and the job
  `python -m dataplane.indexer --store --mapping --out [--ingest FILE]`.
- **The updater** (`dataplane/updater/`; S8): `Replica(store, segment_dir, device="cpu", budget=None, batch=32,
  refresh_interval=0.5, merge_policy=TieredMergePolicy(), commit_dir=None, commit_every=5.0, keep_commits=3)` holds
  the live `Index`, a base segment plus delta segments; `segment_dir` is a commit file, a segment, or a directory (its
  newest commit, else its newest segment). Loading checks `device_nbytes + work_nbytes <= budget`, then refreshes.
  - `refresh()` (`poll()` is the same step) = `dataplane.updater.delta.apply_delta(index, store, index.version)`: one
    `Store.delta` read; patches to keys not upserted in the range → `Index.patch` (one call per field, the last value
    wins); if any key was upserted, its docs at head become one delta segment (`build_segment(..., version=head,
    vector_models=)`, with the store's encoder) added by `Index.add_segment(delta, version=head, delete_ids=deleted)`,
    which tombstones the older copies; else deletes → `Index.delete` and `set_version(head)`. Upserts are visible
    after the refresh, not after a rebuild (`pending_upserts` is 0).
  - `merge()` = `delta.merge_once`: `TieredMergePolicy.plan(index.segments)`; the budget check
    (`index.device_nbytes + inputs' device_nbytes + work_nbytes <= budget`, else the merge waits); `snapshot_segments`
    under the replica's lock, `merge_segments(..., version=V)` outside it, then under the lock the records in
    `(V, index.version]` for the merged keys are replayed onto the merged arrays (patches into its hot columns;
    upserted or deleted keys become `delete_ids`) and one `replace_segments`.
  - `commit()` = `Index.save_commit(commit_dir)` (the delta segments are written with it), then `gc_commits`.
    **The commit hole (§4) is closed by replay on load:** loading a commit replays the patches logged after its oldest
    segment's version, each only onto keys whose live copy is in a segment built before that patch; then the log
    after the commit is refreshed.
  - `start(interval=None, merges=True)`: a refresher thread every `refresh_interval` and a merger thread (merges when
    due; commits every `commit_every` s and after each merge, with `commit_dir`). `stop()`.
  - Serving's `Ports.refresh` hook is `dataplane.updater.serving_refresh(managed_index)`: one `apply_delta` from the
    oldest pending upsert, then one merge if due.
  `swap(new_segment_dir)` checks `current device_nbytes + new device_nbytes + work_nbytes <= budget` (the double
  buffer, §0; 2x when the two are the same size) and raises `engine.backend.OverBudget` otherwise, loads the second
  `Index`, replays changes after its version (outside the lock, then the rest under it), then swaps one reference;
  searches in flight finish on the old one. A swap to an older commit or segment is the rollback. `budget=None` is
  `Backend.memory_budget()`. Also `stats()` (with `segments`, `deleted`, `merges`), `live_ids()`, `on_visible(fn)`.
  Freshness = time from `apply` returning to the refresh making it visible.
- **The freshness hook** (for B1-4): `dataplane.freshness.run(store, replica, n_changes=10_000, batch=100,
  interval=0.005, queries=None, check_rebuild=True) -> FreshnessReport` writes hot-field patches to random live ids,
  measures each change's apply-to-visible latency (p50, p99, max) and compares every query's response with a fresh
  rebuild's (`compare_responses`: totals, ids in order, scores at `rtol=1e-5`, sort values but the last element, the
  internal doc id).

## 4. Segment format (indexer → engine) · owner architect

A segment is a **directory** of `.npy` arrays (numpy format, mmap-able) and `.parquet` string tables, described by
`manifest.json`. Code: `engine/segment.py`.

```python
@dataclass
class SegmentArrays:            # the in-memory form; numpy arrays, host side
    manifest: dict
    ids: numpy.ndarray          # object[N], the _id of each doc id
    text: TextArrays            # term_ptr int64[V+1], doc_ids int32[P], tf uint8[P,F], field_len int32[N,F],
                                # df int32[V], df_field int32[V,F], avg_len float32[F], vocab: list[str] (sorted)
    keywords: dict[str, KeywordColumn]    # codes int32[N] (0 = missing), values: list[str] (values[0] == "", sorted)
    numerics: dict[str, NumericColumn]    # values float32|int32|int64|bool[N], missing: bitmap
    vectors: dict[str, numpy.ndarray]     # float16 or float32 [N, D], L2-normalised
    live: numpy.ndarray                   # bitmap uint32[ceil(N/32)]
def build_segment(catalogue: pandas.DataFrame, mapping: dict) -> SegmentArrays   # engine/build.py, the reference builder
def write_segment(seg: SegmentArrays, path: str | Path) -> Path
def read_segment(path: str | Path, verify: bool = True) -> SegmentArrays         # raises CorruptSegment
```

Layout (`seg-<version>-<8 hex>/`):

| File | dtype, shape | Meaning |
|---|---|---|
| `ids.parquet` | `_id` string, `first_offset` int64; N rows | row = internal doc id |
| `live.npy` | uint32 [ceil(N/32)] | live bitmap, bit `d` = word `d >> 5`, bit `d & 31` (LSB first) |
| `text/vocab.parquet` | `term` string; V rows, sorted | row = term id |
| `text/term_ptr.npy` | int64 [V+1] | CSR row pointers: postings of term `t` are `[term_ptr[t], term_ptr[t+1])` |
| `text/doc_ids.npy` | int32 [P] | ascending within each term |
| `text/tf.npy` | uint8 [P, F] | term frequency per text field (F in manifest order), clipped at 255; a posting exists if any field's tf > 0 |
| `text/field_len.npy` | int32 [N, F] | tokens per field |
| `text/df.npy`, `text/df_field.npy` | int32 [V], int32 [V, F] | document frequency, overall and per field |
| `text/avg_len.npy` | float32 [F] | mean field length |
| `text/pos_ptr.npy`, `text/positions.npy` (optional, S7-1) | int64 [P+1]; uint32 [T] | positions, CSR over postings: posting p's are `positions[pos_ptr[p]:pos_ptr[p+1]]`, each `(field << 24) \| position`, ascending; absent in segments built before S7 or with `positions: false`, which still load (phrases are then refused) |
| `present/<field>.npy` (optional, S7-9 fix) | uint32 bitmap | for text and keyword fields: the docs with a stored value (exists); older segments fall back to tokens (text) and codes (keyword) |
| `src/<field>.parquet` (optional, S7-5) | `text` string; N rows | a text field's raw text, for highlight (host only; `_meta.engine.source`, default true); older segments load without it (highlight is then refused) |
| `kw/<field>.codes.npy` | int32 [N] | 0 = missing; codes order = sorted value order |
| `kw/<field>.values.parquet` | `value` string | row = code; row 0 is `""` |
| `num/<field>.npy` | float32, int32, int64 or bool [N] | the column |
| `num/<field>.missing.npy` | uint32 bitmap | missing values |
| `vec/<field>.npy` | float16 or float32 [N, D] | L2-normalised vectors |
| `vec/<field>.encoder.terms.parquet`, `.encoder.idf.npy`, `.encoder.components.npy` | string; float32 [V]; float32 [V, D] | the fitted query encoder (`tfidf-svd-<d>`), when the engine produced the vectors |

`manifest.json`:

```json
{"format": "s3seg", "format_version": 1, "segment_id": "seg-1044-3fa2c9d1", "version": 1044,
 "n_docs": 1215854, "created": "2026-11-09T12:00:00Z", "analyzer": "s3_english@1",
 "mapping": {"...": "the mapping dict"}, "mapping_sha256": "…",
 "text_fields": ["product_title", "product_brand", "product_color", "product_bullet_point", "product_description"],
 "keyword_fields": ["product_brand.keyword", "product_color.keyword", "category"],
 "numeric_fields": {"price": "float32", "stock": "int32"}, "vector_fields": {"embedding": {"dim": 128, "dtype": "float16", "model": "tfidf-svd-128"}},
 "hot": ["price", "stock"],
 "files": {"text/tf.npy": {"sha256": "…", "bytes": 123, "dtype": "uint8", "shape": [123, 5]}},
 "nbytes": 0, "device_nbytes": 0,
 "checksum": "sha256 of the canonical JSON (sorted keys, no spaces) of `files`"}
```

- **Version** = the store offset the segment includes up to; an `Index` built straight from a catalogue has version 0.
- **Statistics** (`df`, `avg_len`, `N`) are stored per segment as built; the index's statistics are global over live
  documents (S8, below). The idf is `log(1 + (N - df + 0.5) / (df + 0.5))`.
- **`device_nbytes`** = bytes of every array the engine moves to the device (all of the above but the string tables)
  plus derived arrays (idf). `work_nbytes` (per batch) is `Q * N * 4 * 2` for the lexical score buffer and match counts
  at the configured batch size.
- **Checksum:** `read_segment(verify=True)` recomputes each file's sha256 and the manifest checksum; any mismatch
  raises `engine.segment.CorruptSegment`.
- v1 was **one segment per index**. S8 makes an index an ordered list of v1 segments (the format of each is unchanged);
  see "Multi-segment indexes (S8)" below.

**Multi-segment indexes (S8; ruling on S8-1, `docs/proposals/2026-11-10-s8-1-multi-segment.md`).**
- An index is an ordered list of segments, with a commit point `commit-<version>.json` (`{"format": "s3commit",
  "format_version": 1, "version", "segments": [...], "deletes": {segment_id: "del-<version>-<segment_id>.npy"},
  "checksum"}`); a delete file is a uint32 live bitmap. Each `_id` is live in at most one segment.
- **Nested children in a commit (S9-5):** a delete file covers the segment's parent docs only. A child's liveness is
  never stored in a commit; it is re-derived on load: a child is live exactly when its parent copy is live (and its
  own segment's `live`, if any, is set). So a commit taken after an edit or a delete reloads with the old copy's
  children dead, and each child `_id` is live in at most one segment, as each parent `_id` is.
- **Statistics are global and over live documents only** (this replaces "statistics include deleted documents until
  the next build", also for one segment): N = live docs; `df` and `df_field` summed over segments minus the deleted
  docs' terms; `avg_len_f` = int64 sum of live `field_len_f` / N in float64, then float32; idf in float64 then float32.
  Recomputed on add, replace and delete, never per query. Term ids are global and append-only; keyword values have a
  global dictionary per field (facet keys are strings, sorts use the global rank). One encoder per vector field.
- **Doc order is `first_offset` order**: the engine places docs in append-only slots ordered by `first_offset` (an
  edit keeps its slot), so ties break by `first_offset` ascending across segments, and a rebuild has the same order.
- API (additive): `Index.from_segments(paths, device)`, `Index.from_commit(path, device)`,
  `Index.add_segment(seg | path, version, delete_ids=())` (atomic; unknown delete ids ignored),
  `Index.replace_segments(old_ids, new | None, version=None, delete_ids=())`, `Index.remove_segment(segment_id)`,
  `Index.snapshot_segments(segment_ids) -> (segments, version)` (references to immutable arrays, copies of live bits
  and hot columns), `Index.segments` (`[{segment_id, version, n_docs, live, device_nbytes}]`),
  `Index.save_commit(dir) -> Path`, `engine.build.merge_segments(segs, mapping) -> SegmentArrays`.
- **One set of BM25F parameters per index** (2026-11-13): every part is scored with the first segment's `k1` and `b`
  (one number or the per-field map, §2), so `Index.add_segment`, `replace_segments` and `from_segments` refuse, with a
  `ValueError` naming both, a segment built with other parameters, as they already refuse other fields or another
  encoder. To change `_meta.engine.bm25`, rebuild every segment.
- **The commit checksum** (S8-2, the data plane's definition adopted): sha256 of the canonical JSON (sorted keys,
  separators `,` and `:`) of every field of the commit except `checksum`, with one more key,
  `"delete_sha256": {segment_id: sha256 of that delete file's bytes}`, hashed but not stored. The engine has its own
  reader and writer of these bytes (`engine/commit.py`; the engine does not import the data plane); the shared test
  vector is `engine/tests/test_s8_segments.py::test_commit_checksum_vector` (a commit at version 7 of
  `seg-5-aaaa0000` and `seg-7-bbbb1111`, the first with the delete file `np.save` of
  `uint32 [0xFFFFFFFE, 0x00000003]`).
- **A commit holds deletes, not hot-column patches**: a patch applied after a segment's build lives only in memory,
  so a replica that loads a commit must replay the log from the oldest segment's version (patches are idempotent),
  or the data plane must rebuild or merge the segment before committing. **Chosen (dataplane-engineer, 2026-11-10):
  replay on load** (§3, "The updater"): a replica loading a commit replays the patches logged after its oldest
  segment's version, each only onto keys whose live copy is in a segment built before that patch.
- `Index.save(path)` writes one segment of the live docs (a merge); `Index.n_docs` is the live count.

The engine side (`engine/api.py`):

```python
class Index:
    @classmethod
    def from_segment(cls, path: str | Path, device: str = "cpu") -> "Index"
    version: int; n_docs: int; device: str; mapping: dict; device_nbytes: int
    def search(self, request: dict) -> dict
    def search_batch(self, requests: list[dict]) -> list[dict]
    def patch(self, field: str, ids: Sequence[str], values: Sequence) -> None   # hot fields only; NotHot otherwise
    def delete(self, ids: Sequence[str]) -> None                                # clears live bits
    def set_version(self, version: int) -> None                                 # the updater, after a poll
    def save(self, path) -> Path          # a segment of the current state (patches, deletes, version)
    def add_ranker(self, name: str, model) -> None   # for rescore.s3_ranker
    def candidate_features(self, request: dict, window: int) -> tuple[list[str], numpy.ndarray]
```

The S1 rules for the data plane (settled 2026-11-09):

- `build_segment(catalogue, mapping, version=0, vector_models=None)` reads only mapped columns and the id column;
  other columns, including the store's `_`-prefixed ones, are ignored, except `_first_offset` (per doc; `0..N-1` when
  absent). Numeric columns may be numpy dtypes, object columns with `None`, or pandas' nullable `Int32`, `Int64`,
  `Float32`, `boolean`; missing values go to the missing bitmap. Vectors given as a column may have `None` rows
  (stored as zero vectors). `vector_models={field: name}` (or `catalogue.attrs["vector_models"]`) names the encoder
  that produced given vectors and is written as `vector_fields.<f>.model`; without it the model is `"given"`. A
  `neural` query needs an encoder stored in the segment (only `tfidf-svd-<d>` fitted at build in v1); with given
  vectors only `knn` works.
- **The version:** `build_segment(..., version=)` sets `manifest["version"]`; a caller may also set
  `seg.manifest["version"]` before writing. `write_segment` writes whatever it holds and derives
  `segment_id = seg-<version>-<first 8 hex of the checksum>` from it and the content.
- `write_segment(seg, path)`: `path` is the segment directory itself; it must not exist or be an empty directory
  (`FileExistsError` otherwise); written to a sibling `<name>.tmp`, then renamed. `Index.save(path)` follows the same
  rule.
- `Index.patch` and `Index.delete`: `NoSuchKey` (a `KeyError`) for ids the segment never held; already-deleted ids
  are ignored by both (a patch to a deleted id is not applied). `None` or NaN in a patch makes the value missing.
  An id repeated in one `patch` call takes its last value, on every device (2026-11-13: cuda does not order a
  scatter with repeated indices, so the engine keeps one write per row before it scatters).
- **Concurrency (S8-2c, copy-on-write):** an `Index` is thread-safe and searches never wait on writers. A search (or
  `search_batch`, `candidate_features`) reads the current state reference once and runs on that state to the end,
  without a lock. Writers (`patch`, `delete`, `set_version`, `add_segment`, `replace_segments`, `remove_segment`,
  `save`, `save_commit`) are serialised by a writer lock; each changes a clone of the state (a `patch` copies only
  the patched hot column; a structural change copies the slot columns and, on first write, the touched segments'
  live bits; immutable arrays are shared) and publishes it by swapping one reference. **Visibility rule:** a change
  is visible to every search that starts after the writer's call returns, and to no search that started before it;
  a search sees one whole state, never part of a change.

## 5. The search request and response (serving → engine) · owner architect

**The API** (`engine/api.py`; the referee's contract tests are written against exactly this):

```python
def build_index(catalogue: pandas.DataFrame, mapping: dict, device: str = "cpu") -> Index
def available_devices() -> list[str]
# engine/errors.py (imports nothing; all re-exported by engine.api; serving matches them by class name)
class UnsupportedQuery(ValueError):   # .path (JSON path, e.g. "query.bool.must[1].fuzzy"), .reason
class InvalidRequest(UnsupportedQuery) # malformed but in-subset (bad types, from+size over 10,000, unknown field)
class InvalidMapping(ValueError)      # .path, .reason
class NotHot(ValueError)              # Index.patch on a field not in _meta.engine.hot
class NoSuchKey(KeyError)             # Index.patch / Index.delete with an _id the segment never held
class DeviceUnavailable(RuntimeError) # also engine.backend.DeviceUnavailable (the same class)
class OverBudget(MemoryError)         # also engine.backend.OverBudget
class CorruptSegment(IOError)         # also engine.segment.CorruptSegment
Index.search(request: dict) -> dict
Index.search_batch(requests: list[dict]) -> list[dict]   # same result as [search(r) for r in requests], batched
Index.from_segment(path, device) -> Index
```

**The request**: a subset of the OpenSearch query DSL. Anything else, including unknown keys, raises
`UnsupportedQuery` naming its path.

| Key | Supported | Notes |
|---|---|---|
| `query.match` | `{field: "text"}` or `{field: {"query", "operator": "or"\|"and", "minimum_should_match", "boost"}}` | text fields: BM25 on one field; on a keyword or numeric field it is a `term` (as OpenSearch) |
| `query.multi_match` | `query`, `fields` (`"f^w"` boosts), `type` `best_fields`\|`most_fields`\|`cross_fields`, `operator`, `minimum_should_match`, `tie_breaker` | **all types are scored as BM25F** with field weights = boosts (the prototype's scorer); hit sets follow `operator`/`minimum_should_match` over the analysed query terms |
| `query.bool` | `must`, `should`, `filter`, `must_not`, `minimum_should_match`, `boost` | scores: sum of `must` and `should`; `filter`/`must_not` do not score |
| `query.term`, `query.terms` | keyword, numeric, boolean fields | constant score `1.0 * boost` in a scoring context (alone, `must`, `should`); 0 in `filter` and `must_not` |
| `query.range` | `gte`, `gt`, `lte`, `lt` on numeric and keyword fields | constant score, as `term` |
| `query.knn` | `{field: {"vector": [...], "k": int, "filter": query, "method": "exact", "method_parameters": {}}}` | exact; `_score = (1 + cos) / 2` (OpenSearch `cosinesimil`); `filter` is a pre-filter |
| knn `method` (S9-1; on `knn` and `neural`) | `"exact"` (default), `"cagra"`, `"ivf_pq"`; `method_parameters`: cagra `itopk_size`, `search_width`; ivf_pq `n_probes` (positive integers; others `InvalidRequest`) | cagra and ivf_pq run cuVS on a `cuda` index with cuVS installed: an index per segment and field, built on first use and built afresh for a merged segment; the `filter` and deletes are cuVS's bitset prefilter; each segment returns up to `k` docs, and the top `k` of those by exact similarity are the hits (their `_score` is exact, only the set found is approximate). Elsewhere the method runs as `exact`. The response says which ran (`knn`, below). **Version tolerance:** the filter argument of cuVS's `search` and the bitset constructor (`cuvs.neighbors.filters.from_bitset`) are discovered from the installed cuVS, never from its version. **Post-filter:** when a segment has a filter or deletes and the method's `search` takes no filter (cuVS 26.06's `ivf_pq.search`), that segment asks for `k' = min(n, 8192, ceil(k / max(selectivity, 0.01)) * 2)` candidates (selectivity: the segment's valid docs / its docs), drops those failing, and keeps the top `k` by exact similarity; if fewer than `k` pass, that segment answers exactly. A segment with nothing to filter is searched without a filter. If cuVS fails (a build or a search raises), the clause runs exact |
| `query.match_all` | `{}` | score 1.0 |
| `query.match_phrase` (S7-1) | `{field: "text"}` or `{field: {"query", "slop": 0, "boost"}}` | text fields with positions; Lucene's rule: some choice of one position per term with `max(v) - min(v) <= slop`, `v = position - query offset` (stopwords keep their offsets); score = BM25 with the phrase frequency as tf and the sum of the terms' field idf (with slop > 0 the frequency is `sum 1 / (1 + matchLength)` over matching starts, an approximation of Lucene's); on a keyword or numeric field it is a `term` |
| `query.fuzzy` (S7-2) | `{field: "v"}` or `{field: {"value", "fuzziness": "AUTO", "prefix_length": 0, "max_expansions": 50, "transpositions": true, "boost"}}` | term-level (the value is not analysed); expanded in the field's terms dictionary (`engine/dictionary.py`): Levenshtein, adjacent transpositions count 1, at most 2 edits, `AUTO` = 0 / 1 / 2 edits for 1-2 / 3-5 / 6+ characters (`AUTO:low,high` too), the `max_expansions` most similar terms (ties by term order); text fields: the BM25 scores of the expansions present add (Lucene blends their df; ours does not); keyword fields: a term over the expanded values (constant score); numeric fields refused |
| `match` / `multi_match` `fuzziness` (S7-2) | `fuzziness`, `prefix_length`, `max_expansions`, `transpositions` | each analysed query term becomes a clause of its expansions in the queried fields' dictionary; `operator` and `minimum_should_match` count clauses; fuzziness that resolves to 0 everywhere is a plain match; `cross_fields` with fuzziness is `InvalidRequest` (as OpenSearch) |
| `query.prefix` (S7-2) | `{field: "v"}` or `{field: {"value", "boost"}}` | term-level; text fields: every indexed term of the field starting with it; keyword fields: every value starting with it; constant score `1.0 * boost` |
| `query.match_bool_prefix` (S7-2) | `{field: "text"}` or `{field: {"query", "operator", "minimum_should_match", "max_expansions": 50, "boost"}}` | text fields; analysed terms but the last are BM25 clauses, the last is a prefix clause (its first `max_expansions` terms in term order, constant 1.0); combined like `match` |
| `query.exists` (S7-3) | `{"field", "boost"}` | text: a stored value, even one that analyses to no token; keyword: a stored value (an empty string included; over `ignore_above` counts as none); numeric: not missing; knn_vector: a non-zero vector; constant score |
| `query.geo_distance` (S9-4) | `{"distance": "12km" or metres, field: point, "distance_type": "arc", "boost"}` | haversine on OpenSearch's mean earth radius (6,371,008.7714 m) from the point to the doc's decoded location, `<=` the distance; units mi, yd, ft, in, km, m, cm, mm, nmi/NM; `plane`, `validation_method` refused; constant score |
| `query.nested` (S9-5) | `{"path", "query", "score_mode": "avg", "boost"}` | `query` runs on the path's children (fields `path.sub`); a parent matches when one of its live children does; its score is the children's scores by `score_mode`: avg, sum, max, min, or none (0); `inner_hits`, `ignore_unmapped` refused |
| `nested` aggregation (S9-5) | `{"nested": {"path"}, "aggs": {...}}` at the top level | `{"doc_count": children of the matching parents, ...}`; its sub-aggregations run on those children (a terms under it may have one level of its own) |
| `query.geo_bounding_box` (S9-4) | `{field: {"top_left", "bottom_right"} or {"top_right", "bottom_left"} or {"top", "left", "bottom", "right"}}` | Lucene's box: encoded bounds (bottom and left rounded up, top and right down), inclusive; left > right crosses the dateline; top < bottom is `InvalidRequest`; `wkt` refused; constant score |
| `_geo_distance` sort (S9-4) | `{"_geo_distance": {field: point, "order": "asc", "unit": "m", "distance_type": "arc"}}` | the distance in `unit`; a doc without a location sorts as +infinity (last ascending, first descending, as OpenSearch) and its sort value is `None`; one point only; `mode` refused |
| `query.ids` (S7-3) | `{"values": [...], "boost"}` | the docs with these `_id`s; constant score |
| `post_filter` (S7-3) | a query | filters the hits and `hits.total` after aggregations are computed (multi-select facets) |
| `collapse` (S7-3) | `{"field"}`: keyword or numeric | the top hit per value in sort order (missing values form one group); from/size count groups; `hits.total` counts docs; each hit gains `"fields": {field: [value]}`; with `search_after` refused (`UnsupportedQuery`), with `rescore` `InvalidRequest` |
| metric aggregations (S7-4) | `min`, `max`, `avg`, `sum`, `stats`, `value_count`, `cardinality`: `{"field"}` | numeric fields (value_count and cardinality also keyword); values read as float64 (float fields' float32 values exactly); `{"value": x}`, `None` for min/max/avg with no values, 0.0 for sum; `stats` is `{count, min, max, avg, sum}`; cardinality is exact; `missing`, `script` and the like refused |
| `range` aggregation (S7-4) | `{"field", "ranges": [{"from", "to", "key"}]}` | buckets sorted by (from, to); `from` inclusive, `to` exclusive (compared in float64, unlike the range query's float32 bounds, as OpenSearch); key `"<from>-<to>"` with `*` for an open end (Python's float repr, e.g. `"10.0-20.0"`) unless given; `keyed` refused |
| `histogram` aggregation (S7-4) | `{"field", "interval", "offset": 0, "min_doc_count": 0}` | key `floor((v - offset) / interval) * interval + offset` (a float), ascending; with min_doc_count 0 the empty buckets between the first and last are returned; bounds and `keyed` refused |
| sub-aggregations (S7-4) | `"aggs"` inside a `terms` aggregation, one level | any of the above per bucket, over that bucket's docs |
| `highlight` (S7-5) | `{"fields": {field: {...}} or [{field: {...}}], "fragment_size": 100, "number_of_fragments": 5, "pre_tags": ["<em>"], "post_tags": ["</em>"], "type": "plain", "require_field_match": true}` | text fields, from the segment's source text (`src/`); the plain highlighter: words whose analysed token is a term of the query's clauses on that field (all fields when `require_field_match` is false; never `must_not`'s; fuzzy, prefix and synonym expansions included) are wrapped in the tags; fragments cut at the token that ends past each multiple of `fragment_size`, the best `number_of_fragments` by distinct terms returned in text order, only fragments with a term; `number_of_fragments: 0` returns the whole text; each hit gains `"highlight": {field: [fragments]}` (fields without a fragment left out); other types and options refused |
| `query.function_score` (S7-5) | `{"query", "functions": [{"filter", "field_value_factor": {"field", "factor": 1, "modifier": "none", "missing"}, "weight"}], "score_mode": "multiply", "boost_mode": "multiply", "max_boost", "min_score", "boost"}`, or one function at the top level | each function applies where its filter matches; `field_value_factor` = modifier(factor x value) (none, log, log1p, log2p: base 10; ln, ln1p, ln2p; square, sqrt, reciprocal), times `weight`; a doc without the value and no `missing`, or a negative, infinite or NaN result, is `InvalidRequest` (OpenSearch errors too); score_mode multiply, sum, avg (sum / sum of weights), first, max, min over the applying functions, 1 where none applies; capped at `max_boost`; boost_mode multiply, replace, sum, avg, max, min with the query score; times `boost`; `min_score` drops docs below it; other functions (script_score, random_score, decay) refused |
| `query.neural` (S3) | `{field: {"query_text", "k"}}` | encoded by the field's encoder |
| `query.hybrid` + `rank.rrf` (S1-5, extension) | `{"hybrid": {"queries": [...]}}`, `"rank": {"rrf": {"rank_constant": 60, "window_size": 20000}}` | RRF fusion; `_score` is the RRF score |
| `rescore` (S1-5, extension) | `{"window_size": 1000, "s3_ranker": {"model": "<name>"}}` | the learned ranker over the top window |
| `aggs` / `aggregations` | `{name: {"terms": {"field", "size": 10}}}` on keyword and numeric fields | over the **full match set**, independent of from/size; buckets by count desc, then key asc; keys are strings (keyword), ints (integer, long), floats (float), bools (boolean) |
| `sort` | `"_score"`, `{"_score": {"order"}}`, `{field: {"order": "asc"\|"desc", "missing": "_last"\|"_first"}}`, `"field"` | default `[_score desc]`; ties always end by doc id asc |
| `from`, `size` | ints, `from + size <= 10,000` | default 0, 10 |
| `search_after` | the last hit's `sort` list | with `from` 0 |
| `explain` | bool | adds `explain` |
| `_source`, `track_total_hits` | accepted and ignored (totals are always exact; serving adds `_source` from the store) | |

Scoring, exactly:
- BM25F per document: `pseudo_tf = sum_f w_f * tf_f / (1 - b + b * len_f / avg_len_f)`;
  `score = sum_terms idf * pseudo_tf * (k1 + 1) / (k1 + pseudo_tf)`. Single-field `match` is the same with one field and
  that field's idf (`df_field`). Query terms are deduplicated. **Against OpenSearch** (2.19, default BM25, which keeps
  the `(k1 + 1)` factor), single-field scores differ by design in three ways, reproduced to 1.4e-7 by the steward's
  `referee/lucene_model.py`: OpenSearch's N counts only documents that have the field, its avgdl averages over those
  documents, and it stores field lengths in one byte (lossy above 24 tokens). The engine uses all N documents, the
  mean over all documents, and exact lengths.
- **idf:** `match` uses its field's df (`df_field`). `multi_match` uses the global df (the term in any text field of
  the mapping, the segment's `df`), except a `multi_match` naming exactly one field (with a weight above 0), which uses
  that field's df (the prototype's `single_field` rule). Field weights never change the df.
- **Constant scores:** `term`, `terms`, `range` and `match_all` score `1.0 * boost` in a scoring context, and 0 in
  `filter` and `must_not`. A `bool` with only `filter`/`must_not` clauses scores 0. (OpenSearch scores a keyword
  `term` by idf; the conformance suite compares hit sets there, not scores.)
- `bool`: `must` and `should` scores add; without `must` or `filter`, at least one `should` must match unless
  `minimum_should_match` says otherwise; with them, `should` is optional.
- `minimum_should_match`: an int, a negative int (all but n), a percentage rounded down, a negative percentage; for
  `match` it counts analysed query terms including unknown ones, so `operator: and` with an unknown term matches
  nothing.
- **Values by field type** (as OpenSearch): a float field's term value and range bounds are rounded to float32 and
  compared with the stored float32 (so a price equal to a bound is inside `gte` and `lte`); an integer field refuses a
  fractional term value (`InvalidRequest`) and treats a fractional bound by comparison (`gte 2.5` is `gte 3`); a
  boolean field takes `true`/`false` or `"true"`/`"false"`; a keyword field takes the value's string.
- `boost` multiplies a clause's score. Scores are float32.

**The response:**

```json
{"hits": {"total": 1234,
          "hits": [{"_id": "B0001", "_score": 12.5, "sort": [12.5, 40211]}]},
 "aggregations": {"brands": {"buckets": [{"key": "Levi's", "doc_count": 31}], "sum_other_doc_count": 112}},
 "took_ms": {"parse": 0.1, "plan": 0.0, "retrieve": 3.2, "filter": 0.4, "fuse": 0.0, "rank": 0.0, "facet": 0.6,
             "page": 0.2, "total": 4.6},
 "explain": [{"_id": "B0001", "value": 12.5, "description": "sum of", "details": []}]}
```

- `hits.total` is an int (exact). `_score` is a float; for a non-score sort it is still the query score.
- `sort` is on every hit: the values of the sort keys, then the internal doc id as the last element (the tie-breaker
  that makes `search_after` exact). Clients pass it back verbatim. From S8 the last element is the doc's
  `first_offset` (the global tie key; for an index built from a catalogue it is 0..N-1, the doc id, as before). A missing field value sorts as `None`. Under
  `rescore`, the first value is the tier (1 inside the rescored window, 0 below it), so the window comes first.
- `aggregations` is present only when requested. `took_ms` always has `total`, plus each stage that ran; stage names
  are the closed set above plus `encode` (additive). Inside a `search_batch` of two or more requests, a request's own stages are host times
  (the device is not synchronised between plans, S6-11); its shared stages, and every stage of a single request, are
  device times as before.
- `explain` (when requested) is a list in hit order, Lucene-shaped: `{_id, op, value, description, details}` per hit,
  with per term and per field contributions and, under `rescore`, each feature's contribution. Every node has `op`,
  which says how its value comes from its `details` (`description` is for people; tools check by `op`):
  - `"sum"`: the sum of the details' values; `"product"`: their product; `"max"`: the largest;
  - `"value"`: a leaf, or a value given by the formula in `description`; its details, if any, are inputs, not terms;
  - `"ranker"`: a learned ranker's score = `base + each feature's contribution`; `details` are the base
    (`description: "ranker base value"`) then one node per feature with `feature` (its name) and `feature_value`
    (`null` when absent); they sum to the value (exact path attribution for the GBDT).
  A node may carry `source`, the tree its value came from (the lexical tree behind the `lex_*` features of a rescored
  hit; the sub-query behind each RRF term); `source` is not part of the arithmetic. The root's value equals the
  hit's `_score` (within 1e-4); a rescored hit below the window is a `"value"` node with the first-stage tree as
  `source`.
- **Rankers on disk:** `engine.api.save_ranker(model, path, feature_names=None)` and `load_ranker(path)`. JSON, no
  pickle: `{"format": "s3-ranker", "format_version": 1, "kind": "gbdt", "feature_names": [...] | null,
  "model": GBDT.to_dict()}`. `load_ranker` also reads a bare `GBDT.to_dict()` JSON (serving's interim format) and
  refuses other formats or versions with `ValueError`. Register a loaded model with `Index.add_ranker(name, model)`.
- No hit beyond `total`: an empty page is `{"total": n, "hits": []}`.
- `knn` (S9-1; present when the query has `knn` or `neural` clauses): `{field: {"requested": method, "ran": method}}`,
  the method asked for and the one that ran. `ran` is the method; `"<method>+postfilter"` when any segment was
  post-filtered (above); or `"exact"` (no cuVS or GPU, a cuVS failure, or every segment answered exactly after a
  post-filter). Whenever an approximate method did not run as itself (or some segments answered exactly), the object
  also has `reason`, a sentence naming why (e.g. `"not a cuda index (device cpu)"`). Several clauses on one field
  give a list of these objects, in request order (a depth-first walk of the query, then `post_filter`).

**Columnar hits (S6-7, additive; 2026-11-13, architect).** At depth 2,000 the engine's server spent its one Python
core building and encoding a dict per hit (the matched load test of 2026-11-13: 50 QPS sustained, the GPU at 7%). A
caller that encodes responses itself (serving) may ask for the hits as columns instead:

```python
Index.search(request, hits="columns") -> dict
Index.search_batch(requests, hits="columns") -> list[dict]     # hits="dicts" (the default) is unchanged
engine.api.expand_hits(response) -> dict                        # columns -> the dict form, exactly
```

- The response is the dict form's in every key except `hits.hits`, which is replaced by `hits.columns`:
  `{"total": n, "columns": {"_id": list[str], "_score": numpy float32[h], "sort": [column, ...]}}` for the `h` hits
  of the page, in hit order.
- `sort` has one column per position of a hit's `sort` list (the tier under `rescore`, each sort key, then
  `first_offset`). A column is `{"values": ..., "missing": numpy bool[h] | None}`. `values` is a numpy array
  (`int64` for the tier, `_doc`, `first_offset` and integer fields; `float64` for `_score`, float fields and
  `_geo_distance`; `bool` for boolean fields), or a `list[str]` for a keyword field. `missing` marks the hits whose
  value is `None` in the dict form (their `values` entry is unspecified); it is `None` when no hit is missing.
- A request with `collapse`, `highlight` or `explain` (per-hit extras) is answered in the dict form even when
  columns are asked for: a response has exactly one of `hits.hits` and `hits.columns`, and the caller checks which.
- `expand_hits(resp)` gives exactly what `hits="dicts"` gives for the same request on the same state (`_score` is
  `float(values[i])` of the float32; the referee's contract requests are the test). A response already in the dict
  form is returned unchanged.
- Serving owns the bytes (JSON or otherwise) it builds from the columns; the HTTP response it sends is unchanged
  (section 7). Columns are host arrays (numpy), never device tensors.
- **Ids as one bytes array (S6-12, additive, opt-in; 2026-11-13, architect).** `search(request, hits="columns",
  ids="bytes")` and `search_batch(requests, hits="columns", ids="bytes")` return `hits.columns._id` as one numpy
  array of dtype `S<w>` (h entries): each id UTF-8 encoded, `w` the index's longest encoded id, shorter ids padded
  with NUL bytes (numpy drops trailing NULs when an entry is read; ids never end in NUL). `ids="str"` (the default)
  is the `list[str]` above. It applies only with `hits="columns"` (`ValueError` otherwise) and to the responses in
  the columnar form. Why: a GPU process pickles 64 replies of 2,000 ids per batch, and a list of 2,000 str pickles
  about seven times slower than one bytes array (23.8 against 3.3 ms per batch of 64 on CPU; unpickling 13.9
  against 1.7 ms). `engine.api.expand_hits` decodes it (`_id` = `entry.decode("utf-8")`), so the dict form, and
  serving's bytes, are unchanged. The engine keeps a bytes copy of its ids, updated with each segment added, so a
  page reads it with one gather.

**The engine in one GPU process (S6-10, engine side; 2026-11-13, architect).** Serving may run one GPU process that
holds the only `Index` (one index copy on the device) and calls `search_batch` for requests gathered from several CPU
front processes, which do HTTP and encoding (section 7: the process architecture, the transport and batch formation
are the platform-engineer's). What the engine promises that process:

- **The call:** `Index.search_batch(requests, hits="columns", errors="return") -> list`, one entry per request, in
  request order, the same length as `requests`. `hits` is as above ("Columnar hits"). `errors`:
  - `"raise"` (the default, as today): the first refused request raises its error and the batch returns nothing;
  - `"return"`: a request refused while it is parsed or planned (`UnsupportedQuery`, `InvalidRequest`, and any other
    `engine.errors` class raised for that request alone, e.g. an unknown ranker in `rescore`) gets
    `{"error": {"type": <class name>, "path": <JSON path>, "reason": <sentence>}}` in its slot, and every other request
    of the batch is answered as if it had come alone. A failure that is not one request's (`OverBudget`,
    `DeviceUnavailable`, a CUDA error, a bug) still raises for the whole batch; the caller may retry its requests
    singly (section 7 already does) or fail them all.
- **Equality:** every entry equals `search(request, hits=...)` alone on the same state, except `took_ms`
  (`search_batch == [search]`, tested in `engine/tests/test_batched_executor.py` on every device). A batch's
  composition never changes a result: not its size, not its order, not which requests share it.
- **One state per batch:** the batch reads the index state once (S8-2c), so all its requests see the same version and
  the same writes; a write that returns while a batch runs is visible to the next batch, not to this one.
- **What is batched, and what stays per request** (for serving's batch formation; results are the same either way):
  - batched across the requests of a call: lexical retrieval, query encoding, dense similarities, exact unfiltered
    knn top-k, hybrid RRF windows and fusion, rescoring (features and the ranker), `hits.total`, plain score-sorted
    pages, and terms aggregations without sub-aggregations;
  - per request inside the call: parsing and planning, filters, field sorts and `search_after`, `collapse`,
    `highlight`, `explain` (and a rescore with `explain`), filtered or cuVS knn, every other aggregation, and
    building the response. A request of this kind makes its batch slower, never different: serving may route it
    to a batch of its own.
  - a batch of one takes the per-request path throughout.
- **Size:** `search_batch` takes any number of requests. It stacks at most `Executor.BATCH_ROWS` (64) requests'
  `[N]` rows at a time, so the device memory a call adds is about 64 x N x 10 bytes on top of the index (about
  0.8 GB at N = 1.2M), whatever the batch size; larger batches run in chunks of 64. Throughput on an A100 stopped
  growing past 64 in every profile so far (`results/pipeline-profile-*.json`), so 64 is the recommended largest batch.
- **Threads:** `search_batch` is thread-safe and never waits on writers, but calls from several threads compete for
  the one GPU and batch less; the GPU process should make one call at a time and let batches grow instead.
- **Returned objects:** responses hold host values only (Python scalars, lists, numpy arrays in `hits.columns`), no
  device tensors and no references into the index, so they can be pickled or copied to another process as they are.
- **Writes** (`patch`, `delete`, `add_segment`, `replace_segments`, ...) go to the same `Index` in the GPU process;
  front processes hold no `Index`.

## 6. The plan (DSL → operators) · owner architect

Two passes (S3):
- `engine/dsl/parse.py: parse(request: dict) -> Request` (`engine/dsl/ast.py`): syntax only, no index. Keys, shapes
  and value types; anything outside the subset is `UnsupportedQuery` naming its path, malformed input
  `InvalidRequest`. Field names are not resolved and text is not analysed.
- `engine/plan/plan.py: plan(request: dict | Request, state: IndexState) -> Plan`: resolves fields and kinds (a field
  the index lacks is `InvalidRequest`, a field of the wrong kind `UnsupportedQuery`), analyses text, works out
  `minimum_should_match` and BM25F's idf mode, converts term values and range bounds to the field's type, applies the
  sort defaults, and checks vector dimensions and encoders.
- `engine/plan/execute.py: Executor(state, rankers).run(plans: list[Plan]) -> list[dict]`: the batched executor. Plans are frozen dataclasses (hashable, so the executor groups
equal shapes into one batch).

```python
Node = Lexical | Dense | Term | Range | MatchAll | Bool | Hybrid
Lexical(fields: tuple[tuple[str, float], ...], terms: tuple[str, ...], operator: str, msm: int, idf: "global"|"field", boost: float)
LexicalGroups(fields, groups: tuple[tuple[str, ...], ...], modes: tuple["sum"|"synonym"|"const", ...], msm, idf, boost)  # S7-2
Phrase(field, slots: tuple[tuple[str, ...], ...], offsets: tuple[int, ...], slop: int, boost)                           # S7-1
Dense(field: str, vector: tuple[float, ...] | None, text: str | None, k: int, filter: Node | None, boost: float)
Term(field: str, values: tuple)            Range(field: str, lo, hi, lo_inc: bool, hi_inc: bool)
MatchAll(boost: float)                    Bool(must, should, filter, must_not: tuple[Node, ...], msm: int, boost: float)
Hybrid(queries: tuple[Node, ...], rrf_k: int, window: int)
Plan(root: Node, aggs: tuple[TermsAgg, ...], sort: tuple[SortKey, ...], from_: int, size: int,
     search_after: tuple | None, rescore: Rescore | None, explain: bool)
```

Execution stages (each timed into `took_ms`): **retrieve** (Lexical → scores `f32[Q,N]` and match counts; Dense → top-k
ids and sims), **filter** (Term/Range/Bool filter/must_not → bool masks `[Q,N]`, AND with live), **fuse** (Hybrid RRF;
Bool score sums), **rank** (rescore window), **facet** (terms aggs over the final match mask), **page** (sort, tie by doc
id, `search_after`, from/size).

Operators (`engine/ops/`, all `(backend, ...) ->` arrays on the backend's device):

```python
bm25f(b, seg, terms: list[list[int]], w: f32[F], bb: f32[F], k1: float, idf: arr) -> (scores f32[Q,N], nmatch i32[Q,N])
dense_exact(b, emb, q: f32[Q,D], k: int, mask: bool[Q,N] | None) -> (ids i32[Q,k], sims f32[Q,k])   # -1 pads
term_mask(b, col, values) -> bool[N];  range_mask(b, col, lo, hi, lo_inc, hi_inc, missing) -> bool[N]
facet_terms(b, codes i32[N], mask bool[Q,N], n_values: int) -> i64[Q, n_values]
rrf(b, ranked_ids: list[i32[Q,K]], k: int, window: int) -> (ids i32[Q,W], scores f32[Q,W])
features(b, ...) -> f32[Q, W, n_features]        # the prototype's FEATURES list, named in the ranker's manifest
page(b, keys: list[arr[Q,N]], mask bool[Q,N], from_: int, size: int, after: tuple | None) -> (ids i32[Q,size], sort_values)
```

## 7. The admin API and the `SearchIndex` resource (platform → data plane) · owner platform-engineer

**Serving** (`serving/`): `serving.app.create_app(config: ServingConfig, ports: Ports | None = None) -> App`, on the
standard library (`docs/decisions/2026-11-09-serving-and-operator.md`). `App.handle(method, path, body, headers) ->
Response(status, body, content_type, headers)` is the whole API; `serving.app.make_server(app, host, port)` serves it
(`python -m serving`). `serving.ports.Ports` is what serving needs from the engine and the data plane (open the store,
ingest, validate and enrich changes, build a segment from the store, load an `Index`); the defaults call
`dataplane.kv.Store`, `dataplane.intake.ingest/validate/enrich`, `dataplane.indexer.build` and
`engine.api.Index.from_segment` exactly as §3 to §5 define them.

| Method, path | Body → response |
|---|---|
| `POST /indexes/{name}/_search` | §5 request → §5 response (+ `_source` per hit when asked) |
| `POST /indexes/{name}/_msearch` | `[request, ...]` → `[response, ...]` (one `search_batch`) |
| `PUT /indexes/{name}` | `{"mapping": {...}, "source": {"path": "..."}}` → `{"name", "version"}`: intake, build, load |
| `POST /indexes/{name}/_docs` | `[change record without offset, ...]` → `{"first_offset", "last_offset", "rejected": [...], "applied": {"patched", "deleted", "pending_upserts"}}`: validate, `store.apply`, then replay the log onto the live index (patches and deletes are visible at once; upserts after the next reindex) |
| `POST /indexes/{name}/_reindex` | `{}` → `{"segment_id", "version"}`: build from the store, swap |
| `GET /indexes/{name}/versions` | → `[{"segment_id", "version", "created", "live": bool}]` |
| `POST /indexes/{name}/_rollback` | `{"segment_id"}` → `{"segment_id", "version"}`: swap to a kept segment, replay the log after it |
| `POST /indexes/{name}/_bulk` | OpenSearch NDJSON (`index`, `create`, `update` with a partial `doc` / `doc_as_upsert` / `upsert`, `delete`; `?refresh=false\|true\|wait_for`) → `{"took", "errors", "items": [{op: {"_index", "_id", "status", "result", "_seq_no"} \| {..., "error": {"type", "reason", "s3_reason", "field"}}}]}`; see **Bulk** below |
| `POST /indexes/{name}/_refresh` | `{}` → `{"refreshed", "segment_id", "version"}`: make every stored upsert visible now |
| `GET`, `PUT /indexes/{name}/_settings` | `{"index": {"refresh_interval": "500ms"}}` (`500ms`, `5s`, `2m`, `-1` = no background refresh); default `S3_SERVING_REFRESH_INTERVAL` = `500ms` (the S8-1 ruling); stored in `<index>/settings.json` |
| `GET /indexes/{name}/_mapping` | → `{name: {"mappings": <the mapping>, "s3_fields": {field: {"type", "searchable", "filterable", "facetable", "sortable", "dense", "hot"[, "dimension"]}}}}`; sub-fields appear as `f.keyword`. Facetable = keyword; filterable and sortable = keyword and numeric/boolean; searchable = text |
| `POST /indexes/{name}/_rankers` | `{"name", "path"}` → `{"name", "rankers": [...]}`: loads a ranker file (the JSON of `GBDT.to_dict()`; `engine.api` has no save/load of its own, so this is interim) from under the serving data dir into `Index.add_ranker`; kept across swaps. At startup: `S3_SERVING_RANKERS=name=path,...` |
| `GET /indexes`, `GET /indexes/{name}` | what is loaded: segment id, version, docs, device, cache counters |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | liveness; ready once every configured index has a segment loaded (503 while loading or draining); Prometheus text (below) |

- **RPC.** `python -m serving --rpc-port N` (`S3_SERVING_RPC_PORT`) also listens for length-prefixed JSON frames
  (`serving/rpc.py`, `docs/decisions/2026-11-10-rpc-and-operator-watches.md`): request `{"id", "method", "path", "body",
  "headers"}`, response `{"id", "status", "body", "headers"}`, with the same method and path as the HTTP route and the
  same `App.handle`, so responses equal HTTP's (checked on the 200 conformance requests). Client: `serving.rpc.RpcClient`.
  The `SearchIndex` Service also exposes port 9090 as `rpc`.
- **Response bytes and columnar hits (S6-7).** Serving asks the engine for `hits="columns"` (§5) whenever the loaded
  index's `search`/`search_batch` takes `hits` (`S3_SERVING_COLUMNS=1`, the default; `0` asks for the dict form) and
  encodes the columns itself (`serving/encode.py`). The HTTP and RPC bodies are unchanged: byte for byte what
  `json.dumps(engine.api.expand_hits(resp), separators=(",", ":"))` gives (checked against the engine on every
  device in `serving/tests/test_encode.py`). A response the engine answers in the dict form (collapse, highlight,
  explain) is encoded as before; `_source` expands that one response to dicts first. The cache holds what the engine
  returned.
- **Workers (S6-7).** `python -m serving --workers N` (`S3_SERVING_WORKERS`, default 1): a supervisor binds the HTTP
  (and RPC) port once and starts N `python -m serving` workers that share the listening socket (`serving/workers.py`).
  Each worker loads its own copy of the preloaded indexes (N copies in GPU memory on cuda) and has its own cache,
  scheduler and `/metrics`; every response carries `X-S3-Worker: <i>`. **N > 1 is read-only:** `GET` routes,
  `_search`, `_msearch`, `_mapping` and `versions` are served, every other route is 405 `method_not_allowed` (each
  worker's index is its own, so a write would reach one of them). Worker 0 accepts at once (`/healthz` while loading,
  `/readyz` 503 until loaded); the others accept only after their preload. SIGTERM to the supervisor drains every
  worker; a worker that dies is started again. For the replica Deployment, which preloads a segment the indexer Job
  built and takes no writes.
- **One GPU process (S6-10s; proposed 2026-11-13 by the platform-engineer, agreed by the architect with seven changes,
  built).** `python -m serving --fronts F --gpu-processes K` (`S3_SERVING_FRONTS`, `S3_SERVING_GPU_PROCESSES`; K
  defaults to 1). It replaces N workers, each with its own index copy and CUDA context (S6-7 used 40.4 of 41 GB at 7
  workers; without MPS, the contexts taking turns cost most of the GPU time).
  - **Fronts** (`serving/front.py`) are F CPU processes on the shared listening socket. They parse HTTP and JSON, keep
    the cache, and encode columns (`serving/encode.py`). They answer `/healthz` and `/metrics` (their own) themselves,
    and forward every other route to GPU process 0. A front holds no `Index`. Each search goes to the GPU process
    with the fewest requests in flight.
  - **GPU processes** (`serving/gpu.py`) each hold one copy of each preloaded index and a full `App`. A batch loop
    makes one `search_batch(requests, hits="columns", errors="return")` call at a time, with at most 64 requests:
    whatever has queued when the last call returned, with no batch window. When the engine takes `ids` (S6-12) the GPU process and
    the single-process scheduler ask for `ids="bytes"`: `_id` is one numpy `S<w>` array. It pickles about 7x faster,
    and the encoder writes its bytes as they are when they are plain ASCII (else it decodes and escapes them); the
    HTTP bytes are unchanged. **Only `explain` and `highlight` requests
    run in batches of their own.** Filters, field sorts, aggregations and collapse share a batch: §5 makes them cost
    more, never answer differently. Route more only if a measurement shows it is needed.
  - **Errors.** An `{"error": {"type": <engine class name>, ...}}` entry maps to §7's snake_case type and status
    (`serving.errors.classify_entry`: `UnsupportedQuery` gives 400 `unsupported_query`, `InvalidRequest` gives 400
    `invalid_request`). The body is the same as a single process sends. Only when the call itself raises (a failure
    that is not one request's: OverBudget, DeviceUnavailable, CUDA) are the requests tried one at a time, and after 2
    such failures in a row the rest of the batch fails with that error.
  - **Writes, with K = 1.** A front forwards a write to GPU process 0 and answers only after it has been applied. A
    search that starts after the write call returns therefore sees it (§5's one state per batch). Calls run one at a
    time on a call thread next to the batch loop (an `Index` write is copy-on-write). With K > 1 the set is read-only
    and every write or admin call gets 405.
  - **Front caches** are keyed by `(index, segment_id, epoch)`. GPU process 0 republishes these values in
    `versions-0.json` in the state directory within 20 ms of any swap, patch, delete or background refresh. Every
    batch reply also carries the version it read. So a write forwarded by any front leaves no stale entry reachable.
  - **Transport.** One Unix-domain socket per GPU process, in the supervisor's state directory (`mkdtemp`, mode 0700);
    the socket itself is mode 0600. A frame is an 8-byte big-endian length and a pickle. Columns are host numpy
    arrays plus a list of `_id` strings (the largest part), so they pickle as they are. **Unpickling is code
    execution:** the socket is for the service user's own processes only, inside one trust boundary, and is never
    exposed on TCP.
  - **K > 1 runs under CUDA MPS**, with one index copy per process (about 5.8 GB each for ESCI-US on an A100). The MPS
    daemon is the node's (`colab/fair_bench.py --mps=1` starts one). One GPU process makes about 690 QPS lexical and
    180 QPS full pipeline at batch 64 in process (the architect's profiles), so K = 1 likely tops out near 700 QPS
    lexical. The first measurement, with predictions filed first, chooses K.
  - **Readiness:** a front's `/readyz` asks every GPU process. It is ready when all are, and reports `fronts`,
    `gpu_processes`, `gpu_ready` and `worker`.
  - **Checked:** `serving/tests/test_gpu_process.py`, against a single process, real engine, toy dataset:
    - search, `_msearch`, `_mapping`, errors, `_source`, collapse, highlight and explain give the same bytes but for
      `took_ms`, with K = 1 and K = 2;
    - the same results under 32 concurrent connections;
    - a patch through a front is visible to the next search;
    - the modes of the socket and the state directory.
- **Bulk** (`serving/bulk.py`). The body is streamed (Content-Length or chunked), grouped into chunks of `bulk_chunk`
  actions (default 2000), and each chunk is planned, applied to the store as one atomic batch through
  `dataplane.intake.apply_changes` (bad records are rejected one by one) and replayed onto the live index before the next
  chunk is read: memory is one chunk plus the item list. Every action has an item, in order. Semantics as OpenSearch:
  `index` 201 created / 200 updated; `create` 409 `version_conflict_engine_exception` if the id exists; `update` merges a
  partial `doc` (hot fields only: an in-place patch; else the merged document is stored whole), 404
  `document_missing_exception` unless `doc_as_upsert`/`upsert`, `noop` for an empty doc; `delete` 200 `deleted` / 404
  `not_found`; a missing `_id` on `index`/`create` is generated. A data-plane rejection becomes status 400 (404 for
  `no_such_key`) with `error.type` from the reason (`bad_type`, `dims_mismatch`, `too_long` → `mapper_parsing_exception`;
  `unknown_field` → `strict_dynamic_mapping_exception`; `not_hot` → `illegal_argument_exception`; `missing_id`,
  `malformed`, `bad_op` → `action_request_validation_exception`), `error.s3_reason` the named reason and `error.field`.
  Unsupported: `script`, versioning, routing, pipelines. An unparseable action line ends the request with 400 and the
  items already applied; a store failure marks its chunk 5xx and stops. Not available over RPC.
- **Refresh.** A write is durable and its patches and deletes visible at once; an upsert becomes visible at the next
  refresh. `refresh=false` leaves it to the background refresher (every `refresh_interval`, only when something is
  pending; with the full-rebuild backend also at least 4x the last refresh's cost apart); `true` refreshes before
  returning (`"forced_refresh": true`); `wait_for` returns when a refresh covers the write (forces one if the interval
  is `-1` or it takes too long). **The hook:** `serving.ports.Ports.refresh(managed_index) -> {"segment_id", "version"}`;
  the data plane's `default_ports()` sets it to `dataplane.updater.serving_refresh` (a delta segment from the oldest pending
  upsert, then one tiered merge if due; measured below); with it `None` a refresh is a full rebuild from the store and a swap
  (18.7 s on ESCI-sample), and refresh becomes a delta (it runs under the admin lock and may call
  `managed_index._swap_in(new_index, segment_id, "refresh")`).
- **Write metrics:** `s3_index_lag` (store head minus the highest offset fully visible, so upserts waiting for a refresh
  count), `s3_index_pending_upserts`, `s3_refresh_ms`, `s3_bulk_requests_total{refresh}`, `s3_bulk_items_total{op,outcome}`,
  `s3_bulk_rejected_total{reason}`, `s3_bulk_items_per_second` (last request), `s3_bulk_ms`.
- **Batching.** `_search` requests are queued per index; a worker sends up to `max_batch` of them (default 32) to one
  `Index.search_batch`, waiting at most `max_wait_ms` (default 2.0) after the first arrives. `_msearch` is one
  `search_batch` itself. If a batch raises, its requests are retried singly so a bad request fails alone.
- **Query cache.** LRU (`cache_size` entries per index, default 1024) keyed by (segment id, epoch, canonical request
  JSON); the epoch moves on every swap, patch, delete and `_docs` call, so nothing stale is reachable.
- **`_source`.** A request carrying `_source` (true, a list of fields, or `{includes, excludes}`) gets each hit's stored
  document from the store (vectors removed); a replica without the store answers 400 with path `_source`.
- **Response headers on `_search`:** `X-Request-Id` (echoed or generated), `X-S3-Cache: hit|miss`, `X-S3-Queue-Ms`
  (time in the scheduler), `X-S3-Serve-Ms`. The engine's `took_ms` is passed through unchanged.
- **Metrics** (names are the contract the benchmark reads): `s3_http_requests_total{route,status}`,
  `s3_search_requests_total{index,outcome}`, `s3_cache_hits_total`, `s3_cache_misses_total`, `s3_cache_entries`,
  `s3_batches_total`, `s3_batch_size` (histogram), `s3_queue_wait_ms` (histogram), `s3_stage_ms{stage}` (histogram of
  the engine's `took_ms` per stage), `s3_request_ms` (histogram), `s3_errors_total{type}`, `s3_index_version`,
  `s3_index_docs`, `s3_index_ready`, `s3_swaps_total{kind=load|reindex|rollback}`, `s3_patches_total`,
  `s3_queue_depth`.
- **Swaps.** `_reindex` and `_rollback` load the new `Index` first (the budget check, §0, may refuse with
  `over_budget` and leave the old one serving), replay the change log after its version, then replace one reference;
  searches in flight finish on the old index. Segments are kept (`keep_segments`, default 5; the live one never pruned).
- **Configuration** (`serving.config.ServingConfig`, env `S3_SERVING_*`, `S3_SEARCH_DEVICE`): data dir, device,
  `max_batch`, `max_wait_ms`, `cache_size`, `keep_segments`, `S3_SERVING_PRELOAD=name[:segment_id]`.

Errors: HTTP 400 with `{"error": {"type": "unsupported_query" | "invalid_request" | "invalid_mapping" | "not_hot", "path", "reason"}}`;
409 for `over_budget`, `index_exists`, `device_unavailable`; 404 for `index_not_found`, `segment_not_found`; 405 for a wrong
method; 503 for `not_ready` and `overloaded`. The engine's exceptions are recognised by class name (§5), so the
mapping needs no import.

**The `SearchIndex` custom resource** (`search.s3.dev/v1alpha1`, `deploy/`):

```yaml
apiVersion: search.s3.dev/v1alpha1
kind: SearchIndex
metadata: {name: esci-sample}
spec:
  mapping: {configMapRef: {name: esci-mapping, key: mapping.json}}
  source: {path: /data/esci-sample/products.parquet}      # or {pvc: ..., path: ...}
  device: cpu                                               # cpu | torch-cpu | cuda
  replicas: 2
  resources: {requests: {cpu: "1", memory: 2Gi}, limits: {memory: 4Gi}}   # + nvidia.com/gpu: 1 when device is cuda
  segmentVersion: null                                      # null: latest; a segment_id pins it (rollback)
  reindexToken: ""                                          # change it to build again from the same inputs
  storage: {size: 5Gi, accessModes: [ReadWriteOnce]}        # the segment volume; {existingClaim: name} uses one
  serving: {maxBatch: 32, maxWaitMs: 2.0, cacheSize: 1024, workers: 1}
  image: s3-search:cpu                                      # default by device: s3-search:cpu | s3-search:cuda
status:
  phase: Pending | Indexing | Ready | Swapping | Failed
  segmentId: seg-1044-3fa2c9d1        # what the replicas have fully rolled to
  latestSegmentId: seg-1044-3fa2c9d1  # the newest build
  version: 1044
  readyReplicas: 2
  segments: [{segmentId, version, build}]   # the last 10 builds: what segmentVersion may pin
  indexJob: {name, hash, state, segmentId, version}
  conditions: [Valid, MappingAvailable, IndexBuilt, Available, Ready]
```

- The operator (`deploy/s3operator/`, standard library, `docs/decisions/2026-11-09-serving-and-operator.md`) reconciles
  a `SearchIndex` into a `Job` (the indexer, `python -m serving build`, which calls the data plane's intake and
  `indexer.build` through `serving/ports.py` and reports `{segment_id, version}` as its termination message), a
  `Deployment` of replicas (`python -m serving`, loading `S3_SERVING_PRELOAD=<name>:<segment_id>`), a `Service`, and a
  PVC for segments. The Job is named by a hash of the mapping content, source, image and `reindexToken`, so a change to
  any of them rebuilds and a change to `replicas` or `segmentVersion` does not. A new segment (or a pin) changes the
  segment id in the Deployment's pod template, so it rolls out by a rolling update with `maxUnavailable: 0`, readiness on
  `/readyz`, and a SIGTERM that fails readiness and drains before exit. Phases: `Indexing` while a build runs (replicas
  keep serving), `Swapping` while replicas roll to another segment, `Failed` for an invalid spec or a failed build (the
  last segment keeps serving).
- **It never assumes a GPU.** Only when `device: cuda` does it add `nvidia.com/gpu` limits from `resources`, a
  `nodeSelector` (`nvidia.com/gpu.present: "true"`) and the `nvidia.com/gpu` toleration. One `Dockerfile` with
  `ARG BASE=cpu|cuda` selects the base image and `requirements-gpu.txt`.
- The operator runs on watches (SearchIndex, and the Deployments and Jobs it owns) with a periodic resync as the backstop,
  and elects a leader with a `coordination.k8s.io/v1` Lease `s3-operator` (flags `--lease-duration 15 --renew-deadline 10
  --retry-period 2`; `--no-leader-elect` for a single replica); the manifest runs 2 replicas. Decision record
  `docs/decisions/2026-11-10-rpc-and-operator-watches.md`.
- CRD: `deploy/crd/searchindex.yaml`; operator manifests: `deploy/manifests/operator.yaml`; samples:
  `deploy/samples/`; image: `deploy/Dockerfile` (`ARG BASE=cpu|cuda`, built from the repository root).
- Tests: the reconciler against a real kube-apiserver and etcd (envtest binaries at
  `/tmp/claude-0/envtest/controller-tools/envtest/` in the dev container, where kind cannot run pods); the full kind
  end-to-end in GitHub Actions.

## Changes

- 2026-11-09, architect: v1 written from the lab's skeleton (sections 0 and 1 to 7 filled; section 0 added for the
  GPU-portable backend).
- 2026-11-09, architect (S1 landed; rulings on the evaluation-steward's reading of the contract, `referee/contract/
  README.md` on the steward's branch):
  1. **Word pattern:** the prototype's, `[a-z0-9]+(?:'[a-z0-9]+)?`, then a trailing `'s` is cut, then `'` removed.
     Section 2 said `[a-z0-9']+`, which was a slip; the 0.304 reproduction used the prototype's. `snowballstemmer`'s
     `porter` agrees with the referee's own Porter stemmer on all 1,000 toy texts and 3,560 other words checked.
  3. **idf:** confirmed as the steward read it (section 5, "idf").
  4. **Constant scores:** confirmed as the steward read it (section 5, "Constant scores").
  Also confirmed: `product_id` may be (and in the referee's mappings is) a keyword property; sort defaults `_score`
  desc, fields asc, `missing: _last`; knn `_score = (1 + cos) / 2`, total not promised; terms facets on numeric
  fields with typed keys (added in S1); `.path` names the offending key; cross-implementation tolerance
  `1e-4 * max(1, |ref|)` with tie groups (section 0's `rtol=1e-5` stays the cross-device bound).
- 2026-11-09, architect (rulings on the dataplane-engineer's eight questions; section 4 "The S1 rules for the data
  plane"): (1) `_first_offset` from the snapshot, else `0..N-1`; unmapped and `_`-prefixed columns ignored. (2) Either
  `build_segment(version=)` or setting `seg.manifest["version"]`; `write_segment` writes what the manifest holds and
  derives `segment_id` from it. (3) `write_segment` takes the segment directory, which may exist if empty. (4)
  `NoSuchKey` (a `KeyError`) for unknown ids; deleted ids ignored by `patch` and `delete`. (5) `Index` is
  thread-safe by one re-entrant lock: a search never sees half a patch. (6) The data plane may import `engine.api`;
  every named error is in `engine.errors`. (7) `build_segment(vector_models=)` records the encoder of given vectors.
  (8) Nullable pandas dtypes accepted. Why: these are what the S2 code needs to call the engine without a fake, and
  each keeps the engine's invariants (doc-id order, checksums, atomic writes).
- 2026-11-09, architect: all named errors moved to `engine.errors` (serving matches them by class name); `NoSuchKey`
  added. `Index` gained `save`, `add_ranker`, `candidate_features`, `segment_id`. Section 6: parse and plan are one
  pass in S1. Section 5: rescore puts a tier first in `sort`.
- 2026-11-09, dataplane-engineer: sections 2 and 3 amended with the S2 code (additive). Section 2: `read_batch`
  yields a `Rejection` for a malformed line; four reasons added (`malformed`, `bad_op`, `not_hot`, `no_such_key`);
  coercion and unknown-field rules written down; `apply_changes` for the change feed; the encoder hook and one encoder
  per store. Section 3: atomic batches (a refused change refuses its batch, so offsets stay gapless), `changes(docs=)`,
  the snapshot frame's columns and dtypes (`_first_offset`, `_last_offset`), the indexer's version, temp-and-rename
  writes, merges, gc and job, the updater's coalescing and its swap budget (current plus new, which is 2x for equal
  sizes), and the freshness hook. Why: these are what the code does, and the engine, serving and the steward's B1-4
  harness need them to call it. Sections 0 and 4 to 7 are unchanged; what they still need is in the S2 report.
- 2026-11-09, platform-engineer: §7 amended while building S4. `create_app` returns an `App` (stdlib, not ASGI);
  `_docs` also returns `applied`; added `GET /indexes`, the named error types, the batching, cache, `_source`, header and
  metric contracts; the indexer Job runs `python -m serving build` (it needs ingest and build together, through the
  data plane's own functions) rather than `python -m dataplane.indexer`; the `SearchIndex` spec gains `reindexToken`,
  `storage`, `serving`, `source.pvc`, `imagePullPolicy`, and its status gains `latestSegmentId`, `segments`, `indexJob`.
  All additive. Needs from others: `dataplane.kv.Store`, `dataplane.intake.{ingest,validate,enrich}`,
  `dataplane.indexer.build` and `engine.api.Index` (`from_segment`, `search`, `search_batch`, `patch`, `delete`,
  `set_version`, `.version`, `.n_docs`) exactly as §3 to §5 define them; engine exceptions named `UnsupportedQuery`,
  `InvalidRequest`, `InvalidMapping`, `OverBudget`, `DeviceUnavailable`, `NotHot`, `NoSuchKey`, `CorruptSegment`.
- 2026-11-10, platform-engineer: added `GET /indexes/{name}/_mapping` (field capabilities for the demo page) and `POST /indexes/{name}/_rankers` (+ `S3_SERVING_RANKERS`); `GET /indexes` info lists `rankers`. Additive. Asks the architect for a ranker save/load in `engine.api`.
- 2026-11-10, architect (S5 judgement, defects 2 and 3; additive to section 5): every explain node gains a
  machine-readable `op` (`sum`, `product`, `max`, `value`, `ranker`), and may carry `source`. A rescored hit is
  explained as `op: "ranker"`: the ranker's score = base + each feature's contribution, each with its value, the
  lexical tree kept as the `source` of the `lex_*` features (before, it was one node wrapping the lexical tree, with
  no contributions). `GBDT.contributions(X)` gives exact path attribution. `engine.api.save_ranker` and
  `load_ranker` give rankers a JSON file format (`s3-ranker` v1, wrapping `GBDT.to_dict()`, no pickle). Why: section 5
  promised per-feature contributions, and the tools and serving need to read explanations and rankers without
  parsing prose or pickles. Checked by `engine/tests/test_explain.py` against the pre-registered prediction (100
  lexical and 100 rescored explanations sum to the hit's score within 1e-4, on toy).
- 2026-11-10, dataplane-engineer (defect 4): section 2's encoder rule. Intake now uses the engine's encoder by name,
  fits it once per store and keeps it there (`Store.blob`, `Store.set_blob`, additive); the indexer passes
  `vector_models` and the fitted encoder to the segment. Why: a segment of intake-made vectors recorded model
  "given" and no encoder, so `neural` raised `UnsupportedQuery` where `build_index` answered.
- 2026-11-10, architect (S3): section 6 split into two passes, `parse(request) -> Request` (syntax, `engine/dsl/ast.py`,
  no index) and `plan(request, state) -> Plan` (semantics), as section 6 first said; `plan` still accepts a dict.
  Section 5: values are converted by field type as OpenSearch does (float bounds to float32, no fractional term on an
  integer field), and `match` on a keyword or numeric field is a `term`. Why: S3-1 and S3-2; the float32 rule removes
  a boundary disagreement with OpenSearch (and with a pandas mask) on float fields. No request that S1 answered is
  answered differently except a float field's exact-boundary term or range, and a fractional term on an integer field
  (now refused, before an empty answer).
- 2026-11-10, platform-engineer: RPC serving (S7-7), additive: `rpc` listener, client, Service port 9090.
- 2026-11-10, platform-engineer: the operator on watches with Lease leader election (S7-7); no spec or status change.
- 2026-11-10, architect (GPU-friendliness audit): section 0 gains the host-transfer rule (one transfer for the page,
  one per terms aggregation; rescoring on the device) and the `accumulate` option (float32 keeps BM25F
  batch-invariant, not the dense GEMM). Section 5 corrected: OpenSearch's default BM25 keeps `(k1 + 1)`; the lexical
  gap to it is N and avgdl over documents that have the field, and one-byte lengths, by design (the steward's
  reproduction, `referee/lucene_model.py`). Why: the coordinator's audit (numeric codes recomputed on the host per
  request, rescoring on the host, FP64 on T4-class GPUs) and the steward's first contact with OpenSearch 2.19.1.
- 2026-11-10, architect (S7-1): section 4 gains optional positions (`text/pos_ptr.npy`, `text/positions.npy`; older
  segments load without them); section 2 gains `_meta.engine.positions` and the position rule; section 5 gains
  `match_phrase` with `slop`. Additive. Why: the founder's feature coverage (S7).
- 2026-11-10, platform-engineer: `_bulk`, `_refresh`, `_settings` (refresh_interval), `?refresh=` on `_bulk` and `_docs`, write metrics and the `Ports.refresh` / `Ports.apply_changes` hooks (S8-6), additive. Needs from the dataplane-engineer: `Ports.refresh` implemented on delta segments (S8-4).
- 2026-11-10, architect (S7-2): section 5 gains `fuzzy`, fuzziness on `match` and `multi_match`, `prefix` and
  `match_bool_prefix`, expanded in the terms dictionary with OpenSearch's defaults; section 6 gains the
  `LexicalGroups` plan node (clauses of alternative terms: "sum", "synonym", "const"), scored by the new
  `bm25f_groups` operator on the backend. Additive. Why: S7 (typo tolerance and autocomplete).
- 2026-11-10, architect (S8-3, build speed; no format or result change): `engine.build` analyses each field in one
  regex pass per document and one dictionary pass over all its words, and builds postings, tf and positions from one
  stable sort of the tokens' posting keys (the text arrays are identical to before, checked on ESCI-sample);
  `tfidf-svd-<d>` analyses once, multiplies the sparse matrix one column of X at a time (`bincount`), orthonormalises
  by CholeskyQR2 and takes the final SVD from a small eigenproblem (the vectors equal the old ones to 2e-10);
  `engine._blas.limit_to_affinity()` sets OpenBLAS's threads to the CPU affinity mask (it ran 4 threads on 2 cores).
  On 2 cores, same machine: build_segment of ESCI-sample with positions 12.3-12.8 s before, 9.5-9.8 s after (bench
  build); the encoder's fit + encode 47-77 s before, 7.3-8.2 s after.
- 2026-11-10, architect (ruling on S8-1, the dataplane-engineer's multi-segment proposal): accepted with amendments
  (`docs/proposals/2026-11-10-s8-1-multi-segment.md`, Ruling). Section 4: an index is an ordered list of v1 segments
  with commit points; statistics are global over live docs (replacing "deleted docs count until the next build");
  doc order is `first_offset` order by append-only slots; the multi-segment API. Section 5: `sort`'s last element is
  `first_offset` (unchanged for catalogue-built indexes). Why: S8's fast writes need deltas whose results equal a
  rebuild's.
- 2026-11-10, architect (S8-2): the engine searches an ordered list of segments with global live-only statistics
  and first_offset slots (`engine/index_state.py`: `Part` per segment, `IndexState._relayout`); `add_segment`,
  `replace_segments`, `remove_segment`, `snapshot_segments`, `segments`, `from_segments`, `from_commit`,
  `save_commit`; `engine.build.merge_segments` merges arrays (postings, positions, vectors) without re-analysis;
  `engine/commit.py` reads and writes the data plane's commit format with its checksum definition. Section 4 records
  the checksum's bytes, the shared test vector, and that commits do not carry hot-column patches.
- 2026-11-10, architect (S7-3): section 5 gains `exists`, `ids`, `post_filter` and `collapse`; a collapsed hit
  carries `fields`. The last `sort` value is now the doc's `first_offset` in every response (S8 ruling), and
  `search_after` compares it. Additive. Why: S7 (multi-select facets, one hit per brand).
- 2026-11-10, architect (S7-4): section 5 gains the metric aggregations, `range`, `histogram` and one level of
  sub-aggregations under `terms` (`engine/aggs.py`, on the backend, one transfer per aggregation). Additive. Why: S7.
- 2026-11-10, architect (S7-5): section 4 gains the optional source text (`src/<field>.parquet`, listed in the
  manifest's `source_fields`); section 2 gains `_meta.engine.source`; section 5 gains `highlight` (plain) and
  `function_score` with `field_value_factor` and `weight`. Additive. Why: S7.
- 2026-11-10, architect (S7-6): section 2 gains `_meta.engine.synonyms` (query-time, one-word, Solr lines); the
  planner turns a term with synonyms into a "synonym" clause of `LexicalGroups` (section 6). A mapping without
  synonyms plans exactly as before. Additive. Why: S7.
- 2026-11-10, architect (S8-2b, incremental segments; no contract change): `add_segment`, `replace_segments`,
  `remove_segment` and `delete` no longer rebuild the index's layout. Slots are append-only in `first_offset` order:
  an edited key (same `first_offset`) keeps its slot, a new key takes a new slot at the end, a deleted doc leaves a
  hole (live false) until holes pass half the slots, when one full relayout compacts them (order-preserving). Global
  term ids are append-only; `df`, `df_field` and the length sums change by the delta's live docs minus the
  tombstoned docs' terms (a doc -> postings view per segment, built at load); idf and relative lengths are
  recomputed in O(V) and O(N) numpy; keyword dictionaries take new values by sorted insertion with an old -> new code
  remap. The new segment's arrays move to the device before the Index lock is taken. A merge's replace keeps the
  merged docs' slots. An upsert whose `first_offset` is not past every existing one (not a store's) falls back to a
  full relayout. ESCI-sample, 2 cores, deltas of 100 docs (70 edits, 30 new) plus 10 deletes: add_segment p50
  22.6 ms (was 417-538 ms in the dataplane-engineer's stream), a merge's replace of 20 deltas 79 ms.
- 2026-11-10, architect (S7-6 fix): synonym rules with an empty left side, an empty right side or an empty entry
  (`"lamp =>"`, `"=> light"`, `" , "`), or two `=>`, are `InvalidMapping` naming the rule, as OpenSearch refuses them.
- 2026-11-10, architect (fix for the steward's S7-9 finding): `exists` on text and keyword fields follows the stored
  value, not its tokens (a value of stop words or punctuation, or an empty string, exists, as OpenSearch counts it);
  segments gain optional `present/<field>.npy` bitmaps for it. Additive.

- 2026-11-10, architect (S8-2c): copy-on-write states and a writer lock replace the Index RLock (section 4,
  "Concurrency", with the visibility rule); a multi-segment index scores all segments' postings in one pass
  (`ops.bm25f_multi`). Why: under 200 changes/s, searches waited on the RLock for whole refreshes.
- 2026-11-10, architect (S9-4): section 2 gains `geo_point`; section 5 gains `geo_distance`, `geo_bounding_box` and
  the `_geo_distance` sort; the backend gains `sin`, `cos`, `arcsin`. Additive.
- 2026-11-10, architect (S9-5): section 2 gains `nested`; section 4 gains the child segments (`nested/<path>/`,
  listed in the manifest's `nested_fields`); section 5 gains the `nested` query and aggregation; the backend gains
  `segment_max` and `segment_min`. Additive.
- 2026-11-10, architect (S9-2, S9-3): the embeddings spec gains `"fit": "numpy" | "gpu" | "cuml" | "auto"` (the SVD on
  the GPU with CuPy; numpy stays the default) and `Index.add_ranker(name, model, runtime=)` gains "xgboost", "fil"
  and "auto" (the GBDT exported to XGBoost's JSON). Optional imports; decision record
  `docs/decisions/2026-11-10-s9-gpu-dependencies.md`.
- 2026-11-10, architect (S9-1): `knn` and `neural` gain `method` (exact | cagra | ivf_pq) and `method_parameters`;
  the response gains the top-level `knn.<field> = {"requested", "ran"}`, the shape the evaluation-steward proposed,
  ratified with one amendment: several clauses on one field give a list in request order. cuVS is optional
  (`engine/ann.py`); exact is the fallback. Additive.
- 2026-11-10, dataplane-engineer (S9, geo_point and nested through the data plane): section 2 gains intake for both
  types (the accepted point forms, WKT included; nested children validated by the nested mapping) and four rejection
  reasons (`bad_geo`, `geo_out_of_range`, `bad_nested`, `too_many_nested`). Additive. Why: the engine supported
  both (S9-4, S9-5) but intake refused them, so no geo or nested index could be created through serving.
- 2026-11-10, architect (S9-5 fix, found by the dataplane-engineer end to end): section 4 states the rule for
  nested children in a commit: delete files cover parents only, and children's live bits are re-derived from their
  parents on load (`Index.from_commit` revived the children of a parent copy tombstoned by an edit). No format change.
- 2026-11-10, architect (S9-1, the founder's first Colab run, cuVS 26.06): cuVS's filter argument is discovered from
  the installed functions; a search without one (26.06's `ivf_pq.search`) is post-filtered per segment (the rule in
  section 5's knn `method` row), `ran` gains `"<method>+postfilter"`, and the `knn` objects gain `reason` when an
  approximate method did not run as itself; a cuVS failure runs the clause exact instead of raising. Lexical sums over
  fields and synonym groups are elementwise adds in field (group) order, so every backend gives the same bits.
  Additive.
- 2026-11-11, architect (build speed, ESCI-US): `build_segment` analyses each distinct field value once and, for a
  large catalogue in a single-threaded process without CUDA started, in forked worker processes (`S3_BUILD_WORKERS`, default the CPU
  affinity; 1 turns it off); missing checks, keyword columns, presence bitmaps and the slot and id maps are
  vectorised. The segment is unchanged byte for byte (its checksum is equal on toy and ESCI-sample). No format change.
- 2026-11-13, architect (S6-7, the matched load test): section 5 gains **columnar hits**, `hits="columns"` on
  `Index.search` and `Index.search_batch` and `engine.api.expand_hits`, so serving can encode deep pages without a
  Python dict per hit. Additive and opt-in: the default response is unchanged, and so is serving's HTTP response.
  Also recorded today: one set of BM25F parameters per index (section 4), the last-write rule for a repeated id in one
  `patch` call, and that nested text fields keep the default BM25F parameters (section 2).
- 2026-11-13, platform-engineer (S6-7, the server side): section 7 gains **response bytes and columnar hits** (serving
  asks for `hits="columns"` and encodes them to the same bytes as the dict form) and **workers** (`--workers N`, N
  read-only processes behind one port). The HTTP and RPC responses are unchanged but for a new `X-S3-Worker` header.
- 2026-11-13, architect (S6-10, the engine in one GPU process; asked by the coordinator after the platform-engineer
  and the architect proposed the same shape): section 5 says what `search_batch` promises a GPU process serving
  several front processes. New, additive: `search_batch(..., errors="return")` puts a refused request's error in its
  slot instead of failing the batch (default `"raise"`, unchanged). The process architecture is section 7's.
- 2026-11-13, platform-engineer (S6-10s): section 7 gains **one GPU process**, proposed and then built with the
  architect's seven changes:
  - only explain and highlight run in batches of their own;
  - errors map to §7's types, and the one-at-a-time retry is capped;
  - a write is answered only after it is applied;
  - front caches are keyed by the index version;
  - the transport's trust boundary is stated;
  - K > 1 runs under MPS from the start;
  - columns are pickled as they are.
- 2026-11-13, architect (S6-12, from the platform-engineer's S6-10s finding that one GPU process spends its own core on
  pickling replies): section 5's columnar hits gain `ids="bytes"`, `_id` as one numpy `S<w>` array instead of a list
  of str. Opt-in and additive; `expand_hits` decodes it; nothing else changes.
- 2026-11-13, platform-engineer (S6-12s): serving asks for `ids="bytes"` when the engine takes it, and encodes it to the
  same bytes (section 7, "One GPU process").
