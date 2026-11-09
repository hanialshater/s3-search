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
`dataplane` imports `engine.analysis`, `engine.segment`, `engine.build`, `engine.backend`; `serving` imports `engine.api`
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

- **Types (v1):** `text` (searchable; BM25F field), `keyword` (term/terms filters, terms aggs, sort; single-valued
  in v1), `float`, `integer`, `long`, `boolean` (term/range filters, sort), `knn_vector` (exact dense; the `method`
  is accepted for OpenSearch and ignored: the engine is exact). Sub-fields are addressed as `product_brand.keyword`.
- **Analyzer:** one, `s3_english` (the prototype's: lowercase, `[a-z0-9']+` words, strip `'s` and `'`, English
  stopwords, Porter stemmer via `snowballstemmer`, which S1 adds to `requirements.txt`). Code:
  `engine.analysis.analyze(text: str, analyzer: str = "s3_english") -> list[str]`. The conformance suite defines the
  same analyzer in the OpenSearch index settings.
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
is not an object, a patch without `fields`), `bad_op`, `not_hot` and `no_such_key` (the change feed). Rejected records
go to `rejected.jsonl`, one `{"record": ..., "rejection": ...}` per line.

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
- **Encoders.** `encoder=None` resolves each vector field's `_meta.engine.embeddings.<field>.model` through
  `dataplane.intake.encoders.register(name, factory)`; an unregistered name gets the deterministic `HashingEncoder`
  (signed feature hashing of words and bigrams; `name` `hash-<dim>`), the development default. A store records the
  encoder of each vector field (`store.meta("encoder.<field>")`) and refuses to mix in another one.

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
- **The updater** (`dataplane/updater/`): `Replica(store, segment_dir, device="cpu", budget=None, batch=32)` holds the
  live `Index` and a reference to swap; `segment_dir` is a segment or a directory of them (the latest is loaded).
  Loading checks `device_nbytes + work_nbytes <= budget`, then replays the log. `poll()` reads
  `store.changes(after=index.version)`: `patch` → `Index.patch`, `delete` → `Index.delete`, `upsert` → counted as
  pending (visible only after the next build and swap). Within a poll the changes are coalesced: the last value of a
  (field, id) wins, one `Index.patch` call per field, one `Index.delete` call, and an id the index does not hold live
  (upserted or re-inserted since the segment) is not patched: it waits for the build like its upsert.
  `swap(new_segment_dir)` checks `current device_nbytes + new device_nbytes + work_nbytes <= budget` (the double
  buffer, §0; 2x when the two are the same size) and raises `engine.backend.OverBudget` otherwise, loads the second
  `Index`, replays changes after its version (outside the lock, then the rest under it), then swaps one reference;
  searches in flight finish on the old one. A swap to an older segment is the rollback. `budget=None` is
  `Backend.memory_budget()` (the §0 rule, computed by the updater until the backends land). Also `start(interval)` /
  `stop()` (a background poller), `stats()`, `live_ids()`, `on_visible(fn)`. Freshness = time from `apply` returning
  to `poll` making it visible.
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
| `kw/<field>.codes.npy` | int32 [N] | 0 = missing; codes order = sorted value order |
| `kw/<field>.values.parquet` | `value` string | row = code; row 0 is `""` |
| `num/<field>.npy` | float32, int32, int64 or bool [N] | the column |
| `num/<field>.missing.npy` | uint32 bitmap | missing values |
| `vec/<field>.npy` | float16 or float32 [N, D] | L2-normalised vectors |

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
- **Statistics** (`df`, `avg_len`, `N`) include deleted documents until the next build, as Lucene does until a merge.
  The idf is `log(1 + (N - df + 0.5) / (df + 0.5))`, computed at load.
- **`device_nbytes`** = bytes of every array the engine moves to the device (all of the above but the string tables)
  plus derived arrays (idf). `work_nbytes` (per batch) is `Q * N * 4 * 2` for the lexical score buffer and match counts
  at the configured batch size.
- **Checksum:** `read_segment(verify=True)` recomputes each file's sha256 and the manifest checksum; any mismatch
  raises `engine.segment.CorruptSegment`.
- v1 is **one segment per index**; multi-segment indexes are a later `format_version` (additive).

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
```

## 5. The search request and response (serving → engine) · owner architect

**The API** (`engine/api.py`; the referee's contract tests are written against exactly this):

```python
def build_index(catalogue: pandas.DataFrame, mapping: dict, device: str = "cpu") -> Index
def available_devices() -> list[str]
class UnsupportedQuery(ValueError):   # .path (JSON path, e.g. "query.bool.must[1].fuzzy"), .reason
class InvalidRequest(UnsupportedQuery) # malformed but in-subset (bad types, from+size over 10,000)
class InvalidMapping(ValueError)
Index.search(request: dict) -> dict
Index.search_batch(requests: list[dict]) -> list[dict]   # same result as [search(r) for r in requests], batched
Index.from_segment(path, device) -> Index
```

**The request**: a subset of the OpenSearch query DSL. Anything else, including unknown keys, raises
`UnsupportedQuery` naming its path.

| Key | Supported | Notes |
|---|---|---|
| `query.match` | `{field: "text"}` or `{field: {"query", "operator": "or"\|"and", "minimum_should_match", "boost"}}` | text fields; BM25 on one field |
| `query.multi_match` | `query`, `fields` (`"f^w"` boosts), `type` `best_fields`\|`most_fields`\|`cross_fields`, `operator`, `minimum_should_match`, `tie_breaker` | **all types are scored as BM25F** with field weights = boosts (the prototype's scorer); hit sets follow `operator`/`minimum_should_match` over the analysed query terms |
| `query.bool` | `must`, `should`, `filter`, `must_not`, `minimum_should_match`, `boost` | scores: sum of `must` and `should`; `filter`/`must_not` do not score |
| `query.term`, `query.terms` | keyword, numeric, boolean fields | filter context; score 1.0 alone (OpenSearch), 0 inside `filter` |
| `query.range` | `gte`, `gt`, `lte`, `lt` on numeric and keyword fields | |
| `query.knn` | `{field: {"vector": [...], "k": int, "filter": query}}` | exact; `_score = (1 + cos) / 2` (OpenSearch `cosinesimil`); `filter` is a pre-filter |
| `query.match_all` | `{}` | score 1.0 |
| `query.neural` (S3) | `{field: {"query_text", "k"}}` | encoded by the field's encoder |
| `query.hybrid` + `rank.rrf` (S1-5, extension) | `{"hybrid": {"queries": [...]}}`, `"rank": {"rrf": {"rank_constant": 60, "window_size": 20000}}` | RRF fusion; `_score` is the RRF score |
| `rescore` (S1-5, extension) | `{"window_size": 1000, "s3_ranker": {"model": "<name>"}}` | the learned ranker over the top window |
| `aggs` / `aggregations` | `{name: {"terms": {"field", "size": 10}}}` | over the **full match set**, independent of from/size; buckets by count desc, then key asc |
| `sort` | `"_score"`, `{"_score": {"order"}}`, `{field: {"order": "asc"\|"desc", "missing": "_last"\|"_first"}}`, `"field"` | default `[_score desc]`; ties always end by doc id asc |
| `from`, `size` | ints, `from + size <= 10,000` | default 0, 10 |
| `search_after` | the last hit's `sort` list | with `from` 0 |
| `explain` | bool | adds `explain` |
| `_source`, `track_total_hits` | accepted and ignored (totals are always exact; serving adds `_source` from the store) | |

Scoring, exactly:
- BM25F per document: `pseudo_tf = sum_f w_f * tf_f / (1 - b + b * len_f / avg_len_f)`;
  `score = sum_terms idf * pseudo_tf * (k1 + 1) / (k1 + pseudo_tf)`. Single-field `match` is the same with one field and
  that field's idf (`df_field`): equal to Lucene's BM25 times `(k1 + 1)` (Lucene drops the constant) up to Lucene's
  lossy length encoding. Query terms are deduplicated.
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
  that makes `search_after` exact). Clients pass it back verbatim.
- `aggregations` is present only when requested. `took_ms` always has `total`, plus each stage that ran; stage names
  are the closed set above plus `encode` (additive).
- `explain` (when requested) is a list in hit order, Lucene-shaped: `{value, description, details}` per hit, with per
  term and per field contributions and, under `rescore`, each feature's contribution.
- No hit beyond `total`: an empty page is `{"total": n, "hits": []}`.

## 6. The plan (DSL → operators) · owner architect

`engine/dsl/parse.py: parse(request: dict) -> Request` (validates; raises the named errors).
`engine/plan/plan.py: plan(req: Request, index_meta) -> Plan`. `engine/plan/execute.py: execute(plans: list[Plan],
state: IndexState, backend: Backend) -> list[dict]`. Plans are frozen dataclasses (hashable, so the executor groups
equal shapes into one batch).

```python
Node = Lexical | Dense | Term | Range | MatchAll | Bool | Hybrid
Lexical(fields: tuple[tuple[str, float], ...], terms: tuple[str, ...], operator: str, msm: int, idf: "global"|"field", boost: float)
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

**Serving** (`serving/`): `serving.app.create_app(config: ServingConfig) -> ASGI app`; stdlib-friendly, the HTTP
framework is the platform-engineer's call (a decision record if it adds a dependency).

| Method, path | Body → response |
|---|---|
| `POST /indexes/{name}/_search` | §5 request → §5 response (+ `_source` per hit when asked) |
| `POST /indexes/{name}/_msearch` | `[request, ...]` → `[response, ...]` (one `search_batch`) |
| `PUT /indexes/{name}` | `{"mapping": {...}, "source": {"path": "..."}}` → `{"name", "version"}`: intake, build, load |
| `POST /indexes/{name}/_docs` | `[change record without offset, ...]` → `{"first_offset", "last_offset", "rejected": [...]}` |
| `POST /indexes/{name}/_reindex` | `{}` → `{"segment_id", "version"}`: build from the store, swap |
| `GET /indexes/{name}/versions` | → `[{"segment_id", "version", "created", "live": bool}]` |
| `POST /indexes/{name}/_rollback` | `{"segment_id"}` → `{"segment_id", "version"}`: swap to a kept segment, replay the log after it |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | liveness; ready once a segment is loaded; Prometheus text (stage timings, counters) |

Errors: HTTP 400 with `{"error": {"type": "unsupported_query" | "invalid_request" | "invalid_mapping", "path", "reason"}}`;
409 for `over_budget`; 404 for an unknown index or segment.

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
  segmentVersion: null                                      # null: latest; a segment_id pins (rollback)
  image: s3-search:cpu
status:
  phase: Pending | Indexing | Ready | Swapping | Failed
  segmentId: seg-1044-3fa2c9d1
  version: 1044
  readyReplicas: 2
  conditions: [...]
```

- The operator reconciles a `SearchIndex` into a `Job` (the indexer: `python -m dataplane.indexer`), a
  `Deployment` of replicas (`python -m serving`), a `Service`, and a PVC for segments. A new segment rolls out by a
  rolling update with `maxUnavailable: 0`; `segmentVersion` pins one (rollback).
- **It never assumes a GPU.** Only when `device: cuda` does it add `nvidia.com/gpu` limits from `resources`, a
  `nodeSelector` (`nvidia.com/gpu.present: "true"`) and the `nvidia.com/gpu` toleration. One `Dockerfile` with
  `ARG BASE=cpu|cuda` selects the base image and `requirements-gpu.txt`.
- Tests: the reconciler against a real kube-apiserver and etcd (envtest binaries at
  `/tmp/claude-0/envtest/controller-tools/envtest/` in the dev container, where kind cannot run pods); the full kind
  end-to-end in GitHub Actions.

## Changes

- 2026-11-09, architect: v1 written from the lab's skeleton (sections 0 and 1 to 7 filled; section 0 added for the
  GPU-portable backend).
- 2026-11-09, dataplane-engineer: sections 2 and 3 amended with the S2 code (additive). Section 2: `read_batch`
  yields a `Rejection` for a malformed line; four reasons added (`malformed`, `bad_op`, `not_hot`, `no_such_key`);
  coercion and unknown-field rules written down; `apply_changes` for the change feed; the encoder hook and one encoder
  per store. Section 3: atomic batches (a refused change refuses its batch, so offsets stay gapless), `changes(docs=)`,
  the snapshot frame's columns and dtypes (`_first_offset`, `_last_offset`), the indexer's version, temp-and-rename
  writes, merges, gc and job, the updater's coalescing and its swap budget (current plus new, which is 2x for equal
  sizes), and the freshness hook. Why: these are what the code does, and the engine, serving and the steward's B1-4
  harness need them to call it. Sections 0 and 4 to 7 are unchanged; what they still need is in the S2 report.
