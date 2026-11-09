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
- **Analyzer:** one, `s3_english` (the prototype's: lowercase, `[a-z0-9]+(?:'[a-z0-9]+)?` words, strip a trailing `'s`, then `'`, English
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
def read_batch(path: str | Path, mapping: dict) -> Iterator[dict]   # .parquet or .jsonl; one source doc per row
def validate(doc: dict, mapping: dict) -> dict | Rejection         # coerced doc, or why not
def enrich(docs: list[dict], mapping: dict, encoder: Encoder | None) -> list[dict]   # adds vectors
def ingest(path, mapping, store: Store, rejected: Path) -> IntakeReport  # read → validate → enrich → store.apply
@dataclass
class Rejection: _id: str | None; reason: str; field: str | None; detail: str
@dataclass
class IntakeReport: accepted: int; rejected: int; first_offset: int; last_offset: int
```

Rejection reasons (closed set, additive): `missing_id`, `duplicate_id_in_batch`, `bad_type`, `dims_mismatch`,
`unknown_field` (the mapping's `dynamic` is `strict`), `too_long` (keyword over `ignore_above` is *kept* but not indexed
as keyword, as OpenSearch does; this reason is for ids over 512 bytes). Rejected records go to `rejected.jsonl`, one
`{"record": ..., "rejection": ...}` per line.

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
    def __init__(self, path: str | Path, mapping: dict): ...
    def apply(self, changes: Iterable[dict]) -> tuple[int, int]        # (first, last) offset committed
    def head(self) -> int                                              # last committed offset
    def get(self, ids: Sequence[str]) -> list[dict | None]
    def changes(self, after: int, limit: int = 10_000) -> list[dict]   # records with offset > after, ascending
    def snapshot(self, at: int | None = None) -> pandas.DataFrame     # live docs as of offset `at`, ordered by first_offset
```

- **The indexer** (`dataplane/indexer/`): `build(store, mapping, out_dir, device_budget=None) -> Path` takes
  `snapshot(head)`, calls `engine.build.build_segment(snapshot, mapping)`, and writes it with
  `engine.segment.write_segment`. The segment's `version` is that head offset. Doc ids are row order of the snapshot,
  i.e. ascending `first_offset`, so a rebuild keeps the relative order of surviving documents (ties stay stable).
- **The updater** (`dataplane/updater/`): `Replica(store, segment_dir, device)` holds the live `Index` and a reference
  to swap. `poll()` reads `store.changes(after=index.version)`: `patch` → `Index.patch`, `delete` → `Index.delete`,
  `upsert` → counted as pending (visible only after the next build and swap). `swap(new_segment_dir)` loads the second
  `Index` (after the 2x budget check, §0), replays changes after its version, then swaps one reference; searches in
  flight finish on the old one. Freshness = time from `apply` returning to `poll` making it visible.

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
- **Concurrency:** an `Index` is thread-safe. `search`, `search_batch`, `patch`, `delete`, `set_version`, `save`
  and `candidate_features` take one re-entrant lock per `Index`, so a search sees a whole `patch`/`delete` call or
  none of it. Searches on one `Index` therefore run one batch at a time; serving batches with `search_batch`.

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
| `query.match` | `{field: "text"}` or `{field: {"query", "operator": "or"\|"and", "minimum_should_match", "boost"}}` | text fields; BM25 on one field |
| `query.multi_match` | `query`, `fields` (`"f^w"` boosts), `type` `best_fields`\|`most_fields`\|`cross_fields`, `operator`, `minimum_should_match`, `tie_breaker` | **all types are scored as BM25F** with field weights = boosts (the prototype's scorer); hit sets follow `operator`/`minimum_should_match` over the analysed query terms |
| `query.bool` | `must`, `should`, `filter`, `must_not`, `minimum_should_match`, `boost` | scores: sum of `must` and `should`; `filter`/`must_not` do not score |
| `query.term`, `query.terms` | keyword, numeric, boolean fields | constant score `1.0 * boost` in a scoring context (alone, `must`, `should`); 0 in `filter` and `must_not` |
| `query.range` | `gte`, `gt`, `lte`, `lt` on numeric and keyword fields | constant score, as `term` |
| `query.knn` | `{field: {"vector": [...], "k": int, "filter": query}}` | exact; `_score = (1 + cos) / 2` (OpenSearch `cosinesimil`); `filter` is a pre-filter |
| `query.match_all` | `{}` | score 1.0 |
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
  that field's idf (`df_field`): equal to Lucene's BM25 times `(k1 + 1)` (Lucene drops the constant) up to Lucene's
  lossy length encoding. Query terms are deduplicated.
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
  that makes `search_after` exact). Clients pass it back verbatim. A missing field value sorts as `None`. Under
  `rescore`, the first value is the tier (1 inside the rescored window, 0 below it), so the window comes first.
- `aggregations` is present only when requested. `took_ms` always has `total`, plus each stage that ran; stage names
  are the closed set above plus `encode` (additive).
- `explain` (when requested) is a list in hit order, Lucene-shaped: `{value, description, details}` per hit, with per
  term and per field contributions and, under `rescore`, each feature's contribution.
- No hit beyond `total`: an empty page is `{"total": n, "hits": []}`.

## 6. The plan (DSL → operators) · owner architect

`engine/dsl/parse.py: parse(request: dict, state: IndexState) -> Plan` (validates; raises the named errors; in S1
parsing and planning are one pass, because the parser needs the index's field kinds), wrapped by
`engine/plan/plan.py: plan(request, state) -> Plan`. `engine/plan/execute.py: Executor(state, rankers).run(plans:
list[Plan]) -> list[dict]`. Plans are frozen dataclasses (hashable, so the executor groups
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
