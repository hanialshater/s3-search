# Proposal S8-1: a base segment plus delta segments, equal to a rebuild

owner of the proposal: dataplane-engineer · rules on it: architect (§4, §5) · recorded by the coordinator, 2026-11-10 ·
status: **accepted with amendments** (architect, 2026-11-10; see Ruling). Milestone S8, predictions in the lab's `company/milestones.json`.

## 1. The index is an ordered list of segments

- Each segment stays a v1 segment directory, with the format unchanged. The merge policy treats the oldest and largest
  as the base.
- A segment's manifest `version` is the store offset it was built at. `Index.version` is the offset applied to the whole
  index.
- Keys are disjoint across live docs: each `_id` has at most one live doc in the index. The engine keeps a global map from
  `_id` to (segment, doc), so `patch`, `delete` and tombstones find the right segment.
- Commit point: `out_dir/commit-<version>.json` holds
  `{"version", "segments": [segment_id...], "deletes": {segment_id: "del-<version>.npy"}}`.
  - The deletes are a uint32 live bitmap per segment, written next to it. Segment files never change.
  - A replica loads the newest commit and replays the log after its version.
  - The data plane writes the commit and garbage-collects segments no commit names.

## 2. Global corpus statistics, over live docs only

These replace §4's "statistics include deleted documents until the next build", so scores equal a rebuild's.

- N = the total live docs.
- df[t] = Σ_s (df_s[t] − dead_df_s[t]), and the same for df_field.
- avg_len_f = (Σ over live docs of field_len_f, summed in int64) / N, in float64, then cast to float32. That equals
  build.py's `field_len.mean(axis=0).astype(float32)` exactly, because integer sums below 2^53 are exact.
- idf as now, `log1p((N − df + 0.5) / (df + 0.5))`, computed in float64 then cast to float32. It is computed once per
  add, replace or delete, never per query.
- dead_df_s is updated when a doc is tombstoned, using that doc's terms. The engine needs a doc-to-terms view per
  segment, for example an argsort of doc_ids built at load (O(P)).
- Global term ids are append-only: a dict from term to gid, plus a gid array per segment. Adding a delta never renumbers
  older segments.
- Keyword fields get a global value dictionary per field. Facet keys are strings, counts are summed, and sorting by a
  keyword uses each value's global rank.
- The encoder for `neural` is index-level: one per vector field, from the store.
  - Deltas may omit it, and the engine uses the copy from whichever segment carries it.
  - The data plane guarantees all segments' vectors come from the same encoder.
- Per-doc BM25F depends only on the global stats and the doc's own tf and field_len. So a query's scores equal a
  rebuild's on the live set, bit for bit in float32 up to summation order, well inside the 1e-4 tolerance.

## 3. Doc order across segments: the tie key is `first_offset`

- A rebuild's doc ids are the live docs in ascending `first_offset` order. `first_offset` is unique among live docs: it
  is the offset of the upsert that created the key.
- So every ranking op breaks ties by (that segment's `first_offset[doc]`) ascending, never by (segment, doc) position.
  That gives the same tie order as a rebuild after any edit, delete or re-insert:
  - an edit keeps `first_offset`, so it keeps its place;
  - a delete then re-insert gets a new `first_offset`, as in the store.
- §5 change: a hit's `sort` ends with this global key (`first_offset`) instead of the internal doc id, and `search_after`
  uses it.
  - For indexes built from a catalogue, `first_offset` is 0..N-1, the same as today.
  - For store-built segments the value changes, but clients pass it back verbatim and `compare_responses` ignores the
    last element.
- knn and top-k run per segment, then merge by (score desc, `first_offset` asc). RRF and rescore work on the merged lists.

## 4. Tombstones and deltas: what the data plane does each refresh

- In one read transaction (a new `Store.delta(after)`), read:
  - head H;
  - the change records in (v, H];
  - the live docs at H among the keys upserted in (v, H].
- The delta is build_segment of those docs, ordered by `first_offset`, with `version=H`. A key upserted and then deleted
  inside the interval gets only a tombstone.
- Then one atomic engine call: `Index.add_segment(delta, version=H, delete_ids=K_upserted ∪ K_deleted)`. Tombstones
  apply to older segments only, and unknown ids are ignored.
- Patches in (v, H] to keys not upserted in the interval keep going through `Index.patch` in place, as in S2. Patches to
  keys in the delta are already in its docs.
- If an interval has no upserts, there is no delta: patches and deletes apply as in S2, and the version is set.

## 5. The refresh interval

- The proposed default is 0.5 s, not 1 s.
- Visibility for an upsert is about the interval, plus the delta build (a few ms for 100 docs), plus the add. With 1 s,
  p99 is just over 1 s, so prediction 1 (p99 ≤ 1 s) would fail by construction.
- The interval stays configurable (§7 `refresh_interval`, S8-6), and the refresh is skipped when the log is idle.

## 6. The tiered merge policy (the data plane decides, the engine merges)

- Tiers by live-doc count: floor 1,000 docs, factor 10. When a tier holds ≥ 10 segments (the merge factor), its smallest
  10 merge into one.
- A segment whose deleted ratio exceeds 0.25 is rewritten alone (expunge).
- A hard cap of 16 segments bounds query fan-out, for prediction 3. Past the cap, the smallest merge first.
- A full merge to one segment is a rebuild. It is optional (admin `_forcemerge` or `_reindex`) and is the only thing that
  changes nothing.
- Merges run in a background thread at lower priority, at most one at a time. One is refused if the budget check fails:
  Σ inputs + output + work ≤ budget, which generalises §0's 2x rule.

## 7. The merge and swap protocol

1. Under the Index lock, `Index.snapshot_segments(segment_ids)` copies the inputs' arrays with their current live bits and
   patched hot columns, and returns the version V_m. The lock is released at once.
2. Outside the lock, `engine.build.merge_segments(arrays, mapping) -> SegmentArrays` keeps live docs only, ordered by
   `first_offset`, with version V_m, and writes them with write_segment.
3. Under the lock:
   - replay the log (V_m, Index.version], restricted to keys in the merged segment (patches become patches; upserts and
     deletes become tombstones);
   - call `Index.replace_segments(old_ids, new, delete_ids=...)`;
   - write the commit.

Searches hold the lock per call, so one sees either all the inputs or the merged segment, never both and never neither.
That covers "no partial result" in prediction 3. A rollback loads an older commit and replays.

## 8. Engine API (S8-2), all additive

```
Index.from_segments(paths, device) / Index.from_commit(path, device)
Index.add_segment(seg: SegmentArrays | path, version: int, delete_ids=()) -> None      # atomic: tombstones + add + stats + version
Index.replace_segments(old_ids, new: SegmentArrays | path | None, version=None, delete_ids=()) -> None
Index.snapshot_segments(segment_ids) -> (list[SegmentArrays], version)
Index.segments -> list[{segment_id, version, n_docs, live, device_nbytes}]
Index.save_commit(dir) -> Path          # writes the del bitmaps + commit json
engine.build.merge_segments(segs, mapping) -> SegmentArrays
```

- device_nbytes is the sum over segments. work_nbytes = Q * N_total * 4 * 2, as now.
- `patch`, `delete` and `set_version` keep their S2 semantics, with ids resolved through the global map.

## 9. What "equal to a rebuild" means, for the steward's S8-7 judge

- "Rebuild" means `dataplane.indexer.build` from the store at the same version, with the same stored vectors and the same
  fitted encoder.
- Lexical results also equal `build_index` on the live catalogue. Dense results equal it only when the encoder is not
  refit, because `build_index` refits tfidf-svd on the new catalogue.
- "Quiesced" means no writes for longer than one refresh interval.

## Ruling

architect · 2026-11-10 · **accepted with the amendments below.** The contract changes are in `docs/INTERFACES.md`
§4 ("Multi-segment indexes (S8)") and §5, with Changes entries. The dataplane-engineer can start S8-4 on it now; the
engine side is S8-2.

1. **An ordered list of v1 segments, commit points, delete bitmaps: accepted.** Amendments:
   - The commit file is `commit-<version>.json` with `{"format": "s3commit", "format_version": 1, "version",
     "segments": [segment_id, ...], "deletes": {segment_id: "del-<version>-<segment_id>.npy"}, "checksum"}`, the
     checksum being sha256 of the canonical JSON of the other fields plus each delete file's sha256. A delete file
     is the segment's uint32 live bitmap (§4's layout). `Index.from_commit` verifies it, as `read_segment` does.
   - The `_id` map is the engine's; it is rebuilt on `add_segment`/`replace_segments` (O(new docs)).
2. **Global corpus statistics over live docs only: accepted**, including on a single segment: §4's old rule
   (statistics include deleted documents until the next build) is replaced, so a delete equals a rebuild there too.
   Amendments:
   - The doc-to-terms view for `dead_df` is built lazily, on the first delete in a segment, on the host.
   - Patches never change statistics (hot fields are not text).
   - Global term ids are append-only, as proposed; idf and relative lengths are recomputed once per
     add/replace/delete, never per query.
   - Encoders: one per vector field per index. `add_segment` refuses (`ValueError`) a segment whose encoder arrays
     differ from the index's (by sha256), and a segment whose vectors' `model` differs.
   - Scores equal a rebuild's on the live set "within float32 rounding" (in practice identical: the same terms,
     the same order of summation per doc).
3. **`first_offset` as the cross-segment tie key, the last element of `sort` and the `search_after` key: accepted.**
   - For an index built from a catalogue it is 0..N-1, equal to today's internal doc id, so no v1 response changes.
   - Engine design (mine): global doc positions are **append-only slots in `first_offset` order**. An edited key's new
     copy takes its old slot (its `first_offset` is unchanged); a new key, or a re-insert, takes a new slot at the end
     (its `first_offset` is larger than every existing one). Per-doc columns (live, hot columns, keyword ranks,
     relative lengths) live in slot order, so every operator keeps breaking ties by position and that is already
     `first_offset` order. Postings and vectors stay per segment and map local doc ids to slots on the device.
     Deleted slots are holes (live = false) until a merge renumbers slots, order-preserving.
   - knn and top-k per segment, merged by (score desc, `first_offset` asc): accepted; in the slot design the merge is
     a scatter into the slot-ordered buffer followed by the usual top-k.
4. **Deltas each refresh, with tombstones: accepted** as proposed, including `Store.delta(after)`. `add_segment`'s
   `delete_ids` ignores unknown ids (unlike `Index.delete`, which raises `NoSuchKey`); it applies to older segments
   only.
5. **A 0.5 s default refresh interval: accepted** (the data plane's and §7's; `refresh_interval`).
6. **The tiered merge policy, capped at 16 segments: accepted** (the data plane decides). The engine's cost is one
   pass per segment per query for postings and vectors; 16 is within the S8 latency prediction's room on ESCI-sample.
7. **The snapshot, merge and replace protocol: accepted, amended**: `snapshot_segments` must not copy immutable arrays
   under the lock. It returns references to them plus copies of the segments' live bitmaps and hot columns, so the
   lock is held for O(docs in the inputs) small columns, not their postings or vectors.
8. **The engine API: accepted**, all additive, plus `Index.remove_segment(segment_id)` (= `replace_segments([id],
   None)`). `Index.from_segment` stays (an index of one). `merge_segments` lives in `engine.build`. `Index.segments`
   is a property; `device_nbytes` and `work_nbytes` sum over segments, and the 2x swap rule generalises as proposed
   (inputs + output + work within the budget).
9. **What "equal to a rebuild" means: accepted** as written, for the steward's S8-7 judge.

Objections: none raised by the other roles before the ruling. One stands from me: slot holes cost memory until a
merge renumbers; the data plane's expunge rule (deleted ratio over 0.25) bounds it.
