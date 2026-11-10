"""S8-2: a base plus delta segments equals a rebuild (ids, order, scores within 1e-4, totals, aggregations), with
global statistics over live docs and first_offset as the tie key; commits, merges, snapshots."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from engine.api import Index
from engine.build import build_segment, merge_segments
from engine.commit import commit_checksum, read_commit
from engine.tests.conftest import toy_catalogue, toy_mapping

TEXT = ["product_title", "product_description"]


def mapping():
    m = toy_mapping(dim=8)
    m["_meta"]["engine"].pop("embeddings")              # vectors come from the store, not refit per segment
    return m


def vec(pid):
    rng = np.random.default_rng(int(pid[1:]))
    v = rng.normal(size=8).astype(np.float32)
    return v / np.linalg.norm(v)


def frame(rows):
    df = pd.DataFrame(rows).sort_values("_first_offset", kind="stable").reset_index(drop=True)
    df["embedding"] = [vec(p) for p in df["product_id"]]
    return df


def scenario(seed=0):
    """A store's history: base, two deltas (new docs, edits keeping first_offset, deletes), and the final
    catalogue (what a rebuild reads)."""
    rng = np.random.default_rng(seed)
    cat = toy_catalogue(160, seed).to_dict("records")
    for i, r in enumerate(cat):
        r["_first_offset"] = i
    live = {r["product_id"]: dict(r) for r in cat[:100]}
    base = frame(list(live.values()))
    deltas, offset = [], 1000
    pool = cat[100:]
    for step in range(2):
        changed = {}
        for r in pool[step * 30:(step + 1) * 30]:                 # new keys: new first_offset
            r = dict(r, _first_offset=offset)
            offset += 1
            live[r["product_id"]] = changed[r["product_id"]] = r
        for pid in rng.choice(sorted(live), 12, replace=False):  # edits: same first_offset, new text and price
            r = dict(live[pid], product_title=live[pid]["product_title"] + " wool black",
                     price=float(rng.uniform(1, 100)))
            live[pid] = changed[pid] = r
        dels = [p for p in rng.choice(sorted(set(live) - set(changed)), 6, replace=False)]
        for p in dels:
            live.pop(p)
        deltas.append((frame(list(changed.values())), dels, 10 + step))
    return base, deltas, frame(list(live.values()))


REQS = [
    {"query": {"match": {"product_title": "black running shoes"}}, "size": 50},
    {"query": {"multi_match": {"query": "wool socks kids", "fields": ["product_title^3", "product_description"]}},
     "size": 40, "aggs": {"b": {"terms": {"field": "product_brand.keyword"}}, "s": {"terms": {"field": "stock"}}}},
    {"query": {"match": {"product_description": {"query": "leather jacket", "operator": "and"}}}, "size": 50},
    {"query": {"match_phrase": {"product_title": {"query": "wool black", "slop": 1}}}, "size": 60},
    {"query": {"fuzzy": {"product_title": {"value": "jaket", "fuzziness": 1}}}, "size": 60},
    {"query": {"prefix": {"product_title": "sh"}}, "size": 60},
    {"query": {"bool": {"must": [{"match": {"product_title": "shoes"}}],
                        "filter": [{"range": {"price": {"gte": 10, "lte": 70}}}]}}, "sort": [{"price": "desc"}],
     "size": 60},
    {"query": {"match_all": {}}, "sort": [{"product_brand.keyword": "asc"}, {"stock": "desc"}], "size": 200},
    {"query": {"term": {"category": "socks"}}, "size": 200},              # all scores tie: first_offset order
    {"query": {"knn": {"embedding": {"vector": vec("P0007").tolist(), "k": 15}}}, "size": 15},
]


def same(a, b):
    ha, hb = a["hits"]["hits"], b["hits"]["hits"]
    assert a["hits"]["total"] == b["hits"]["total"]
    assert [h["_id"] for h in ha] == [h["_id"] for h in hb]
    assert np.allclose([h["_score"] for h in ha], [h["_score"] for h in hb], rtol=0, atol=1e-4)
    assert [h["sort"][:-1] for h in ha] == [h["sort"][:-1] for h in hb] or "sort" not in str(a)
    assert a.get("aggregations") == b.get("aggregations")


def incremental(base, deltas, device="cpu"):
    ix = Index(build_segment(base, mapping(), version=1), device)
    for df, dels, v in deltas:
        ix.add_segment(build_segment(df, mapping(), version=v), version=v,
                       delete_ids=list(dels) + list(df["product_id"]))
    return ix


@pytest.mark.parametrize("seed", [0, 1])
def test_base_plus_deltas_equals_a_rebuild(seed, device):
    base, deltas, final = scenario(seed)
    ix = incremental(base, deltas, device)
    rebuilt = Index(build_segment(final, mapping(), version=11), device)
    assert len(ix.segments) == 3 and ix.n_docs == rebuilt.n_docs == len(final) and ix.version == 11
    for r in REQS:
        same(ix.search(r), rebuilt.search(r))
    # the tie key: the last sort element is first_offset, the same as the rebuild's
    a = ix.search(REQS[8])["hits"]["hits"]
    assert [h["sort"][-1] for h in a] == sorted(h["sort"][-1] for h in a)


def test_delete_relays_out_statistics(device):
    base, deltas, final = scenario(0)
    ix = Index(build_segment(base, mapping()), device)
    gone = list(base["product_id"][:10])
    ix.delete(gone)
    rebuilt = Index(build_segment(base[~base["product_id"].isin(gone)], mapping()), device)
    for r in REQS[:4]:
        same(ix.search(r), rebuilt.search(r))


def test_merge_and_replace_and_remove(device):
    base, deltas, final = scenario(0)
    ix = incremental(base, deltas, device)
    ix.patch("price", [final["product_id"][3]], [12.5])
    segs, v = ix.snapshot_segments()
    merged = merge_segments(segs, version=v)
    ix2 = incremental(base, deltas, device)
    ix2.patch("price", [final["product_id"][3]], [12.5])
    ix2.replace_segments([s["segment_id"] for s in ix2.segments], merged)
    assert len(ix2.segments) == 1
    for r in REQS:
        same(ix.search(r), ix2.search(r))
    # remove the newest delta: its docs are gone (no resurrection of the tombstoned copies)
    last = ix.segments[-1]["segment_id"]
    n_last = ix.segments[-1]["live"]
    ix.remove_segment(last)
    assert ix.n_docs == len(final) - n_last


def test_snapshot_shares_immutable_arrays():
    base, deltas, _ = scenario(0)
    ix = incremental(base, deltas)
    segs, v = ix.snapshot_segments([ix.segments[0]["segment_id"]])
    part = ix._st.parts[0]
    assert segs[0].text.doc_ids is part.seg.text.doc_ids and segs[0].vectors["embedding"] is part.seg.vectors["embedding"]
    assert segs[0].live is not part.seg.live and v == ix.version


def test_commit_round_trip(tmp_path, device):
    base, deltas, final = scenario(1)
    ix = incremental(base, deltas, device)
    path = ix.save_commit(tmp_path)
    c = json.loads(path.read_text())
    assert c["format"] == "s3commit" and c["version"] == 11 and len(c["segments"]) == 3 and c["deletes"]
    back = Index.from_commit(path, device)
    assert back.version == 11
    for r in REQS:
        same(ix.search(r), back.search(r))
    (tmp_path / c["deletes"][c["segments"][0]]).write_bytes(b"x")
    with pytest.raises(Exception):
        read_commit(path)


def test_commit_checksum_vector(tmp_path):
    """The shared test vector: the data plane's commit writer must give the same checksum for these bytes."""
    bits = np.array([0xFFFFFFFE, 0x00000003], dtype=np.uint32)
    with open(tmp_path / "del-7-seg-5-aaaa0000.npy", "wb") as fh:
        np.save(fh, bits, allow_pickle=False)
    commit = {"format": "s3commit", "format_version": 1, "version": 7, "segments": ["seg-5-aaaa0000", "seg-7-bbbb1111"],
              "deletes": {"seg-5-aaaa0000": "del-7-seg-5-aaaa0000.npy"}}
    file_sha = hashlib.sha256((tmp_path / "del-7-seg-5-aaaa0000.npy").read_bytes()).hexdigest()
    body = dict(commit, delete_sha256={"seg-5-aaaa0000": file_sha})
    want = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert commit_checksum(commit, tmp_path) == want


def test_add_segment_refuses_another_encoder(catalogue):
    m = toy_mapping(dim=8)
    a = catalogue.iloc[:60].assign(_first_offset=range(60))
    b = catalogue.iloc[60:].assign(_first_offset=range(60, len(catalogue)))
    ix = Index(build_segment(a, m))
    with pytest.raises(ValueError):
        ix.add_segment(build_segment(b, m), version=2)             # refit on other docs: another encoder
    assert ix.n_docs == 60 and len(ix.segments) == 1


def test_add_segment_is_atomic_on_a_duplicate():
    base, deltas, _ = scenario(0)
    ix = Index(build_segment(base, mapping()))
    before = ix.search(REQS[0])
    dup = base.iloc[:3]
    with pytest.raises(ValueError):
        ix.replace_segments([], build_segment(dup, mapping()))        # live twice, not tombstoned
    same(ix.search(REQS[0]), before)
