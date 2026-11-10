"""geo_point and nested through the data plane (S9-4, S9-5), on the real engine: validation, intake, the store, deltas,
merges, commits, a hot patch on a parent with children, and serving's _bulk update of a nested array; each state is
checked against a rebuild from the store and against engine.api.build_index on the same catalogue."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from dataplane.freshness import compare_responses
from dataplane.indexer import TieredMergePolicy, build
from dataplane.intake import Rejection, apply_changes, ingest, validate
from dataplane.mapping import InvalidMapping, parse_mapping
from dataplane.tests.conftest import MAPPING, make_catalogue
from dataplane.updater import Replica

SELLERS = [f"seller-{i:02d}" for i in range(8)]
NY = {"lat": 40.71, "lon": -74.01}
REQS = [
    {"query": {"bool": {"filter": [{"geo_distance": {"distance": "500km", "location": NY}}]}},
     "sort": [{"_geo_distance": {"location": NY, "order": "asc"}}], "size": 100},
    {"query": {"geo_bounding_box": {"location": {"top_left": {"lat": 45, "lon": -80}, "bottom_right": {"lat": 35, "lon": -70}}}},
     "size": 100},
    {"query": {"nested": {"path": "offers", "score_mode": "max", "query": {"bool": {
        "must": [{"match": {"offers.note": "fast"}}],
        "filter": [{"terms": {"offers.seller": ["seller-03", "seller-07"]}}, {"range": {"offers.price": {"lte": 200}}}]}}}},
     "size": 100},
    {"size": 0, "aggs": {"o": {"nested": {"path": "offers"}, "aggs": {"s": {"terms": {"field": "offers.seller"}}}}}},
    {"query": {"bool": {"must": [{"match": {"product_title": "lamp"}}],
                        "filter": [{"nested": {"path": "offers", "query": {"range": {"offers.stock": {"gt": 10}}}}}]}},
     "sort": [{"price": {"order": "asc"}}], "size": 100},
]


def geo_mapping() -> dict:
    m = copy.deepcopy(MAPPING)
    m["properties"]["location"] = {"type": "geo_point"}
    m["properties"]["offers"] = {"type": "nested", "properties": {
        "seller": {"type": "keyword"}, "price": {"type": "float"}, "stock": {"type": "integer"}, "note": {"type": "text"}}}
    return m


def offers(rng, k: int | None = None) -> list[dict]:
    n = int(rng.integers(0, 5)) if k is None else k
    return [{"seller": SELLERS[int(rng.integers(len(SELLERS)))], "price": round(float(rng.uniform(1, 400)), 2),
             "stock": int(rng.integers(0, 40)), "note": ["fast shipping", "slow boat", "fast and cheap"][int(rng.integers(3))]}
            for _ in range(n)]


def geo_catalogue(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat = make_catalogue(n)
    cat["location"] = [{"lat": round(float(rng.uniform(30, 50)), 4), "lon": round(float(rng.uniform(-90, -60)), 4)}
                       for _ in range(n)]
    cat["offers"] = [offers(rng) for _ in range(n)]
    return cat


def equals(index, other, reqs=REQS, tie_keys=True):
    """Same responses; with ``tie_keys`` also the same first_offsets (only between store-built indexes: a catalogue
    numbers its rows from 0, the store's offsets start at 1)."""
    for i, q in enumerate(reqs):
        a, b = index.search(q), other.search(q)
        assert compare_responses(a, b) is None, (i, compare_responses(a, b))
        if tie_keys:
            assert [h["sort"][-1] for h in a["hits"]["hits"]] == [h["sort"][-1] for h in b["hits"]["hits"]], i


def rebuild(store, tmp_path, name):
    from engine.api import Index
    return Index.from_segment(build(store, None, tmp_path / name))


# ---- the mapping and validation --------------------------------------------------------------------------------------

@pytest.mark.parametrize("edit, path", [
    (lambda p: p.update(location={"type": "geo_point", "ignore_malformed": True}), "properties.location.ignore_malformed"),
    (lambda p: p.update(offers={"type": "nested"}), "properties.offers.properties"),
    (lambda p: p["offers"]["properties"].update(v={"type": "knn_vector", "dimension": 2}), "properties.offers.properties.v.type"),
    (lambda p: p["offers"]["properties"].update(g={"type": "geo_point"}), "properties.offers.properties.g.type"),
    (lambda p: p["offers"]["properties"].update(n={"type": "nested", "properties": {"x": {"type": "keyword"}}}),
     "properties.offers.properties.n.type"),
    (lambda p: p.update(offers={**p["offers"], "include_in_parent": True}), "properties.offers.include_in_parent"),
])
def test_bad_geo_and_nested_mappings_name_their_path(edit, path):
    m = geo_mapping()
    edit(m["properties"])
    with pytest.raises(InvalidMapping) as e:
        parse_mapping(m)
    assert e.value.path == path


@pytest.mark.parametrize("value", [{"lat": 40.7, "lon": -74.0}, {"lat": "40.7", "lon": "-74.0"}, "40.7,-74.0",
                                   " 40.7 , -74.0 ", [-74.0, 40.7], np.array([-74.0, 40.7]), "POINT (-74.0 40.7)",
                                   "point(-74 40.7)"])
def test_geo_point_forms(value):
    d = validate({"product_id": "a", "location": value}, geo_mapping())
    assert d["location"] == {"lat": 40.7, "lon": -74.0}


@pytest.mark.parametrize("value, reason", [
    ("dr5regw3", "bad_geo"), ("40.7", "bad_geo"), ({"lat": 1}, "bad_geo"), ({"lat": 1, "lon": 2, "z": 3}, "bad_geo"),
    ([1, 2, 3], "bad_geo"), (True, "bad_geo"), (12.5, "bad_geo"), ({"lat": "n", "lon": 0}, "bad_geo"),
    ({"lat": float("inf"), "lon": 0}, "bad_geo"), ({"lat": 91, "lon": 0}, "geo_out_of_range"),
    ([181, 0], "geo_out_of_range"), ("POINT (200 10)", "geo_out_of_range"),
])
def test_geo_point_rejections_are_named(value, reason):
    r = validate({"product_id": "a", "location": value}, geo_mapping())
    assert isinstance(r, Rejection) and (r.reason, r.field) == (reason, "location")


def test_nested_values_are_validated_by_the_nested_mapping():
    m = geo_mapping()
    d = validate({"product_id": "a", "offers": [{"seller": "s1", "price": "3.5", "stock": 2.0, "extra": 1},
                                                {"seller": None, "note": "fast"}]}, m)
    assert d["offers"] == [{"seller": "s1", "price": 3.5, "stock": 2, "extra": 1}, {"note": "fast"}]
    assert validate({"product_id": "a", "offers": {"seller": "s2"}}, m)["offers"] == [{"seller": "s2"}]
    assert validate({"product_id": "a", "offers": []}, m)["offers"] == []
    for v, reason, field in [([1], "bad_nested", "offers"), ("x", "bad_nested", "offers"),
                             ([{"price": "x"}], "bad_type", "offers.price"), ([{"stock": 2 ** 40}], "bad_type", "offers.stock"),
                             ([{}] * 65_536, "too_many_nested", "offers")]:
        r = validate({"product_id": "a", "offers": v}, m)
        assert (r.reason, r.field) == (reason, field), v
    m["dynamic"] = "strict"
    r = validate({"product_id": "a", "offers": [{"extra": 1}]}, m)
    assert (r.reason, r.field) == ("unknown_field", "offers.extra")


# ---- end to end -------------------------------------------------------------------------------------------------------

@pytest.fixture
def geo_store(tmp_path):
    from dataplane.kv import Store
    s = Store(tmp_path / "geo.sqlite", geo_mapping())
    yield s
    s.close()


def test_ingest_store_and_build_equal_build_index(tmp_path, geo_store):
    from engine.api import build_index
    cat = geo_catalogue()
    cat.to_parquet(tmp_path / "c.parquet", index=False)
    rep = ingest(tmp_path / "c.parquet", geo_store.mapping, geo_store, tmp_path / "rej.jsonl")
    assert (rep.accepted, rep.rejected) == (200, 0)
    doc = geo_store.get(["P00003"])[0]
    assert doc["location"] == cat.location[3] and doc["offers"] == cat.offers[3]
    snap = geo_store.snapshot()
    assert snap.location[3] == cat.location[3] and snap.offers[3] == cat.offers[3]
    seg = build(geo_store, None, tmp_path / "segs")
    man = json.loads((seg / "manifest.json").read_text())
    assert man["geo_fields"] == ["location"] and man["nested_paths"] == ["offers"]
    from engine.api import Index
    equals(Index.from_segment(seg), build_index(cat, geo_mapping()), tie_keys=False)


def stream(tmp_path, geo_store):
    """A base, then 9 refreshes of new products (points in every form), offer-array edits, deletes and price patches
    on parents with children, with merges every third refresh; equal to a rebuild throughout."""
    rng = np.random.default_rng(11)
    cat = geo_catalogue()
    rep = apply_changes(({"op": "upsert", "_id": r["product_id"], "doc": r} for r in cat.to_dict("records")),
                        geo_store.mapping, geo_store)
    assert not rep.rejected
    out = tmp_path / "segs"
    r = Replica(geo_store, build(geo_store, None, out), budget=1 << 40, commit_dir=out,
                merge_policy=TieredMergePolicy(floor=5, factor=10, merge_factor=4))
    forms = [lambda p: p, lambda p: f"{p['lat']},{p['lon']}", lambda p: [p["lon"], p["lat"]],
             lambda p: f"POINT ({p['lon']} {p['lat']})"]
    for k in range(9):
        ch = []
        for i in range(6):                                            # new products, each point in another form
            pt = {"lat": round(float(rng.uniform(38, 43)), 4), "lon": round(float(rng.uniform(-76, -71)), 4)}
            ch.append({"op": "upsert", "_id": f"G{k}-{i}", "doc": {
                "product_title": f"acme lamp {k} {i}", "price": float(rng.uniform(1, 300)), "stock": 3,
                "location": forms[(k + i) % 4](pt), "offers": offers(rng, k=int(rng.integers(0, 4)))}})
        old = dict(cat.iloc[k].to_dict())                              # an edit replacing the whole offers array
        ch.append({"op": "upsert", "_id": old["product_id"], "doc": {**old, "offers": offers(rng, k=2),
                                                                     "location": "POINT (-73.9 40.8)"}})
        ch.append({"op": "delete", "_id": f"P{100 + k:05d}"})
        ch.append({"op": "patch", "_id": f"P{150 + k:05d}", "fields": {"price": 1.0 + k}})   # a parent with children
        rep = apply_changes(ch, geo_store.mapping, geo_store)
        assert not rep.rejected, rep.rejected
        r.refresh()
        if k % 3 == 2:
            while r.merge() is not None:
                pass
    assert r.merges
    equals(r.index, rebuild(geo_store, tmp_path, "rebuild1"))
    # the patched parents keep their children: a nested filter still finds them, with the new price
    kids = geo_store.get(["P00152"])[0]["offers"]
    if kids:
        q = {"query": {"bool": {"filter": [{"ids": {"values": ["P00152"]}},
                                           {"nested": {"path": "offers", "query": {"match_all": {}}}}]}},
             "sort": [{"price": {"order": "asc"}}]}
        hits = r.search(q)["hits"]["hits"]
        assert [h["_id"] for h in hits] == ["P00152"] and hits[0]["sort"][0] == 3.0
    r.refresh()
    equals(r.index, rebuild(geo_store, tmp_path, "rebuild2"))
    return r, out


def test_deltas_merges_and_a_parent_patch_equal_a_rebuild(tmp_path, geo_store):
    stream(tmp_path, geo_store)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="engine: Index.from_commit revives the nested children of a parent copy tombstoned by "
                          "add_segment (an edit), so the edited parent's children are 'live in two segments'")
def test_commit_reload_with_edited_nested_parents(tmp_path, geo_store):
    r, out = stream(tmp_path, geo_store)
    c = r.commit()
    apply_changes([{"op": "patch", "_id": "P00160", "fields": {"price": 99.0}},
                   {"op": "upsert", "_id": "LAST", "doc": {"product_title": "lamp", "location": [-74, 40.7],
                                                           "offers": [{"seller": "seller-03", "price": 5, "note": "fast"}]}}],
                  geo_store.mapping, geo_store)
    r2 = Replica(geo_store, c, budget=1 << 40)
    r.refresh()
    fresh = rebuild(geo_store, tmp_path, "rebuild3")
    equals(r2.index, fresh)
    equals(r.index, fresh)


def test_bulk_update_replaces_the_whole_nested_array(tmp_path):
    from serving.config import ServingConfig
    from serving.manager import ManagedIndex
    from serving.metrics import standard
    from serving.ports import default_ports
    cat = geo_catalogue(60)
    cat.to_parquet(tmp_path / "c.parquet", index=False)
    cfg = ServingConfig(data_dir=str(tmp_path / "srv"), refresh_interval="-1")
    mi = ManagedIndex("geo", tmp_path / "srv" / "geo", geo_mapping(), cfg, default_ports(), standard())
    mi.create({"path": str(tmp_path / "c.parquet")})
    new = [{"seller": "seller-07", "price": 9.0, "stock": 1, "note": "fast"}]
    lines = [json.dumps({"update": {"_id": "P00004"}}), json.dumps({"doc": {"offers": new}}),
             json.dumps({"update": {"_id": "P00005"}}), json.dumps({"doc": {"location": "POINT (-73.5 40.6)"}}),
             json.dumps({"index": {"_id": "P00006"}}), json.dumps({**cat.iloc[6].to_dict(), "location": "geohash"})]
    from serving import bulk as bulk_mod
    status, body = mi.bulk(bulk_mod.parse_actions(iter(l.encode() for l in lines)), "true")
    items = body["items"]
    assert [list(i.values())[0].get("status") for i in items][:2] == [200, 200], items
    err = list(items[2].values())[0]
    assert err["status"] == 400 and "bad_geo" in json.dumps(err), err
    store = mi._get_store()
    assert store.get(["P00004"])[0]["offers"] == new                      # replaced, not appended
    assert store.get(["P00005"])[0]["location"] == {"lat": 40.6, "lon": -73.5}
    q = {"query": {"nested": {"path": "offers", "query": {"term": {"offers.seller": "seller-07"}}}}, "size": 100}
    got = {h["_id"] for h in mi.index.search(q)["hits"]["hits"]}
    assert "P00004" in got
    equals(mi.index, rebuild(store, tmp_path, "rebuild"))
