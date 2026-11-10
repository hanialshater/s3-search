"""S9-4: geo_point, geo_distance, geo_bounding_box and _geo_distance sort against brute force (Lucene's encoding,
haversine on OpenSearch's earth radius) on toy, on every device and across segments."""
import copy
import math

import numpy as np
import pytest

from engine.api import Index, InvalidMapping, InvalidRequest, UnsupportedQuery, build_index
from engine.build import build_segment
from engine.geo import EARTH_RADIUS_M, LAT_DECODE, LON_DECODE, encode_lat, encode_lon
from engine.tests.conftest import toy_catalogue, toy_mapping


def with_geo(cat, seed=0):
    rng = np.random.default_rng(seed)
    cat = cat.copy()
    forms = []
    for i in range(len(cat)):
        if i % 11 == 0:
            forms.append(None)
            continue
        lat, lon = float(rng.uniform(-60, 70)), float(rng.uniform(-179.9, 179.9))
        if i < 6:                                            # near the dateline and a pole
            lat, lon = [(10.0, 179.95), (10.0, -179.95), (89.99, 3.0), (0.0, 0.0), (-45.5, 120.25), (52.37, 4.9)][i]
        forms.append([{"lat": lat, "lon": lon}, [lon, lat], f"{lat},{lon}"][i % 3])
    cat["location"] = forms
    return cat


def geo_mapping():
    m = toy_mapping()
    m["properties"]["location"] = {"type": "geo_point"}
    return m


@pytest.fixture(scope="module")
def gcat():
    return with_geo(toy_catalogue())


@pytest.fixture(scope="module")
def gindex(gcat, device):
    return build_index(gcat, geo_mapping(), device)


def point(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return v["lat"], v["lon"]
    if isinstance(v, list):
        return v[1], v[0]
    a, b = v.split(",")
    return float(a), float(b)


def decoded(v):
    lat, lon = point(v)
    return encode_lat(lat) * LAT_DECODE, encode_lon(lon) * LON_DECODE


def hav(lat1, lon1, lat2, lon2):
    r = math.pi / 180
    h = math.sin((lat2 - lat1) * r / 2) ** 2 + math.cos(lat1 * r) * math.cos(lat2 * r) * math.sin((lon2 - lon1) * r / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, h)))


def ids_of(r):
    return [h["_id"] for h in r["hits"]["hits"]]


@pytest.mark.parametrize("center,dist", [((52.0, 5.0), "1500km"), ((10.0, 180.0), "100km"), ((0, 0), 2_000_000),
                                         ((89.0, 0.0), "500mi"), ((-40, 120), "3000NM")])
def test_geo_distance(gindex, gcat, center, dist):
    from engine.geo import parse_distance
    meters = parse_distance(dist)
    want = [pid for pid, v in zip(gcat["product_id"], gcat["location"])
            if v is not None and hav(center[0], center[1], *decoded(v)) <= meters]
    r = gindex.search({"query": {"bool": {"filter": [{"geo_distance": {"distance": dist, "location": {
        "lat": center[0], "lon": center[1]}}}]}}, "size": 500, "sort": ["_doc"]})
    assert ids_of(r) == want


@pytest.mark.parametrize("box", [(60, -10, 40, 30), (20, 170, 0, -170), (90, -180, -90, 180), (5, 4, 4, 5)])
def test_geo_bounding_box(gindex, gcat, box):
    from engine.geo import encode_lat_ceil, encode_lon_ceil
    top, left, bottom, right = box
    want = []
    for pid, v in zip(gcat["product_id"], gcat["location"]):
        if v is None:
            continue
        la, lo = encode_lat(point(v)[0]), encode_lon(point(v)[1])
        ok_lat = encode_lat_ceil(bottom) <= la <= encode_lat(top)
        if left <= right:
            ok_lon = encode_lon_ceil(left) <= lo <= encode_lon(right)
        else:
            ok_lon = lo >= encode_lon_ceil(left) or lo <= encode_lon(right)
        if ok_lat and ok_lon:
            want.append(pid)
    q = {"geo_bounding_box": {"location": {"top_left": {"lat": top, "lon": left},
                                            "bottom_right": {"lat": bottom, "lon": right}}}}
    r = gindex.search({"query": {"bool": {"filter": [q]}}, "size": 500, "sort": ["_doc"]})
    assert ids_of(r) == want
    q2 = {"geo_bounding_box": {"location": {"top": top, "left": left, "bottom": bottom, "right": right}}}
    assert ids_of(gindex.search({"query": q2, "size": 500, "sort": ["_doc"]})) == want


@pytest.mark.parametrize("order,unit", [("asc", "km"), ("desc", "m"), ("asc", "mi")])
def test_geo_distance_sort(gindex, gcat, order, unit):
    from engine.geo import UNITS
    c = (48.85, 2.35)
    rows = []
    for d, (pid, v) in enumerate(zip(gcat["product_id"], gcat["location"])):
        dist = math.inf if v is None else hav(c[0], c[1], *decoded(v)) / UNITS[unit]
        rows.append((dist, d, pid))
    rows.sort(key=lambda x: (-x[0] if order == "desc" else x[0], x[1]))
    r = gindex.search({"query": {"match_all": {}}, "size": 500,
                       "sort": [{"_geo_distance": {"location": [c[1], c[0]], "order": order, "unit": unit}}]})
    assert ids_of(r) == [p for _, _, p in rows]
    for h, (dist, _, _) in zip(r["hits"]["hits"], rows):
        v = h["sort"][0]
        assert (v is None and math.isinf(dist)) or math.isclose(v, dist, rel_tol=1e-9)
    page = gindex.search({"query": {"match_all": {}}, "size": 7, "sort": [{"_geo_distance": {
        "location": [c[1], c[0]], "order": order, "unit": unit}}], "search_after": r["hits"]["hits"][9]["sort"]})
    assert ids_of(page) == [p for _, _, p in rows[10:17]]


def test_exists_and_refusals(gindex, gcat):
    r = gindex.search({"query": {"exists": {"field": "location"}}, "size": 0})
    assert r["hits"]["total"] == int(gcat["location"].notna().sum())
    for bad, err in [({"geo_distance": {"distance": "10km", "location": "u33dc"}}, InvalidRequest),
                     ({"geo_distance": {"distance": "10 parsecs", "location": [0, 0]}}, InvalidRequest),
                     ({"geo_distance": {"distance": "10km", "location": [0, 0], "distance_type": "plane"}},
                      UnsupportedQuery),
                     ({"geo_bounding_box": {"location": {"wkt": "BBOX (0, 1, 1, 0)"}}}, UnsupportedQuery),
                     ({"geo_bounding_box": {"location": {"top_left": [0, 0], "bottom_right": [1, 5]}}}, InvalidRequest),
                     ({"geo_distance": {"distance": "1km", "price": [0, 0]}}, UnsupportedQuery)]:
        with pytest.raises(err):
            gindex.search({"query": bad})
    with pytest.raises(InvalidMapping):
        build_index(gcat.assign(location=["999,999"] * len(gcat)), geo_mapping())


def test_geo_across_segments_equals_a_rebuild(gcat, device):
    m = geo_mapping()
    m["_meta"]["engine"].pop("embeddings")
    cat = gcat.drop(columns=[]).assign(_first_offset=range(len(gcat)))
    base, rest = cat.iloc[:80], cat.iloc[80:]
    edit = cat.iloc[[3, 7, 20]].copy()
    edit["location"] = [{"lat": 1.0, "lon": 2.0}, None, "5.5,6.5"]
    delta = __import__("pandas").concat([edit, rest]).sort_values("_first_offset")
    ix = Index(build_segment(base, m), device)
    ix.add_segment(build_segment(delta, m, version=2), version=2, delete_ids=list(delta["product_id"]))
    final = cat.copy().set_index("product_id")
    final.loc[edit["product_id"], "location"] = edit.set_index("product_id")["location"]
    rebuilt = Index(build_segment(final.reset_index(), m), device)
    for q in [{"query": {"geo_distance": {"distance": "3000km", "location": [2.0, 1.0]}}, "size": 200},
              {"query": {"match_all": {}}, "size": 200, "sort": [{"_geo_distance": {"location": "1,2"}}]},
              {"query": {"geo_bounding_box": {"location": {"top": 30, "left": -20, "bottom": -10, "right": 40}}},
               "size": 200}]:
        a, b = ix.search(q)["hits"], rebuilt.search(q)["hits"]
        assert [h["_id"] for h in a["hits"]] == [h["_id"] for h in b["hits"]] and a["total"] == b["total"]
