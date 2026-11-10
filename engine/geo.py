"""Geo points (S9-4), as OpenSearch and Lucene store and compare them.

- A ``geo_point`` is stored as two int32 columns, Lucene's encoding (``GeoEncodingUtils``): latitude
  ``floor(lat / (180 / 2^32))``, longitude ``floor(lon / (360 / 2^32))`` (90 and 180 nudged down one ulp); every
  comparison reads the decoded values ``encoded * (180 / 2^32)`` (and 360 for longitude), as OpenSearch does.
- Distances: haversine on a sphere of OpenSearch's mean earth radius, 6,371,008.7714 m, in float64.
- Units: OpenSearch's (``mi``, ``yd``, ``ft``, ``in``, ``km``, ``m``, ``cm``, ``mm``, ``nmi``/``NM``); a bare number is
  metres.
- Points: ``{"lat", "lon"}``, ``[lon, lat]`` or ``"lat,lon"`` (a geohash is refused).
"""
from __future__ import annotations

import math
import re

import numpy as np

EARTH_RADIUS_M = 6371008.7714
LAT_DECODE = 180.0 / (1 << 32)
LON_DECODE = 360.0 / (1 << 32)
UNITS = {"mi": 1609.344, "miles": 1609.344, "yd": 0.9144, "yards": 0.9144, "ft": 0.3048, "feet": 0.3048,
         "in": 0.0254, "inch": 0.0254, "km": 1000.0, "kilometers": 1000.0, "m": 1.0, "meters": 1.0,
         "cm": 0.01, "centimeters": 0.01, "mm": 0.001, "millimeters": 0.001, "nmi": 1852.0, "NM": 1852.0,
         "nauticalmiles": 1852.0}
INT_MAX, INT_MIN = 2 ** 31 - 1, -(2 ** 31)


class BadGeo(ValueError):
    pass


def parse_point(v):
    """(lat, lon) from OpenSearch's point forms; raises BadGeo."""
    if isinstance(v, dict):
        if set(v) != {"lat", "lon"}:
            raise BadGeo("a point is {lat, lon}")
        lat, lon = v["lat"], v["lon"]
    elif isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise BadGeo("a point is [lon, lat]")
        lon, lat = v
    elif isinstance(v, str):
        parts = v.split(",")
        if len(parts) != 2:
            raise BadGeo("a point string is 'lat,lon' (geohashes are not supported)")
        lat, lon = parts
    else:
        raise BadGeo("a point is {lat, lon}, [lon, lat] or 'lat,lon'")
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        raise BadGeo("latitude and longitude must be numbers") from None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0) or math.isnan(lat) or math.isnan(lon):
        raise BadGeo(f"point ({lat}, {lon}) is out of range")
    return lat, lon


def parse_distance(v, default_unit: str = "m") -> float:
    """Metres from a number (in ``default_unit``) or a string like ``"12km"``."""
    if isinstance(v, bool):
        raise BadGeo("a distance is a number or a string like '12km'")
    if isinstance(v, (int, float)):
        return float(v) * UNITS[default_unit]
    if isinstance(v, str):
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)\s*", v)
        if m:
            unit = m.group(2) or default_unit
            if unit in UNITS:
                return float(m.group(1)) * UNITS[unit]
    raise BadGeo(f"distance {v!r}: a number and a unit ({', '.join(sorted(set(UNITS)))})")


def encode_lat(lat: float) -> int:
    if lat == 90.0:
        lat = math.nextafter(90.0, 0.0)
    return int(math.floor(lat / LAT_DECODE))


def encode_lon(lon: float) -> int:
    if lon == 180.0:
        lon = math.nextafter(180.0, 0.0)
    return int(math.floor(lon / LON_DECODE))


def encode_lat_ceil(lat: float) -> int:
    return min(INT_MAX, int(math.ceil(lat / LAT_DECODE)))


def encode_lon_ceil(lon: float) -> int:
    return min(INT_MAX, int(math.ceil(lon / LON_DECODE)))


def haversine_m(b, lat_e, lon_e, lat: float, lon: float):
    """Distances in metres from (lat, lon) to the docs' encoded points, on the backend, float64."""
    la = b.astype(lat_e, "float64") * LAT_DECODE
    lo = b.astype(lon_e, "float64") * LON_DECODE
    r = math.pi / 180.0
    dlat = (la - lat) * r
    dlon = (lo - lon) * r
    h = b.sin(dlat * 0.5) ** 2 + math.cos(lat * r) * b.cos(la * r) * b.sin(dlon * 0.5) ** 2
    h = b.where(h > 1.0, 1.0, h)
    return 2.0 * EARTH_RADIUS_M * b.arcsin(h ** 0.5)


def encode_column(values):
    """(lat_e int32[N], lon_e int32[N], missing bool[N]) from a column of points (None/NaN: missing)."""
    n = len(values)
    lat_e, lon_e, miss = np.zeros(n, np.int32), np.zeros(n, np.int32), np.zeros(n, bool)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and v != v) or type(v).__name__ == "NAType":
            miss[i] = True
            continue
        lat, lon = parse_point(v)
        lat_e[i], lon_e[i] = encode_lat(lat), encode_lon(lon)
    return lat_e, lon_e, miss


def lat_col(field: str) -> str:
    return f"{field}#lat"


def lon_col(field: str) -> str:
    return f"{field}#lon"
