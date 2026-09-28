"""Geodesy helpers shared by the SVMI+ pipelines.

camera_poses.txt stores positions as (down, north, east) metres relative to the
first frame. The north/east columns are exact multiples of 1e-4 deg latitude and
1e-5 deg longitude, so we invert with the same metres-per-degree constants the
dataset used, which recovers the logged GNSS fix exactly.
"""
import math

# Takeoff point supplied by the dataset owner; used as the local origin.
ORIGIN_LAT = 49.967068
ORIGIN_LON = 2.266771

M_PER_DEG_LAT = 111229.5357  # 11.12295357 m per 1e-4 deg in the log
M_PER_DEG_LON = 71745.380    # 0.71745380 m per 1e-5 deg in the log

TILE_SIZE = 256
EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137.0


def local_to_latlon(north, east, lat0=ORIGIN_LAT, lon0=ORIGIN_LON):
    return lat0 + north / M_PER_DEG_LAT, lon0 + east / M_PER_DEG_LON


def latlon_to_local(lat, lon, lat0=ORIGIN_LAT, lon0=ORIGIN_LON):
    return (lat - lat0) * M_PER_DEG_LAT, (lon - lon0) * M_PER_DEG_LON


def latlon_to_world_px(lat, lon, zoom):
    """Web Mercator global pixel coordinates (x right, y down) at a zoom level."""
    n = TILE_SIZE * 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def world_px_to_latlon(x, y, zoom):
    n = TILE_SIZE * 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def ground_resolution(lat, zoom):
    """Metres per Web Mercator pixel at a latitude."""
    return EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE_SIZE * 2 ** zoom)
