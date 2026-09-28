"""Download an IGN orthophoto mosaic (Geoplateforme WMTS) around the flight.

Writes data/map/ign_z{zoom}.jpg plus a JSON sidecar describing its Web Mercator
footprint, so any pixel can be mapped to lat/lon and back.
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmi import geo, poses  # noqa: E402

WMTS = ("https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
        "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM"
        "&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}&FORMAT=image/jpeg")


def fetch(session, z, x, y, cache):
    path = cache / f"{z}_{x}_{y}.jpg"
    if path.exists():
        return path.read_bytes()
    for attempt in range(5):
        try:
            r = session.get(WMTS.format(z=z, x=x, y=y), timeout=30)
            if r.status_code == 200:
                path.write_bytes(r.content)
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"tile {z}/{x}/{y} failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=int, default=20)
    ap.add_argument("--margin", type=float, default=250.0, help="metres around the trajectory")
    ap.add_argument("--name", default=None, help="output basename (default ign_z{zoom})")
    args = ap.parse_args()

    frames = poses.load_poses()
    north = [f["north"] for f in frames]
    east = [f["east"] for f in frames]
    lat_max, lon_min = geo.local_to_latlon(max(north) + args.margin, min(east) - args.margin)
    lat_min, lon_max = geo.local_to_latlon(min(north) - args.margin, max(east) + args.margin)

    z = args.zoom
    x0, y0 = geo.latlon_to_world_px(lat_max, lon_min, z)
    x1, y1 = geo.latlon_to_world_px(lat_min, lon_max, z)
    tx0, ty0 = int(x0 // geo.TILE_SIZE), int(y0 // geo.TILE_SIZE)
    tx1, ty1 = int(x1 // geo.TILE_SIZE), int(y1 // geo.TILE_SIZE)
    tiles = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
    print(f"zoom {z}: {len(tiles)} tiles, {geo.ground_resolution(geo.ORIGIN_LAT, z):.3f} m/px")

    cache = ROOT / "data/map/tile_cache"
    cache.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "SVMI-CCVPE research script"
    with ThreadPoolExecutor(4) as pool:
        blobs = list(pool.map(lambda t: fetch(session, z, t[0], t[1], cache), tiles))

    w = (tx1 - tx0 + 1) * geo.TILE_SIZE
    h = (ty1 - ty0 + 1) * geo.TILE_SIZE
    mosaic = Image.new("RGB", (w, h))
    for (x, y), blob in zip(tiles, blobs):
        mosaic.paste(Image.open(io.BytesIO(blob)).convert("RGB"),
                     ((x - tx0) * geo.TILE_SIZE, (y - ty0) * geo.TILE_SIZE))

    out = ROOT / f"data/map/{args.name or f'ign_z{z}'}.jpg"
    mosaic.save(out, quality=92)
    nw = geo.world_px_to_latlon(tx0 * geo.TILE_SIZE, ty0 * geo.TILE_SIZE, z)
    se = geo.world_px_to_latlon((tx1 + 1) * geo.TILE_SIZE, (ty1 + 1) * geo.TILE_SIZE, z)
    meta = {
        "image": out.name,
        "zoom": z,
        "origin_world_px": [tx0 * geo.TILE_SIZE, ty0 * geo.TILE_SIZE],
        "size_px": [w, h],
        "north_west": {"lat": nw[0], "lon": nw[1]},
        "south_east": {"lat": se[0], "lon": se[1]},
        "m_per_px": geo.ground_resolution(geo.ORIGIN_LAT, z),
        "source": "IGN Geoplateforme ORTHOIMAGERY.ORTHOPHOTOS (Licence Ouverte 2.0)",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} {w}x{h}")


if __name__ == "__main__":
    main()
