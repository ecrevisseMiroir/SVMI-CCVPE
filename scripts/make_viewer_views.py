"""Render the model-input views for the web viewer.

- data/viewer_pano/NNNNNN.jpg  : 640x320 equirectangular panorama (CCVPE input)
- data/viewer_nadir/NNNNNN.jpg : 512x384 nadir pinhole view, 60 deg HFOV (Game4Loc input)

Both use the same renderers and settings as ccvpe/run_ccvpe.py and
game4loc/run_game4loc.py (level-flight assumption, no ground truth).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmi import fisheye, nadir, poses  # noqa: E402

OUT_PANO = ROOT / "data/viewer_pano"
OUT_NADIR = ROOT / "data/viewer_nadir"
_renderers = None


def _init():
    global _renderers
    _renderers = (fisheye.PanoRenderer(320, 640, supersample=2), nadir.NadirRenderer(512, 384, 60.0))


def render(i):
    pano_r, nadir_r = _renderers
    img = cv2.imread(str(poses.IMAGES_DIR / f"{i:06d}.png"))
    cv2.imwrite(str(OUT_PANO / f"{i:06d}.jpg"), pano_r.render(img), [cv2.IMWRITE_JPEG_QUALITY, 82])
    cv2.imwrite(str(OUT_NADIR / f"{i:06d}.jpg"), nadir_r.render(img), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_PANO.mkdir(parents=True, exist_ok=True)
    OUT_NADIR.mkdir(parents=True, exist_ok=True)
    n = len(poses.load_poses())
    todo = [i for i in range(n) if args.force
            or not (OUT_PANO / f"{i:06d}.jpg").exists() or not (OUT_NADIR / f"{i:06d}.jpg").exists()]
    print(f"{n} frames, {len(todo)} to render")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for k, _ in enumerate(ex.map(render, todo, chunksize=16), 1):
            if k % 500 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
