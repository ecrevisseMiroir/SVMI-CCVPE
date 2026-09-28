"""Prepare lightweight assets for the web viewer in viewer/.

- data/viewer_frames/NNNNNN.jpg : every SVMI+ frame downscaled to JPEG
- data/map/ign_z19_viewer.jpg   : downscaled orthophoto for smooth canvas panning
- data/map/ign_z19_viewer.json  : its scale relative to data/map/ign_z19.jpg

Viewer-image pixel = scale * (mosaic pixel of ign_z19.jpg), so all geo maths in
data/map/ign_z19.json still applies after dividing by `scale`.

Usage:
    .venv/bin/python scripts/make_viewer_frames.py [--size 640x360] [--quality 80]
"""
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "SVMI/SVMISplus/SVMISplus/Matrice600Pro/SixDOFs/Images"
OUT = ROOT / "data/viewer_frames"
MAP_DIR = ROOT / "data/map"

Image.MAX_IMAGE_PIXELS = None


def convert(job):
    src, dst, size, quality = job
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.LANCZOS)
        tmp = dst.with_suffix(".tmp")
        im.save(tmp, "JPEG", quality=quality, optimize=True, progressive=False)
        os.replace(tmp, dst)
    return dst.name


def make_frames(size, quality, workers, force):
    OUT.mkdir(parents=True, exist_ok=True)
    # only NNNNNN.png (skips macOS "._*.png" resource-fork files)
    srcs = sorted(p for p in SRC.glob("*.png") if re.fullmatch(r"\d{6}\.png", p.name))
    if not srcs:
        raise SystemExit(f"no PNG frames found in {SRC}")
    jobs = []
    for src in srcs:
        dst = OUT / (src.stem + ".jpg")
        if force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            jobs.append((src, dst, size, quality))
    print(f"{len(srcs)} frames, {len(jobs)} to convert -> {OUT.relative_to(ROOT)}")
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, _ in enumerate(ex.map(convert, jobs, chunksize=16), 1):
                if k % 200 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)}", flush=True)
    total = sum(p.stat().st_size for p in OUT.glob("[0-9]*.jpg"))
    print(f"frames: {total / 1e6:.1f} MB total")


def make_map(scale, quality):
    meta = json.loads((MAP_DIR / "ign_z19.json").read_text())
    src = MAP_DIR / meta.get("image", "ign_z19.jpg")
    with Image.open(src) as im:
        full = im.size
        size = (round(full[0] * scale), round(full[1] * scale))
        small = im.convert("RGB").resize(size, Image.LANCZOS)
    dst = MAP_DIR / "ign_z19_viewer.jpg"
    small.save(dst, "JPEG", quality=quality, optimize=True)
    side = {
        "image": dst.name,
        "source_image": src.name,
        "scale": size[0] / full[0],  # viewer px per source mosaic px
        "size_px": list(size),
        "source_size_px": list(full),
    }
    (MAP_DIR / "ign_z19_viewer.json").write_text(json.dumps(side, indent=2) + "\n")
    print(f"map: {dst.relative_to(ROOT)} {size[0]}x{size[1]} "
          f"({dst.stat().st_size / 1e6:.1f} MB), scale {side['scale']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", default="640x360", help="frame size WxH")
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--map-scale", type=float, default=0.5)
    ap.add_argument("--map-quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--force", action="store_true", help="re-encode existing frames")
    ap.add_argument("--skip-frames", action="store_true")
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))
    if not args.skip_frames:
        make_frames((w, h), args.quality, args.workers, args.force)
    if not args.skip_map:
        make_map(args.map_scale, args.map_quality)


if __name__ == "__main__":
    main()
