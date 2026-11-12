"""Match every panorama patch against the WHOLE IGN area with DINOv2 (zero-shot).

1. The wide mosaic (data/map/ign_z19_wide.jpg) is downscaled by --map-scale and
   cut into DINOv2 windows; the patch tokens of all windows form one feature grid
   covering the full area (one token per 14x14 map pixel).
2. For each frame, the dual-fisheye panorama goes through DINOv2; every panorama
   patch is compared (cosine similarity) to every map patch, and its top-K map
   patches are stored, plus a flag saying whether the best one is mutual (that map
   patch prefers this panorama patch over all others in the frame).

No ground truth is used for matching. GT positions are written only so the web
page can optionally show where the drone really was.

Outputs, read by viewer/dino.html:
    data/dino/meta.json    grids, frame list, record layout
    data/dino/matches.bin  per frame: int32 idx[Np*K], uint8 sim[Np*K], uint8 mutual[Np]
    data/dino/map.jpg      the downscaled map the features were computed on
    data/dino/pano/NNNNNN.jpg

    python3 dino/precompute_global_matches.py --stride 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dino"))

from render_dino_matches import ARCH, PATCH, load_dino, patch_features  # noqa: E402
from svmi import fisheye, poses, tiles  # noqa: E402

OUT = ROOT / "data/dino"
PANO_W, PANO_H = 640, 320


def map_feature_grid(model, rgb, device, win):
    """Patch tokens for a large image, computed window by window. Returns (rows, cols, C) float16."""
    h, w = rgb.shape[:2]
    rows, cols = h // PATCH, w // PATCH
    grid = None
    for y0 in range(0, h, win):
        for x0 in range(0, w, win):
            crop = rgb[y0:y0 + win, x0:x0 + win]
            f = patch_features(model, crop, crop.shape[:2], device).half().cpu()
            if grid is None:
                grid = torch.empty(rows, cols, f.shape[-1], dtype=torch.float16)
            grid[y0 // PATCH:y0 // PATCH + f.shape[0], x0 // PATCH:x0 // PATCH + f.shape[1]] = f
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=list(ARCH), default="vits14")
    ap.add_argument("--map-scale", type=float, default=0.5,
                    help="downscale of the 0.192 m/px mosaic before DINOv2 (0.5 -> 5.4 m per map patch)")
    ap.add_argument("--window", type=int, default=504, help="DINOv2 window on the map, multiple of 14")
    ap.add_argument("--pano-hw", type=int, nargs=2, default=[322, 644])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--level", choices=["none", "imu"], default="none")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    assert args.window % PATCH == 0 and all(v % PATCH == 0 for v in args.pano_hw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(args.arch, device)
    gt = poses.load_poses()
    idx = list(range(args.start, args.end or len(gt), args.stride))
    (OUT / "pano").mkdir(parents=True, exist_ok=True)

    # --- whole-area map features -------------------------------------------------
    mosaic = tiles.Mosaic()
    small = cv2.resize(mosaic.image, None, fx=args.map_scale, fy=args.map_scale, interpolation=cv2.INTER_AREA)
    small = small[: small.shape[0] // PATCH * PATCH, : small.shape[1] // PATCH * PATCH]
    cv2.imwrite(str(OUT / "map.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    t0 = time.time()
    grid = map_feature_grid(model, small, device, args.window)
    map_rows, map_cols, dim = grid.shape
    S = grid.reshape(-1, dim).float().to(device)              # (Ns, C), already L2-normalised
    print(f"map: {small.shape[1]}x{small.shape[0]} px, {map_rows}x{map_cols} patches "
          f"({len(S)} total) in {time.time() - t0:.0f}s", flush=True)

    m_per_px = mosaic.meta["m_per_px"] / args.map_scale
    pano_rows, pano_cols = args.pano_hw[0] // PATCH, args.pano_hw[1] // PATCH
    n_p, k = pano_rows * pano_cols, args.top_k
    record = n_p * k * 4 + n_p * k + n_p
    record += (-record) % 4                                     # keep Int32Array offsets aligned

    meta = {
        "arch": args.arch, "patch": PATCH, "top_k": k, "record_bytes": record,
        "layout": "per frame: int32 idx[Np*K] (row-major map patch id), uint8 sim[Np*K] "
                  "(cosine*255, clipped to [0,1]), uint8 mutual[Np], zero padding to 4 bytes",
        "pano": {"image_dir": "pano", "w": PANO_W, "h": PANO_H, "rows": pano_rows, "cols": pano_cols},
        "map": {"image": "map.jpg", "w": int(small.shape[1]), "h": int(small.shape[0]),
                "rows": map_rows, "cols": map_cols, "m_per_px": m_per_px, "scale": args.map_scale},
        "frames": [],
    }

    renderer = fisheye.PanoRenderer(PANO_H, PANO_W, supersample=2)
    ids = torch.arange(n_p, device=device)
    t0 = time.time()
    with open(OUT / "matches.bin", "wb") as fbin:
        for n, i in enumerate(idx):
            img = cv2.imread(str(poses.IMAGES_DIR / gt[i]["image"]))
            R = fisheye.leveling_rotation(np.array(gt[i]["R_cam_from_dne"])) if args.level == "imu" else None
            pano = renderer.render(img, R)
            cv2.imwrite(str(OUT / "pano" / f"{i:06d}.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 85])

            P = patch_features(model, cv2.cvtColor(pano, cv2.COLOR_BGR2RGB), args.pano_hw, device).reshape(-1, dim)
            with torch.no_grad():
                sim = P @ S.T                                   # (Np, Ns)
                val, ind = sim.topk(k, dim=1)
                mutual = sim.argmax(0)[ind[:, 0]] == ids
            buf = bytearray(record)
            a = ind.to(torch.int32).cpu().numpy().tobytes()
            b = (val.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()
            c = mutual.to(torch.uint8).cpu().numpy().tobytes()
            buf[:len(a)] = a
            buf[len(a):len(a) + len(b)] = b
            buf[len(a) + len(b):len(a) + len(b) + len(c)] = c
            fbin.write(buf)

            mx, my = mosaic.local_to_px(gt[i]["north"], gt[i]["east"])
            meta["frames"].append({"i": i, "alt": round(gt[i]["alt"], 1),
                                   "gt_x": round(mx * args.map_scale, 1), "gt_y": round(my * args.map_scale, 1),
                                   "heading": round(gt[i]["heading_deg"], 1)})
            if n % 20 == 0 or n == len(idx) - 1:
                fbin.flush()
                (OUT / "meta.json").write_text(json.dumps(meta))   # partial runs stay viewable
                print(f"{n + 1}/{len(idx)}  {(n + 1) / (time.time() - t0):.2f} fr/s", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
