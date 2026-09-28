"""Zero-shot CCVPE (Xia et al., T-PAMI 2023) on SVMI+.

For every frame:
  ground  = 360 deg panorama rendered from the dual-fisheye image (320x640,
            centre column = left-fisheye optical axis, no GT used by default),
  aerial  = 640x640 north-up IGN tile, resized to 512 as in CCVPE's VIGOR
            pipeline, centred on the GT position plus a random offset of up to
            +-1/4 tile (VIGOR "positive" protocol: the query lies in the central
            half of the tile; the model is told nothing else).
Outputs the arg-max location of the heatmap and the orientation at that
location, with unknown orientation (ori_noise = 180).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/CCVPE"))

from svmi import fisheye, poses, results, tiles  # noqa: E402

WEIGHTS = ROOT / "third_party/ccvpe_weights/VIGOR"
TILE_RAW = 640  # VIGOR aerial images are 640x640 before CCVPE resizes them to 512
TILE_IN = 512


def unit_offsets(n, seed=0):
    """Deterministic per-frame offsets in [-1, 1]^2, shared by every run."""
    return np.random.default_rng(seed).uniform(-1.0, 1.0, size=(n, 2))


def load_model(area, device):
    from models import CVM_VIGOR_ori_prior
    model = CVM_VIGOR_ori_prior(device, ori_noise=180, circular_padding=True)
    model.load_state_dict(torch.load(WEIGHTS / area / "model.pt", map_location="cpu"))
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=["samearea", "crossarea"], default="samearea")
    ap.add_argument("--m-per-px", type=float, default=0.111,
                    help="ground resolution of the 640 px tile (VIGOR ~0.10-0.12)")
    ap.add_argument("--level", choices=["none", "imu"], default="none",
                    help="imu: remove roll/pitch with the logged attitude before rendering")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-examples", type=int, default=6, help="dump N debug composites")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = poses.load_poses()
    idx = list(range(0, len(gt), args.stride))
    offsets = unit_offsets(len(gt)) * (TILE_RAW * args.m_per_px / 4)
    mosaic = tiles.Mosaic()
    renderer = fisheye.PanoRenderer(320, 640, supersample=2)
    model = load_model(args.area, device)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_grd = transforms.Compose([transforms.ToTensor(), norm])
    to_sat = transforms.Compose([transforms.Resize([TILE_IN, TILE_IN]), transforms.ToTensor(), norm])

    run_id = f"ccvpe_vigor_{args.area}_{args.m_per_px:g}mpp_{args.level}"
    debug_dir = ROOT / "results" / run_id / "examples"
    debug_dir.mkdir(parents=True, exist_ok=True)
    example_ids = set(idx[:: max(1, len(idx) // max(1, args.save_examples))][: args.save_examples])

    frames, t0 = [], time.time()
    for b in range(0, len(idx), args.batch):
        batch = idx[b:b + args.batch]
        grds, sats, centres = [], [], []
        for i in batch:
            img = cv2.imread(str(poses.IMAGES_DIR / gt[i]["image"]))
            R_level = None
            if args.level == "imu":
                R_level = fisheye.leveling_rotation(np.array(gt[i]["R_cam_from_dne"]))
            pano = cv2.cvtColor(renderer.render(img, R_level), cv2.COLOR_BGR2RGB)
            cn = gt[i]["north"] + offsets[i, 0]
            ce = gt[i]["east"] + offsets[i, 1]
            sat = mosaic.tile(cn, ce, TILE_RAW, args.m_per_px)
            grds.append(to_grd(Image.fromarray(pano)))
            sats.append(to_sat(Image.fromarray(sat)))
            centres.append((cn, ce, pano, sat))
        with torch.no_grad():
            out = model(torch.stack(grds).to(device), torch.stack(sats).to(device))
        heatmap = out[1].cpu().numpy()[:, 0]
        ori = out[2].cpu().numpy()
        for k, i in enumerate(batch):
            r, c = np.unravel_index(heatmap[k].argmax(), heatmap[k].shape)
            cn, ce, pano, sat = centres[k]
            scale = args.m_per_px * TILE_RAW / TILE_IN
            pn, pe = tiles.tile_px_to_local(r, c, cn, ce, TILE_IN, scale)
            cos_t, sin_t = ori[k, :, r, c]
            theta = math.degrees(math.atan2(sin_t, cos_t))  # CCVPE: CCW from north, of the pano centre
            heading = (-theta) % 360.0
            f = results.make_frame(gt[i], pn, pe, heading,
                                   tile_centre={"north": cn, "east": ce},
                                   tile_centre_err_m=math.hypot(cn - gt[i]["north"], ce - gt[i]["east"]),
                                   peak_prob=float(heatmap[k, r, c]))
            frames.append(f)
            if i in example_ids:
                _save_example(debug_dir / f"{i:06d}.jpg", pano, sat, heatmap[k], (r, c), gt[i], cn, ce,
                              scale, heading)
        if (b // args.batch) % 50 == 0:
            done = b + len(batch)
            print(f"{done}/{len(idx)}  {done / (time.time() - t0):.1f} fr/s  "
                  f"median err so far {np.median([f['err_m'] for f in frames]):.1f} m", flush=True)

    params = {"weights": f"VIGOR/{args.area}", "tile_px": TILE_RAW, "m_per_px": args.m_per_px,
              "tile_extent_m": TILE_RAW * args.m_per_px, "max_offset_m": TILE_RAW * args.m_per_px / 4,
              "level": args.level, "ori_noise": 180, "stride": args.stride}
    colors = {("samearea", 0.111): "#e4572e", ("samearea", 0.25): "#ff8c42", ("samearea", 0.5): "#ffd23f",
              ("crossarea", 0.111): "#9b5de5", ("crossarea", 0.25): "#c77dff", ("crossarea", 0.5): "#e0aaff"}
    color = "#2ec4b6" if args.level == "imu" else colors.get((args.area, args.m_per_px), "#e4572e")
    name = f"CCVPE VIGOR-{args.area} ({args.m_per_px:g} m/px{', IMU-levelled' if args.level == 'imu' else ''})"
    summary = results.write_run(run_id, name, params, frames, gt, color)
    centre = np.array([f["tile_centre_err_m"] for f in frames])
    summary_path = ROOT / "results" / run_id / "summary.txt"
    lines = [name] + [f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()]
    lines.append(f"  baseline tile-centre median err m: {np.median(centre):.3f} (mean {centre.mean():.3f})")
    lines.append("  baseline random heading median err deg: 90")
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _save_example(path, pano, sat, heat, rc, g, cn, ce, scale, heading):
    sat = cv2.resize(sat, (TILE_IN, TILE_IN))
    h = cv2.applyColorMap((255 * heat / heat.max()).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(sat, cv2.COLOR_RGB2BGR), 0.6, h, 0.4, 0)
    gr = int(round((cn - g["north"]) / scale + TILE_IN / 2 - 0.5))
    gc = int(round((g["east"] - ce) / scale + TILE_IN / 2 - 0.5))
    r, c = rc
    for (y, x), col, hd in (((gr, gc), (0, 255, 0), g["heading_deg"]), ((r, c), (0, 0, 255), heading)):
        cv2.circle(over, (x, y), 8, col, 2)
        a = math.radians(hd)
        cv2.line(over, (x, y), (int(x + 30 * math.sin(a)), int(y - 30 * math.cos(a))), col, 2)
    pano_bgr = cv2.cvtColor(cv2.resize(pano, (TILE_IN * 2, TILE_IN)), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), np.hstack([pano_bgr, over]))


if __name__ == "__main__":
    main()
