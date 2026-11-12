"""DINOv2 patch matching between the SVMI+ panorama and a satellite patch.

Per frame:
  left   panorama rendered from the dual fisheye (same renderer as CCVPE),
  right  north-up IGN tile centred on the ground-truth position, nothing drawn on it,
  lines  mutual-nearest-neighbour matches between DINOv2 patch tokens
         (cosine similarity), coloured by the panorama azimuth they start from.

This is zero-shot: DINOv2 was never trained to relate a ground-level panorama
to an aerial view, so expect semantically plausible matches (field to field,
tree line to tree line), not geometrically exact ones.

Needs third_party/dinov2 (code) and third_party/dinov2_weights/<arch>_pretrain.pth.

    python3 dino/render_dino_matches.py --stride 5 --out results/dino_matches.mp4
    python3 dino/render_dino_matches.py --stride 5 --show        # live window
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/dinov2"))

from svmi import fisheye, poses, tiles  # noqa: E402

WEIGHTS = ROOT / "third_party/dinov2_weights"
ARCH = {"vits14": "vit_small", "vitb14": "vit_base", "vitl14": "vit_large"}
PATCH = 14
TILE_RAW = 640          # aerial tile in pixels before resizing for the model
PANO_W, PANO_H = 640, 320
GAP, HEADER = 140, 40
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_dino(arch, device):
    from dinov2.models import vision_transformer as vits
    model = getattr(vits, ARCH[arch])(patch_size=PATCH, img_size=518, init_values=1.0, block_chunks=0)
    state = torch.load(WEIGHTS / f"dinov2_{arch}_pretrain.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def patch_features(model, rgb, size_hw, device):
    """L2-normalised patch tokens, shape (rows, cols, C), for an RGB uint8 image."""
    h, w = size_hw
    x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
    tok = model.forward_features(x)["x_norm_patchtokens"][0]
    tok = F.normalize(tok, dim=-1)
    return tok.reshape(h // PATCH, w // PATCH, -1)


def mutual_matches(fp, fs, top, min_sim, skip_rows):
    """Mutual nearest neighbours pano -> sat. Returns [(pr, pc, sr, sc, sim)], best first."""
    pr_n, pc_n, c = fp.shape
    sr_n, sc_n, _ = fs.shape
    sim = fp.reshape(-1, c) @ fs.reshape(-1, c).T            # (Np, Ns) cosine similarity
    best_s = sim.argmax(1)                                    # each pano patch's best sat patch
    best_p = sim.argmax(0)                                    # each sat patch's best pano patch
    ids = torch.arange(sim.shape[0], device=sim.device)
    val = sim[ids, best_s]
    keep = (best_p[best_s] == ids) & (val >= min_sim) & (ids // pc_n >= skip_rows)
    ids, val = ids[keep], val[keep]
    order = val.argsort(descending=True)[:top]
    ids, val = ids[order].cpu().numpy(), val[order].cpu().numpy()
    sidx = best_s[torch.as_tensor(ids, device=sim.device)].cpu().numpy() if len(ids) else ids
    return [(int(i // pc_n), int(i % pc_n), int(s // sc_n), int(s % sc_n), float(v))
            for i, s, v in zip(ids, sidx, val)]


def compose(pano_bgr, sat_bgr, matches, grid_p, grid_s, caption):
    """Canvas: panorama left, clean satellite patch right, match lines across the gap."""
    side = TILE_RAW
    canvas = np.full((HEADER + side, PANO_W + GAP + side, 3), 24, np.uint8)
    py = HEADER + (side - PANO_H) // 2
    sx = PANO_W + GAP
    canvas[py:py + PANO_H, :PANO_W] = pano_bgr
    canvas[HEADER:HEADER + side, sx:sx + side] = sat_bgr

    overlay = canvas.copy()
    for pr, pc, sr, sc, _ in matches:
        p = (int((pc + 0.5) / grid_p[1] * PANO_W), int(py + (pr + 0.5) / grid_p[0] * PANO_H))
        s = (int(sx + (sc + 0.5) / grid_s[1] * side), int(HEADER + (sr + 0.5) / grid_s[0] * side))
        hue = int(179 * (pc + 0.5) / grid_p[1])
        col = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.line(overlay, p, s, col, 1, cv2.LINE_AA)
        cv2.circle(overlay, p, 3, col, -1, cv2.LINE_AA)
        cv2.circle(overlay, s, 3, col, -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0)
    # lines are semi-transparent; redraw nothing else on the images themselves

    cv2.putText(canvas, caption, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "panorama (centre = forward)", (8, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=list(ARCH), default="vits14")
    ap.add_argument("--m-per-px", type=float, default=0.111, help="ground resolution of the aerial tile")
    ap.add_argument("--sat-size", type=int, default=448, help="aerial input to DINOv2 (multiple of 14)")
    ap.add_argument("--pano-hw", type=int, nargs=2, default=[322, 644], help="panorama input (multiples of 14)")
    ap.add_argument("--top", type=int, default=60, help="max matches drawn")
    ap.add_argument("--min-sim", type=float, default=0.0, help="drop matches below this cosine similarity")
    ap.add_argument("--skip-top-rows", type=int, default=0,
                    help="ignore this many panorama patch rows at the top (sky / zenith)")
    ap.add_argument("--offset-frac", type=float, default=0.0,
                    help="0 = tile centred on GT; 1 = CCVPE's random offset of up to 1/4 tile")
    ap.add_argument("--level", choices=["none", "imu"], default="none")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=None, help="write an mp4 here")
    ap.add_argument("--png-dir", type=Path, default=None, help="also dump every composed frame as PNG")
    ap.add_argument("--show", action="store_true", help="live window (space = pause, q = quit)")
    args = ap.parse_args()
    assert args.sat_size % PATCH == 0 and all(v % PATCH == 0 for v in args.pano_hw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = poses.load_poses()
    idx = list(range(args.start, args.end or len(gt), args.stride))
    offsets = np.random.default_rng(0).uniform(-1, 1, size=(len(gt), 2)) \
        * (TILE_RAW * args.m_per_px / 4) * args.offset_frac
    mosaic = tiles.Mosaic()
    renderer = fisheye.PanoRenderer(PANO_H, PANO_W, supersample=2)
    model = load_dino(args.arch, device)
    grid_p = (args.pano_hw[0] // PATCH, args.pano_hw[1] // PATCH)
    grid_s = (args.sat_size // PATCH,) * 2

    writer = None
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.png_dir:
        args.png_dir.mkdir(parents=True, exist_ok=True)

    t0, paused = time.time(), False
    for n, i in enumerate(idx):
        img = cv2.imread(str(poses.IMAGES_DIR / gt[i]["image"]))
        R_level = fisheye.leveling_rotation(np.array(gt[i]["R_cam_from_dne"])) if args.level == "imu" else None
        pano_bgr = renderer.render(img, R_level)
        cn, ce = gt[i]["north"] + offsets[i, 0], gt[i]["east"] + offsets[i, 1]
        sat_rgb = mosaic.tile(cn, ce, TILE_RAW, args.m_per_px)

        fp = patch_features(model, cv2.cvtColor(pano_bgr, cv2.COLOR_BGR2RGB), args.pano_hw, device)
        fs = patch_features(model, sat_rgb, (args.sat_size,) * 2, device)
        matches = mutual_matches(fp, fs, args.top, args.min_sim, args.skip_top_rows)

        mean_sim = np.mean([m[4] for m in matches]) if matches else float("nan")
        caption = (f"frame {i}   alt {gt[i]['alt']:.0f} m   {len(matches)} mutual matches   "
                   f"mean cos {mean_sim:.2f}   DINOv2 {args.arch}   {args.m_per_px:g} m/px")
        frame = compose(pano_bgr, cv2.cvtColor(sat_rgb, cv2.COLOR_RGB2BGR), matches, grid_p, grid_s, caption)

        if args.out:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(frame)
        if args.png_dir:
            cv2.imwrite(str(args.png_dir / f"{i:06d}.png"), frame)
        if args.show:
            cv2.imshow("DINOv2 pano <-> satellite", frame)
            while True:
                k = cv2.waitKey(0 if paused else 1) & 0xFF
                if k == ord("q"):
                    n = len(idx)
                    break
                if k == ord(" "):
                    paused = not paused
                if not paused or k in (ord("n"), ord(" ")):
                    break
            if k == ord("q"):
                break
        if n % 10 == 0:
            print(f"{n + 1}/{len(idx)}  {(n + 1) / (time.time() - t0):.2f} fr/s", flush=True)

    if writer:
        writer.release()
        print(f"wrote {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
