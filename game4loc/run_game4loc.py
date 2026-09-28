"""Zero-shot Game4Loc (Ji et al., AAAI 2025) retrieval on SVMI+.

Query   = virtual nadir pinhole view (GTA-UAV style: 512x384, 60 deg HFOV,
          resized to 384x384) rendered from the dual fisheye, assuming level
          flight and using no ground truth.
Gallery = north-up IGN tiles at several ground sizes, on a dense grid covering
          the whole flight area (trajectory bbox + margin).
The prediction is the centre of the best-matching tile (top-1). Game4Loc does
not estimate heading. Ground truth is only used for scoring.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmi import nadir, poses, results, tiles  # noqa: E402

MODEL = "vit_base_patch16_rope_reg1_gap_256.sbb_in1k"
IMG = 384


def load_model(area, device):
    ckpt = hf_hub_download("Yux1ang/gta_uav_pretrained_models", f"vit_base_eva_gta_{area}.pth",
                           local_dir=ROOT / "third_party/game4loc_weights")
    net = timm.create_model(MODEL, pretrained=False, num_classes=0, img_size=IMG)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def preprocess(rgb):
    x = cv2.resize(rgb, (IMG, IMG), interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32) / 255.0
    return torch.from_numpy((x - 0.5) / 0.5).permute(2, 0, 1)


@torch.no_grad()
def embed(net, images, device, batch=64):
    feats = []
    for b in range(0, len(images), batch):
        x = torch.stack([preprocess(im) for im in images[b:b + batch]]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = net(x)
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", choices=["same_area", "cross_area"], default="cross_area")
    ap.add_argument("--hfov", type=float, default=60.0)
    ap.add_argument("--tile-sizes", type=float, nargs="+", default=[60, 120, 240], help="gallery tile sizes (m)")
    ap.add_argument("--overlap", type=float, default=0.75, help="gallery stride = size * (1 - overlap)")
    ap.add_argument("--margin", type=float, default=250.0, help="search area = trajectory bbox + margin (m)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--save-examples", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = poses.load_poses()
    net = load_model(args.area, device)
    mosaic = tiles.Mosaic()

    north = [g["north"] for g in gt]
    east = [g["east"] for g in gt]
    n0, n1 = min(north) - args.margin, max(north) + args.margin
    e0, e1 = min(east) - args.margin, max(east) + args.margin
    gallery_imgs, gallery_meta = [], []
    for size in args.tile_sizes:
        step = size * (1 - args.overlap)
        for cn in np.arange(n0 + size / 2, n1 - size / 2 + 1e-6, step):
            for ce in np.arange(e0 + size / 2, e1 - size / 2 + 1e-6, step):
                gallery_imgs.append(mosaic.tile(cn, ce, IMG, size / IMG))
                gallery_meta.append((cn, ce, size))
    t0 = time.time()
    g_feat = embed(net, gallery_imgs, device).to(device)
    print(f"gallery: {len(gallery_meta)} tiles over {n1 - n0:.0f} x {e1 - e0:.0f} m "
          f"({time.time() - t0:.1f} s)", flush=True)

    renderer = nadir.NadirRenderer(512, 384, args.hfov)
    idx = list(range(0, len(gt), args.stride))
    run_id = f"game4loc_{args.area}_hfov{args.hfov:g}"
    debug_dir = ROOT / "results" / run_id / "examples"
    debug_dir.mkdir(parents=True, exist_ok=True)
    example_ids = set(idx[:: max(1, len(idx) // max(1, args.save_examples))][: args.save_examples])

    frames = []
    for b in range(0, len(idx), 64):
        batch = idx[b:b + 64]
        queries = [cv2.cvtColor(renderer.render(cv2.imread(str(poses.IMAGES_DIR / gt[i]["image"]))),
                                cv2.COLOR_BGR2RGB) for i in batch]
        q = embed(net, queries, device).to(device)
        sim = q @ g_feat.T
        top = sim.topk(5, dim=1)
        for k, i in enumerate(batch):
            j = int(top.indices[k, 0])
            cn, ce, size = gallery_meta[j]
            frames.append(results.make_frame(
                gt[i], cn, ce, None, tile_size_m=size, score=float(top.values[k, 0]),
                top5=[{"north": gallery_meta[int(t)][0], "east": gallery_meta[int(t)][1],
                       "size_m": gallery_meta[int(t)][2], "score": float(s)}
                      for t, s in zip(top.indices[k], top.values[k])]))
            if i in example_ids:
                gt_tile = mosaic.tile(gt[i]["north"], gt[i]["east"], IMG, frames[-1]["tile_size_m"] / IMG)
                panel = [cv2.resize(queries[k], (IMG, IMG)), gallery_imgs[j], gt_tile]
                cv2.imwrite(str(debug_dir / f"{i:06d}.jpg"), cv2.cvtColor(np.hstack(panel), cv2.COLOR_RGB2BGR))
        print(f"{len(frames)}/{len(idx)}  median err so far {np.median([f['err_m'] for f in frames]):.1f} m",
              flush=True)

    params = {"weights": f"GTA-UAV {args.area}", "hfov_deg": args.hfov, "tile_sizes_m": args.tile_sizes,
              "overlap": args.overlap, "search_area_m": [n1 - n0, e1 - e0], "gallery_size": len(gallery_meta)}
    name = f"Game4Loc GTA-{args.area.replace('_', '-')} (retrieval, {args.hfov:g}° nadir)"
    color = {"cross_area": "#3a86ff", "same_area": "#00b4d8"}[args.area]
    summary = results.write_run(run_id, name, params, frames, gt, color)
    rng = np.random.default_rng(0)
    rand = [math.hypot(*(np.array(gallery_meta[rng.integers(len(gallery_meta))][:2]) - (g["north"], g["east"])))
            for g in gt[::args.stride]]
    lines = [name] + [f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()]
    lines.append(f"  baseline random gallery tile median err m: {np.median(rand):.3f}")
    (ROOT / "results" / run_id / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
