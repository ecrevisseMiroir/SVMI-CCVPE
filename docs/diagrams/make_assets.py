"""Image assets for the explanatory diagrams (one example frame, real data).

Writes docs/diagrams/assets/*.jpg|png: raw fisheye, CCVPE panorama, orthophoto
context with the tile outline, the 512x512 aerial tile, CCVPE's output heatmap
with GT and predicted poses, and the GT trajectory over the orthophoto.
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "ccvpe"), str(ROOT / "third_party/CCVPE")]

from run_ccvpe import TILE_IN, TILE_RAW, load_model, unit_offsets  # noqa: E402
from svmi import fisheye, poses, tiles  # noqa: E402

OUT = Path(__file__).resolve().parent / "assets"
FRAME = 1500
M_PER_PX = 0.111
GT_BGR = (233, 165, 14)      # #0ea5e9
PRED_BGR = (246, 92, 139)    # #8b5cf6 (cross-area)


def arrow(img, x, y, hdg, color, length=34, r=9, thick=3):
    a = math.radians(hdg)
    tip = (int(x + length * math.sin(a)), int(y - length * math.cos(a)))
    cv2.circle(img, (int(x), int(y)), r + 2, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
    cv2.arrowedLine(img, (int(x), int(y)), tip, (255, 255, 255), thick + 3, cv2.LINE_AA, tipLength=0.35)
    cv2.arrowedLine(img, (int(x), int(y)), tip, color, thick, cv2.LINE_AA, tipLength=0.35)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = poses.load_poses()
    g = gt[FRAME]
    raw = cv2.imread(str(poses.IMAGES_DIR / g["image"]))
    cv2.imwrite(str(OUT / "fisheye.jpg"), cv2.resize(raw, (960, 540), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    pano = fisheye.PanoRenderer(320, 640, supersample=2).render(raw)
    cv2.imwrite(str(OUT / "pano.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 92])

    mosaic = tiles.Mosaic()
    off = unit_offsets(len(gt))[FRAME] * (TILE_RAW * M_PER_PX / 4)
    cn, ce = g["north"] + off[0], g["east"] + off[1]
    tile_rgb = mosaic.tile(cn, ce, TILE_RAW, M_PER_PX)
    tile512 = cv2.resize(tile_rgb, (TILE_IN, TILE_IN), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "tile.jpg"), cv2.cvtColor(tile512, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # orthophoto context: 300 m square around the tile, tile outlined
    ctx_m, ctx_px = 300.0, 600
    ctx = cv2.cvtColor(mosaic.tile(cn, ce, ctx_px, ctx_m / ctx_px), cv2.COLOR_RGB2BGR)
    half = TILE_RAW * M_PER_PX / 2 / (ctx_m / ctx_px)
    c = ctx_px / 2
    cv2.rectangle(ctx, (int(c - half), int(c - half)), (int(c + half), int(c + half)), (255, 255, 255), 5, cv2.LINE_AA)
    cv2.rectangle(ctx, (int(c - half), int(c - half)), (int(c + half), int(c + half)), (74, 163, 22), 3, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "ortho_context.jpg"), ctx, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # CCVPE forward pass on this frame (cross-area weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("crossarea", device)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    grd = transforms.Compose([transforms.ToTensor(), norm])(Image.fromarray(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)))
    sat = transforms.Compose([transforms.Resize([TILE_IN, TILE_IN]), transforms.ToTensor(), norm])(
        Image.fromarray(tile_rgb))
    with torch.no_grad():
        out = model(grd[None].to(device), sat[None].to(device))
    heat = out[1][0, 0].cpu().numpy()
    ori = out[2][0].cpu().numpy()
    r, cpx = np.unravel_index(heat.argmax(), heat.shape)
    theta = math.degrees(math.atan2(ori[1, r, cpx], ori[0, r, cpx]))
    pred_hdg = (-theta) % 360
    scale = M_PER_PX * TILE_RAW / TILE_IN
    gr = (cn - g["north"]) / scale + TILE_IN / 2 - 0.5
    gc = (g["east"] - ce) / scale + TILE_IN / 2 - 0.5
    pn = cn - (r + 0.5 - TILE_IN / 2) * scale
    pe = ce + (cpx + 0.5 - TILE_IN / 2) * scale
    err = math.hypot(pn - g["north"], pe - g["east"])

    h = heat / heat.max()
    h_col = cv2.applyColorMap((255 * h).astype(np.uint8), cv2.COLORMAP_INFERNO)
    alpha = np.clip(h * 1.6, 0, 0.85)[..., None]
    base = cv2.cvtColor(tile512, cv2.COLOR_RGB2BGR).astype(np.float32) * 0.75
    over = (base * (1 - alpha) + h_col.astype(np.float32) * alpha).astype(np.uint8)
    arrow(over, gc, gr, g["heading_deg"], GT_BGR)
    arrow(over, cpx, r, pred_hdg, PRED_BGR)
    cv2.imwrite(str(OUT / "output.jpg"), over, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # GT trajectory (and cross-area predictions) over the orthophoto
    n = np.array([f["north"] for f in gt])
    e = np.array([f["east"] for f in gt])
    mid_n, mid_e = (n.max() + n.min()) / 2, (e.max() + e.min()) / 2
    span = max(n.max() - n.min(), e.max() - e.min()) + 120
    px = 900
    res = span / px
    traj = cv2.cvtColor(mosaic.tile(mid_n, mid_e, px, res), cv2.COLOR_RGB2BGR)
    to_px = lambda N, E: (int((E - mid_e) / res + px / 2), int((mid_n - N) / res + px / 2))
    preds = json.loads((ROOT / "results/ccvpe_vigor_crossarea_0.111mpp_none/predictions.json").read_text())["frames"]
    with_pred = traj.copy()
    layer = with_pred.copy()
    for f in preds:
        cv2.circle(layer, to_px(f["north"], f["east"]), 3, PRED_BGR, -1, cv2.LINE_AA)
    with_pred = cv2.addWeighted(layer, 0.8, with_pred, 0.2, 0)
    for img, name in ((traj, "trajectory_gt.jpg"), (with_pred, "trajectory_pred.jpg")):
        pts = np.array([to_px(a, b) for a, b in zip(n, e)], np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 6, cv2.LINE_AA)
        cv2.polylines(img, [pts], False, GT_BGR, 3, cv2.LINE_AA)
        x0, y0 = to_px(0, 0)
        cv2.circle(img, (x0, y0), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x0, y0), 8, (40, 40, 40), -1, cv2.LINE_AA)
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])

    meta = {"frame": FRAME, "gt": {"north": g["north"], "east": g["east"], "heading_deg": g["heading_deg"],
                                   "alt": g["alt"]},
            "pred": {"north": pn, "east": pe, "heading_deg": pred_hdg}, "err_m": err,
            "heading_err_deg": min(abs(pred_hdg - g["heading_deg"]) % 360, 360 - abs(pred_hdg - g["heading_deg"]) % 360),
            "tile_centre": {"north": cn, "east": ce}, "tile_extent_m": TILE_RAW * M_PER_PX}
    (OUT / "example.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
