"""Build the explanatory diagrams (SVG + PNG) from docs/diagrams/assets.

Run make_assets.py first. Output: docs/diagrams/0N_*.svg and .png (16:9).
"""
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
W, H = 1600, 900
FONT = "Inter, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

INK, MUTED, FAINT = "#111827", "#4b5563", "#9ca3af"
LANE = {
    "ground": ("#2563eb", "#eff6ff"),
    "aerial": ("#15803d", "#f0fdf4"),
    "model": ("#c2410c", "#fff7ed"),
    "pose": ("#7c3aed", "#f5f3ff"),
    "gt": ("#0369a1", "#ecfeff"),
    "neutral": ("#374151", "#f9fafb"),
}
GT_COL, CROSS_COL, SAME_COL = "#0ea5e9", "#8b5cf6", "#e4572e"


def img_uri(name, max_side=None, fmt="JPEG"):
    im = Image.open(ASSETS / name).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, fmt, quality=88)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=18, color=INK, weight=400, anchor="start", italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
            f'text-anchor="{anchor}"{st}>{esc(s)}</text>')


def lines(x, y, items, size=17, color=MUTED, gap=1.45, weight=400, anchor="start"):
    return "".join(text(x, y + i * size * gap, s, size, color, weight, anchor) for i, s in enumerate(items))


def box(x, y, w, h, lane, rx=14, dash=False):
    stroke, fill = LANE[lane]
    d = ' stroke-dasharray="8 6"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>'


def image(x, y, w, h, uri, border="#d1d5db", rx=8):
    cid = f"c{abs(hash((x, y, w, h))) % 10**8}"
    return (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>'
            f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" preserveAspectRatio="xMidYMid slice" '
            f'clip-path="url(#{cid})"/>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{border}" stroke-width="1.5"/>')


def arrow(x1, y1, x2, y2, color="#374151", width=3, label=None, label_pos=None, label_lines=None, size=15):
    s = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
         f'marker-end="url(#head-{color[1:]})"/>')
    if label or label_lines:
        lx, ly = label_pos or ((x1 + x2) / 2, (y1 + y2) / 2 - 12)
        for i, t in enumerate(label_lines or [label]):
            s += (f'<text x="{lx}" y="{ly + i * size * 1.35}" font-size="{size}" fill="{color}" text-anchor="middle" '
                  f'paint-order="stroke" stroke="#ffffff" stroke-width="5" stroke-linejoin="round">{esc(t)}</text>')
    return s


def path_arrow(d, color="#374151", width=3):
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'marker-end="url(#head-{color[1:]})"/>')


def badge(x, y, n, color):
    return (f'<circle cx="{x}" cy="{y}" r="17" fill="{color}"/>'
            + text(x, y + 6, str(n), 17, "#ffffff", 700, "middle"))


def legend_dot(x, y, color, label, line=False):
    mark = (f'<line x1="{x - 9}" y1="{y - 5}" x2="{x + 9}" y2="{y - 5}" stroke="{color}" stroke-width="4"/>' if line
            else f'<circle cx="{x}" cy="{y - 5}" r="7" fill="{color}" stroke="#fff" stroke-width="2"/>')
    return mark + text(x + 18, y, label, 15, MUTED)


def svg(title, subtitle, body, footer):
    colors = {c for c, _ in LANE.values()} | {"#374151", GT_COL, CROSS_COL}
    markers = "".join(
        f'<marker id="head-{c[1:]}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for c in colors)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}"><defs>{markers}</defs>'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
            + text(60, 72, title, 34, INK, 700) + text(60, 108, subtitle, 18, MUTED)
            + body + text(60, H - 28, footer, 13, FAINT) + "</svg>")


# --------------------------------------------------------------------------- figures
def fig_overview(ex):
    cards = [
        ("model", "Pretrained CCVPE", None,
         ["Official code + pretrained", "weights (no training by us)", "",
          "Trained on VIGOR (street-view", "panoramas + aerial images,", "4 US cities)", "",
          "2 weight sets used:", "• same-area  • cross-area", "",
          "Applied to SVMI+ as is", "(zero-shot)"]),
        ("ground", "Prepare inputs", "pano_tile",
         ["Per frame:", "dual fisheye → 360° panorama", "IGN orthophoto → 71 m tile", "around a coarse location prior"]),
        ("pose", "Run inference", "output.jpg",
         ["CCVPE(panorama, tile) →", "location heatmap + heading", "→ pose (x, y, heading)", "for all 2799 frames"]),
        ("gt", "Evaluate & visualise", "viewer",
         ["Compare with SVMI+ ground", "truth (6-DoF log + takeoff fix)", "Web UI player: images,", "GT pose, CCVPE poses"]),
    ]
    body, x0, cw, gap, y0, ch = "", 60, 335, 47, 160, 610
    pano, tile = img_uri("pano.jpg", 900), img_uri("tile.jpg", 600)
    for k, (lane, title, thumb, items) in enumerate(cards):
        x = x0 + k * (cw + gap)
        stroke, _ = LANE[lane]
        body += box(x, y0, cw, ch, lane) + badge(x + 32, y0 + 38, k + 1, stroke)
        body += text(x + 60, y0 + 45, title, 21, INK, 700)
        ty = y0 + 80
        if thumb == "pano_tile":
            body += image(x + 20, ty, cw - 40, (cw - 40) / 2, pano)
            body += image(x + 20 + (cw - 40 - 190) / 2, ty + (cw - 40) / 2 + 18, 190, 190, tile)
            ly = ty + (cw - 40) / 2 + 240
        elif thumb == "output.jpg":
            body += image(x + 30, ty, cw - 60, cw - 60, img_uri("output.jpg", 700))
            ly = ty + cw - 60 + 36
        elif thumb == "viewer":
            body += image(x + 20, ty, cw - 40, (cw - 40) * 0.694, img_uri("viewer.png", 1200))
            ly = ty + (cw - 40) * 0.694 + 36
        else:
            body += (f'<rect x="{x + 20}" y="{ty}" width="{cw - 40}" height="64" rx="8" fill="#111827"/>'
                     + text(x + 36, ty + 28, "$ git clone github.com/", 15, "#e5e7eb")
                     + text(x + 36, ty + 49, "    tudelft-iv/CCVPE", 15, "#e5e7eb"))
            ly = ty + 100
        body += lines(x + 24, ly, items, 17)
        if k < 3:
            body += arrow(x + cw + 6, y0 + ch / 2, x + cw + gap - 6, y0 + ch / 2, "#374151", 3)
    return svg("What we did: zero-shot CCVPE on SVMI+",
               "A cross-view model trained on street-level data, applied unchanged to a drone flight",
               body, "CCVPE: Xia, Booij, Kooij, T-PAMI 2023 · VIGOR: Zhu, Yang, Chen, CVPR 2021 · "
                     "Orthophoto: IGN Géoplateforme (Licence Ouverte 2.0)")


def fig_io(ex):
    g, b = LANE["ground"][0], LANE["aerial"][0]
    body = ""
    # lane titles
    body += text(60, 160, "Ground image", 20, g, 700) + text(60, 498, "Aerial image", 20, b, 700)
    # ground lane
    body += image(60, 180, 336, 189, img_uri("fisheye.jpg", 1000))
    body += lines(60, 395, ["Raw dual fisheye, 1280×720", "(Ricoh Theta S under the drone)"], 15)
    body += arrow(410, 275, 528, 275, g, 3, label_lines=["reproject with the", "camera calibration"],
                  label_pos=(469, 220))
    body += image(540, 185, 360, 180, img_uri("pano.jpg", 1000))
    body += lines(540, 390, ["360° panorama, RGB 320×640", "centre column = drone forward"], 15)
    # aerial lane
    body += image(60, 518, 280, 280, img_uri("ortho_context.jpg", 800))
    body += lines(60, 825, ["IGN orthophoto (0.19 m/px),", "green box = tile for this frame"], 15)
    body += arrow(352, 658, 578, 658, b, 3,
                  label_lines=["crop 640×640 px at 0.111 m/px", "(≈71 m) around the location", "prior, resize to 512×512"],
                  label_pos=(465, 588))
    body += image(590, 528, 260, 260, img_uri("tile.jpg", 700))
    body += lines(590, 815, ["Aerial tile, RGB 512×512", "north up"], 15)
    # model
    mx, my, mw, mh = 965, 215, 255, 520
    body += box(mx, my, mw, mh, "model")
    body += text(mx + mw / 2, my + 44, "CCVPE", 26, LANE["model"][0], 700, "middle")
    blocks = [("Ground encoder", "EfficientNet-B0"), ("Aerial encoder", "EfficientNet-B0"),
              ("Descriptor matching", "per aerial location"), ("Localization decoder", "→ heatmap"),
              ("Orientation decoder", "→ heading per location")]
    for k, (t1, t2) in enumerate(blocks):
        by = my + 70 + k * 88
        body += (f'<rect x="{mx + 18}" y="{by}" width="{mw - 36}" height="72" rx="10" fill="#ffffff" '
                 f'stroke="#fdba74" stroke-width="1.5"/>')
        body += text(mx + mw / 2, by + 31, t1, 17, INK, 600, "middle") + text(mx + mw / 2, by + 55, t2, 14, MUTED,
                                                                            anchor="middle")
    body += arrow(912, 275, mx - 8, 320, g, 3)
    body += arrow(862, 658, mx - 8, 600, b, 3)
    # output
    ox = 1275
    body += arrow(mx + mw + 6, 315, ox - 10, 315, LANE["pose"][0], 3)
    body += text(ox, 160, "Output", 20, LANE["pose"][0], 700)
    body += image(ox, 180, 270, 270, img_uri("output.jpg", 700))
    body += legend_dot(ox + 10, 482, GT_COL, "ground truth pose")
    body += legend_dot(ox + 10, 507, CROSS_COL, "CCVPE pose (heatmap peak)")
    body += box(ox, 528, 270, 270, "pose")
    body += text(ox + 135, 562, "Pose (3-DoF)", 20, LANE["pose"][0], 700, "middle")
    body += lines(ox + 18, 596, ["x, y: heatmap peak in the tile", "→ metres → lat/lon", "",
                                 "heading: orientation at the", "peak, degrees from north", "",
                                 f"this frame: error {ex['err_m']:.1f} m,", f"heading error {ex['heading_err_deg']:.1f}°"],
                  15, gap=1.4)
    return svg("CCVPE inputs and output (one frame)",
               "Frame 1500, 54 m above ground · both inputs are RGB images; the output is a 3-DoF pose",
               body, "Heatmap = CCVPE's probability of the camera position over the tile · "
                     "the tile is placed around the true position plus a random offset of up to ±18 m "
                     "(VIGOR test protocol)")


def fig_loop(ex, summ):
    body = ""
    # frame stack
    fe = img_uri("fisheye.jpg", 600)
    for k in range(3):
        body += image(60 + k * 22, 190 + k * 22, 300, 169, fe)
    body += lines(60, 440, ["2799 frames", "4 min 36 s flight, ~10 fps"], 17, INK)
    body += arrow(420, 300, 498, 300, "#374151", 3)
    # loop box
    lx, ly, lw, lh = 510, 160, 520, 470
    body += box(lx, ly, lw, lh, "model")
    body += text(lx + 28, ly + 46, "for each frame i:", 22, LANE["model"][0], 700)
    steps = [
        ("1", "panorama_i ← reproject fisheye_i", None),
        ("2", "tile_i ← orthophoto around prior_i", None),
        ("3", "heatmap_i, heading_i ← CCVPE(", "panorama_i, tile_i)"),
        ("4", "pose_i = (north, east, heading)", "at the heatmap peak"),
        ("5", "error_i = | pose_i − ground truth_i |", None),
    ]
    y = ly + 100
    for n, s1, s2 in steps:
        body += badge(lx + 44, y - 6, n, LANE["model"][0]) + text(lx + 76, y, s1, 18, INK)
        if s2:
            y += 28
            body += text(lx + 76, y, s2, 18, INK)
        y += 62
    # loop-back arrow, inside the box on the right
    ax = lx + lw - 40
    body += path_arrow(f"M {ax - 20} {ly + lh - 40} C {ax + 25} {ly + lh - 40}, "
                       f"{ax + 25} {ly + 60}, {ax - 14} {ly + 60}", LANE["model"][0], 3)
    body += text(ax - 4, ly + lh / 2 + 5, "next i", 15, LANE["model"][0], 600, "end")
    mx = 1120
    body += arrow(lx + lw + 8, 400, mx - 12, 400, "#374151", 3)
    # map
    body += image(mx, 160, 420, 420, img_uri("trajectory_pred.jpg", 900))
    body += legend_dot(mx + 10, 610, GT_COL, "ground-truth trajectory", line=True)
    body += legend_dot(mx + 10, 636, CROSS_COL, "CCVPE cross-area estimates (one dot per frame)")
    # numbers
    cards = [
        ("2 × 2799", "inferences (same-area,\ncross-area weights)"),
        ("≈ 29 fps", "on an RTX 3080\n(about 1.6 min per run)"),
        (f"{summ['cross']:.1f} m / {summ['same']:.1f} m", "median error,\ncross-area / same-area"),
        (f"{summ['centre']:.1f} m", "median error of always\nanswering the tile centre"),
    ]
    for k, (big, small) in enumerate(cards):
        x = 60 + k * 248
        body += box(x, 680, 230, 150, "neutral")
        body += text(x + 18, 726, big, 25, INK, 700)
        body += lines(x + 18, 762, small.split("\n"), 15)
    return svg("Inference over the whole flight",
               "Each frame is processed independently: no tracking, no use of the previous estimate",
               body, "prior_i = ground-truth position + fixed random offset of up to ±18 m (the tile is "
                     "guaranteed to contain the drone, as in the VIGOR test protocol)")


def fig_gt(ex):
    c = LANE["gt"][0]
    body = ""
    items = [
        ("SVMI+ camera_poses.txt", ["2799 rows, one per image:", "position (m) relative to frame 0,",
                                    "rotation as axis-angle"]),
        ("Takeoff GNSS fix", ["49.967068 N, 2.266771 E", "(local origin)"]),
        ("Convention fixes", ["column 2 is south, not north", "heading = 180° − logged value",
                              "altitude relative to frame 0", "(checked with sun direction and", "forest landmarks)"]),
    ]
    y = 160
    for k, (title, ls) in enumerate(items):
        h = 50 + len(ls) * 24
        body += box(60, y, 360, h, "gt") + text(80, y + 34, title, 19, c, 700) + lines(80, y + 62, ls, 16)
        body += arrow(428, y + h / 2, 488, 430, c, 2.5)
        y += h + 22
    # GT on orthophoto
    body += text(500, 160, "Ground-truth trajectory", 20, INK, 700)
    body += text(500, 186, "drawn on the IGN orthophoto", 16, MUTED)
    body += image(500, 205, 400, 400, img_uri("trajectory_gt.jpg", 900))
    body += legend_dot(510, 636, GT_COL, "trajectory", line=True) + legend_dot(650, 636, "#282828", "takeoff")
    body += arrow(912, 405, 978, 405, "#374151", 3)
    # viewer
    vx, vy, vw = 990, 205, 560
    vh = vw * 1000 / 1440
    body += text(vx, 160, "Web UI player", 20, INK, 700)
    body += text(vx, 186, "python3 -m http.server → localhost:8000/viewer/", 16, MUTED)
    body += image(vx, vy, vw, vh, img_uri("viewer.png", 1600))
    callouts = [  # (x, y in screenshot px of 1440x1000, label)
        (300, 85, "1"), (680, 85, "2"), (1045, 215, "3"), (1215, 315, "4"), (1405, 795, "5")]
    for sx, sy, n in callouts:
        body += badge(vx + sx * vw / 1440, vy + sy * vh / 1000, n, "#111827")
    legend = ["raw dual-fisheye frame", "panorama fed to CCVPE", "ground-truth pose (cyan)",
              "CCVPE poses: cross-area (purple), same-area (orange)",
              "per-frame error chart; play, scrub, step, click the path to jump"]
    for k, t in enumerate(legend):
        y = vy + vh + 42 + k * 30
        body += badge(vx + 14, y - 6, k + 1, "#111827") + text(vx + 42, y, t, 16, INK)
    return svg("Ground truth and the web player",
               "The 6-DoF log is only used for scoring, for the tile prior, and for display",
               body, "Ground-truth heading = panorama centre direction, clockwise from north · "
                     "GNSS north is quantized in 11 m steps")


def render_png(svg_path, png_path):
    opts = webdriver.FirefoxOptions()
    opts.add_argument("-headless")
    opts.set_preference("layout.css.devPixelsPerPx", "2")
    d = webdriver.Firefox(options=opts)
    try:
        d.set_window_size(W + 100, H + 200)
        d.get(svg_path.as_uri())
        time.sleep(1.5)
        d.find_element(By.CSS_SELECTOR, "svg").screenshot(str(png_path))
    finally:
        d.quit()


def main():
    ex = json.loads((ASSETS / "example.json").read_text())
    root = HERE.parents[1]
    s = lambda run: json.loads((root / f"results/{run}/predictions.json").read_text())["summary"]["median_err_m"]
    frames = json.loads((root / "results/ccvpe_vigor_crossarea_0.111mpp_none/predictions.json").read_text())["frames"]
    import numpy as np
    summ = {"cross": s("ccvpe_vigor_crossarea_0.111mpp_none"), "same": s("ccvpe_vigor_samearea_0.111mpp_none"),
            "centre": float(np.median([f["tile_centre_err_m"] for f in frames]))}
    figs = {"01_overview": fig_overview(ex), "02_inputs_outputs": fig_io(ex),
            "03_inference_loop": fig_loop(ex, summ), "04_ground_truth_viewer": fig_gt(ex)}
    for name, content in figs.items():
        p = HERE / f"{name}.svg"
        p.write_text(content)
        render_png(p, HERE / f"{name}.png")
        print("wrote", p.name, "and", p.with_suffix(".png").name)


if __name__ == "__main__":
    main()
