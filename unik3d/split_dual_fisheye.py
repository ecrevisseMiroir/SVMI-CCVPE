"""Split Ricoh Theta S dual-fisheye frames (1280x720) into left/right 640x640 crops.

Crop = rows 0:640, cols 0:640 (left, calib "camera 1") and 640:1280 (right, calib "camera 0").
Matches cam_left.json / cam_right.json (right cx already shifted by -640).
Usage: python split_dual_fisheye.py IMAGES_DIR OUT_DIR
"""
import sys, os
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
for side in ("left", "right"):
    os.makedirs(os.path.join(out, side), exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.endswith(".png") or f.startswith("._"):
        continue
    im = Image.open(os.path.join(src, f)).convert("RGB")
    im.crop((0, 0, 640, 640)).save(os.path.join(out, "left", f))
    im.crop((640, 0, 1280, 640)).save(os.path.join(out, "right", f))
