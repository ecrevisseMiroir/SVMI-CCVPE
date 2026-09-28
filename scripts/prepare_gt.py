"""Write data/gt.json: per-frame ground truth in local metres and lat/lon."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svmi import geo, poses  # noqa: E402


def main():
    frames = poses.load_poses()
    for f in frames:
        f.pop("R_cam_from_dne")
    out = ROOT / "data/gt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "origin": {"lat": geo.ORIGIN_LAT, "lon": geo.ORIGIN_LON},
        "heading_convention": "panorama centre (left-fisheye optical axis), degrees clockwise from north",
        "alt_convention": "metres above the takeoff ground",
        "frames": frames,
    }))
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
