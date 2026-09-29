"""Prediction files consumed by the viewer (results/<run>/predictions.json)."""
import json
import math
from pathlib import Path

import numpy as np

from . import geo

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def angle_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def summarize(frames, gt, key="err_m"):
    errs = np.array([f[key] for f in frames if f.get(key) is not None])
    heads = np.array([f["heading_err_deg"] for f in frames if f.get("heading_err_deg") is not None])
    s = {}
    if len(errs):
        s.update({"n": int(len(errs)), "mean_err_m": float(errs.mean()), "median_err_m": float(np.median(errs)),
                  "p90_err_m": float(np.percentile(errs, 90)),
                  "within_5m": float((errs < 5).mean()), "within_10m": float((errs < 10).mean())})
        alt = np.array([gt[f["i"]]["alt"] for f in frames if f.get(key) is not None])
        for name, lo, hi in (("ground_lt5m", -1e9, 5), ("low_5_30m", 5, 30), ("high_gt30m", 30, 1e9)):
            m = (alt >= lo) & (alt < hi)
            if m.any():
                s[f"median_err_m_{name}"] = float(np.median(errs[m]))
    if len(heads):
        s.update({"mean_heading_err_deg": float(heads.mean()), "median_heading_err_deg": float(np.median(heads))})
    return s


def make_frame(gt_frame, north, east, heading_deg=None, **extra):
    lat, lon = geo.local_to_latlon(north, east)
    f = {
        "i": gt_frame["i"],
        "lat": lat, "lon": lon, "north": float(north), "east": float(east),
        "heading_deg": None if heading_deg is None else float(heading_deg % 360.0),
        "err_m": float(math.hypot(north - gt_frame["north"], east - gt_frame["east"])),
        "heading_err_deg": None if heading_deg is None else angle_diff(heading_deg, gt_frame["heading_deg"]),
    }
    f.update(extra)
    return f


def write_run(run_id, method, params, frames, gt, color):
    out_dir = RESULTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(frames, gt)
    (out_dir / "predictions.json").write_text(json.dumps(
        {"method": method, "params": params, "summary": summary, "frames": frames}))
    manifest_path = RESULTS / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"methods": []}
    file = f"{run_id}/predictions.json"
    previous = next((m for m in manifest["methods"] if m["file"] == file), {})
    entry = {"name": method, "file": file, "color": color}
    if "visible" in previous:
        entry["visible"] = previous["visible"]
    manifest["methods"] = [m for m in manifest["methods"] if m["file"] != file] + [entry]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return summary
