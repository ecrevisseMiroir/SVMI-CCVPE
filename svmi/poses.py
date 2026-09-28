"""Ground-truth loading for SVMI+ (Matrice600Pro/SixDOFs).

Axis conventions, established empirically against the images and the IGN
orthophoto (see README "Ground-truth conventions"):

* Columns 1-3 of camera_poses.txt are (down, south, east) in metres. Column 2
  is labelled "north" in the dataset Readme, but the images show the drone
  flying toward the forest that lies SOUTH of the takeoff point whenever
  column 2 increases, and the sun only stays fixed in the world when the
  heading is mirrored accordingly.
* Column 1 is an absolute GPS altitude reading; the drone sits on the ground
  at frame 0, so height above ground is column-1 value relative to frame 0.
* The axis-angle rotation R maps a direction expressed in that same
  (down, south, east) frame into the left-fisheye camera frame
  (+x up, +y right, +z optical axis): d_cam = R @ d_world.
"""
from pathlib import Path

import cv2
import numpy as np

from . import geo

DATASET_ROOT = Path(__file__).resolve().parents[1] / "SVMI/SVMISplus/SVMISplus/Matrice600Pro"
IMAGES_DIR = DATASET_ROOT / "SixDOFs/Images"
POSES_FILE = DATASET_ROOT / "SixDOFs/camera_poses.txt"

# Log frame (down, south, east) -> true (down, north, east).
LOG_TO_DNE = np.diag([1.0, -1.0, 1.0])


def load_poses(path=POSES_FILE):
    """Return a list of per-frame ground-truth dicts in true (north, east) metres.

    heading_deg is the left-fisheye optical axis (the panorama centre column)
    projected on the ground, in degrees clockwise from north.
    R_cam_from_dne maps (down, north, east) directions into the camera frame.
    """
    raw = np.loadtxt(path)
    ground = -raw[0, 0]
    frames = []
    for i, row in enumerate(raw):
        down, north, east = LOG_TO_DNE @ row[:3]
        R_log = cv2.Rodrigues(row[3:6])[0]
        R = R_log @ LOG_TO_DNE
        optical_axis = R[2, :]  # camera +z expressed in (down, north, east)
        heading = float(np.degrees(np.arctan2(optical_axis[2], optical_axis[1])) % 360.0)
        lat, lon = geo.local_to_latlon(north, east)
        frames.append({
            "i": i,
            "image": f"{i:06d}.png",
            "north": float(north),
            "east": float(east),
            "alt": float(-down - ground),
            "lat": lat,
            "lon": lon,
            "heading_deg": heading,
            "R_cam_from_dne": R.tolist(),
        })
    return frames
