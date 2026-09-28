"""Virtual downward-looking pinhole camera rendered from the dual fisheye.

Image "up" is the drone's forward axis (panorama centre); no attitude from the
ground truth is used, i.e. the drone is assumed to fly level.
"""
import math

import cv2
import numpy as np

from . import fisheye


class NadirRenderer:
    def __init__(self, width=512, height=384, hfov_deg=60.0):
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        v, u = np.meshgrid(np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij")
        x = (u - width / 2) / f   # right
        y = (v - height / 2) / f  # image down = backwards
        d = np.stack([-np.ones_like(x), x, -y], axis=-1)  # (up, right, forward) in the left-fisheye frame
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        use_left = d[..., 2] >= 0
        ul, vl = fisheye._project(d, fisheye.K_LEFT, fisheye.XI_LEFT)
        ur, vr = fisheye._project(d @ fisheye.R_RIGHT_FROM_LEFT.T, fisheye.K_RIGHT, fisheye.XI_RIGHT)
        self.mx = np.where(use_left, ul, ur).astype(np.float32)
        self.my = np.where(use_left, vl, vr).astype(np.float32)

    def render(self, img):
        return cv2.remap(img, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
