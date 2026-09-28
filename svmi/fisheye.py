"""Dual-fisheye (Ricoh Theta S) to equirectangular panorama for SVMI+.

Calibration (camera_calibration.txt) uses the unified spherical model (Mei):
a unit-sphere point X_s projects to K @ [X_s.x/(X_s.z+xi), X_s.y/(X_s.z+xi), 1].

Frame convention (derived from the pose log and the images): the pose camera is
the LEFT fisheye (calibration "camera 1"). Its +x axis points up (the camera
hangs upside down with its long side vertical), +z is the optical axis, roughly
the drone's forward axis, and +y points to the right. The right fisheye
("camera 0") looks backwards; c2Mc1 maps left-camera coordinates into it.

The panorama follows street-view convention: the top row is the zenith, the
centre column is the pose camera's optical axis, and azimuth grows clockwise
(seen from above) from left to right.
"""
import cv2
import numpy as np

# camera 1 = left half of the image, camera 0 = right half
K_LEFT = np.array([[569.1013833, 0, 317.5131787], [0, 571.5333211, 322.4985417], [0, 0, 1]])
XI_LEFT = 1.967618682
K_RIGHT = np.array([[540.1395384, 0, 966.9491055], [0, 553.5242477, 320.4407121], [0, 0, 1]])
XI_RIGHT = 1.81043672
R_RIGHT_FROM_LEFT = np.array([
    [-0.9987124667, -0.000276324765, -0.05072802434],
    [-0.00125077543, 0.9998152918, 0.01917857638],
    [0.05071335495, 0.0192173327, -0.998528342],
])


def _project(dirs, K, xi):
    z = dirs[..., 2] + xi
    u = K[0, 0] * dirs[..., 0] / z + K[0, 2]
    v = K[1, 1] * dirs[..., 1] / z + K[1, 2]
    return u, v


def pano_directions(height, width):
    """Unit directions (up, right, forward) per pixel in the level body frame."""
    v, u = np.meshgrid(np.arange(height) + 0.5, np.arange(width) + 0.5, indexing="ij")
    lat = np.pi / 2 - np.pi * v / height
    azi = 2 * np.pi * (u / width - 0.5)
    return np.stack([np.sin(lat), np.cos(lat) * np.sin(azi), np.cos(lat) * np.cos(azi)], axis=-1)


class PanoRenderer:
    """Precomputes remap tables; call render(img, R_level) per frame.

    R_level rotates level-frame directions (up, right, forward) into the pose
    camera frame. Identity assumes the drone flies level; pass
    leveling_rotation(R_gt) to remove roll/pitch using the IMU attitude.
    """

    def __init__(self, height=320, width=640, supersample=2):
        self.h, self.w, self.ss = height, width, supersample
        self.dirs = pano_directions(height * supersample, width * supersample).astype(np.float64)
        self._cache_key = None

    def maps(self, R_level=None):
        d = self.dirs if R_level is None else self.dirs @ R_level.T
        use_left = d[..., 2] >= 0
        ul, vl = _project(d, K_LEFT, XI_LEFT)
        dr = d @ R_RIGHT_FROM_LEFT.T
        ur, vr = _project(dr, K_RIGHT, XI_RIGHT)
        mx = np.where(use_left, ul, ur).astype(np.float32)
        my = np.where(use_left, vl, vr).astype(np.float32)
        return mx, my

    def render(self, img, R_level=None):
        key = None if R_level is None else R_level.tobytes()
        if key != self._cache_key or not hasattr(self, "_maps"):
            self._maps = self.maps(R_level)
            self._cache_key = key
        pano = cv2.remap(img, *self._maps, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if self.ss > 1:
            pano = cv2.resize(pano, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return pano


def leveling_rotation(R_cam_from_dne):
    """Level-frame -> camera-frame rotation that removes roll/pitch but keeps yaw.

    R_cam_from_dne maps (down, north, east) directions into the camera frame
    (see svmi.poses). The level frame shares the camera's heading, so no yaw
    information leaks into the panorama.
    """
    optical = R_cam_from_dne[2, :]
    heading = np.arctan2(optical[2], optical[1])
    up_w = np.array([-1.0, 0.0, 0.0])
    fwd_w = np.array([0.0, np.cos(heading), np.sin(heading)])
    right_w = np.cross(fwd_w, up_w)  # up x right = forward  =>  right = forward x up
    level_axes_w = np.stack([up_w, right_w, fwd_w], axis=1)  # columns: level axes in world
    return R_cam_from_dne @ level_axes_w
