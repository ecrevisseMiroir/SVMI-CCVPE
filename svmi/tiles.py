"""North-up aerial tiles cut from the IGN mosaic, with exact pixel<->metre maps."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import geo

ROOT = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = None


class Mosaic:
    def __init__(self, name="ign_z19_wide"):
        self.meta = json.loads((ROOT / f"data/map/{name}.json").read_text())
        self.zoom = self.meta["zoom"]
        self.image = cv2.cvtColor(cv2.imread(str(ROOT / f"data/map/{self.meta['image']}")), cv2.COLOR_BGR2RGB)

    def local_to_px(self, north, east):
        lat, lon = geo.local_to_latlon(north, east)
        x, y = geo.latlon_to_world_px(lat, lon, self.zoom)
        ox, oy = self.meta["origin_world_px"]
        return x - ox, y - oy

    def tile(self, center_north, center_east, size_px, m_per_px):
        """RGB tile (size_px x size_px), north up, centred on a local position.

        Pixel (row, col) of the tile covers local position
        north = center_north - (row + 0.5 - size_px / 2) * m_per_px,
        east  = center_east  + (col + 0.5 - size_px / 2) * m_per_px.
        """
        half = size_px / 2
        corners_tile = np.float32([[0, 0], [size_px, 0], [0, size_px]])
        corners_map = np.float32([
            self.local_to_px(center_north + half * m_per_px, center_east - half * m_per_px),
            self.local_to_px(center_north + half * m_per_px, center_east + half * m_per_px),
            self.local_to_px(center_north - half * m_per_px, center_east - half * m_per_px),
        ])
        A = cv2.getAffineTransform(corners_tile, corners_map)
        interp = cv2.INTER_AREA if m_per_px > self.meta["m_per_px"] else cv2.INTER_CUBIC
        return cv2.warpAffine(self.image, A, (size_px, size_px),
                              flags=cv2.WARP_INVERSE_MAP | interp, borderMode=cv2.BORDER_REFLECT)


def tile_px_to_local(row, col, center_north, center_east, size_px, m_per_px):
    """Inverse of Mosaic.tile's pixel convention (row/col may be fractional pixel centres)."""
    half = size_px / 2
    return (center_north - (row + 0.5 - half) * m_per_px,
            center_east + (col + 0.5 - half) * m_per_px)
