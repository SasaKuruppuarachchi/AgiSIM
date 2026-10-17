"""Calibrated, subsampled surround preview; no Isaac, ROS or dataset I/O.

The equirectangular image spans +180 to -180 degrees azimuth (left to right)
and +35 to -55 degrees elevation (top to bottom), in drone body axes. Forward
is at the centre; the vertical centre is the cameras' -10 degree elevation.
A 360 degree panorama cannot be represented by one rectilinear pinhole view.

Inverse KB4/equidistant warps remove each lens's radial distortion. All cameras
sample the same body-frame ray in overlaps, with feathered, normalized weights.
The warp assumes distant scenery: physically separated lenses have parallax
on nearby objects, so this is a visualization, not a new calibrated dataset
camera. Blind regions stay black. The original dataset images are untouched.
"""

from functools import lru_cache

import cv2
import numpy as np

from agipix_splat_geometry import HEIGHT, MOUNTS, WIDTH, body_to_camera_cv, project_equidistant


PREVIEW_WIDTH = 1024
PREVIEW_HEIGHT = 256
TOP_ELEVATION_DEG = 35.0
BOTTOM_ELEVATION_DEG = -55.0
SOURCE_SUBSAMPLE = 2
FEATHER_PIXELS = 80.0  # Measured in the original 1280x800 fisheye images.
JPEG_QUALITY = 80


def encode_preview_jpeg(rgb):
    """Encode the RGB panorama for ROS CompressedImage without changing size."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Could not encode the splat panorama as JPEG")
    return encoded.tobytes()


class SplatPanorama:
    """Build fixed remap/blend tables once; only resize/remap/blend per frame."""

    def __init__(self):
        yaw = np.deg2rad(180.0 - (np.arange(PREVIEW_WIDTH) + 0.5) * 360.0 / PREVIEW_WIDTH)
        elevation = np.deg2rad(TOP_ELEVATION_DEG + (np.arange(PREVIEW_HEIGHT) + 0.5)
                              * (BOTTOM_ELEVATION_DEG - TOP_ELEVATION_DEG) / PREVIEW_HEIGHT)
        yaw, elevation = np.meshgrid(yaw, elevation)
        rays = np.stack((np.cos(elevation) * np.cos(yaw),
                         np.cos(elevation) * np.sin(yaw), np.sin(elevation)), axis=-1)
        maps, weights = {}, {}
        for mount in MOUNTS:
            camera_rays = rays @ body_to_camera_cv(mount)[:3, :3]
            uv = project_equidistant(camera_rays)
            u, v = uv[..., 0], uv[..., 1]
            edge = np.minimum.reduce((u - 0.5, WIDTH - 0.5 - u, v - 0.5, HEIGHT - 0.5 - v))
            valid = (camera_rays[..., 2] > 0) & (edge > 0)
            # Favor the optical centre; fade smoothly before rectangular image
            # boundaries. Normalization prevents bright/dark bands in overlaps.
            feather = np.clip(edge / FEATHER_PIXELS, 0.0, 1.0)
            feather = feather * feather * (3.0 - 2.0 * feather)
            weights[mount.name] = np.where(valid, feather * camera_rays[..., 2] ** 4, 0.0)
            # Projection uses pixel-boundary coordinates; OpenCV array centres
            # are integer indices. Account for the area-downsampled source too.
            maps[mount.name] = ((u / SOURCE_SUBSAMPLE - 0.5).astype(np.float32),
                                (v / SOURCE_SUBSAMPLE - 0.5).astype(np.float32))
        total = sum(weights.values())
        self.coverage = np.sum([w > 0 for w in weights.values()], axis=0)
        self._maps = maps
        self._weights = {name: np.divide(w, total, out=np.zeros_like(w), where=total > 0)
                         .astype(np.float32)[..., None] for name, w in weights.items()}

    def stitch(self, images):
        result = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.float32)
        for mount in MOUNTS:
            image = images[mount.name]
            if image.shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"{mount.name}: expected {WIDTH}x{HEIGHT} RGB")
            small = cv2.resize(image, (WIDTH // SOURCE_SUBSAMPLE, HEIGHT // SOURCE_SUBSAMPLE),
                               interpolation=cv2.INTER_AREA)
            warped = cv2.remap(small, *self._maps[mount.name], interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            result += warped * self._weights[mount.name]
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _default_panorama():
    return SplatPanorama()


def make_preview(images):
    return _default_panorama().stitch(images)
