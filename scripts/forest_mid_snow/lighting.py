"""CPU-only preparation of a single visible, shadowed forest HDRI dome.

The source photograph contains a forest's own canopy occlusion. A small linear
radiance floor lifts its darkest directions before our scene canopy occludes
them again. This is an explicit lighting approximation, not an exposure edit:
the HDR panorama and its highlights remain in linear floating-point radiance.
"""
from datetime import datetime
import hashlib
from pathlib import Path
import shutil

import numpy as np


def radiance_stats(rgb):
    """Solid-angle and upward-irradiance statistics of a latlong RGB image."""
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB latlong image")
    if not np.isfinite(rgb).all() or np.any(rgb < 0):
        raise ValueError("HDR radiance must be finite and nonnegative")
    height, width, _ = rgb.shape
    luminance = rgb @ np.array([.2126, .7152, .0722], np.float32)
    theta = (np.arange(height, dtype=np.float64) + .5) * np.pi / height
    solid_angle = np.sin(theta)
    upward = np.maximum(np.cos(theta), 0) * solid_angle
    weighted = lambda values, weights: float(np.sum(values * weights[:, None], dtype=np.float64)
                                            / (width * weights.sum()))
    horizon = np.abs(theta - np.pi / 2) < np.deg2rad(10)
    return dict(shape=list(rgb.shape), minimum=float(luminance.min()),
                median=float(np.median(luminance)), p95=float(np.quantile(luminance, .95)),
                p99=float(np.quantile(luminance, .99)), maximum=float(luminance.max()),
                solid_angle_mean=weighted(luminance, solid_angle),
                upper_hemisphere_mean=weighted(luminance, solid_angle * (theta < np.pi / 2)),
                horizon_mean=weighted(luminance, solid_angle * horizon),
                upward_irradiance_over_pi=weighted(luminance, upward))


def write_ambient_hdri(source, destination, *, ambient=.25, ambient_color=(.96, .98, 1.),
                       expected_md5=None, log=print):
    """Write ``source_rgb + ambient * ambient_color`` as linear Radiance HDR.

    ``ambient`` is in source texture radiance units, *before* the dome intensity
    multiplier. A .25 offset and dome intensity 6000 represent an added constant
    radiance of 1500, not a hidden unshadowed light. Use one visible DomeLight,
    regular geometric shadows, and the returned output filename for its texture.

    Source assets are never modified. Existing outputs move to archive/ first.
    ``expected_md5`` can be passed from the already verified CC0 API manifest.
    """
    import cv2
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Source HDRI must remain unchanged; use a derived output path")
    if destination.suffix.lower() != ".hdr":
        raise ValueError("Destination must use Radiance .hdr format")
    color = np.asarray(ambient_color, np.float32)
    if (not np.isfinite(ambient) or ambient < 0 or color.shape != (3,)
            or not np.isfinite(color).all() or np.any(color < 0)):
        raise ValueError("Ambient radiance and RGB color must be finite and nonnegative")
    with source.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if expected_md5 is not None and md5.lower() != expected_md5.lower():
        raise ValueError(f"Source HDRI MD5 mismatch: {source}")
    bgr = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.float32 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Source must decode as a floating-point RGB HDR image")
    rgb = bgr[..., ::-1]
    before = radiance_stats(rgb)
    result = np.ascontiguousarray(rgb + float(ambient) * color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        archive = destination.parent / "archive"
        archive.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        shutil.move(destination, archive / f"{destination.stem}_{timestamp}.hdr")
    if not cv2.imwrite(str(destination), np.ascontiguousarray(result[..., ::-1])):
        raise RuntimeError(f"Failed to encode linear HDR: {destination}")
    decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.float32 or decoded.shape != result.shape:
        raise RuntimeError("Derived HDRI failed floating-point roundtrip validation")
    decoded = decoded[..., ::-1]
    # RGBE uses one exponent per pixel, so normalize error by the brightest
    # channel rather than a near-zero channel with unavoidable quantization.
    error = float(np.max(np.abs(decoded - result) / np.maximum(result.max(axis=2, keepdims=True), 1e-8)))
    if error > 1 / 128 + 1e-6:
        raise RuntimeError(f"Unexpected Radiance HDR roundtrip error: {error:.4%}")
    with destination.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    stats = dict(source=source.name, source_md5=md5, output=destination.name,
                 output_sha256=sha256, output_bytes=destination.stat().st_size,
                 format="linear Radiance RGBE HDR", ambient_radiance=float(ambient),
                 ambient_color_rgb=color.tolist(), before=before, after=radiance_stats(decoded),
                 max_relative_rgbe_quantization_error=error)
    log(f"ambient HDRI: offset {ambient:g}; upward irradiance/pi "
        f"{before['upward_irradiance_over_pi']:.4f}→"
        f"{stats['after']['upward_irradiance_over_pi']:.4f}; "
        f"RGBE roundtrip error {error:.3%}")
    return stats
