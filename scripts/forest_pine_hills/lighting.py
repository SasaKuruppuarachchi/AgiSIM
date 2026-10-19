"""A single HDR sky and an aligned, energy-matched 0.53-degree sun.

USD 25.05 changed DomeLight orientation and DistantLight normalization. These
lights explicitly opt into that interpretation. Latlong directions follow the
OpenEXR/USD convention, rotated from Y-up to this world's Z-up; a yaw rotation
places the detected solar lobe at the requested world azimuth without tilting
the horizon. The HDR solar excess is moved to the DistantLight, not counted twice.

Source HDRIs are verified CC0 data, never modified. Image decoding uses OpenCV;
the derived sky remains linear floating point Radiance HDR (no tone mapping).
"""
import hashlib
import math
from pathlib import Path

import numpy as np

SUN_DIAMETER_DEG = 0.53
DEFAULT_SKY_INTENSITY = 6000.0
DEFAULT_SUN_AZIMUTH_DEG = 225.0


def latlong_directions(height, width):
    """OpenEXR latlong pixel-centre directions in the dome's local Y-up frame."""
    latitude = np.pi / 2 - (np.arange(height, dtype=np.float64) + .5) * np.pi / height
    longitude = np.pi - (np.arange(width, dtype=np.float64) + .5) * 2 * np.pi / width
    cos_lat = np.cos(latitude)[:, None]
    return np.stack((cos_lat * np.sin(longitude)[None, :],
                     np.broadcast_to(np.sin(latitude)[:, None], (height, width)),
                     cos_lat * np.cos(longitude)[None, :]), axis=-1)


def split_sun(rgb, *, radius_degrees=1.5):
    """Separate the compact upper-hemisphere solar excess from its nearby sky.

    The annular median estimates the underlying sky. The removed RGB radiance
    is integrated against solid angle and cosine to yield the normal-incidence
    illuminance represented by a normalized DistantLight. The 0.53-degree ideal
    solar disk concentrates a source capture's finite-resolution solar lobe,
    preserving its integrated energy and colour. A low-contrast/cloud-obscured
    sky is rejected instead of inventing a sun direction from arbitrary clouds.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[1] != 2 * rgb.shape[0]
            or not np.isfinite(rgb).all() or np.any(rgb < 0)):
        raise ValueError("Expected finite, nonnegative 2:1 RGB latlong radiance")
    if not math.isfinite(radius_degrees) or not .3 <= radius_degrees <= 5:
        raise ValueError("Solar extraction radius must be between 0.3 and 5 degrees")
    height, width, _ = rgb.shape
    directions = latlong_directions(height, width)
    luminance = rgb @ np.array([.2126, .7152, .0722], np.float32)
    peak = np.unravel_index(np.argmax(luminance[:height // 2]), (height // 2, width))
    direction = directions[peak]
    angles = np.arccos(np.clip(directions @ direction, -1, 1))
    compact = (angles < np.deg2rad(radius_degrees)) & (luminance > luminance[peak] * .5)
    direction = np.sum(directions[compact] * luminance[compact, None], axis=0)
    direction /= np.linalg.norm(direction)
    angles = np.arccos(np.clip(directions @ direction, -1, 1))
    annulus = (angles > np.deg2rad(radius_degrees * 1.5)) & (angles < np.deg2rad(radius_degrees * 2.5))
    if not np.any(annulus):
        raise ValueError("HDRI resolution is too small to isolate a solar lobe")
    background = np.median(rgb[annulus], axis=0)
    contrast = float(luminance[peak] / max(float(background @ [.2126, .7152, .0722]), 1e-8))
    if contrast < 3:
        raise ValueError(f"HDRI has no resolved solar hotspot (peak/nearby sky = {contrast:.2f})")
    blend = np.clip((np.deg2rad(radius_degrees) - angles) / np.deg2rad(radius_degrees * .4), 0, 1)
    blend = blend * blend * (3 - 2 * blend)
    excess = np.maximum(rgb - background, 0) * blend[..., None]
    sky = np.maximum(rgb - excess, 0).astype(np.float32)
    solid_angle = np.sqrt(np.maximum(1 - directions[..., 1] ** 2, 0)) * (np.pi / height) * (2 * np.pi / width)
    projection = np.maximum(directions @ direction, 0)
    irradiance = np.sum(excess * (solid_angle * projection)[..., None], axis=(0, 1))
    if not np.all(irradiance > 0):
        raise ValueError("Solar lobe has no positive integrated RGB energy")
    metadata = dict(source_peak_pixel_rc=list(map(int, peak)),
                    source_peak_luminance=float(luminance[peak]),
                    source_sun_direction_y_up=direction.tolist(),
                    sun_elevation_deg=math.degrees(math.asin(float(direction[1]))),
                    solar_extraction_radius_deg=radius_degrees,
                    source_solar_normal_irradiance_rgb=irradiance.tolist(),
                    source_sky_at_sun_rgb=background.tolist(),
                    solar_peak_to_sky_ratio=contrast)
    return sky, metadata


def prepare_sky(source, destination, *, expected_md5):
    """Verify and decode a local HDR, then write the derived sun-separated sky."""
    import cv2
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.suffix.lower() != ".hdr":
        raise ValueError("Write a separate derived .hdr file; preserve the source")
    digest = hashlib.md5(source.read_bytes()).hexdigest()
    if digest != expected_md5.lower():
        raise ValueError(f"HDRI MD5 mismatch: {source}")
    bgr = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if bgr is None or bgr.dtype != np.float32:
        raise ValueError(f"Unable to decode floating-point HDRI: {source}")
    sky, metadata = split_sun(bgr[..., ::-1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # The shared builder archives the previous texture directory first.
        raise FileExistsError(f"Archive the previous build before writing {destination}")
    if not cv2.imwrite(str(destination), np.ascontiguousarray(sky[..., ::-1])):
        raise RuntimeError(f"Failed to write derived HDRI: {destination}")
    decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != np.float32 or decoded.shape != sky.shape:
        raise RuntimeError("Derived HDRI roundtrip failed")
    error = np.abs(decoded[..., ::-1] - sky) / np.maximum(sky.max(axis=2, keepdims=True), 1e-8)
    if float(error.max()) > 1 / 128 + 1e-6:
        raise RuntimeError("Derived HDRI has excessive RGBE quantization error")
    metadata.update(source=source.name, source_md5=digest, output=destination.name,
                    output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                    output_bytes=destination.stat().st_size,
                    max_relative_rgbe_quantization_error=float(error.max()))
    return metadata


def build_lighting(stage, hdri_source, tex_dir, tex_rel, *, expected_md5,
                   sky_intensity=DEFAULT_SKY_INTENSITY,
                   sun_azimuth_deg=DEFAULT_SUN_AZIMUTH_DEG, solar_gain=1.0):
    """Replace the new scene's shared-builder lights; return full provenance.

    ``solar_gain=1`` preserves the captured sky/sun energy ratio. A different
    gain is an explicitly reported artistic relighting adjustment, not a claim
    of a calibrated capture. No additional fill or unshadowed light is created.
    ``tex_rel`` is the new scene texture-directory prefix relative to its USD.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    if (not math.isfinite(sky_intensity) or sky_intensity <= 0
            or not math.isfinite(solar_gain) or solar_gain <= 0
            or not math.isfinite(sun_azimuth_deg)):
        raise ValueError("Lighting controls must be finite; intensity/gain must be positive")
    destination = Path(tex_dir) / "pine_hills_sky_without_sun.hdr"
    metadata = prepare_sky(hdri_source, destination, expected_md5=expected_md5)
    local = np.asarray(metadata["source_sun_direction_y_up"])
    base = np.array([local[0], -local[2], local[1]])  # Rx(+90 degrees)
    yaw = math.radians(sun_azimuth_deg) - math.atan2(base[1], base[0])
    c, s = math.cos(yaw), math.sin(yaw)
    # Column-vector Y-up -> Z-up followed by a Z yaw. USD uses row vectors.
    rotation = np.array([[c, 0, s], [s, 0, -c], [0, 1, 0]], dtype=np.float64)
    sun_direction = rotation @ local
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.T
    UsdGeom.Scope.Define(stage, "/World/Lights")
    for prim in stage.GetPrimAtPath("/World/Lights").GetChildren():
        if prim.GetName() not in ("Sky", "Sun"):
            prim.SetActive(False)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    dome.GetPrim().SetActive(True)
    dome.CreateIntensityAttr(float(sky_intensity))
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0))
    dome.CreateTextureFileAttr(Sdf.AssetPath(tex_rel + destination.name))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    UsdGeom.Xformable(dome).MakeMatrixXform().Set(Gf.Matrix4d(tuple(map(tuple, matrix))))
    dome.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
    dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(True)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.GetPrim().SetActive(True)
    irradiance = np.asarray(metadata["source_solar_normal_irradiance_rgb"]) * sky_intensity * solar_gain
    intensity = float(irradiance.max())
    color = irradiance / intensity
    sun.CreateIntensityAttr(intensity)
    sun.CreateNormalizeAttr(True)
    sun.CreateExposureAttr(0.0)
    sun.CreateAngleAttr(SUN_DIAMETER_DEG)
    sun.CreateColorAttr(Gf.Vec3f(*map(float, color)))
    sun.GetPrim().CreateAttribute("omni:rtx:usdluxVersion", Sdf.ValueTypeNames.Int).Set(2505)
    orientation = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*map(float, -sun_direction)))
    UsdGeom.Xformable(sun).MakeMatrixXform().Set(Gf.Matrix4d().SetRotate(orientation))
    metadata.update(sky_intensity=float(sky_intensity), sun_intensity_lux=intensity,
                    sun_color_rgb=color.tolist(), sun_direction_world=sun_direction.tolist(),
                    sun_azimuth_deg=float(sun_azimuth_deg), dome_yaw_deg=math.degrees(yaw),
                    solar_gain=float(solar_gain), sun_diameter_deg=SUN_DIAMETER_DEG,
                    dome_count=1, shadow_linking="default geometric shadows",
                    usdlux_version=2505, sun_energy_moved_from_hdri=True)
    return metadata
