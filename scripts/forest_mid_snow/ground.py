"""Bake a moss floor with coherent, thin melting-snow islands (no Kit startup).

The atlas is ordinary OmniPBR: one colour, tangent normal and roughness map.
Its UVs cover the entire stand, so snow boundaries never follow mesh triangles
or repeat with the source texture tiles. Ground geometry/collision is unchanged.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def srgb_to_linear(value):
    value = np.asarray(value, np.float32)
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)


def normalized(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def _smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


@dataclass
class SnowMask:
    """Image-oriented fields: row zero is world +Y; column zero is world -X."""
    values: np.ndarray
    area: float
    litter: np.ndarray
    variation: np.ndarray

    def sample(self, x, y, field=None):
        from scipy.ndimage import map_coordinates
        x, y = np.broadcast_arrays(np.asarray(x, np.float32), np.asarray(y, np.float32))
        data = self.values if field is None else field
        n = len(data)
        coords = np.array([(self.area / 2 - y) * (n - 1) / self.area,
                           (x + self.area / 2) * (n - 1) / self.area])
        return map_coordinates(data, coords.reshape(2, -1), order=1,
                               mode="nearest").reshape(x.shape)


@dataclass
class GroundResult:
    material: object
    summary: dict
    snow_mask: SnowMask


def make_snow_mask(*, area=100., seed=2026, snow_fraction=.23, tree_xy=(),
                   terrain_z=None, size=1025):
    """Calibrate a continuous mask; trees and exposed hummocks melt first.

    ``terrain_z`` follows FI's X-major grid, unlike the image-oriented output.
    The requested fraction is the mean snow blend, including feathered edges.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.spatial import cKDTree
    if not 0 <= snow_fraction <= 1:
        raise ValueError("snow_fraction must be between zero and one")
    if area <= 0 or size < 17:
        raise ValueError("area must be positive and mask size at least 17")
    rng = np.random.default_rng(seed + 4107)
    spacing = area / (size - 1)

    def noise(scale):
        f = gaussian_filter(rng.normal(size=(size, size)).astype(np.float32),
                            max(.65, scale / spacing), mode="reflect")
        return (f - f.mean()) / max(float(f.std()), 1e-6)

    # Broad islands, lobed boundaries, small broken melt holes. No checkerboard
    # or thresholding per terrain triangle, and no uniform snow near spawn.
    score = .45 * noise(1.4) + .30 * noise(.55) + .20 * noise(.18) + .10 * noise(.06)
    ys = np.linspace(area / 2, -area / 2, size, dtype=np.float32)
    xs = np.linspace(-area / 2, area / 2, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    distance = np.full((size, size), 100., np.float32)
    trees = np.asarray(tree_xy, np.float32).reshape(-1, 2)
    if len(trees):
        distance[:] = cKDTree(trees).query(np.c_[X.ravel(), Y.ravel()])[0].reshape(X.shape)
        score -= .65 * np.exp(-.5 * (distance / .75) ** 2)
    if terrain_z is not None:
        grid = np.asarray(terrain_z, np.float32)
        coords = np.array([(X + area / 2) * (len(grid) - 1) / area,
                           (Y + area / 2) * (len(grid) - 1) / area])
        heights = map_coordinates(grid, coords, order=1, mode="nearest")
        relief = heights - gaussian_filter(heights, 3 / spacing)
        score -= .24 * np.clip(relief / max(float(relief.std()), .025), -2, 2)
    # Persistently bare immediate root collars; no tree mesh changes required.
    available = _smoothstep((distance - .40) / .35)
    max_fraction = float(available.mean())
    # The CLI permits a fully snowy floor. Relax bare root collars only when
    # requested coverage exceeds the capacity of the default melting pattern;
    # ordinary mid-snow settings retain the exact same collar exclusions.
    if snow_fraction > max_fraction:
        excess = (snow_fraction - max_fraction) / max(1 - max_fraction, 1e-8)
        available += (1 - available) * (1 - (1 - excess) ** 2)
    lo, hi = float(score.min() - 1), float(score.max() + 1)
    for _ in range(25):
        threshold = .5 * (lo + hi)
        mask = _smoothstep((score - threshold) / .16 + .5) * available
        if float(mask.mean()) > snow_fraction:
            lo = threshold
        else:
            hi = threshold
    mask = _smoothstep((score - .5 * (lo + hi)) / .16 + .5) * available
    if snow_fraction == 0:
        mask.fill(0)
    elif snow_fraction == 1:
        mask.fill(1)
    litter = (.24 * np.exp(-.5 * (distance / .85) ** 2)
              + .055 * _smoothstep(noise(1.8) - .25)) * (1 - mask)
    variation = np.clip(1 + .07 * noise(5) + .035 * noise(.7), .83, 1.14)
    return SnowMask(mask.astype(np.float32), float(area), litter.astype(np.float32),
                    variation.astype(np.float32))


def _verified_maps(source, asset):
    """Read data files only, validating the three inputs against the saved API MD5s."""
    source = Path(source).resolve()
    manifest = json.loads((source / "polyhaven_manifest.json").read_text())
    record = manifest[asset]
    if not record["license"].replace("-", " ").startswith("CC0"):
        raise ValueError(f"Non-CC0 input: {asset}")
    selected = {}
    for entry in record["files"]:
        name = Path(entry["file"]).name.lower()
        kind = ("diffuse" if "_diff_" in name or "_diffuse_" in name else
                "normal" if "_nor_gl_" in name else "roughness" if "_rough_" in name else None)
        if kind is None:
            continue
        path = (source / entry["file"]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("Texture path escapes asset directory")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != entry["md5"]:
            raise ValueError(f"MD5 mismatch: {path}")
        selected[kind] = path
    if set(selected) != {"diffuse", "normal", "roughness"}:
        raise ValueError(f"Incomplete PBR maps for {asset}")
    return selected


def _read_map(path, kind, size):
    from PIL import Image
    with Image.open(path) as image:
        data = np.asarray(image.convert("L" if kind == "roughness" else "RGB"), np.float32) / 255
    if kind == "diffuse":
        data = srgb_to_linear(data)
    elif kind == "normal":
        data = data * 2 - 1
    # Downsample in the map's actual value space before tiling to avoid severe
    # aliasing of tiny needles. In particular, do not average sRGB as radiance.
    channels = [data] if data.ndim == 2 else [data[..., i] for i in range(3)]
    resized = [np.asarray(Image.fromarray(c).resize((size, size), Image.Resampling.LANCZOS))
               for c in channels]
    result = resized[0] if data.ndim == 2 else np.stack(resized, axis=-1)
    return normalized(result) if kind == "normal" else np.clip(result, 0, 1)


def _tile_sample(tile, x, y, period):
    from scipy.ndimage import map_coordinates
    x, y = np.broadcast_arrays(x, y)
    coords = np.array([np.remainder(-y / period, 1) * len(tile) - .5,
                       np.remainder(x / period, 1) * len(tile) - .5])
    if tile.ndim == 2:
        return map_coordinates(tile, coords, order=1, mode="grid-wrap")
    return np.stack([map_coordinates(tile[..., c], coords, order=1, mode="grid-wrap")
                     for c in range(3)], axis=-1)


def bake_atlas(output, snow_mask, maps, *, resolution=8192, stripe=128, log=print):
    """Bake verified maps in bounded-memory stripes; return texture filenames/stats.

    ``maps`` is {moss/snow/needles: {diffuse/normal/roughness: Path}}. The 8k
    atlas resolves ~12 mm over 100 m; a 4k option reduces texture memory fourfold.
    """
    from PIL import Image
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resolution < 64 or stripe < 1:
        raise ValueError("resolution must be at least 64; stripe must be positive")
    periods = {"moss": 3., "snow": 2., "needles": 2.}
    tiles = {name: {kind: _read_map(path, kind, max(8, int(np.ceil(resolution * periods[name] / snow_mask.area))))
                    for kind, path in files.items()} for name, files in maps.items()}
    atlas = {kind: np.empty((resolution, resolution) + (() if kind == "roughness" else (3,)), np.uint8)
             for kind in ("diffuse", "normal", "roughness")}
    pix = snow_mask.area / resolution
    xs = -snow_mask.area / 2 + (np.arange(resolution, dtype=np.float32) + .5) * pix
    # A two-centimetre melt edge is encoded only as tangent-normal relief. This
    # does not move the terrain after placement or alter any physics/GT heights.
    dy, dx = np.gradient(.018 * snow_mask.values ** 1.5, snow_mask.area / (len(snow_mask.values) - 1))
    dy = -dy  # image rows descend in world Y
    coverage = 0.
    for row in range(0, resolution, stripe):
        end = min(row + stripe, resolution)
        ys = snow_mask.area / 2 - (np.arange(row, end, dtype=np.float32) + .5) * pix
        X, Y = np.broadcast_arrays(xs[None, :], ys[:, None])
        snow = snow_mask.sample(X, Y)
        litter = snow_mask.sample(X, Y, snow_mask.litter)
        variation = snow_mask.sample(X, Y, snow_mask.variation)
        coverage += float(snow.sum(dtype=np.float64))
        for kind in atlas:
            moss = _tile_sample(tiles["moss"][kind], X, Y, periods["moss"])
            needles = _tile_sample(tiles["needles"][kind], X, Y, periods["needles"])
            winter = _tile_sample(tiles["snow"][kind], X, Y, periods["snow"])
            s = snow if kind == "roughness" else snow[..., None]
            l = litter if kind == "roughness" else litter[..., None]
            if kind == "diffuse":
                # Preserve the olive moss photograph; cool yellow/autumn tones
                # mildly. Settled melting snow is dimmer than fresh powder.
                moss *= np.array([.32, .48, .28], np.float32)
                floor = (moss * (1 - l) + needles * l) * variation[..., None]
                winter = np.clip(winter * np.array([1.35, 1.40, 1.45], np.float32), 0, .90)
                mixed = linear_to_srgb(floor * (1 - s) + winter * s)
            elif kind == "normal":
                mixed = normalized(normalized(moss * (1 - l) + needles * l) * (1 - s) + winter * s)
                mixed[..., 0] -= snow_mask.sample(X, Y, dx) * mixed[..., 2]
                mixed[..., 1] -= snow_mask.sample(X, Y, dy) * mixed[..., 2]
                mixed = normalized(mixed) * .5 + .5
            else:
                mixed = (moss * (1 - l) + needles * l) * (1 - s) + winter * s
            atlas[kind][row:end] = np.rint(np.clip(mixed, 0, 1) * 255).astype(np.uint8)
        if row == 0 or end == resolution or end % 2048 == 0:
            log(f"ground atlas rows {end}/{resolution}")
    filenames = {}
    for kind, data in atlas.items():
        name = f"mid_snow_ground_{kind}_{resolution}.png"
        Image.fromarray(data).save(output / name, compress_level=3)
        filenames[kind] = name
    # Small diagnostic mask is also useful to audit snow coverage offline.
    mask_name = "mid_snow_ground_mask.png"
    Image.fromarray(np.rint(snow_mask.values * 255).astype(np.uint8)).save(output / mask_name)
    return dict(files=filenames, resolution=resolution, texel_size_m=pix,
                snow_fraction=coverage / resolution ** 2, mask=mask_name,
                texture_bytes=sum((output / name).stat().st_size for name in filenames.values()),
                source_tile_metres=periods, snow_relief="normal map only; terrain unchanged")


def apply_patchy_ground(stage, stand, tex_dir, tex_rel, forest_source, snow_source,
                       log=print, *, snow_fraction=.23, resolution=8192):
    """Bind the atlas to FI's existing collider mesh; return mask for placements."""
    from pxr import Sdf, UsdGeom, UsdShade, Vt
    from scene import define_material
    terrain = stand["terrain"]
    mask = make_snow_mask(area=float(terrain["area"]), seed=stand["summary"]["seed"],
                          snow_fraction=snow_fraction, terrain_z=terrain["z"],
                          tree_xy=[(tree["x"], tree["y"]) for tree in stand["trees"]])
    maps = {"moss": _verified_maps(forest_source, "forest_leaves_02"),
            "needles": _verified_maps(forest_source, "forrest_ground_03"),
            "snow": _verified_maps(snow_source, "snow_02")}
    stats = bake_atlas(tex_dir, mask, maps, resolution=resolution, log=log)
    spec = dict(diffuse=stats["files"]["diffuse"], normal=stats["files"]["normal"],
                rough_tex=stats["files"]["roughness"], rough=.88)
    material = define_material(stage, Sdf.Path("/World/Looks/mid_snow_ground"), spec, tex_rel)
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Terrain/ground"))
    points = np.asarray(ground.GetPointsAttr().Get())
    uv = (points[:, :2] + mask.area / 2) / mask.area
    primvar = UsdGeom.PrimvarsAPI(ground).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                   UsdGeom.Tokens.vertex)
    primvar.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, np.float32)))
    if primvar.IsIndexed():
        primvar.BlockIndices()
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(material)
    for subset in UsdGeom.Subset.GetAllGeomSubsets(ground):
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    stats.update(requested_snow_fraction=snow_fraction, moss_asset="forest_leaves_02",
                 snow_asset="snow_02", litter_asset="forrest_ground_03",
                 terrain_geometry_changed=False)
    log(f"ground: {stats['snow_fraction']:.1%} melting snow, {resolution}² atlas; moss dominant")
    return GroundResult(material, stats, mask)
