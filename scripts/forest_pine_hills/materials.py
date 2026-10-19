"""Offline USD/MDL authoring for FOREST_PINE_HILLS only.

No renderer settings, sensors, or runtime callbacks are created here. The tiled
materials retain source-frequency detail; the only terrain-wide data are smooth
vertex blend weights, not a colour/normal atlas. See MATERIALS_RESEARCH.txt for local 6.0
API provenance and intentional limitations of the sensor material presets.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


# Coupled values intentionally limit renderer-level geometry variants. USD's
# native prototype count alone does not establish RTX memory sharing.
INSTANCE_PALETTE = (
    ((.84, .89, .80), .22), ((.92, .96, .88), .17),
    ((1.00, 1.00, .94), .10), ((1.09, 1.05, .97), .06),
    ((.91, .94, .98), .24), ((1.04, 1.09, .98), .13),
    ((.98, .94, .84), .08), ((1.11, 1.10, 1.04), .05),
)


def override_instance_variation(stage, palette_size=0):
    """Session-overlay A/B helper; caller owns edit target and never must save.

    0 blocks root tint/wetness; 1 gives every root the same pair; 2 or 8 quantizes
    to shared pairs. Vertex terrain masks are deliberately left untouched.
    Returns USD statistics, not a claim about renderer memory consumption.
    This helper targets the initial experimental root-primvar scene; the final
    transform-based shader deliberately ignores these root attributes.
    """
    from pxr import Gf, UsdGeom
    if palette_size not in (0, 1, 2, 8):
        raise ValueError("Use palette_size=0, 1, 2 or 8")
    before = len(stage.GetPrototypes())
    changed = 0
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(("/World/Forest/", "/World/layout/Forest/")):
            continue
        tint = prim.GetAttribute("primvars:pineHillsTint")
        wet = prim.GetAttribute("primvars:pineHillsWetness")
        if not tint or UsdGeom.Primvar(tint).GetInterpolation() != UsdGeom.Tokens.constant:
            continue
        if palette_size == 0:
            tint.Block()
            if wet:
                wet.Block()
        else:
            # Stable path hash avoids running the builder's random sampler again.
            index = int.from_bytes(hashlib.sha256(str(prim.GetPath()).encode()).digest()[:4], "little") % palette_size
            palette = (INSTANCE_PALETTE[1], INSTANCE_PALETTE[7]) if palette_size == 2 else INSTANCE_PALETTE
            colour, moisture = palette[index] if palette_size > 1 else ((1., 1., 1.), .1)
            tint.Set(Gf.Vec3f(*colour))
            if wet:
                wet.Set(moisture)
        changed += 1
    return {"roots_changed": changed, "palette_size": palette_size,
            "usd_prototypes_before": before, "usd_prototypes_after": len(stage.GetPrototypes())}


def write_variation_overlay(asset, output, palette_size=0):
    """Write a small test layer referencing the unchanged scene asset."""
    from pxr import Sdf, Usd
    asset, output = Path(asset).resolve(), Path(output).resolve()
    if asset == output:
        raise ValueError("An A/B overlay must not replace the source USD")
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [str(asset)]
    layer.defaultPrim = "World"
    layer.documentation = "Temporary instance-variation A/B overlay; source scene is unchanged."
    stage = Usd.Stage.Open(layer)
    result = override_instance_variation(stage, palette_size)
    layer.Save()
    return dict(result, path=str(output), source=str(asset), bytes=output.stat().st_size)


def write_transform_variation_overlay(asset, output):
    """Experimental A/B: hash existing instance transforms, shared primvar gain.

    This candidate avoids distinct root primvar values entirely. The MDL standard
    exposes the object's instance transform, and the gain is one constant value
    on each shared prototype. Parent must verify actual RTX appearance and cost.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    result = write_variation_overlay(asset, output, 0)
    output = Path(output).resolve()
    mdl_path = output.with_name("PineHillsTransform.mdl")
    # The measured transform candidate is now the source default. Copy its
    # current bytes verbatim so future trial overlays cannot retain stale code.
    mdl_path.write_text(Path(__file__).with_name("PineHills.mdl").read_text())
    stage = Usd.Stage.Open(str(output))
    changed = 0
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.PrimAllPrimsPredicate):
        if not prim.IsA(UsdShade.Shader) or prim.IsInstanceProxy():
            continue
        shader = UsdShade.Shader(prim)
        if shader.GetSourceAssetSubIdentifier("mdl") in ("Foliage", "Surface"):
            shader.SetSourceAsset(Sdf.AssetPath(str(mdl_path)), "mdl")
            changed += 1
    root = stage.GetPrimAtPath("/World/Prototypes")
    for proto in root.GetAllChildren():
        UsdGeom.PrimvarsAPI(proto).CreatePrimvar("pineHillsVariationStrength", Sdf.ValueTypeNames.Float,
                                                UsdGeom.Tokens.constant).Set(.14)
    stage.GetRootLayer().Save()
    return dict(result, shader_overrides=changed, shader=str(mdl_path),
                bytes=output.stat().st_size, variation="world-origin hash; shared prototype strength")


def terrain_fields(points, trees):
    """Return needle/moss/soil/rock weights plus wetness on a regular XY mesh."""
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64)
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    if len(xs) * len(ys) != len(points):
        raise ValueError("Pine terrain material requires a regular heightfield mesh")
    xi, yi = np.searchsorted(xs, points[:, 0]), np.searchsorted(ys, points[:, 1])
    z = np.empty((len(ys), len(xs)))
    z[yi, xi] = points[:, 2]
    smooth = gaussian_filter(z, max(1.0, .8 / float(np.median(np.diff(xs)))))
    gy, gx = np.gradient(smooth, ys, xs)
    slope = np.arctan(np.hypot(gx, gy))
    curvature = np.gradient(gx, xs, axis=1) + np.gradient(gy, ys, axis=0)
    slope = slope[yi, xi]
    curvature = curvature[yi, xi]
    canopy = np.zeros(len(points))
    tree_xy = np.array([[t["x"], t["y"]] for t in trees], dtype=float).reshape(-1, 2)
    if len(tree_xy):
        distance, _ = cKDTree(tree_xy).query(points[:, :2], k=min(6, len(tree_xy)))
        distance = np.asarray(distance).reshape(len(points), -1)
        canopy = 1.0 - np.exp(-np.exp(-distance ** 2 / (2 * 3.4 ** 2)).sum(axis=1) * .6)
    height = (points[:, 2] - np.percentile(points[:, 2], 5)) / max(np.ptp(points[:, 2]), 1)
    wet = np.clip(.10 + .52 * canopy + .60 * curvature - .20 * height - .25 * slope, 0, .85)
    # Low-frequency non-repeating patches only drive mixing; detail is in the MDL.
    phase = .5 + .20 * np.sin(points[:, 0] * .41 + np.sin(points[:, 1] * .18))
    phase += .12 * np.sin(points[:, 1] * .67 - points[:, 0] * .23)
    rock = np.clip((slope - np.deg2rad(15)) / np.deg2rad(17), 0, .82) * np.clip(height + .35, .2, 1)
    moss = np.clip(.08 + wet * .82 + (phase - .5) * .42 - rock * .65, .03, .72)
    soil = np.clip(.06 + np.maximum(curvature, 0) * .2 + slope * .10, .04, .23)
    # A ragged root/needle strip matches the reference and the scatter module's
    # fern-edge/root placement. This is an ecological opening, not a flat road.
    center = .55 * points[:, 1] + .45 * np.sin(points[:, 1] / 4.5)
    half_width = .75 + .25 * np.sin(.8 * points[:, 1])
    trail = np.exp(-.5 * ((points[:, 0] - center) / half_width) ** 2)
    trail *= np.clip((32 - np.abs(points[:, 1])) / 8, 0, 1)
    moss *= 1 - .96 * trail
    soil = soil * (1 - trail) + .64 * trail
    needles = np.maximum(1 - moss - soil - rock, .02)
    weights = np.stack([needles, moss, soil, rock], axis=1)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32), wet.astype(np.float32)


def nonvisual(material, base):
    """Author both renderer and SimReady forms used by the installed Isaac 6.0."""
    from pxr import Sdf
    if base not in {"bark", "wood", "leaf_grass", "dead_leaf_grass", "dirt", "stone"}:
        raise ValueError(f"Unsupported forest non-visual base: {base}")
    prim = material.GetPrim()
    for suffix, value in (("base", base), ("coating", "none"), ("attributes", "none")):
        prim.CreateAttribute(f"inputs:nonvisual:{suffix}", Sdf.ValueTypeNames.String, custom=True).Set(value)
        prim.CreateAttribute(f"omni:simready:nonvisual:{suffix}", Sdf.ValueTypeNames.Token, custom=True).Set(value)


class Textures:
    def __init__(self, source, directory, relative):
        self.source = Path(source).resolve()
        self.directory = Path(directory)
        self.relative = str(relative)
        self.manifest = json.loads((self.source / "polyhaven_manifest.json").read_text())
        self.records = {}
        self.copied = {}
        for asset, record in self.manifest.items():
            if not record["license"].replace("-", " ").startswith("CC0"):
                raise ValueError(f"Non-CC0 source: {asset}")
            for entry in record["files"]:
                self.records[Path(entry["file"]).name] = entry

    def copy(self, name):
        """Reuse original verified 2K pixels, even if shared builder made a 1K copy."""
        if name in self.copied:
            return self.copied[name]
        entry = self.records.get(Path(name).name)
        if entry is None:
            # Locally generated birch/bilberry maps are not downloaded inputs.
            return self.relative + Path(name).name
        source = (self.source / entry["file"]).resolve()
        if not source.is_relative_to(self.source):
            raise ValueError("Asset path escapes source directory")
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "md5").hexdigest()
        if digest != entry["md5"]:
            raise ValueError(f"MD5 mismatch: {source}")
        destination = self.directory / ("pine_" + source.name)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
        elif hashlib.md5(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing texture differs from verified source: {destination}")
        value = self.relative + destination.name
        self.copied[name] = value
        return value

    def asset_map(self, asset, fragment):
        return self.copy(next(Path(e["file"]).name for e in self.manifest[asset]["files"]
                              if fragment in Path(e["file"]).name))


def _shader(stage, material, mdl_asset, function, values):
    from pxr import Gf, Sdf, UsdShade
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("PineHillsShader"))
    shader.SetSourceAsset(Sdf.AssetPath(mdl_asset), "mdl")
    shader.SetSourceAssetSubIdentifier(function, "mdl")
    for name, value in values.items():
        if isinstance(value, str):
            inp = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
            inp.Set(Sdf.AssetPath(value))
            inp.GetAttr().SetColorSpace("sRGB" if any(k in name for k in ("color", "diffuse")) else "raw")
        elif isinstance(value, bool):
            shader.CreateInput(name, Sdf.ValueTypeNames.Bool).Set(value)
        elif isinstance(value, (tuple, list)):
            shader.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        else:
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(float(value))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                   material.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    return shader


def _prototype_variation(stage):
    """A shared gain per prototype; individual tint comes from MDL transforms."""
    from pxr import Sdf, UsdGeom
    count = 0
    for proto in stage.GetPrimAtPath("/World/Prototypes").GetAllChildren():
        UsdGeom.PrimvarsAPI(proto).CreatePrimvar("pineHillsVariationStrength", Sdf.ValueTypeNames.Float,
                                                UsdGeom.Tokens.constant).Set(.14)
        count += 1
    return count


def author_materials(stage, stand, tex_dir, tex_rel, source, terrain=None):
    """Upgrade the new stage in-place, after scatter and before its final save."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
    textures = Textures(source, tex_dir, tex_rel)
    mdl_source = Path(__file__).with_name("PineHills.mdl")
    mdl_destination = Path(tex_dir) / "PineHills.mdl"
    shutil.copyfile(mdl_source, mdl_destination)
    mdl_asset = str(tex_rel) + mdl_destination.name

    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Terrain/ground"))
    if not ground:
        raise ValueError("Missing /World/Terrain/ground")
    points = np.asarray(ground.GetPointsAttr().Get())
    weights, wetness = terrain_fields(points, stand["trees"])
    pv = UsdGeom.PrimvarsAPI(ground)
    pv.CreatePrimvar("pineHillsCover", Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex).Set(Vt.Vec4fArray.FromNumpy(weights))
    pv.CreatePrimvar("pineHillsWetness", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(wetness))
    ground_values = {}
    for prefix, asset in (("needles", "forrest_ground_03"), ("moss", "forest_leaves_02"), ("rock", "rock_moss_set_01")):
        for key, fragment in (("color", "_diff"), ("normal", "_nor_gl_"), ("rough", "_rough_")):
            ground_values[prefix + "_" + key] = textures.asset_map(asset, fragment)
    ground_values.update(needles_period=2.0, moss_period=2.0, rock_period=2.0, detail_period=.22,
                         detail_enabled=True, diffuse_gain=.65, moss_tint=(.48, .58, .29))
    # One visual shader is shared, but dominant-cover subsets give the lidar an
    # actual per-surface preset rather than making the entire floor generic dirt.
    # Retain superseded subsets as inactive prims. Leaving zero-face subsets
    # active keeps their old OmniPBR material slots in RTX even though USD's
    # face binding queries correctly report the replacement partition.
    for subset in UsdGeom.Subset.GetGeomSubsets(ground, UsdGeom.Tokens.face, "materialBind"):
        subset.GetPrim().SetActive(False)
    triangles = np.asarray(ground.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    dominant = weights[triangles].mean(axis=1).argmax(axis=1)
    ground_materials = []
    for index, (label, base) in enumerate(zip(("needles", "moss", "soil", "rock"),
                                             ("dead_leaf_grass", "leaf_grass", "dirt", "stone"))):
        material = UsdShade.Material.Define(stage, f"/World/Looks/pine_hills_ground_{label}")
        _shader(stage, material, mdl_asset, "Ground", ground_values)
        nonvisual(material, base)
        indices = np.flatnonzero(dominant == index).tolist()
        if indices:
            # An authoring repeat must reactivate the existing replacement prim
            # before creating/setting its properties. Never leave an empty bound
            # subset active, including a cover absent from a small test terrain.
            sub_path = ground.GetPath().AppendChild(f"pine_hills_{label}")
            stage.OverridePrim(sub_path).SetActive(True)
            sub = UsdGeom.Subset.CreateGeomSubset(ground, f"pine_hills_{label}", UsdGeom.Tokens.face,
                                                indices, "materialBind", UsdGeom.Tokens.partition)
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(material)
        ground_materials.append(material)
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_materials[0])

    counts = Counter()
    bark_detail = textures.asset_map("pine_tree_01", "pine_tree_01_bark_nor_gl_")
    moss_detail = textures.asset_map("forest_leaves_02", "_nor_gl_")
    # Traverse class prototypes explicitly; ordinary stage traversal skips them.
    materials = [UsdShade.Material(p) for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.PrimAllPrimsPredicate)
                 if p.IsA(UsdShade.Material) and not p.IsInstanceProxy() and not p.GetPath().name.startswith("pine_hills_ground")]
    for material in materials:
        old = UsdShade.Shader(stage.GetPrimAtPath(material.GetPath().AppendChild("Shader")))
        if not old:
            continue
        name = material.GetPath().name.lower()
        if name.startswith("floor_"):
            continue  # obsolete shared-builder ground subsets are inactive
        is_foliage = any(x in name for x in ("twig", "fern", "grass", "leaf")) and "dead" not in name
        is_moss = "moss" in name and "rock" not in name
        is_rock = "rock" in name
        values = {}
        for old_name, new_name in (("diffuse_texture", "diffuse_image"), ("normalmap_texture", "normal_image"),
                                   ("ORM_texture", "orm_image"), ("reflectionroughness_texture", "rough_image")):
            inp = old.GetInput(old_name)
            value = inp.Get() if inp else None
            if value and value.path and (new_name != "rough_image" or not is_foliage):
                values[new_name] = textures.copy(Path(value.path).name)
        tint = old.GetInput("diffuse_tint")
        values["tint"] = tuple(tint.Get()) if tint and tint.Get() is not None else (1., 1., 1.)
        if is_foliage and "diffuse_image" in values:
            values["translucency"] = .30 if "fern" in name else .20
            _shader(stage, material, mdl_asset, "Foliage", values)
            base = "leaf_grass"
            counts["thin_walled_foliage_materials"] += 1
        else:
            values["detail_normal"] = moss_detail if is_moss else bark_detail
            values["detail_weight"] = .18 if is_moss else (.07 if is_rock else .12)
            values["sheen"] = .22 if is_moss else 0.
            constant = old.GetInput("diffuse_color_constant")
            if constant and constant.Get() is not None:
                values["constant_color"] = tuple(constant.Get())
            _shader(stage, material, mdl_asset, "Surface", values)
            base = "leaf_grass" if is_moss else "stone" if is_rock else "bark" if any(k in name for k in ("bark", "trunk", "roots")) else "wood"
            counts["detailed_surface_materials"] += 1
        nonvisual(material, base)
        counts["nonvisual_" + base] += 1
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Prototypes"), Usd.PrimAllPrimsPredicate):
        if prim.IsA(UsdGeom.Mesh) and any(x in prim.GetName() for x in ("foliage", "litter", "sapling")):
            UsdGeom.Mesh(prim).CreateDoubleSidedAttr(True)
    variation_count = _prototype_variation(stage)
    report = {
        "mdl": mdl_asset, "mdl_sha256": hashlib.sha256(mdl_source.read_bytes()).hexdigest(),
        "ground_mapping": "world-space triplanar; 2 m repeating 2K source maps; smooth vertex blend weights",
        "ground_source_texel_mm": 2000 / 2048, "ground_detail_period_m": .22,
        "ground_diffuse_gain": .65, "ground_moss_tint": [.48, .58, .29],
        "ground_blend_inputs": ["slope", "curvature", "height", "canopy occlusion", "spatial variation", "ragged root trail"],
        "mean_cover": dict(zip(("needles", "moss", "soil", "rock"), map(float, weights.mean(axis=0)))),
        "wetness_range": [float(wetness.min()), float(wetness.max())],
        "variation": "per-instance object-to-world origin hash; shared prototype strength primvar",
        "prototype_strength_primvars": variation_count, "unique_instance_color_primvars": 0,
        "verified_2k_textures": len(textures.copied),
        "nonvisual_materials": "Isaac 6.0 presets; species-specific measured spectral calibration not claimed",
        "detail_distance_filter": "RTX texture mip filtering; no per-camera callback or global settings",
        **dict(counts),
    }
    stand.setdefault("pine_hills", {})["materials"] = report
    return report
