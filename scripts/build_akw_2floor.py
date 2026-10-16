#!/usr/bin/env python3
"""Build the sim-ready two-floor AKW environment.

Takes the sim-ready level-1 stage (akw_lvl1_climb.usd) and replaces its building
content with the designer's two-floor export (Enrich_two_floor.usdc), so level 1
keeps the exact world frame used by spawn points, benchmark tracks and planners.

What the build does:
  * fits the rigid transform designer-frame -> level-1 frame from meshes common
    to both files (aborts unless >= 10 meshes agree to < 1 mm, no scaling);
  * copies the designer scene under /World/akw with that transform, dropping the
    game/export leftovers (Player rig on the spawn point, empty ROV and
    Global_Volume, the Blender sun and dome);
  * re-adds the obstacles added only in akw_lvl1_climb (extra locker bodies and
    desk) at their original world poses;
  * re-materials the surfaces whose designer textures were never delivered with
    OmniPBR and Isaac Sim asset-library textures, projected in world space at a
    fixed tile size (the designer UVs stretch 0..1 over each whole object), and
    keeps the level-1 fixture emission;
  * replaces the Blender point lights (point sources inside closed fixture
    meshes, ~1 nit) with sphere lights just below each fixture;
  * adds static colliders: an exact triangle mesh (approximation "none") on every mesh,
    so physics matches the geometry the RTX lidar sees (add_static_colliders).

Runs on the USD libraries bundled with Isaac Sim (no Kit startup needed):
    python3 scripts/build_akw_2floor.py [--isaacsim ~/isaacsim] [--assets $ISAACSIM_ASSET_ROOT]

Textures are downscaled copies written next to the output under textures/akw/.
"""

import argparse
import glob
import os
import sys

ENVS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "extensions", "pegasus.simulator",
                        "pegasus", "simulator", "assets", "Worlds", "isaac_envs")

# Designer prims that are not part of the environment.
DROP_DESIGN_PRIMS = ("Player", "ROV", "Global_Volume", "Directional_Light", "env_light")

# Meshes that exist only in akw_lvl1_climb: (mesh path in the level-1 file, sibling in the
# designer file whose material the copy reuses).
CLIMB_ADDITIONS = (
    ("uploads_files_887512_locker/cadena_locker_body_01", "uploads_files_887512_locker/cadena_locker_body_1"),
    ("uploads_files_887512_locker/cadena_locker_body_02", "uploads_files_887512_locker/cadena_locker_body_1"),
    ("Student_Study_Set__AMS04___2_/default_01", "Student_Study_Set__AMS04___2_/default_2"),
)


# Designer material -> level-1 material (base name) that replaces it (brightened fixture emission).
MATERIALS_FROM_BASE = {"light": "light"}
MISSING_TEXTURE_COLOR = (0.55, 0.55, 0.55)  # diffuse used where a texture file is missing

# Isaac Sim asset-library texture packs, relative to the asset root (.../Assets/Isaac/6.0).
RIVERMARK = "Isaac/Environments/Outdoor/Rivermark/dsready_content/nv_content/common_assets/shared_textures"
WAREHOUSE = "Isaac/Environments/Modular_Warehouse/Props/Source/textures"
WAREHOUSE_NEW = "Isaac/Environments/Modular_Warehouse_New/Source/Textures"

# Designer material -> OmniPBR replacement. tile: metres covered by one texture repeat.
TEXTURED_MATERIALS = {
    # Walls also covers the floors and ceilings of the shell meshes, so it must read in any orientation.
    "Walls": dict(albedo=(RIVERMARK, "concrete_clean_a.png"), normal=(RIVERMARK, "concrete_clean_n.png"),
                  tile=2.5, roughness=0.8, tint=(1.0, 0.96, 0.88), brightness=1.35),
    "wall_small": dict(albedo=(RIVERMARK, "opaque_concrete_panels_building_110_a.png"),
                       normal=(RIVERMARK, "opaque_concrete_panels_building_110_n.png"), tile=3.0, roughness=0.8,
                       brightness=1.2),
    "Ground": dict(albedo=(RIVERMARK, "concrete_slab_01_a.png"), normal=(RIVERMARK, "concrete_slab_01_n.png"),
                   tile=3.0, roughness=0.7, brightness=1.2),
    "cube_concrete": dict(albedo=(WAREHOUSE, "t_concrete_foundation_Albedo.png"),
                          normal=(WAREHOUSE, "t_concrete_foundation_Normal.png"), tile=2.0, roughness=0.85),
    "vent": dict(albedo=(RIVERMARK, "metal_steel_galvanized_01_a.png"),
                 normal=(RIVERMARK, "metal_steel_galvanized_01_n.png"), tile=1.0, roughness=0.45, metallic=1.0),
    "Door": dict(albedo=(WAREHOUSE_NEW, "t_metal_painted_rough_b_Albedo.png"),
                 normal=(WAREHOUSE_NEW, "t_metal_painted_rough_b_Normal.png"), tile=1.0, roughness=0.5,
                 tint=(0.55, 0.62, 0.66)),
    "pump": dict(albedo=(WAREHOUSE_NEW, "T_MetalPainted_Gray_Worn_Albedo.png"),
                 normal=(WAREHOUSE_NEW, "t_metalpainted_gray_metallic_a_normal.png"), tile=1.0, roughness=0.45,
                 tint=(0.35, 0.5, 0.85)),
    "red": dict(albedo=(WAREHOUSE_NEW, "t_metal_painted_rough_b_Albedo.png"),
                normal=(WAREHOUSE_NEW, "t_metal_painted_rough_b_Normal.png"), tile=1.0, roughness=0.45,
                tint=(0.85, 0.12, 0.08)),
    "drone": dict(albedo=(WAREHOUSE_NEW, "T_GlassLens_Colored_Glossy_B_Albedo.png"), tile=0.2, roughness=0.1),
}
TEXTURE_MAX_SIZE = 2048  # px; library textures are up to 4k

LIGHT_INTENSITY = 1.0e5  # nits, sphere light below each wall/low fixture (~2 m above level 1)
HIGH_BAY_INTENSITY = 5.0e5  # nits, level-2 roof fixtures hang ~15 m above the floor
LEVEL2_FLOOR_Z = 15.5  # m, level-1 frame
LIGHT_RADIUS = 0.08  # m
LIGHT_DROP = 0.12  # m below the fixture's lowest point


def _bootstrap_usd(isaacsim):
    """Re-exec under Isaac Sim's bundled Python with its USD libraries on the path."""
    if os.environ.get("_AKW_BUILD_BOOTSTRAPPED"):
        return
    ext = os.path.join(isaacsim, "extscache")
    libs = [glob.glob(os.path.join(ext, pattern))[0] for pattern in
            ("omni.usd.libs-*", "omni.usd.schema.physx-*")]
    numpy_dir = glob.glob(os.path.join(ext, "omni.kit.pip_archive-*", "pip_prebundle"))[0]
    env = dict(os.environ, _AKW_BUILD_BOOTSTRAPPED="1",
               PYTHONPATH=os.pathsep.join(libs + [numpy_dir]),
               LD_LIBRARY_PATH=os.pathsep.join([os.path.join(p, "bin") for p in libs]
                                               + [os.environ.get("LD_LIBRARY_PATH", "")]))
    python = os.path.join(isaacsim, "kit", "python", "bin", "python3")
    # Re-exec the entry script (also when imported by scripts/add_akw_c_colliders.py).
    os.execve(python, [python, os.path.abspath(sys.argv[0])] + sys.argv[1:], env)


def add_static_colliders(stage, root):
    """Exact triangle-mesh collider on every mesh under root; returns the mesh count. Idempotent.

    Every collider in these stages is static (the drone is the only dynamic body), so PhysX
    accepts triangle meshes for all of them. Convex hulls of concave props (turbine, cylinder
    racks, desks) were up to 2.6 m larger than the visible surface: invisible obstacles that the
    lidar map shows as free (super_parity_results/akw2f_scene_01, hull_excess / hull_vs_render).
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
            count += 1
    return count


def fit_design_to_level1(base_stage, design_stage, base_root, design_root):
    """Rigid transform (R, t) with p_level1 = R @ p_design + t, from vertex-matched meshes."""
    import numpy as np
    from pxr import Usd, UsdGeom

    xa, xb = UsdGeom.XformCache(), UsdGeom.XformCache()

    def world_points(cache, prim):
        pts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        return (np.c_[pts, np.ones(len(pts))] @ np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]

    pairs = []
    for top in base_stage.GetPrimAtPath(base_root).GetChildren():
        other = design_stage.GetPrimAtPath(design_root.AppendChild(top.GetName()))
        if not other:
            continue
        design_meshes = [p for p in Usd.PrimRange(other) if p.IsA(UsdGeom.Mesh)]
        for mesh in (p for p in Usd.PrimRange(top) if p.IsA(UsdGeom.Mesh)):
            a = world_points(xa, mesh)
            same = [p for p in design_meshes if len(UsdGeom.Mesh(p).GetPointsAttr().Get() or ()) == len(a)]
            if len(a) >= 8 and len(same) == 1:
                pairs.append((a, world_points(xb, same[0])))

    def kabsch(a, b):
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = np.linalg.svd((b - cb).T @ (a - ca))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        return r, ca - r @ cb

    # Per-pair fits; keep the consensus (moved objects and symmetric meshes disagree).
    fits = [kabsch(a, b) for a, b in pairs]
    keys = [tuple(np.round(np.r_[r.ravel(), t], 2)) for r, t in fits]
    consensus = max(set(keys), key=keys.count)
    inliers = [p for p, k in zip(pairs, keys) if k == consensus]
    if len(inliers) < 10:
        raise SystemExit(f"transform fit: only {len(inliers)} meshes agree")
    a = np.concatenate([p[0] for p in inliers])
    b = np.concatenate([p[1] for p in inliers])
    r, t = kabsch(a, b)
    residual = np.linalg.norm(b @ r.T + t - a, axis=1).max()
    if residual > 1e-3 or abs(np.linalg.det(r) - 1.0) > 1e-6:
        raise SystemExit(f"transform fit: residual {residual:.4f} m")
    yaw = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    print(f"design->level1: yaw {yaw:.4f} deg, t {np.round(t, 4)}, {len(inliers)} meshes, max residual {residual * 1e3:.2f} mm")
    return r, t, yaw


def import_texture(src, dst_dir, max_size=TEXTURE_MAX_SIZE):
    """Downscaled copy of a library texture next to the output; returns its file name."""
    from PIL import Image

    name = os.path.basename(src)
    dst = os.path.join(dst_dir, name)
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(src):
        image = Image.open(src)
        if max(image.size) > max_size:
            image = image.resize((max_size, max_size), Image.LANCZOS)
        image.save(dst, optimize=True)
    return name


def define_omnipbr(stage, path, albedo, normal, tile, roughness, metallic=0.0, tint=(1.0, 1.0, 1.0),
                   brightness=1.0):
    """OmniPBR material with world-space projected textures (one repeat per `tile` metres)."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(albedo))
    shader.GetInput("diffuse_texture").GetAttr().SetColorSpace("sRGB")
    if normal:
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal))
        shader.GetInput("normalmap_texture").GetAttr().SetColorSpace("raw")
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*tint))
    shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(brightness)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0 / tile, 1.0 / tile))
    out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for output in (material.CreateSurfaceOutput("mdl"), material.CreateDisplacementOutput("mdl"),
                   material.CreateVolumeOutput("mdl")):
        output.ConnectToSource(out)
    return material


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.path.join(ENVS_DIR, "akw_lvl1_climb.usd"))
    parser.add_argument("--design", default=os.path.join(ENVS_DIR, "Enrich_two_floor.usdc"))
    parser.add_argument("--out", default=os.path.join(ENVS_DIR, "akw_2floor.usd"))
    parser.add_argument("--isaacsim", default=os.path.expanduser("~/isaacsim"))
    parser.add_argument("--assets", default=os.environ.get("ISAACSIM_ASSET_ROOT",
                                                           os.path.expanduser("~/isaac_sim_assets/Assets/Isaac/6.0")))
    args = parser.parse_args()
    _bootstrap_usd(args.isaacsim)

    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

    base_stage = Usd.Stage.Open(args.base)
    design_stage = Usd.Stage.Open(args.design)
    base_root, design_root = Sdf.Path("/World/akw"), design_stage.GetDefaultPrim().GetPath()
    rot, trans, yaw = fit_design_to_level1(base_stage, design_stage, base_root, design_root)

    base_layer = base_stage.Flatten()
    design_layer = design_stage.Flatten()
    out = Sdf.Layer.CreateNew(args.out)
    for spec in base_layer.rootPrims:
        Sdf.CopySpec(base_layer, spec.path, out, spec.path)
    out.defaultPrim = base_layer.defaultPrim
    data = dict(base_layer.customLayerData)
    data.pop("omni_layer", None)
    data["akw_2floor_build"] = {"base": os.path.basename(args.base), "design": os.path.basename(args.design),
                                "design_yaw_deg": float(yaw), "design_translation_m": Gf.Vec3d(*trans)}
    out.customLayerData = data
    for key in ("upAxis", "metersPerUnit"):
        out.pseudoRoot.SetInfo(key, base_layer.pseudoRoot.GetInfo(key))

    # Building content: designer scene replaces the level-1 copy. CopySpec remaps
    # material bindings that point inside the copied subtree.
    del out.GetPrimAtPath("/World").nameChildren["akw"]
    Sdf.CopySpec(design_layer, design_root, out, base_root)
    akw_spec = out.GetPrimAtPath(base_root)
    for name in DROP_DESIGN_PRIMS:
        if name in akw_spec.nameChildren:
            del akw_spec.nameChildren[name]

    # Materials: reuse the level-1 tuning, neutral colour for textures that are not on disk.
    base_materials = {}
    for prim in base_stage.Traverse():
        if prim.IsA(UsdShade.Material):
            base_materials.setdefault(prim.GetName().rsplit("_", 1)[0], prim.GetPath())
    design_materials = out.GetPrimAtPath(base_root.AppendChild("_materials"))
    for name, base_name in MATERIALS_FROM_BASE.items():
        del design_materials.nameChildren[name]
        Sdf.CopySpec(base_layer, base_materials[base_name], out, design_materials.path.AppendChild(name))
    print(f"materials from level 1: {', '.join(MATERIALS_FROM_BASE)}")
    for name in TEXTURED_MATERIALS:
        del design_materials.nameChildren[name]

    stage = Usd.Stage.Open(out)
    texture_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "textures", "akw")
    os.makedirs(texture_dir, exist_ok=True)
    for name, spec in TEXTURED_MATERIALS.items():
        maps = {key: "./textures/akw/" + import_texture(os.path.join(args.assets, *spec[key]), texture_dir)
                for key in ("albedo", "normal") if key in spec}
        define_omnipbr(stage, design_materials.path.AppendChild(name), maps["albedo"], maps.get("normal"),
                       spec["tile"], spec["roughness"], spec.get("metallic", 0.0), spec.get("tint", (1.0, 1.0, 1.0)),
                       spec.get("brightness", 1.0))
    print(f"textured materials: {', '.join(TEXTURED_MATERIALS)}")

    akw = UsdGeom.Xformable(stage.GetPrimAtPath(base_root))
    akw.ClearXformOpOrder()
    akw.AddTranslateOp().Set(Gf.Vec3d(*trans))
    akw.AddRotateZOp().Set(float(np.round(yaw, 3)))
    xcache = UsdGeom.XformCache()

    missing = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(base_root)):
        shader = UsdShade.Shader(prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        for shader_input in shader.GetInputs():
            sources = shader_input.GetConnectedSources()[0]
            texture = UsdShade.Shader(sources[0].source.GetPrim()) if sources else None
            if not texture or texture.GetIdAttr().Get() != "UsdUVTexture":
                continue
            asset = texture.GetInput("file").Get()
            if asset is None or not asset.resolvedPath:
                missing.add(asset.path if asset else "?")
                shader_input.DisconnectSource()
                if shader_input.GetBaseName() == "diffuseColor":
                    shader_input.Set(Gf.Vec3f(*MISSING_TEXTURE_COLOR))
    if missing:
        print(f"missing textures (grey fallback; rebuild once delivered): {', '.join(sorted(missing))}")

    # Obstacles added in akw_lvl1_climb, at their original world pose.
    base_xc = UsdGeom.XformCache()
    for src_rel, material_sibling in CLIMB_ADDITIONS:
        src = base_root.AppendPath(src_rel)
        dst_parent = stage.GetPrimAtPath(src.GetParentPath())
        dst = dst_parent.GetPath().AppendChild(src.name)
        Sdf.CopySpec(base_layer, src, out, dst)
        prim = stage.GetPrimAtPath(dst)
        world = base_xc.GetLocalToWorldTransform(base_stage.GetPrimAtPath(src))
        local = world * xcache.GetLocalToWorldTransform(dst_parent).GetInverse()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(local)
        for rel in prim.GetRelationships():
            if rel.GetName().startswith("material:binding"):
                prim.RemoveProperty(rel.GetName())
        sibling = next(p for p in Usd.PrimRange(stage.GetPrimAtPath(base_root.AppendPath(material_sibling)))
                       if p.IsA(UsdGeom.Mesh))
        material, _ = UsdShade.MaterialBindingAPI(sibling).ComputeBoundMaterial()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    xcache.Clear()

    # Fixture lighting: one sphere light under each fixture, originals removed.
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    old_lights = [p for p in Usd.PrimRange(stage.GetPrimAtPath(base_root)) if p.IsA(UsdLux.SphereLight)]
    lights_root = UsdGeom.Scope.Define(stage, "/World/FixtureLights").GetPrim().GetPath()
    for light in old_lights:
        holder = light.GetParent()
        fixture = holder.GetParent()
        box = bbox.ComputeWorldBound(fixture).ComputeAlignedRange()
        mid = box.GetMidpoint()
        new = UsdLux.SphereLight.Define(stage, lights_root.AppendChild(fixture.GetName()))
        UsdGeom.Xformable(new).AddTranslateOp().Set(Gf.Vec3d(mid[0], mid[1], box.GetMin()[2] - LIGHT_DROP))
        new.CreateRadiusAttr(LIGHT_RADIUS)
        high_bay = box.GetMin()[2] - LEVEL2_FLOOR_Z > 8.0
        new.CreateIntensityAttr(HIGH_BAY_INTENSITY if high_bay else LIGHT_INTENSITY)
        new.CreateTreatAsPointAttr(False)
        stage.RemovePrim(holder.GetPath())
    print(f"fixture lights: {len(old_lights)} replaced")

    # Static colliders (the climb additions copied from a base that already has colliders,
    # scripts/add_akw_c_colliders.py, get the same idempotent setting).
    print(f"colliders: {add_static_colliders(stage, base_root)} triangle-mesh")

    out.Save()
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
