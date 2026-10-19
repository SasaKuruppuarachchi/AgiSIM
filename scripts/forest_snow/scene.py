"""Winter-specific hooks for the shared FOREST_FI scene builder (metres, Z-up)."""
from collections import Counter
from pathlib import Path
import json
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from scene import Terrain, copy_texture, define_material, write_mesh
from .materials import winterize_prototypes
from .needles import trim_conifer_needles


class SnowTerrain(Terrain):
    """Original undulation plus 50 broad, smooth wind drifts; spawn remains z=0."""
    def __init__(self, rng):
        super().__init__(rng)
        self.bumps += [(rng.uniform(-49, 49), rng.uniform(-49, 49), rng.uniform(1.2, 3.5),
                        rng.uniform(.08, .38)) for _ in range(50)]
        X, Y = np.meshgrid(self.xs, self.xs, indexing="ij")
        self.Z = self.analytic(X, Y)


def trunk_mesh(segments, sides=12):
    """Closed 12-sided segments following the existing measured stem proxy."""
    angle = np.arange(sides) * (2 * np.pi / sides)
    ring = np.c_[np.cos(angle), np.sin(angle)]
    vertices, faces = [], []
    for segment in segments:
        c, r, h = np.array(segment["c"]), segment["r"], segment["h"]
        xy = c[:2] + ring * r
        v = np.vstack([np.c_[xy, np.full(sides, c[2] - h / 2)],
                       np.c_[xy, np.full(sides, c[2] + h / 2)],
                       c + [0, 0, -h / 2], c + [0, 0, h / 2]])
        offset = sum(len(part) for part in vertices)
        f = []
        for i in range(sides):
            j = (i + 1) % sides
            f += [(i, j, j + sides), (i, j + sides, i + sides),
                  (2 * sides, j, i), (2 * sides + 1, i + sides, j + sides)]
        vertices.append(v)
        faces.append(np.array(f, np.int32) + offset)
    return np.concatenate(vertices), np.concatenate(faces)


def winterize(stage, stand, tex_dir, tex_rel, source, forest_source, log):
    tex_dir = Path(tex_dir)
    manifest = json.loads((source / "polyhaven_manifest.json").read_text())
    maps = {entry["channel"]: source / entry["file"] for entry in manifest["snow_02"]["files"]}
    spec = dict(diffuse=copy_texture(str(maps["Diffuse"]), str(tex_dir), 2048),
                normal=copy_texture(str(maps["nor_gl"]), str(tex_dir), 2048),
                rough_tex=copy_texture(str(maps["Rough"]), str(tex_dir), 2048),
                tile=2, tint=(.93, .97, 1.0), rough=.88)
    snow = define_material(stage, Sdf.Path("/World/Looks/snow"), spec, tex_rel)
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Terrain/ground"))
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(snow)
    for subset in UsdGeom.Subset.GetAllGeomSubsets(ground):
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(snow)

    # Snow hides summer ground cover. A few bare bilberry stems and dry twigs
    # protrude; trees, logs, rocks and their original placements are retained.
    kept = []
    rng = np.random.default_rng(stand["summary"]["seed"] + 91)
    for obj in stand["objects"]:
        remove = obj["kind"] in ("fern", "moss", "grass")
        if obj["kind"] == "bilberry":
            remove = rng.random() > .055 or np.hypot(obj["x"], obj["y"]) < 4
        if obj["kind"] == "drybranch":
            remove = rng.random() > .35
        path = f"/World/Forest/Understory/{obj['kind']}/{obj['id']}"
        if remove:
            stage.GetPrimAtPath(path).SetActive(False)
        else:
            kept.append(obj)
    stand["objects"] = kept

    alpha_stats = trim_conifer_needles(stage, forest_source, log)
    cap_stats = winterize_prototypes(stage, snow, log)
    # Static mesh proxies, with no CollisionAPI on foliage or snow crowns.
    old_root = stage.GetPrimAtPath("/World/Forest/Colliders/Trunks")
    for prim in list(old_root.GetChildren()):
        prim.SetActive(False)
    for tree in stand["trees"]:
        V, F = trunk_mesh(tree["colliders"])
        mesh = write_mesh(stage, f"/World/Forest/Colliders/Trunks/{tree['id']}", V, F, normals=False)
        mesh.GetPrim().SetCustomDataByKey("tree_id", tree["id"])
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Forest/Colliders/Deadwood")):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")

    # Wide overcast source plus a weak soft key; no hard summer sun.
    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lights/Sun"))
    sun.CreateIntensityAttr(700)
    sun.CreateAngleAttr(30)
    sun.CreateColorAttr(Gf.Vec3f(.90, .95, 1))
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/Sky"))
    hdri = source / next(entry["file"] for entry in manifest["snowy_forest"]["files"] if entry["channel"] == "hdri")
    shutil.copy2(hdri, tex_dir)
    dome.CreateTextureFileAttr(Sdf.AssetPath(tex_rel + hdri.name))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(2500)
    dome.CreateColorAttr(Gf.Vec3f(.96, .98, 1))

    # The photographed forest already occludes much of the HDRI sky. Add a
    # neutral overcast illuminant so our modeled canopy does not double that
    # occlusion. Keep normal geometric shadows: unlinked infinite fill lights
    # caused a measured ~2x slowdown in this heavily instanced RTX scene.
    # The reference HDRI remains the visible background and contributes light.
    fill = UsdLux.DomeLight.Define(stage, "/World/Lights/OvercastSky")
    fill.CreateIntensityAttr(60000)
    fill.CreateColorAttr(Gf.Vec3f(.92, .96, 1))
    fill.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    UsdLux.LightAPI(fill).CreateSpecularAttr(0)
    # Lift the source texture's mean linear albedo (~0.38) toward fresh snow,
    # retaining its texture contrast, roughness and normal map.
    stage.GetPrimAtPath("/World/Looks/snow/Shader").CreateAttribute(
        "inputs:albedo_brightness", Sdf.ValueTypeNames.Float).Set(2.)

    triangles, proto_count, triangles_by_proto = 0, 0, {}
    for proto in stage.GetPrimAtPath("/World/Prototypes").GetAllChildren():
        proto_count += 1
        triangles_by_proto[proto.GetName()] = 0
        for prim in Usd.PrimRange(proto, Usd.PrimAllPrimsPredicate):
            if prim.IsActive() and prim.IsA(UsdGeom.Mesh):
                n_faces = len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get())
                triangles += n_faces
                triangles_by_proto[proto.GetName()] += n_faces
    instances, expanded_triangles, active_prototypes = 0, 0, set()
    for prim in stage.Traverse():
        if prim.IsInstance():
            instances += 1
            refs = prim.GetMetadata("references").GetAddedOrExplicitItems()
            proto_name = refs[0].primPath.name
            active_prototypes.add(proto_name)
            expanded_triangles += triangles_by_proto[proto_name]
    terrain_triangles = len(ground.GetFaceVertexCountsAttr().Get())
    summary = stand["summary"]
    summary.update(environment="FOREST_SNOW", understory=dict(Counter(o["kind"] for o in kept)),
                   distinct_triangles=triangles, prototypes=proto_count, render_instances=instances,
                   active_prototypes=len(active_prototypes), terrain_triangles=terrain_triangles,
                   expanded_render_triangles=expanded_triangles + terrain_triangles,
                   snow_caps=cap_stats, floor_cover={"snow_02": 1.0}, drift_mounds=50,
                   needle_alpha_cleanup=alpha_stats,
                   trunk_cylinders=0, trunk_triangle_meshes=len(stand["trees"]),
                   lighting=dict(hdri=hdri.name, dome_intensity=2500, soft_key_intensity=700,
                                 soft_key_angle_degrees=30, shadowed_overcast_dome_intensity=60000,
                                 snow_albedo_brightness=2.0))
    stage.GetRootLayer().documentation = (
        "FOREST_SNOW: 100 x 100 m CC0 boreal winter forest, shared FOREST_FI pipeline. "
        "780 pine/spruce trees, PBR snow ground with wind drifts, real snow-cap geometry, "
        "sparse winter understory and Poly Haven snowy_forest overcast HDRI. "
        "Spawn (0,0,0.07), terrain z=0 within 3m, stem clearing radius 4m. "
        "Static triangle-mesh ground/trunk/deadwood colliders; no foliage collision. "
        "No PhysicsScene. See forest_snow_LICENSES.md for sources and MD5s.")
    log(f"winter scene: {instances} instances, {triangles} distinct prototype triangles; caps {cap_stats}")
