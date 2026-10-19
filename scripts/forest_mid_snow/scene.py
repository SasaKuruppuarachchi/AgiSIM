"""Moss-dominated spruce forest with melting snow, sharing the boreal pipeline."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from scene import Terrain, TerrainZ, copy_texture, define_material, write_mesh, world_pts
from forest_snow.scene import trunk_mesh
from forest_snow.needles import trim_conifer_needles
from .ground import apply_patchy_ground
from .lighting import write_ambient_hdri


class MidSnowTerrain(Terrain):
    """Gentle forest hummocks; the inherited takeoff clearing stays exactly flat."""
    def __init__(self, rng):
        super().__init__(rng)
        self.waves = [(kx, ky, p, a * .65) for kx, ky, p, a in self.waves]
        self.bumps = [(x, y, r, a * .65) for x, y, r, a in self.bumps]
        self.micro = [(kx, ky, p, a * 1.6) for kx, ky, p, a in self.micro]
        X, Y = np.meshgrid(self.xs, self.xs, indexing="ij")
        self.Z = self.analytic(X, Y)


def moss_collars(stage, source, tex_dir, tex_rel):
    """Thin irregular moss on the actual lower trunk surface; shared by instances."""
    manifest = json.loads((source / "polyhaven_manifest.json").read_text())
    maps = manifest["forest_leaves_02"]["files"]
    def texture(fragment):
        path = source / next(entry["file"] for entry in maps if fragment in entry["file"])
        return copy_texture(str(path), str(tex_dir), 2048)
    material = define_material(stage, Sdf.Path("/World/Looks/trunk_moss"),
        dict(diffuse=texture("_diffuse_"), normal=texture("_nor_gl_"), orm=texture("_arm_"),
             tile=.8, tint=(.32, .46, .27), rough=.95), tex_rel)
    total = count = 0
    for proto in stage.GetPrimAtPath("/World/Prototypes").GetAllChildren():
        if "_trunk_" not in proto.GetName():
            continue
        for prim in list(proto.GetAllChildren()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith("trunk_"):
                continue
            mesh = UsdGeom.Mesh(prim)
            V = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            F = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
            C = V[F].mean(1)
            center = V[V[:, 2] < .4, :2].mean(0)
            angle = np.arctan2(C[:, 1] - center[1], C[:, 0] - center[0])
            z = C[:, 2]
            field = .59 + .17 * np.sin(3 * angle + 5 * z) + .13 * np.sin(7 * angle - 9 * z)
            field += .08 * np.sin(11 * angle + 18 * z) - .53 * z
            selected = F[(z > -.15) & (z < 1.6) & (field > .14)]
            if not len(selected):
                continue
            indices, remap = np.unique(selected, return_inverse=True)
            points = V[indices].copy()
            radial = points[:, :2] - center
            radius = np.linalg.norm(radial, axis=1)
            points[:, :2] += radial / np.maximum(radius[:, None], 1e-8) * .004
            uv = np.c_[np.arctan2(radial[:, 1], radial[:, 0]) * np.median(radius), points[:, 2]]
            cap = write_mesh(stage, proto.GetPath().AppendChild(f"litter_moss_{count}"),
                             points, remap.reshape(-1, 3), uv)
            UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(material)
            cap.GetPrim().SetCustomDataByKey("forest_mid_snow:role", "trunk_surface_moss")
            count += 1
            total += len(selected)
    return dict(meshes=count, triangles=total, surface_offset_m=.004)


def open_spruce_crowns(stage):
    """Retain whole connected needle pieces, leaving every woody branch intact."""
    from meshlib import components, compact
    rng = np.random.default_rng(7061)
    stats = {}
    for proto in stage.GetPrimAtPath("/World/Prototypes").GetAllChildren():
        if not proto.GetName().startswith("spruce_"):
            continue
        for prim in list(proto.GetAllChildren()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith("foliage_"):
                continue
            mesh = UsdGeom.Mesh(prim)
            V = np.asarray(mesh.GetPointsAttr().Get(), np.float32)
            F = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
            U = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(), np.float32)
            n, labels, face_labels = components(F, len(V))
            z = np.bincount(labels, weights=V[:, 2], minlength=n) / np.maximum(np.bincount(labels, minlength=n), 1)
            probability = np.interp(z, [0, 3, 6, 10, 14, 25], [.025, .04, .12, .32, .55, .65])
            keep = (rng.random(n) < probability)[face_labels]
            v, f, u = compact(V, F, U, keep)
            if len(f):
                write_mesh(stage, prim.GetPath(), v, f, u, double_sided=True)
            else:
                prim.SetActive(False)
            stats[str(prim.GetPath())] = dict(before=len(F), after=len(f), retained_components=int(np.unique(face_labels[keep]).size))
    return stats


def seat_tree_roots(stage, stand):
    """Bury the scanned root edge on slopes, moving render and trunk proxies together."""
    terrain = TerrainZ(stand["terrain"])
    bases = {}
    shifts = []
    for tree in stand["trees"]:
        name = tree["trunk"]
        if name not in bases:
            proto = stage.GetPrimAtPath(f"/World/Prototypes/{name}")
            mesh = next(UsdGeom.Mesh(p) for p in proto.GetAllChildren() if p.IsA(UsdGeom.Mesh) and p.GetName().startswith("trunk_"))
            V = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            # Boundary of the scanned base, excluding the vertical stem surface.
            low = V[V[:, 2] < .40]
            center = low[:, :2].mean(0)
            r = np.linalg.norm(low[:, :2] - center, axis=1)
            bases[name] = low[r > np.percentile(r, 65)]
        W = world_pts(bases[name], [tree["x"], tree["y"], tree["z"]], tree["yaw"], tree["trunk_scale"])
        sink = max(0., float(np.max(W[:, 2] - terrain.z_at(W[:, 0], W[:, 1]))) + .035)
        tree["z"] -= sink
        tree["height"] -= sink
        tree["crown_base"] -= sink
        tree["collider_top"] -= sink
        tree["extra_root_sink_m"] = sink
        for segment in tree["colliders"]:
            segment["c"][2] -= sink
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
        root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(tree["x"], tree["y"], tree["z"]))
        shifts.append(sink)
    return dict(mean_extra_sink_m=float(np.mean(shifts)), max_extra_sink_m=float(max(shifts)))


def compose(stage, stand, tex_dir, tex_rel, snow_source, forest_source, log,
            *, snow_fraction=.23, resolution=8192):
    tex_dir = Path(tex_dir)
    ground_result = apply_patchy_ground(stage, stand, tex_dir, tex_rel, forest_source,
        snow_source, log, snow_fraction=snow_fraction, resolution=resolution)
    rng = np.random.default_rng(stand["summary"]["seed"] + 117)
    kept = []
    for obj in stand["objects"]:
        kind = obj["kind"]
        snow = float(ground_result.snow_mask.sample(obj["x"], obj["y"]))
        remove = kind in ("fern", "grass")
        if kind == "bilberry":
            remove = rng.random() > .08 or snow > .35
        elif kind == "moss":
            remove = snow > .3
        elif kind == "sapling":
            remove = rng.random() > .45
        if remove:
            stage.GetPrimAtPath(f"/World/Forest/Understory/{kind}/{obj['id']}").SetActive(False)
            collider = stage.GetPrimAtPath(f"/World/Forest/Colliders/Stems/{obj['id']}")
            if collider:
                collider.SetActive(False)
        else:
            kept.append(obj)
    stand["objects"] = kept

    # Early thaw: the evergreen foliage is exposed and bilberry is still leafless.
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Prototypes"), Usd.PrimAllPrimsPredicate):
        if prim.IsA(UsdGeom.Mesh):
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat and mat.GetPrim().GetName() == "bilberry_leaf":
                prim.SetActive(False)
        if prim.IsA(UsdShade.Shader):
            name = prim.GetParent().GetName()
            if name in ("fir_tree_01_trunk_c", "fir_tree_01_bark", "fir_tree_01_dead_branches"):
                prim.GetAttribute("inputs:diffuse_tint").Set(Gf.Vec3f(.96, 1.02, .88))
            elif name in ("fir_tree_01_twig", "fir_sapling_medium_twigs"):
                prim.GetAttribute("inputs:diffuse_tint").Set(Gf.Vec3f(.92, 1.0, .82))
    crown_stats = open_spruce_crowns(stage)
    # The source fir is an open-grown tree. Narrow crowns in this closed stand
    # leave the slender trunks and bare whorls visible, as in the reference.
    for tree in stand["trees"]:
        if tree["species"] != "spruce":
            continue
        tree["crown_scale"][:2] = [s * .60 for s in tree["crown_scale"][:2]]
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
        for prim in root.GetChildren():
            if prim.GetName() == "crown" or prim.GetName().startswith("skirt_"):
                prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*tree["crown_scale"]))
    alpha_stats = trim_conifer_needles(stage, forest_source, log)
    moss_stats = moss_collars(stage, forest_source, tex_dir, tex_rel)
    root_stats = seat_tree_roots(stage, stand)

    # Use the same non-foliage physics proxies as the verified winter scene.
    for prim in list(stage.GetPrimAtPath("/World/Forest/Colliders/Trunks").GetChildren()):
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

    manifest = json.loads((snow_source / "polyhaven_manifest.json").read_text())
    entry = next(f for f in manifest["snowy_forest"]["files"] if f["channel"] == "hdri")
    hdri = snow_source / entry["file"]
    lighting_stats = write_ambient_hdri(hdri, tex_dir / "snowy_forest_ambient_2k.hdr",
        expected_md5=entry["md5"], log=log)
    dome = UsdLux.DomeLight(stage.GetPrimAtPath("/World/Lights/Sky"))
    dome.CreateTextureFileAttr(Sdf.AssetPath(tex_rel + lighting_stats["output"]))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(6000)
    dome.CreateColorAttr(Gf.Vec3f(.98, .99, 1))
    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Lights/Sun"))
    sun.CreateIntensityAttr(2500)
    sun.CreateAngleAttr(25)
    sun.CreateColorAttr(Gf.Vec3f(1, .98, .95))

    by_proto = {}
    for proto in stage.GetPrimAtPath("/World/Prototypes").GetAllChildren():
        by_proto[proto.GetName()] = sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
            for p in Usd.PrimRange(proto, Usd.PrimAllPrimsPredicate) if p.IsActive() and p.IsA(UsdGeom.Mesh))
    instances = []
    for prim in stage.Traverse():
        if prim.IsInstance():
            instances.append(prim.GetMetadata("references").GetAddedOrExplicitItems()[0].primPath.name)
    ground = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Terrain/ground"))
    terrain_triangles = len(ground.GetFaceVertexCountsAttr().Get())
    stand["summary"].update(environment="FOREST_MID_SNOW", understory=dict(Counter(o["kind"] for o in kept)),
        distinct_triangles=sum(by_proto.values()), prototypes=len(by_proto), active_prototypes=len(set(instances)),
        render_instances=len(instances), terrain_triangles=terrain_triangles,
        expanded_render_triangles=sum(by_proto[p] for p in instances) + terrain_triangles,
        needle_alpha_cleanup=alpha_stats, trunk_moss=moss_stats, patchy_ground=ground_result.summary,
        floor_cover=ground_result.summary, trunk_cylinders=0, trunk_triangle_meshes=len(stand["trees"]),
        lighting=dict(hdri=lighting_stats, dome_intensity=6000, visible_domes=1,
                      sun_intensity=2500, sun_angle_degrees=25), crown_snow=False,
        spruce_foliage_thinning=crown_stats, spruce_crown_width_multiplier=.60, root_seating=root_stats)
    stage.GetRootLayer().documentation = (
        "FOREST_MID_SNOW: photo-inspired 100 x 100 m spruce-dominated moss forest in early thaw. "
        "Olive moss, exposed litter and irregular thin melting-snow islands; green crowns and mossy trunk bases. "
        "Shared FOREST_FI placement/prototypes and verified conifer alpha geometry. "
        "Flat takeoff clearing, static triangle-mesh ground/trunk/deadwood colliders; no foliage physics. "
        "CC0 sources and MD5s: forest_mid_snow_LICENSES.md.")
    log(f"mid-snow: {len(instances)} instances; ground {ground_result.summary}; trunk moss {moss_stats}")
