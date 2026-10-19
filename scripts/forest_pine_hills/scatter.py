"""Static ecology/detail authoring over the shared forest stand.

Only this builder calls these hooks. No frame callbacks, sensor changes or
global renderer switches are introduced by the geometry work.
"""
from collections import Counter
import hashlib
import math
import numpy as np

from .terrain import from_stand, normal_at, terrain_description, trail_distance

# Accepted geometry-only render trial. Z scale and all stem radii stay intact.
MATURE_CROWN_XY = {"pine": .78, "spruce": .70}
YOUNG_PINE_FRACTION = .18
YOUNG_PINE_HEIGHT_SCALE = (.60, .80)
YOUNG_PINE_WIDTH_SCALE = (.75, .90)
EVIDENCE_CAMERA_XY = np.array([-6.5, -11.])
EVIDENCE_LOOK_XY = np.array([3.2, 6.])


def support_matrix(position, yaw, scale, normal=(0, 0, 1)):
    """Row-vector affine transform, with local X along the support plane."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    a = np.deg2rad(yaw)
    x = np.array([np.cos(a), np.sin(a), 0.])
    x -= n * np.dot(x, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    m = np.eye(4)
    m[:3, :3] = np.diag(scale) @ np.stack((x, y, n))
    m[3, :3] = position
    return m


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3] + matrix[3, :3]


def canopy_gap(trees, xy):
    """Approximate sky-view from overlapping horizontal crown footprints."""
    from scipy.spatial import cKDTree
    centres = np.array([[t["x"], t["y"]] for t in trees])
    widths = np.array([2.2 * t["crown_scale"][0] for t in trees])
    distances, indices = cKDTree(centres).query(np.atleast_2d(xy), k=min(12, len(trees)))
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    optical_depth = np.sum(.45 * np.exp(-.5 * (distances / widths[indices]) ** 2), axis=1)
    return np.exp(-optical_depth)


def ecology_weights(kind, gap, slope, lowland):
    """Plausible habitat weights; values are relative sampling probabilities."""
    gentle = np.exp(-(slope / .48) ** 2)
    if kind == "fern":
        return gentle * (.25 + gap) * (.35 + lowland)
    if kind == "grass":
        return gentle * (.08 + gap ** 2) * (1.2 - .6 * lowland)
    if kind == "moss":
        return gentle * (.1 + (1 - gap) ** 1.5) * (.4 + lowland)
    if kind == "drybranch":
        return .3 + 1 - gap
    return gentle * (.4 + gap)


def _set_matrix(prim, matrix):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = prim.GetAttribute("xformOp:transform")
    if op:
        op.Set(Gf.Matrix4d(*matrix.ravel().tolist()))
        xf.SetXformOpOrder([UsdGeom.XformOp(op)])
    else:
        xf.AddTransformOp().Set(Gf.Matrix4d(*matrix.ravel().tolist()))


def _geometry(stage, name, cache):
    from pxr import Usd, UsdGeom
    if name not in cache:
        points = [np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
                  for p in Usd.PrimRange(stage.GetPrimAtPath(f"/World/Prototypes/{name}"), Usd.PrimAllPrimsPredicate)
                  if p.IsActive() and p.IsA(UsdGeom.Mesh)]
        v = np.concatenate(points)
        # Deterministic stratified support samples, including the extremal vertices.
        ix = np.unique(np.r_[np.arange(0, len(v), max(1, len(v) // 3500)),
                             np.argmin(v, axis=0), np.argmax(v, axis=0)])
        cache[name] = v[ix]
    return cache[name]


def _colliders(stage, obj):
    paths = []
    for name in ("Deadwood", "Rocks"):
        root = stage.GetPrimAtPath(f"/World/Forest/Colliders/{name}")
        if root:
            paths.extend(p for p in root.GetChildren() if p.GetName().startswith(obj["id"] + "_h"))
    root = stage.GetPrimAtPath(f"/World/Forest/Colliders/Stems/{obj['id']}")
    if root:
        paths.append(root)
    return paths


def _move_object(stage, obj, new_matrix, old_matrix=None):
    """Move render object and its separate physics meshes with one affine map."""
    from pxr import Gf, UsdGeom, UsdPhysics, Vt
    prim = stage.GetPrimAtPath(f"/World/Forest/Understory/{obj['kind']}/{obj['id']}")
    if old_matrix is None:
        old_matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), float)
    delta = np.linalg.inv(old_matrix) @ new_matrix
    for collider in _colliders(stage, obj):
        if collider.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(collider)
            p = transform_points(mesh.GetPointsAttr().Get(), delta).astype(np.float32)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p))
            mesh.CreateExtentAttr([Gf.Vec3f(*map(float, p.min(0))), Gf.Vec3f(*map(float, p.max(0)))])
            if obj["kind"] != "rock":
                UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("none")
        else:
            cm = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(collider), float)
            _set_matrix(collider, cm @ delta)
    _set_matrix(prim, new_matrix)
    obj.update(x=float(new_matrix[3, 0]), y=float(new_matrix[3, 1]), z=float(new_matrix[3, 2]),
               transform=new_matrix.round(10).tolist())


def _new_instance(stage, objects, template, kind, matrix, suffix, *, collider=False):
    from pxr import Sdf, UsdGeom, UsdPhysics
    obj = dict(template, kind=kind, id=f"{kind}_{suffix}", collider=None)
    obj.pop("collider_geom", None)
    root_path = f"/World/Forest/Understory/{kind}/{obj['id']}"
    prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    prim.GetReferences().AddInternalReference(Sdf.Path(f"/World/Prototypes/{obj['proto']}"))
    prim.SetInstanceable(True)
    _set_matrix(prim, matrix)
    obj.update(x=float(matrix[3, 0]), y=float(matrix[3, 1]), z=float(matrix[3, 2]),
               scale=np.linalg.norm(matrix[:3, :3], axis=1).tolist(),
               yaw=math.degrees(math.atan2(matrix[0, 1], matrix[0, 0])),
               transform=matrix.round(10).tolist())
    if collider:
        # Root plates have a separate convex collision hull, never foliage physics.
        from scipy.spatial import ConvexHull
        from scene import write_mesh
        points = transform_points(_geometry(stage, obj["proto"], {}), matrix)
        hull = ConvexHull(points)
        group = "Rocks" if kind.startswith("rock") else "Deadwood"
        mesh = write_mesh(stage, f"/World/Forest/Colliders/{group}/{obj['id']}_h0",
                          points, hull.simplices, normals=False)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        obj["collider"] = "hull"
    objects.append(obj)
    return obj


def _hard_footprints(stage, objects, geometry):
    """Conservative horizontal discs for existing non-foliage collision objects."""
    discs = []
    for obj in objects:
        if not obj.get("collider"):
            continue
        if obj["collider"] == "stem":
            radius = obj.get("collider_geom", {}).get("r", .08)
        else:
            points = _geometry(stage, obj["proto"], geometry)
            if "transform" in obj:
                world = transform_points(points, np.asarray(obj["transform"]))
                radius = float(np.linalg.norm(world[:, :2] - [obj["x"], obj["y"]], axis=1).max())
            else:
                scaled = points * np.asarray(obj["scale"])
                radius = float(np.linalg.norm(scaled, axis=1).max())
        discs.append((obj["x"], obj["y"], radius))
    return np.asarray(discs, float).reshape(-1, 3)


def _free_footprint(xy, radius, trees_xy, tree_radii, hard_discs, *, pad_radius=3.8):
    xy = np.asarray(xy, float)
    if np.linalg.norm(xy) < pad_radius + radius or np.abs(xy).max() > 49 - radius:
        return False
    if len(trees_xy) and np.any(np.linalg.norm(trees_xy - xy, axis=1) < tree_radii + radius):
        return False
    if len(hard_discs) and np.any(np.linalg.norm(hard_discs[:, :2] - xy, axis=1) < hard_discs[:, 2] + radius):
        return False
    return True


def _move_tree_before_seating(stage, tree, xy, terrain):
    """Move stem and all world-space proxy segments before root seating.

    height/crown_base remain AGL dimensions; collider_top is a world-Z value.
    The caller subsequently rebuilds the triangle proxies from these segments.
    """
    from pxr import Gf
    old = np.array([tree["x"], tree["y"], tree["z"]], float)
    dz = float(terrain.z_at(*xy) - terrain.z_at(tree["x"], tree["y"]))
    new = np.array([xy[0], xy[1], tree["z"] + dz], float)
    delta = new - old
    tree.update(x=float(new[0]), y=float(new[1]), z=float(new[2]))
    tree["collider_top"] += dz
    for segment in tree["colliders"]:
        segment["c"] = (np.asarray(segment["c"], float) + delta).tolist()
    root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, new)))
    return dict(id=tree["id"], from_xyz_m=old.tolist(), to_xyz_m=new.tolist(), shift_m=float(np.linalg.norm(delta)))


def vary_tree_age(stage, stand, *, young_fraction=YOUNG_PINE_FRACTION):
    """A younger pine cohort across the plot, scaling whole scanned trees together.

    No prototype is copied. Crown, trunk and any skirts share the same added
    XYZ factors, preserving their attachment heights. Separate world-space
    collider segments receive the exact same affine transform about the root.
    """
    from pxr import Gf
    if not np.isfinite(young_fraction) or not 0 <= young_fraction <= 1:
        raise ValueError("Young pine fraction must be finite and between zero and one")
    seed = stand.get("summary", {}).get("seed", 2026)
    ranked = []
    for tree in stand["trees"]:
        if tree["species"] == "pine":
            digest = hashlib.sha256(f"pine-hills-age:{seed}:{tree['id']}".encode()).digest()
            ranked.append((digest, tree))
    ranked.sort(key=lambda item: item[0])
    count = round(len(ranked) * young_fraction)
    changed = []
    for digest, tree in ranked[:count]:
        uniform = lambda start: int.from_bytes(digest[start:start + 4], "little") / (2 ** 32 - 1)
        fz = YOUNG_PINE_HEIGHT_SCALE[0] + np.ptp(YOUNG_PINE_HEIGHT_SCALE) * uniform(4)
        fxy = YOUNG_PINE_WIDTH_SCALE[0] + np.ptp(YOUNG_PINE_WIDTH_SCALE) * uniform(8)
        factors = np.array([fxy, fxy, fz])
        base = np.array([tree["x"], tree["y"], tree["z"]])
        old_height, old_dbh = tree["height"], tree["dbh"]
        for key in ("crown_scale", "trunk_scale"):
            tree[key] = (np.asarray(tree[key]) * factors).tolist()
        for key in ("height", "target_height", "crown_base"):
            if key in tree:
                tree[key] *= fz
        for key in ("dbh", "target_dbh"):
            if key in tree:
                tree[key] *= fxy
        if "prof_scale" in tree:
            tree["prof_scale"] = [tree["prof_scale"][0] * fxy, tree["prof_scale"][1] * fz]
        tree["collider_top"] = base[2] + (tree["collider_top"] - base[2]) * fz
        for segment in tree["colliders"]:
            segment["c"] = (base + (np.asarray(segment["c"]) - base) * factors).tolist()
            segment["r"] *= fxy
            segment["h"] *= fz
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
        for child in root.GetChildren():
            name = child.GetName()
            if name == "trunk":
                child.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*tree["trunk_scale"]))
            elif name == "crown" or name.startswith("skirt_"):
                child.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*tree["crown_scale"]))
        tree["age_cohort"] = "younger_pine"
        tree["whole_tree_age_scale_xyz"] = factors.tolist()
        changed.append(dict(id=tree["id"], scale_xyz=factors.tolist(),
                            source_height_m=old_height, pre_seating_height_m=tree["height"],
                            source_dbh_m=old_dbh, scaled_dbh_m=tree["dbh"]))
    return dict(method="deterministic varied-age cohort throughout plot, not camera-facing geometry",
                pine_stems=len(ranked), young_pine_stems=count, mature_pine_stems=len(ranked) - count,
                requested_pine_fraction=young_fraction, actual_pine_fraction=count / len(ranked) if ranked else 0,
                configured_height_scale=list(YOUNG_PINE_HEIGHT_SCALE), configured_width_scale=list(YOUNG_PINE_WIDTH_SCALE),
                actual_height_scale_range=[min(t["scale_xyz"][2] for t in changed), max(t["scale_xyz"][2] for t in changed)] if changed else [],
                actual_width_scale_range=[min(t["scale_xyz"][0] for t in changed), max(t["scale_xyz"][0] for t in changed)] if changed else [],
                new_prototype_meshes=0, stems_preserved=len(stand["trees"]), trees=changed)


def open_trail_and_canopy(stage, stand, terrain, geometry, *, crown_xy=None):
    """Keep every mature tree, widening the irregular natural gap beside the track."""
    from pxr import Gf
    crown_xy = MATURE_CROWN_XY if crown_xy is None else crown_xy
    if any(not 0 < float(value) <= 1 for value in crown_xy.values()):
        raise ValueError("Crown width multipliers must be in (0, 1]")
    points = np.array([[t["x"], t["y"]] for t in stand["trees"]])
    selected = (trail_distance(points[:, 0], points[:, 1]) < 1.15) & (points[:, 1] > -12) & (points[:, 1] < 18)
    occupied = points.copy()
    hard = _hard_footprints(stage, stand["objects"], geometry)
    moves, skipped = [], []
    for index in np.flatnonzero(selected):
        tree = stand["trees"][index]
        old = points[index]
        other_points = occupied[np.arange(len(occupied)) != index]
        candidates = []
        for radius in np.linspace(2., 7., 26):
            for angle in np.linspace(0, 2 * np.pi, 96, endpoint=False):
                xy = old + radius * np.array([np.cos(angle), np.sin(angle)])
                distance = float(trail_distance(*xy))
                if distance < 1.8:
                    continue
                # 2.25 m stems apart, plus an independent root-flare/hard-object gap.
                if not _free_footprint(xy, .65, other_points, np.full(len(other_points), 1.6), hard):
                    continue
                candidates.append((radius + .1 * abs(distance - 2.3), xy))
            if candidates:
                break
        if not candidates:
            skipped.append(tree["id"])
            continue
        xy = min(candidates, key=lambda item: item[0])[1]
        occupied[index] = xy
        moves.append(_move_tree_before_seating(stage, tree, xy, terrain))
    # Additional modest shoulder variation breaks the continuous crown ceiling.
    # Keep Z unchanged: scaling crown height would move branch/trunk attachment
    # heights and could expose a seam between independently instanced scans.
    crown_parts = Counter()
    shoulder_variation = []
    for tree in stand["trees"]:
        factor = float(crown_xy.get(tree["species"], 1.))
        shoulder = 1.
        distance = float(trail_distance(tree["x"], tree["y"]))
        if distance < 4 and -14 < tree["y"] < 22:
            digest = hashlib.sha256(("pine-hills-crown:" + tree["id"]).encode()).digest()
            shoulder = .75 + .25 * (int.from_bytes(digest[:4], "little") / (2 ** 32 - 1))
            shoulder_variation.append(dict(id=tree["id"], extra_xy_multiplier=shoulder,
                                           trail_distance_m=distance))
        total = factor * shoulder
        tree["crown_scale"][:2] = [float(v) * total for v in tree["crown_scale"][:2]]
        tree["crown_width_multiplier"] = total
        tree["trail_shoulder_crown_multiplier"] = shoulder
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
        for child in root.GetChildren():
            if child.GetName() == "crown" or child.GetName().startswith("skirt_"):
                child.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*tree["crown_scale"]))
                crown_parts[tree["species"]] += 1
    return dict(crown_width_xy=dict(crown_xy), crown_z_unchanged=True,
                scaled_crown_parts=dict(crown_parts), stems_preserved=len(stand["trees"]),
                relocated_trees=moves, relocation_skipped=skipped,
                trail_shoulder_variation=shoulder_variation,
                trail_shoulder_extra_xy_range=[.75, 1.],
                trail_min_new_stem_offset_m=1.8, min_new_stem_spacing_m=2.25)


def foreground_detail(stage, stand, objects, terrain, geometry):
    """Four grounded root webs and two half-buried rocks in the reference framing.

    Search nearby free placements instead of intersecting a trunk/hard object.
    Every occupied footprint stays outside the flat take-off pad.
    """
    tree_xy = np.array([[t["x"], t["y"]] for t in stand["trees"]])
    tree_r = np.array([t["dbh"] / 2 + .18 for t in stand["trees"]])
    hard = _hard_footprints(stage, objects, geometry)
    direction = EVIDENCE_LOOK_XY - EVIDENCE_CAMERA_XY
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    templates = {kind: [o for o in objects if o["kind"] == kind] for kind in ("roots", "rock")}
    placements = []
    specs = [("trail_roots", f, side) for f, side in ((1.4, -.15), (2.7, .2), (4.1, -.2), (5.5, .1))]
    specs += [("rock", 3., 1.65), ("rock", 5.4, -2.1)]
    for index, (kind, forward, lateral) in enumerate(specs):
        if kind == "rock":
            # Use the small scan variants; 43% of their vertical span is buried.
            choices = sorted(templates["rock"], key=lambda o: np.ptp(_geometry(stage, o["proto"], geometry)[:, :2], axis=0).max())
            template = choices[index % min(4, len(choices))]
            scale = .62 if index == 4 else .52
        else:
            template = templates["roots"][index % len(templates["roots"])]
            scale = 1.12 if index % 2 else .95
        points = _geometry(stage, template["proto"], geometry)
        # 3D radius also bounds the XY footprint after the slope-support tilt.
        radius = float(np.linalg.norm(points, axis=1).max()) * scale
        preferred = EVIDENCE_CAMERA_XY + forward * direction + lateral * left
        candidates = []
        for offset in np.linspace(0., 1.8, 10):
            for angle in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                xy = preferred + offset * np.array([np.cos(angle), np.sin(angle)])
                if not _free_footprint(xy, radius + .08, tree_xy, tree_r, hard):
                    continue
                if kind == "rock" and trail_distance(*xy) < 1.15:
                    continue
                candidates.append((offset, xy))
            if candidates:
                break
        if not candidates:
            raise RuntimeError(f"No collision-free foreground placement for {kind} {index}")
        xy = candidates[0][1]
        yaw = (31 + index * 67) % 360
        matrix = support_matrix([xy[0], xy[1], 0], yaw, [scale] * 3, normal_at(terrain, *xy))
        world = transform_points(points, matrix)
        ground = terrain.z_at(world[:, 0], world[:, 1])
        if kind == "rock":
            matrix[3, 2] = float(terrain.z_at(*xy)) - .43 * np.ptp(points[:, 2]) * scale
        else:
            bottom = points[:, 2] < np.percentile(points[:, 2], 20)
            matrix[3, 2] = float(np.quantile((ground - world[:, 2])[bottom], .25)) - .015
        obj = _new_instance(stage, objects, template, kind, matrix, f"foreground_{index:02d}", collider=kind == "rock")
        obj["reference_foreground_detail"] = True
        if kind == "rock":
            obj["half_buried"] = True
            hard = np.vstack((hard, [xy[0], xy[1], radius]))
        placements.append(dict(id=obj["id"], kind=kind, xyz_m=[obj["x"], obj["y"], obj["z"]],
                               footprint_radius_m=radius, physics_collider=kind == "rock"))
    return dict(reference_camera_xy_m=EVIDENCE_CAMERA_XY.tolist(), placements=placements,
                clear_pad_radius_m=3.8, existing_hard_objects_avoided=True)


def frame_edge_ferns(stage, stand, objects, terrain, geometry, *, count=16):
    """Move existing fern instances to the near track edges; do not add geometry."""
    trees_xy = np.array([[t["x"], t["y"]] for t in stand["trees"]])
    tree_radii = np.array([t["dbh"] / 2 + .15 for t in stand["trees"]])
    hard = _hard_footprints(stage, objects, geometry)
    direction = EVIDENCE_LOOK_XY - EVIDENCE_CAMERA_XY
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    ferns = [o for o in objects if o["kind"] == "fern"]
    candidates = sorted(ferns, key=lambda o: -math.hypot(o["x"] - EVIDENCE_CAMERA_XY[0], o["y"] - EVIDENCE_CAMERA_XY[1]))
    if len(candidates) < count:
        raise ValueError("Not enough existing fern instances for edge repositioning")
    placed = []
    occupied = []
    for index, obj in enumerate(candidates[:count]):
        forward = 1.1 + 4.8 * (index // 2) / max(count // 2 - 1, 1)
        lateral = (-1 if index % 2 else 1) * (1.25 + .18 * (index % 5))
        preferred = EVIDENCE_CAMERA_XY + forward * direction + lateral * left
        points = _geometry(stage, obj["proto"], geometry) * np.asarray(obj["scale"])
        radius = float(np.linalg.norm(points, axis=1).max())
        selected = None
        for offset in np.linspace(0, 1.2, 7):
            for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                xy = preferred + offset * np.array([np.cos(angle), np.sin(angle)])
                if not _free_footprint(xy, radius + .03, trees_xy, tree_radii, hard):
                    continue
                if trail_distance(*xy) < 1.05:
                    continue
                if occupied and np.linalg.norm(np.asarray(occupied) - xy, axis=1).min() < .38:
                    continue
                selected = xy
                break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"Could not find a clear foreground fern pose for {obj['id']}")
        matrix = support_matrix([selected[0], selected[1], terrain.z_at(*selected) - .015],
                                obj["yaw"], obj["scale"], normal_at(terrain, *selected))
        _move_object(stage, obj, matrix)
        obj["reference_frame_edge_fern"] = True
        obj["detail_lod_band"] = "near"
        obj["canopy_gap"] = round(float(canopy_gap(stand["trees"], [selected])[0]), 4)
        occupied.append(selected)
        placed.append(dict(id=obj["id"], xyz_m=[obj["x"], obj["y"], obj["z"]],
                           scale_unchanged=obj["scale"]))
    return dict(repositioned_count=len(placed), total_fern_count_before=len(ferns),
                total_fern_count_after=sum(o["kind"] == "fern" for o in objects),
                new_instances=0, placements=placed)


def _open_scan_boundary(stage, name):
    """True open scan edges after welding UV-split positions to one micrometre.

    Root scans contain an irregular dark soil skirt around their raised wood.
    Raw USD edge counts mistake UV seams for external boundaries, so the
    topology used for this measurement is position-welded only in memory.
    """
    from pxr import Usd, UsdGeom
    proto = stage.GetPrimAtPath(f"/World/Prototypes/{name}")
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(proto), float))
    vertices, triangles, offset = [], [], 0
    for prim in Usd.PrimRange(proto, Usd.PrimAllPrimsPredicate):
        if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        v = np.asarray(mesh.GetPointsAttr().Get(), float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim), float) @ inverse
        vertices.append(transform_points(v, matrix))
        triangles.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3) + offset)
        offset += len(v)
    vertices, triangles = np.concatenate(vertices), np.concatenate(triangles)
    _, first, remap = np.unique(np.round(vertices, 6), axis=0, return_index=True, return_inverse=True)
    welded = vertices[first]
    faces = remap[triangles]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])]
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_indices = np.unique(unique[counts == 1])
    if not len(boundary_indices):
        raise ValueError(f"Expected an open soil-skirt boundary in root scan {name}")
    return welded[boundary_indices], vertices


def seat_trail_root_edges(stage, objects, terrain, *, foreground_metadata=None,
                         percentile=90., target_gap_m=-.002):
    """Seat added root-scan soil skirts; preserve all wood geometry and instances.

    Whole-instance world-Z translation places the 90th-percentile true boundary
    gap 2 mm beneath the triangle heightfield. The percentile deliberately
    permits a few root cut tips to protrude; using the maximum buries the wood.
    This helper is idempotent and may also correct a previously built stage in
    memory. The caller, not this function, owns all USD/JSON persistence.
    """
    from pxr import UsdGeom
    if not np.isfinite(percentile) or not 0 <= percentile <= 100 or not np.isfinite(target_gap_m):
        raise ValueError("Finite boundary percentile and target gap are required")
    scans = {}
    results = []
    placements = {p["id"]: p for p in (foreground_metadata or {}).get("placements", [])}
    for obj in objects:
        if obj["kind"] != "trail_roots":
            continue
        if obj.get("collider") or _colliders(stage, obj):
            raise ValueError(f"Root skirt seating is visual-only; unexpected collider on {obj['id']}")
        if obj["proto"] not in scans:
            scans[obj["proto"]] = _open_scan_boundary(stage, obj["proto"])
        edges, all_vertices = scans[obj["proto"]]
        prim = stage.GetPrimAtPath(f"/World/Forest/Understory/trail_roots/{obj['id']}")
        matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=float)
        edge_world = transform_points(edges, matrix)
        gaps = edge_world[:, 2] - terrain.z_at(edge_world[:, 0], edge_world[:, 1])
        before = float(np.percentile(gaps, percentile))
        sink = max(0., before - target_gap_m)
        if sink < 1e-7:
            sink = 0.
        wood_world = transform_points(all_vertices, matrix)
        peak_before = float(np.max(wood_world[:, 2] - terrain.z_at(wood_world[:, 0], wood_world[:, 1])))
        matrix[3, 2] -= sink
        _set_matrix(prim, matrix)
        obj.update(x=float(matrix[3, 0]), y=float(matrix[3, 1]), z=float(matrix[3, 2]),
                   transform=matrix.round(10).tolist(),
                   trail_root_edge_sink_m=obj.get("trail_root_edge_sink_m", 0.) + sink,
                   trail_root_boundary_p90_gap_m=before - sink)
        if obj["id"] in placements:
            placements[obj["id"]]["xyz_m"] = [obj["x"], obj["y"], obj["z"]]
            placements[obj["id"]]["root_edge_sink_m"] = obj["trail_root_edge_sink_m"]
        results.append(dict(id=obj["id"], proto=obj["proto"], boundary_vertices=len(edges),
                            extra_sink_m=sink, cumulative_sink_m=obj["trail_root_edge_sink_m"],
                            boundary_percentile_before_m=before, boundary_percentile_after_m=before - sink,
                            highest_root_above_ground_before_m=peak_before,
                            highest_root_above_ground_after_m=peak_before - sink))
    return dict(method="position-welded open scan boundary against triangle terrain; whole-instance Z translation",
                percentile=percentile, target_boundary_gap_m=target_gap_m, instances=len(results),
                shifted_instances=sum(r["extra_sink_m"] > 0 for r in results),
                new_or_removed_instances=0, prototype_geometry_changed=False, colliders_changed=0,
                objects=results)


def _crooked_stems(stage, stand, rng):
    """Small lower-stem crooks from existing scans, with matching proxy centres.

    The deformation is zero at the base and above 8 m, so the scanned root
    flare and crown join remain intact. These are procedural variants, not
    additional claimed scan assets.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt
    from scene import world_pts
    variants = {}
    changed = 0
    for tree in stand["trees"]:
        if tree["species"] != "pine" or rng.random() > .30:
            continue
        original = tree["trunk"]
        name = original + "_crooked"
        if original not in variants:
            path = Sdf.Path(f"/World/Prototypes/{name}")
            Sdf.CopySpec(stage.GetRootLayer(), Sdf.Path(f"/World/Prototypes/{original}"), stage.GetRootLayer(), path)
            phase = {"a": 0., "b": 2.1, "c": 4.2}[original.split("_")[1]]
            direction = np.array([np.cos(phase), np.sin(phase)])
            variants[original] = (name, direction)
            for prim in stage.GetPrimAtPath(path).GetAllChildren():
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), float)
                z = points[:, 2]
                bend = .23 * np.sin(np.pi * np.clip((z - .6) / 7.4, 0, 1)) ** 2
                points[:, :2] += bend[:, None] * direction
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                # Recompute normals because the scanned bark's support changed.
                import meshlib
                faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(meshlib.vertex_normals(points, faces)))
                mesh.CreateExtentAttr([Gf.Vec3f(*map(float, points.min(0))), Gf.Vec3f(*map(float, points.max(0)))])
        name, direction = variants[original]
        for segment in tree["colliders"]:
            z_local = (segment["c"][2] - tree["z"]) / tree["trunk_scale"][2]
            bend = .23 * np.sin(np.pi * np.clip((z_local - .6) / 7.4, 0, 1)) ** 2
            offset = world_pts([[bend * direction[0], bend * direction[1], 0]], [0, 0, 0],
                               tree["yaw"], tree["trunk_scale"])[0]
            segment["c"] = (np.asarray(segment["c"]) + offset).tolist()
        tree["trunk"] = name
        prim = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}/trunk")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddInternalReference(Sdf.Path(f"/World/Prototypes/{name}"))
        tree["procedural_lower_stem_crook"] = True
        changed += 1
    return dict(trees=changed, extra_trunk_prototypes=len(variants), max_local_bend_m=.23,
                source_scan_variants=3, new_scan_assets=0)


def author_scatter(stage, stand, log, *, seed=None, crown_xy=None, young_pine_fraction=YOUNG_PINE_FRACTION):
    from pxr import Gf, UsdGeom, UsdPhysics
    from forest_mid_snow.scene import seat_tree_roots
    from forest_snow.scene import trunk_mesh
    from scene import write_mesh
    terrain = from_stand(stand)
    rng = np.random.default_rng(seed if seed is not None else stand["summary"]["seed"] + 511)
    geometry = {}
    age_cohort = vary_tree_age(stage, stand, young_fraction=young_pine_fraction)
    canopy = open_trail_and_canopy(stage, stand, terrain, geometry, crown_xy=crown_xy)
    crooks = _crooked_stems(stage, stand, rng)
    seated = seat_tree_roots(stage, stand)
    for tree in stand["trees"]:
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}")
        root.SetCustomDataByKey("forest_fi", dict(species=tree["species"], dbh_m=round(tree["dbh"], 3),
            height_m=round(tree["height"], 2), crown_base_m=round(tree["crown_base"], 2)))
        root.SetCustomDataByKey("forest_pine_hills:age_cohort", tree.get("age_cohort", "mature"))
    # Regenerate proxies after both root seating and curved-stem edits.
    for prim in stage.GetPrimAtPath("/World/Forest/Colliders/Trunks").GetChildren():
        prim.SetActive(False)
    for tree in stand["trees"]:
        points, faces = trunk_mesh(tree["colliders"])
        mesh = write_mesh(stage, f"/World/Forest/Colliders/Trunks/{tree['id']}", points, faces, normals=False)
        mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        mesh.GetPrim().SetCustomDataByKey("tree_id", tree["id"])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")

    objects = []
    for obj in stand["objects"]:
        kind = obj["kind"]
        gap = float(canopy_gap(stand["trees"], [[obj["x"], obj["y"]]])[0])
        keep = .22 if kind == "bilberry" else .45 if kind == "cut_stump" else 1.
        if kind == "sapling":
            keep = .30 + .4 * gap
        if rng.random() > keep:
            stage.GetPrimAtPath(f"/World/Forest/Understory/{kind}/{obj['id']}").SetActive(False)
            for p in _colliders(stage, obj):
                p.SetActive(False)
            continue
        objects.append(obj)
    # Vegetation sample pool evaluated once, so per-candidate tests remain cheap.
    xy = rng.uniform(-48, 48, (16000, 2))
    from scipy.spatial import cKDTree
    tree_xy = np.array([[t["x"], t["y"]] for t in stand["trees"]])
    near, _ = cKDTree(tree_xy).query(xy)
    xy = xy[(near > .55) & (np.linalg.norm(xy, axis=1) > 4.0)]
    heights = terrain.z_at(xy[:, 0], xy[:, 1])
    normals = normal_at(terrain, xy[:, 0], xy[:, 1])
    slopes = np.linalg.norm(normals[:, :2], axis=1) / normals[:, 2]
    gaps = canopy_gap(stand["trees"], xy)
    lowland = np.clip((np.median(heights) - heights) / 7 + .5, 0, 1)
    radii = np.linalg.norm(xy, axis=1)
    track = trail_distance(xy[:, 0], xy[:, 1])
    # Build-time density LOD: central +/-22 m scan plot detailed, outer stand sparse.
    lod = np.where(radii < 24, 1., np.where(radii < 38, .32, .10))
    templates = {kind: [o for o in objects if o["kind"] == kind]
                 for kind in ("fern", "grass", "moss", "drybranch", "roots")}
    targets = {"fern": 620, "grass": 270, "moss": 680, "drybranch": 430}
    for kind, target in targets.items():
        choices = templates[kind]
        weights = ecology_weights(kind, gaps, slopes, lowland) * lod
        if kind == "fern":
            weights *= .35 + 2.5 * np.exp(-((track - 2.0) / 1.15) ** 2)
        elif kind in ("moss", "grass"):
            weights *= .10 + .9 * np.clip(track / 1.2, 0, 1)
        weights /= weights.sum()
        existing = [o for o in objects if o["kind"] == kind]
        for index in range(target):
            sample = int(rng.choice(len(xy), p=weights))
            x, y = xy[sample] + rng.normal(0, .12, 2)
            yaw = float(rng.uniform(0, 360))
            s = rng.uniform(.65, 1.45) if kind != "drybranch" else rng.uniform(.45, 1.25)
            matrix = support_matrix([x, y, terrain.z_at(x, y) - .015], yaw, [s, s, s], normal_at(terrain, x, y))
            if index < len(existing):
                obj = existing[index]
                _move_object(stage, obj, matrix)
            else:
                obj = _new_instance(stage, objects, choices[index % len(choices)], kind, matrix, f"detail_{index:04d}")
            obj["scale"], obj["yaw"] = [s, s, s], yaw
            obj["canopy_gap"] = round(float(gaps[sample]), 4)
            obj["detail_lod_band"] = "near" if radii[sample] < 24 else "mid" if radii[sample] < 38 else "far"

    log_contact = []
    for obj in objects:
        kind = obj["kind"]
        if kind not in ("log", "rock", "stump", "cut_stump", "roots"):
            continue
        v = _geometry(stage, obj["proto"], geometry)
        x, y = obj["x"], obj["y"]
        normal = normal_at(terrain, x, y) if kind in ("log", "roots", "rock") else np.array([0., 0., 1.])
        matrix = support_matrix([x, y, 0], obj["yaw"], obj["scale"], normal)
        p = transform_points(v, matrix)
        ground = terrain.z_at(p[:, 0], p[:, 1])
        if kind == "rock":
            matrix[3, 2] = terrain.z_at(x, y) - .43 * np.ptp(v[:, 2]) * obj["scale"][2]
        elif kind == "log":
            # Max gap to sampled underside: grounded at a high point, never floating
            # wholesale above the sloping support plane. A 15 mm seat avoids cracks.
            matrix[3, 2] = float(np.max(ground - p[:, 2])) - .015
            local_gap = p[:, 2] + matrix[3, 2] - ground
            log_contact.append(float(np.min(local_gap)))
        else:
            bottom = v[:, 2] < np.percentile(v[:, 2], 20)
            matrix[3, 2] = float(np.quantile((ground - p[:, 2])[bottom], .25)) - .025
        _move_object(stage, obj, matrix)

    # Scattered exposed bedrock at the ridge shoulder: shallowly buried scans,
    # not floating boulders perched on a heightfield. Reuse the small source hulls.
    rock_templates = [o for o in objects if o["kind"] == "rock"]
    for index in range(12):
        template = rock_templates[index % len(rock_templates)]
        for _ in range(40):
            x, y = np.array([28., 9.]) + rng.normal(0, [6., 9.])
            if abs(x) < 47 and abs(y) < 47 and cKDTree(tree_xy).query([x, y])[0] > 1.8:
                break
        else:
            continue
        v = _geometry(stage, template["proto"], geometry)
        s = rng.uniform(.55, .95)
        matrix = support_matrix([x, y, terrain.z_at(x, y) - .48 * np.ptp(v[:, 2]) * s],
                                rng.uniform(0, 360), [s] * 3, normal_at(terrain, x, y))
        # The helper authors a separate hull from the same transformed vertices.
        rock = _new_instance(stage, objects, template, "rock_outcrop", matrix, f"{index:02d}", collider=True)
        rock["half_buried"] = True

    # Six exposed root plates at the uphill ends of fallen logs, same CC0 root scans.
    for index, log_obj in enumerate([o for o in objects if o["kind"] == "log"][:6]):
        matrix = np.asarray(log_obj["transform"])
        v = _geometry(stage, log_obj["proto"], geometry)
        endpoint = transform_points([[v[:, 0].min(), 0, .10]], matrix)[0]
        scale = rng.uniform(.65, 1.0)
        plate = support_matrix(endpoint, log_obj["yaw"] + 90, [scale] * 3,
                               normal=matrix[0, :3] / np.linalg.norm(matrix[0, :3]))
        plate[3, 2] += .30
        _new_instance(stage, objects, templates["roots"][index % len(templates["roots"])],
                      "root_plate", plate, f"{index:02d}", collider=True)

    # Exposed root webs interrupt the needle/litter track. Align to the actual
    # support plane and bury their scanned edges; they stay below flight height.
    for index in range(32):
        y = rng.uniform(-20., 24.)
        x = .55 * y + .45 * np.sin(y / 4.5) + rng.normal(0, .65)
        if math.hypot(x, y) < 4.2:
            continue
        template = templates["roots"][index % len(templates["roots"])]
        s = rng.uniform(.8, 1.2)
        matrix = support_matrix([x, y, terrain.z_at(x, y) - .015], rng.uniform(0, 360),
                                [s, s, s], normal_at(terrain, x, y))
        _new_instance(stage, objects, template, "trail_roots", matrix, f"{index:02d}")

    foreground = foreground_detail(stage, stand, objects, terrain, geometry)
    root_edge_seating = seat_trail_root_edges(stage, objects, terrain, foreground_metadata=foreground)
    edge_ferns = frame_edge_ferns(stage, stand, objects, terrain, geometry)

    # Moss clumps on upward-facing log/rock surfaces, sampled from actual geometry.
    caps = 0
    for obj in [o for o in objects if o["kind"] in ("log", "rock", "rock_outcrop")]:
        v = _geometry(stage, obj["proto"], geometry)
        matrix = np.asarray(obj["transform"])
        w = transform_points(v, matrix)
        upper = w[w[:, 2] > np.percentile(w[:, 2], 80)]
        for index in range(3 if obj["kind"] == "log" else 2):
            point = upper[rng.integers(len(upper))].copy()
            point[2] -= .012
            scale = rng.uniform(.35, .7)
            cap_matrix = support_matrix(point, rng.uniform(0, 360), [scale, scale, scale * .6])
            _new_instance(stage, objects, templates["moss"][caps % len(templates["moss"])],
                          "surface_moss", cap_matrix, f"{caps:03d}")
            caps += 1

    stand["objects"] = objects
    info = dict(terrain=terrain_description(terrain), root_seating=seated, crooked_stems=crooks,
                varied_age_pines=age_cohort,
                canopy_and_trail=canopy, foreground_detail=foreground,
                root_edge_seating=root_edge_seating,
                reference_edge_ferns=edge_ferns,
                understory=dict(Counter(o["kind"] for o in objects)), surface_moss_instances=caps,
                grounded_logs=len(log_contact), log_min_sampled_gap_m=log_contact,
                ground_cover_lod=dict(method="static radial density bands centred on the scan area",
                    bands_m=[24, 38, 71], relative_density=[1., .32, .10], runtime_callbacks=0),
                ecology="Canopy-footprint sky-view, local slope and lowland moisture weights",
                trail=dict(centerline="x=.55*y+.45*sin(y/4.5)", bare_half_width_m=.9,
                           flattened=False, fern_edge_distance_m=2.),
                trunk_triangle_meshes=len(stand["trees"]))
    stand["summary"].update(terrain_note=info["terrain"], scatter=info,
                            understory=info["understory"], trunk_cylinders=0,
                            trunk_triangle_meshes=len(stand["trees"]))
    log(f"pine hills scatter: {len(objects)} objects, {crooks['trees']} crooked stems, {caps} surface moss clumps")
    return info
