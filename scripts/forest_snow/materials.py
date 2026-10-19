"""Geometric snow accumulation on the shared FOREST_FI prototypes.

Caps are low-resolution, connected patches supported by real upward-facing source
triangles. They do not cover an entire crown silhouette, alpha cards, or vertical
trunk walls. The original bark/needle geometry and material bindings are retained.
The generated caps are ordinary opaque render meshes, so RTX lidar sees snow; no
collision or rigid-body schemas are added to foliage or caps.
"""

from collections import defaultdict

import numpy as np


def _split_point_junctions(points, triangles):
    """Separate deposits that only touch at one vertex, keeping side walls closed."""
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary_degree = np.bincount(edges[counts == 1].ravel(), minlength=len(points))
    extra = []
    for v in np.flatnonzero(boundary_degree > 2):
        incident = np.flatnonzero((triangles == v).any(axis=1))
        by_vertex = defaultdict(list)
        for f in incident:
            for u in triangles[f]:
                if u != v:
                    by_vertex[int(u)].append(int(f))
        adj = defaultdict(set)
        for group in by_vertex.values():
            for f in group:
                adj[f].update(group)
        left = set(map(int, incident))
        first = True
        while left:
            frontier, component = [min(left)], set()
            while frontier:
                f = frontier.pop()
                if f in component:
                    continue
                component.add(f)
                frontier.extend(adj[f] - component)
            left.difference_update(component)
            if not first:
                new = len(points) + len(extra)
                extra.append(points[v].copy())
                for f in component:
                    triangles[f][triangles[f] == v] = new
            first = False
    if extra:
        points = np.concatenate([points, np.asarray(extra)])
    return points, triangles


def snow_caps(vertices, faces, *, cell=0.16, thickness=0.075, foliage=False):
    """Return a closed, rounded snow mesh supported by upward source triangles.

    Spatial cells retain the uppermost *actual triangle centroid* and accumulate
    projected support area. Neighbouring occupied cells connect only when their
    heights agree, with no interpolation across empty cells or tall vertical gaps.
    Several levels of a crown are retained, allowing snowfall through crown gaps.
    The rim tapers to 6 mm; interior snow reaches ``thickness`` metres. Isolated
    needles and patches smaller than three triangles remain uncovered.

    All coordinates are in prototype metres. The algorithm is deterministic and
    never changes the source arrays. It runs during the offline scene build only.
    """
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    empty = (np.empty((0, 3), np.float32), np.empty((0, 3), np.int32))
    if not len(faces):
        return empty
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1)
    nz = np.abs(normal[:, 2]) if foliage else normal[:, 2]
    keep = (norm > 1e-12) & (nz > 0.28 * norm)
    # Alpha-cut needle meshes are double-sided. Their downward-wound triangles
    # also have a genuine upper surface; closed branches/rocks use oriented normals.
    centers = tri[keep].mean(axis=1)
    areas = 0.5 * nz[keep]
    if len(centers) < 3:
        return empty
    layer = max(0.18, 2.0 * cell)
    keys = np.floor(centers / [cell, cell, layer]).astype(np.int64)
    unique, inv = np.unique(keys, axis=0, return_inverse=True)
    area_sum = np.bincount(inv, weights=areas)
    order = np.argsort(centers[:, 2], kind="stable")
    chosen = np.zeros(len(unique), np.int64)
    # Repeated assignment retains the highest real surface centroid per cell.
    chosen[inv[order]] = order
    supported = area_sum >= cell * cell * (0.012 if foliage else 0.008)
    keys, points = unique[supported], centers[chosen[supported]]
    if len(points) < 3:
        return empty
    columns = defaultdict(list)
    for i, (x, y, _) in enumerate(keys):
        columns[(int(x), int(y))].append(i)

    max_dz = cell * (0.95 if foliage else 1.45)

    def neighbour(x, y, i):
        candidates = columns.get((x, y), ())
        if not candidates:
            return None
        z = points[i, 2]
        j = min(candidates, key=lambda n: abs(points[n, 2] - z))
        # Reciprocal matching avoids joining two vertically separated boughs to
        # the same neighbour, which would create overlapping/nonmanifold sheets.
        back = min(columns[tuple(map(int, keys[i, :2]))],
                   key=lambda n: abs(points[n, 2] - points[j, 2]))
        return j if back == i and abs(points[j, 2] - z) <= max_dz else None

    triangles = []

    def add(a, b, c):
        if b is None or c is None:
            return
        P = points[[a, b, c]]
        if np.ptp(P[:, 2]) > max_dz:
            return
        if neighbour(*map(int, keys[c, :2]), b) != c:
            return
        n = np.cross(P[1] - P[0], P[2] - P[0])
        if n[2] > cell * cell * 0.03:
            triangles.append((a, b, c))

    for i, (x, y, _) in enumerate(keys):
        b = neighbour(int(x + 1), int(y), i)
        c = neighbour(int(x), int(y + 1), i)
        d = neighbour(int(x + 1), int(y + 1), i)
        if d is None:
            add(i, b, c)
        else:
            add(i, b, d)
            add(i, d, c)
    if not triangles:
        return empty
    triangles = np.unique(np.asarray(triangles, np.int32), axis=0)

    # Discard small disconnected specks; keep broad, coherent deposits on boughs.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = coo_matrix((np.ones(len(triangles) * 2),
                      (np.r_[triangles[:, 0], triangles[:, 1]],
                       np.r_[triangles[:, 1], triangles[:, 2]])),
                     shape=(len(points), len(points)))
    _, labels = connected_components(adj, directed=False)
    face_labels = labels[triangles[:, 0]]
    counts = np.bincount(face_labels, minlength=len(points))
    triangles = triangles[counts[face_labels] >= 3]
    if not len(triangles):
        return empty
    used, remap = np.unique(triangles.ravel(), return_inverse=True)
    points = points[used]
    triangles = remap.reshape(-1, 3).astype(np.int32)
    points, triangles = _split_point_junctions(points, triangles)

    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    _, edge_index, edge_count = np.unique(np.sort(edges, axis=1), axis=0, return_index=True,
                                        return_counts=True)
    border_edges = edges[edge_index[edge_count == 1]]
    boundary = np.zeros(len(points), bool)
    boundary[border_edges.ravel()] = True
    # Graph distance from the rim gives a soft banked edge. Source heights remain
    # untouched: every upper vertex has a known real support immediately below it.
    level = np.full(len(points), 3.0)
    level[boundary] = 0.0
    for _ in range(3):
        a, b = edges[:, 0], edges[:, 1]
        updated = level.copy()
        np.minimum.at(updated, a, level[b] + 1)
        np.minimum.at(updated, b, level[a] + 1)
        level = updated
    lift = 0.006 + (thickness - 0.006) * (1 - np.exp(-(level + 0.15) / 1.1))
    # Small coherent variation avoids a mechanically uniform cap depth.
    lift *= 0.93 + 0.07 * np.sin(points[:, 0] * 5.3 + points[:, 1] * 3.7)
    upper = points.copy()
    upper[:, 2] += lift
    lower = points.copy()
    lower[:, 2] -= 0.006
    n = len(points)
    walls = []
    for a, b in border_edges:
        walls.extend(((b, a, a + n), (b, a + n, b + n)))
    F = np.concatenate([triangles, triangles[:, ::-1] + n, np.asarray(walls, np.int32).reshape(-1, 3)])
    return np.concatenate([upper, lower]).astype(np.float32), F.astype(np.int32)


def winterize_prototypes(stage, snow_material, log=print):
    """Add supported snow caps to the stage's instance source meshes.

    ``snow_material`` is a UsdShade.Material, usually the same rough PBR snow as
    the terrain. Its texture coordinates are in metres. Returns JSON-ready stats.
    Existing prims are retained, and any deciduous leaves are deactivated.
    """
    from pxr import Usd, UsdGeom, UsdShade
    from scene import write_mesh

    root = stage.GetPrimAtPath("/World/Prototypes")
    if not root:
        raise ValueError("Missing /World/Prototypes")
    stats = {"snow_meshes": 0, "snow_triangles": 0, "inactive_leaves": [], "prototypes": {}}
    # Snapshot the range: newly authored cap meshes must not become new sources.
    source_meshes = [p for p in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate)
                     if p.IsActive() and p.IsA(UsdGeom.Mesh)]
    for prim in source_meshes:
        name = prim.GetParent().GetName()
        role = prim.GetName().rsplit("_", 1)[0]
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        mat_name = material.GetPrim().GetName() if material else ""
        if mat_name in ("birch_leaf", "bilberry_leaf"):
            prim.SetActive(False)
            stats["inactive_leaves"].append(str(prim.GetPath()))
            continue
        conifer = name.startswith(("pine_", "spruce_", "uspruce_", "sap_"))
        branch = role == "branch"
        foliage = role == "foliage" and conifer
        solid = role in ("deadwood", "rock")
        if not (foliage or branch or solid):
            continue
        mesh = UsdGeom.Mesh(prim)
        V = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        F = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
        if foliage:
            cell = 0.075 if name.startswith("sap_") else 0.14
            thickness = 0.035 if name.startswith("sap_") else 0.085
        elif solid:
            cell, thickness = (0.045, 0.045) if role == "deadwood" else (0.095, 0.055)
        else:
            cell, thickness = 0.055, 0.032
        SV, SF = snow_caps(V, F, cell=cell, thickness=thickness, foliage=foliage)
        if not len(SF):
            continue
        # Keep role parsing in forest_gt.py compatible (snow_<integer>).
        dest = prim.GetParent().GetPath().AppendChild(f"snow_{stats['snow_meshes']}")
        cap = write_mesh(stage, dest, SV, SF, SV[:, :2], double_sided=False)
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(snow_material)
        cap.GetPrim().SetCustomDataByKey("forest_snow:source", str(prim.GetPath()))
        stats["snow_meshes"] += 1
        stats["snow_triangles"] += len(SF)
        entry = stats["prototypes"].setdefault(name, {"meshes": 0, "triangles": 0})
        entry["meshes"] += 1
        entry["triangles"] += len(SF)
    log(f"snow caps: {stats['snow_meshes']} meshes, {stats['snow_triangles']} distinct triangles; "
        f"{len(stats['inactive_leaves'])} deciduous leaf meshes inactive")
    return stats
