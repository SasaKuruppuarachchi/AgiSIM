"""Snow-only post-decimation alpha trimming of real conifer needle meshes.

The shared FOREST_FI source/caches are never changed. UV templates repeat across
thousands of geometric needles, so clipping is done once per template and then
mapped back to each source triangle with barycentric coordinates.
"""

from pathlib import Path

import numpy as np


_SAMPLES = np.array([[i / 24, j / 24, 1 - (i + j) / 24]
                     for i in range(25) for j in range(25 - i)])
_rng = np.random.default_rng(123)
_s, _r = np.sqrt(_rng.random(256)), _rng.random(256)
_CHECK = np.c_[1 - _s, _s * (1 - _r), _s * _r]


class AlphaClipper:
    def __init__(self, alpha):
        self.alpha = np.asarray(alpha) >= .5
        self.height, self.width = self.alpha.shape
        self.cache = {}

    def opaque(self, uv):
        x = (np.mod(uv[:, 0], 1) * self.width).astype(int)
        y = (np.mod(1 - uv[:, 1], 1) * self.height).astype(int)
        return self.alpha[np.minimum(y, self.height - 1), np.minimum(x, self.width - 1)]

    def template(self, uv):
        from scipy.spatial import ConvexHull, QhullError
        key = tuple(np.round(np.asarray(uv).ravel() * 65536).astype(np.int32))
        if key in self.cache:
            return self.cache[key]
        out = []

        def recurse(B, depth):
            U = B @ uv
            # Keep already opaque triangles unchanged. The final independent
            # area-weighted check must exceed 95% for every processed prototype.
            if self.opaque(_CHECK @ U).mean() >= .92:
                out.append(B)
                return
            candidates = _SAMPLES[self.opaque(_SAMPLES @ U)]
            if len(candidates) < 3:
                return
            try:
                P = candidates[ConvexHull(candidates[:, :2]).vertices]
            except QhullError:
                return
            # Fit one triangle inside the opaque convex hull. This preserves the
            # thin needle shape at the original triangle budget where possible.
            while len(P) > 3:
                a = np.roll(P, 1, axis=0)[:, :2]
                b, c = P[:, :2] - a, np.roll(P, -1, axis=0)[:, :2] - a
                P = np.delete(P, np.argmin(np.abs(b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])), axis=0)
            if self.opaque((_CHECK @ P) @ U).mean() >= .92:
                out.append(P @ B)
                return
            if depth >= 3:
                return
            # Split the longest UV edge; unlike uniform 4-way subdivision this
            # follows the narrow needle and adds few triangles near each boundary.
            edges = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            a, b, c = max(edges, key=lambda e: np.linalg.norm(U[e[0]] - U[e[1]]))
            middle = (B[a] + B[b]) * .5
            recurse(np.array([B[a], middle, B[c]]), depth + 1)
            recurse(np.array([middle, B[b], B[c]]), depth + 1)

        recurse(np.eye(3), 0)
        if out:
            tris = np.asarray(out)
            # Remove numerical slivers and weld vertices within this UV template.
            a, b = tris[:, 1, :2] - tris[:, 0, :2], tris[:, 2, :2] - tris[:, 0, :2]
            area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            tris = tris[area > 1e-8]
            points, indices = np.unique(np.round(tris.reshape(-1, 3), 8), axis=0, return_inverse=True)
            result = points, indices.reshape(-1, 3).astype(np.int32)
        else:
            result = np.empty((0, 3)), np.empty((0, 3), np.int32)
        self.cache[key] = result
        return result

    def trim(self, vertices, faces, uv):
        V, F, UV = np.asarray(vertices), np.asarray(faces), np.asarray(uv)
        keys, inv = np.unique(np.round(UV[F].reshape(-1, 6) * 65536).astype(np.int32),
                              axis=0, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        counts = np.bincount(inv)
        offsets = np.r_[0, np.cumsum(counts)]
        blocks = []
        n_vertices = n_faces = 0
        for i in range(len(keys)):
            face_ids = order[offsets[i]:offsets[i + 1]]
            B, T = self.template(UV[F[face_ids[0]]].astype(float))
            blocks.append((face_ids, B, T))
            n_vertices += len(face_ids) * len(B)
            n_faces += len(face_ids) * len(T)
        out_v, out_uv = np.empty((n_vertices, 3), np.float32), np.empty((n_vertices, 2), np.float32)
        out_f = np.empty((n_faces, 3), np.int32)
        vo = fo = 0
        for ids, B, T in blocks:
            if not len(T):
                continue
            nv, nf = len(ids) * len(B), len(ids) * len(T)
            out_v[vo:vo + nv] = np.einsum("pb,fbc->fpc", B, V[F[ids]]).reshape(-1, 3)
            out_uv[vo:vo + nv] = np.einsum("pb,fbc->fpc", B, UV[F[ids]]).reshape(-1, 2)
            out_f[fo:fo + nf] = (T[None] + (vo + np.arange(len(ids)) * len(B))[:, None, None]).reshape(-1, 3)
            vo += nv
            fo += nf
        return out_v, out_f, out_uv


def _opacity(vertices, faces, uv, clipper):
    from meshlib import tri_area
    area = tri_area(vertices, faces)
    if not len(area) or not area.sum():
        return {"surface_m2": 0.0, "opaque_fraction": 1.0}
    rng = np.random.default_rng(20261018)
    selected = rng.choice(len(faces), 100000, p=area / area.sum())
    s, r = np.sqrt(rng.random(len(selected))), rng.random(len(selected))
    B = np.c_[1 - s, s * (1 - r), s * r]
    U = (B[:, :, None] * uv[faces[selected]]).sum(axis=1)
    return {"surface_m2": float(area.sum()), "opaque_fraction": float(clipper.opaque(U).mean())}


def trim_conifer_needles(stage, source, log=print):
    """Trim the snowy stage only, before ``winterize_prototypes``; return audit stats."""
    from pxr import Usd, UsdGeom, UsdShade
    from meshlib import load_alpha
    from scene import write_mesh
    materials = {
        "pine_tree_01_twig": "pine_tree_01",
        "fir_tree_01_twig": "fir_tree_01",
        "pine_sapling_small_twig": "pine_sapling_small",
        "fir_sapling_twigs": "fir_sapling",
        "fir_sapling_medium_twigs": "fir_sapling_medium",
    }
    clippers, stats = {}, {}
    root = stage.GetPrimAtPath("/World/Prototypes")
    meshes = [p for p in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate) if p.IsActive() and p.IsA(UsdGeom.Mesh)]
    for prim in meshes:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        name = material.GetPrim().GetName() if material else ""
        if name not in materials:
            continue
        if name not in clippers:
            path = Path(source) / "polyhaven" / materials[name] / "alpha" / f"{name}_alpha_1k.png"
            clippers[name] = AlphaClipper(load_alpha(str(path)))
        clipper = clippers[name]
        m = UsdGeom.Mesh(prim)
        V = np.asarray(m.GetPointsAttr().Get(), np.float32)
        F = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int32).reshape(-1, 3)
        U = np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Get(), np.float32)
        before = _opacity(V, F, U, clipper)
        if before["opaque_fraction"] >= .97:
            stats[str(prim.GetPath())] = {"before": dict(before, triangles=len(F)),
                                        "after": dict(before, triangles=len(F)),
                                        "retained_opaque_surface": 1.0, "unchanged": True}
            continue
        SV, SF, SU = clipper.trim(V, F, U)
        after = _opacity(SV, SF, SU, clipper)
        if after["opaque_fraction"] < .95:
            raise RuntimeError(f"Alpha trimming missed target: {prim.GetPath()}: {after}")
        write_mesh(stage, prim.GetPath(), SV, SF, SU, double_sided=True)
        retention = after["surface_m2"] * after["opaque_fraction"] / (before["surface_m2"] * before["opaque_fraction"])
        result = {"before": dict(before, triangles=len(F)), "after": dict(after, triangles=len(SF)),
                  "retained_opaque_surface": retention, "unchanged": False}
        stats[str(prim.GetPath())] = result
        log(f"needle alpha trim {prim.GetParent().GetName()}: {len(F)} -> {len(SF)} triangles; "
            f"opaque {before['opaque_fraction']:.4f} -> {after['opaque_fraction']:.4f}; "
            f"area {before['surface_m2']:.2f} -> {after['surface_m2']:.2f} m2")
    return stats
