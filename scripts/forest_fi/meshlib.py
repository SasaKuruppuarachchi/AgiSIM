"""Mesh helpers for the forest_fi build (numpy/scipy; runs under Isaac Sim's bundled Python).

glTF reading, connected components, alpha-mask culling / card-to-geometry conversion, Blender
decimation (subprocess, cached), trunk profiles and approximate convex decomposition.
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


# ----------------------------------------------------------------------------- parts
class Part:
    """One mesh of a prototype: Z-up metres, per-vertex UV (USD st convention, origin bottom-left)."""

    def __init__(self, role, mat, V, F, UV=None):
        self.role, self.mat = role, mat
        self.V = np.asarray(V, np.float32).reshape(-1, 3)
        self.F = np.asarray(F, np.int32).reshape(-1, 3)
        self.UV = np.zeros((len(self.V), 2), np.float32) if UV is None else np.asarray(UV, np.float32)

    @property
    def tris(self):
        return len(self.F)

    def area(self):
        return float(tri_area(self.V, self.F).sum())

    def copy(self):
        return Part(self.role, self.mat, self.V.copy(), self.F.copy(), self.UV.copy())


def merge_parts(parts, role=None, mat=None):
    Vs, Fs, UVs, off = [], [], [], 0
    for p in parts:
        Vs.append(p.V)
        Fs.append(p.F + off)
        UVs.append(p.UV)
        off += len(p.V)
    return Part(role or parts[0].role, mat or parts[0].mat, np.concatenate(Vs), np.concatenate(Fs),
                np.concatenate(UVs))


def tri_area(V, F):
    V = V.astype(np.float64)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def compact(V, F, UV, keep_faces):
    """Keep the selected faces and drop unreferenced vertices."""
    F = F[keep_faces]
    used = np.zeros(len(V), bool)
    used[F.ravel()] = True
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(used.sum())
    return V[used], remap[F].astype(np.int32), UV[used]


# ------------------------------------------------------------------------------ glTF
def _accessor(g, bins, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    buf = bins[bv["buffer"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    dt = np.dtype(CT[a["componentType"]])
    n = NC[a["type"]]
    stride = bv.get("byteStride", 0)
    if stride and stride != dt.itemsize * n:
        raw = np.frombuffer(buf, np.uint8, count=stride * a["count"], offset=off).reshape(a["count"], stride)
        return raw[:, :dt.itemsize * n].copy().view(dt).reshape(a["count"], n)
    return np.array(np.frombuffer(buf, dt, count=a["count"] * n, offset=off).reshape(a["count"], n))


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)  # (x, y, z) -> (x, -z, y)


def load_gltf(path):
    """{node name: [ {mat, V (Z-up, node TRS applied), F, UV (st)} ]} plus the material table.

    Reads the glTF JSON and binary buffer as data only. Only root-level nodes with meshes are
    expected (true for the Poly Haven exports used here)."""
    g = json.load(open(path))
    d = os.path.dirname(path)
    bins = [np.memmap(os.path.join(d, b["uri"]), np.uint8, "r") for b in g["buffers"]]
    nodes = {}
    for node in g["nodes"]:
        if "mesh" not in node:
            continue
        S = np.diag(node.get("scale", [1, 1, 1]))
        R = _quat_mat(node["rotation"]) if "rotation" in node else np.eye(3)
        T = np.asarray(node.get("translation", [0, 0, 0]), np.float64)
        prims = []
        for p in g["meshes"][node["mesh"]]["primitives"]:
            V = _accessor(g, bins, p["attributes"]["POSITION"]).astype(np.float64)
            V = (V @ (R @ S).T + T) @ Y_UP_TO_Z_UP.T
            if "TEXCOORD_0" in p["attributes"]:
                UV = _accessor(g, bins, p["attributes"]["TEXCOORD_0"]).astype(np.float64)
                UV = np.c_[UV[:, 0], 1.0 - UV[:, 1]]
            else:
                UV = np.zeros((len(V), 2))
            F = _accessor(g, bins, p["indices"]).reshape(-1, 3).astype(np.int32)
            prims.append({"mat": g["materials"][p["material"]]["name"], "V": V.astype(np.float32),
                          "F": F, "UV": UV.astype(np.float32)})
        nodes[node["name"]] = prims
    mats = {}
    for m in g["materials"]:
        pbr = m.get("pbrMetallicRoughness", {})

        def img(tex):
            if tex is None:
                return None
            return os.path.join(d, g["images"][g["textures"][tex["index"]]["source"]]["uri"])
        mats[m["name"]] = {"diffuse": img(pbr.get("baseColorTexture")), "normal": img(m.get("normalTexture")),
                           "orm": img(pbr.get("metallicRoughnessTexture")),
                           "double_sided": bool(m.get("doubleSided", False))}
    return nodes, mats


# ------------------------------------------------------------------------ components
def components(F, n):
    """Connected-component label per vertex and per face."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = F.astype(np.int64)
    A = coo_matrix((np.ones(len(F) * 2), (np.r_[F[:, 0], F[:, 1]], np.r_[F[:, 1], F[:, 2]])), shape=(n, n))
    nc, lab = connected_components(A, directed=False)
    return nc, lab, lab[F[:, 0]]


# ----------------------------------------------------------------------- alpha masks
def load_alpha(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255.0


def alpha_at(alpha, uv):
    """Alpha at USD st coordinates (origin bottom-left, wrap)."""
    H, W = alpha.shape
    px = np.clip((np.mod(uv[:, 0], 1.0) * W).astype(np.int64), 0, W - 1)
    py = np.clip(((1.0 - np.mod(uv[:, 1], 1.0)) * H).astype(np.int64), 0, H - 1)
    return alpha[py, px]


def alpha_to_geometry(part, alpha, subdiv=0, thresh=0.5):
    """Cut an alpha-textured surface to its opaque shape.

    subdiv=0: drop triangles whose 3 corners, edge midpoints and centroid are all transparent
              (removes empty card fragments, keeps real needle geometry).
    subdiv=k: split every triangle into k*k sub-triangles and keep those whose centroid is opaque
              (turns fern / moss cards into frond-shaped geometry; edges are stair-stepped at the
              sub-triangle size)."""
    V, F, UV = part.V.astype(np.float64), part.F.astype(np.int64), part.UV.astype(np.float64)
    if subdiv <= 0:
        bary = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5], [1 / 3, 1 / 3, 1 / 3]])
        op = np.zeros(len(F), bool)
        for b in bary:
            uv = b[0] * UV[F[:, 0]] + b[1] * UV[F[:, 1]] + b[2] * UV[F[:, 2]]
            op |= alpha_at(alpha, uv) >= thresh
        V2, F2, UV2 = compact(part.V, part.F, part.UV, op)
        return Part(part.role, part.mat, V2, F2, UV2)
    k = subdiv
    # barycentric lattice of a triangle split k x k
    pts = [(i, j) for i in range(k + 1) for j in range(k + 1 - i)]
    pid = {p: n for n, p in enumerate(pts)}
    sub = []
    for i in range(k):
        for j in range(k - i):
            sub.append((pid[(i, j)], pid[(i + 1, j)], pid[(i, j + 1)]))
            if i + j < k - 1:
                sub.append((pid[(i + 1, j)], pid[(i + 1, j + 1)], pid[(i, j + 1)]))
    B = np.array([[1 - (i + j) / k, i / k, j / k] for i, j in pts])  # weights of corners 0,1,2
    sub = np.array(sub)
    nV = np.einsum("pb,fbc->fpc", B, V[F])  # (faces, lattice pts, 3)
    nUV = np.einsum("pb,fbc->fpc", B, UV[F])
    cen_uv = nUV[:, sub].mean(2)  # (faces, subtris, 2)
    keep = alpha_at(alpha, cen_uv.reshape(-1, 2)).reshape(cen_uv.shape[:2]) >= thresh
    nf, npt = len(F), len(pts)
    Fs = (np.arange(nf)[:, None, None] * npt + sub[None]).reshape(-1, 3)
    V2, F2, UV2 = compact(nV.reshape(-1, 3), Fs, nUV.reshape(-1, 2), keep.ravel())
    # weld duplicated lattice vertices on shared edges is not needed for rendering; keep as is
    return Part(part.role, part.mat, V2.astype(np.float32), F2, UV2.astype(np.float32))


def drop_components(part, keep_frac, rng):
    """Keep a random subset of the loose pieces (needles) of a foliage part."""
    nc, vl, fl = components(part.F, len(part.V))
    keep_c = rng.random(nc) < keep_frac
    V, F, UV = compact(part.V, part.F, part.UV, keep_c[fl])
    return Part(part.role, part.mat, V, F, UV)


# ------------------------------------------------------------------------ decimation
def blender_bin():
    return os.environ.get("BLENDER", os.path.expanduser("~/tools/blender-4.2/blender"))


def blender_env():
    """Environment for Blender subprocesses: without Isaac Sim's USD/Python library paths."""
    env = {k: v for k, v in os.environ.items() if k not in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}
    return env


def decimate(part, ratio, cache_dir, tag):
    """Quadric collapse decimation in Blender (cached on the input hash)."""
    if ratio >= 0.999 or part.tris == 0:
        return part
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1()
    for a in (part.V, part.F, part.UV):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{ratio:.5f}".encode())
    key = h.hexdigest()[:16]
    src = os.path.join(cache_dir, f"{tag}_{key}_in.npz")
    dst = os.path.join(cache_dir, f"{tag}_{key}_out.npz")
    if not os.path.exists(dst):
        np.savez(src, V=part.V, F=part.F, UV=part.UV)
        cmd = [blender_bin(), "-b", "--factory-startup", "--disable-autoexec", "-noaudio", "--python-exit-code", "1",
               "--python", os.path.join(HERE, "blender_decimate.py"), "--", src, dst, str(ratio)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=blender_env())
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(f"blender decimate failed for {tag}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        archive = os.path.join(cache_dir, "archive")
        os.makedirs(archive, exist_ok=True)
        os.rename(src, os.path.join(archive, f"{time.time_ns()}_{os.path.basename(src)}"))
    d = np.load(dst)
    return Part(part.role, part.mat, d["V"], d["F"], d["UV"])


# ----------------------------------------------------------------------- trunk shape
def trunk_profile(V, dz=0.5, zmax=None):
    """Slices of a (roughly vertical) trunk mesh: rows (z_lo, z_hi, cx, cy, r_mean, r_max)."""
    V = np.asarray(V, np.float64)
    z0 = max(V[:, 2].min(), 0.0)
    z1 = V[:, 2].max() if zmax is None else min(zmax, V[:, 2].max())
    rows = []
    z = z0
    while z < z1 - 1e-6:
        s = V[(V[:, 2] >= z) & (V[:, 2] < z + dz)]
        if len(s) >= 6:
            c = 0.5 * (s[:, :2].min(0) + s[:, :2].max(0))
            r = np.linalg.norm(s[:, :2] - c, axis=1)
            rows.append([z, min(z + dz, z1), c[0], c[1], float(np.mean(r)), float(np.percentile(r, 95))])
        z += dz
    return np.array(rows)


def dbh_from_profile(prof, h=1.3):
    i = np.argmin(np.abs(0.5 * (prof[:, 0] + prof[:, 1]) - h))
    return 2.0 * prof[i, 4]


# ----------------------------------------------------------- approximate convex parts
def convex_pieces(V, k, rng_seed=0, min_pts=8):
    """Approximate convex decomposition: k-means clusters of the vertices, one hull each (<=64 verts).

    Returns a list of (points, triangles) hull meshes. Good enough for logs (split along their
    length) and stumps (body + root flares); not exact for deep concavities."""
    from scipy.cluster.vq import kmeans2
    from scipy.spatial import ConvexHull
    V = np.asarray(V, np.float64)
    if k <= 1:
        groups = [V]
    else:
        _, lab = kmeans2(V, k, seed=rng_seed, minit="++")
        groups = [V[lab == i] for i in range(k) if (lab == i).sum() >= min_pts]
    out = []
    for P in groups:
        try:
            h = ConvexHull(P)
        except Exception:
            continue
        P = P[h.vertices]
        if len(P) > 64:  # PhysX hull vertex limit: keep the farthest-point subset
            sel = [int(np.argmax(np.linalg.norm(P - P.mean(0), axis=1)))]
            dmin = np.linalg.norm(P - P[sel[0]], axis=1)
            while len(sel) < 64:
                j = int(np.argmax(dmin))
                sel.append(j)
                dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
            P = P[sel]
        h = ConvexHull(P)
        out.append((P.astype(np.float32), h.simplices.astype(np.int32)))
    return out


def vertex_normals(V, F):
    V = V.astype(np.float64)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for i in range(3):
        for c in range(3):
            N[:, c] += np.bincount(F[:, i], weights=n[:, c], minlength=len(V))
    L = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.where(L > 1e-20, N / np.maximum(L, 1e-20), np.array([0, 0, 1.0]))
    return N.astype(np.float32)
