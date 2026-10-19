"""Scene assembly for forest_fi.usd (stage `scene` of scripts/build_forest_fi.py).

Layout of the output stage (Z up, metres, defaultPrim /World):
  /World/Prototypes        class prim (abstract: never rendered, never simulated) holding one Xform
                           per prototype with its meshes and its own Looks (materials).
  /World/Terrain/ground    100 x 100 m heightfield mesh, 0.5 m grid, static triangle-mesh collider,
                           three forest-floor materials (GeomSubsets).
  /World/Forest/Trees/<id> tree root (translate + yaw) with instanceable children
                           crown / trunk / skirt_mid / skirt_low (internal references to prototypes).
  /World/Forest/Understory/<kind>/<id>   instanceable understory objects.
  /World/Forest/Colliders  invisible PhysX colliders in world coordinates: trunk cylinders (per
                           tree, following the measured stem profile), sapling stems, convex pieces
                           for logs / stumps, convex hulls for rocks. Foliage has no collider.
  /World/Lights            sun (DistantLight) + sky (DomeLight).
"""
import json
import math
import os
import shutil

import numpy as np

import meshlib as ml
from protos import load_proto

AREA = 100.0  # m, square centred on the origin
GRID = 0.5  # m terrain resolution
CLEAR_FLAT_R = 3.8  # terrain vertices flat (z = 0) inside this radius: the mesh is exactly flat within 3 m
CLEAR_BLEND_R = 7.5  # ... blending back to the undulating terrain by this radius
TREE_CLEAR_R = 4.0  # no stem surface within this distance of the spawn point
SPAWN = (0.0, 0.0, 0.07)
TREE_SINK = 0.06  # tree bases sit this far below the local terrain
SPECIES_FRAC = {"pine": 0.55, "spruce": 0.30, "birch": 0.15}
STEMS_TARGET = 780  # stems in the 1 ha plot (managed, thinned: 600-900 /ha)
MIN_SPACING = 2.0
SUN_INTENSITY = 25000.0
DOME_INTENSITY = 35000.0  # multiplies the HDRI radiance

# Naslund height curves h = 1.3 + (d / (a + b d))^2 (d in cm), fitted to two anchor points each
# (Finnish managed stands: pine 15 cm -> 15.5 m, 35 cm -> 24 m; spruce 15 -> 15 m, 35 -> 25 m;
# birch 15 -> 16 m, 30 -> 22 m).
def _naslund(d1, h1, d2, h2):
    k1, k2 = d1 / math.sqrt(h1 - 1.3), d2 / math.sqrt(h2 - 1.3)
    b = (k2 - k1) / (d2 - d1)
    return k1 - b * d1, b


NASLUND = {"pine": _naslund(15, 15.5, 35, 24.0), "spruce": _naslund(15, 15.0, 35, 25.0),
           "birch": _naslund(15, 16.0, 30, 22.0)}
DBH_DIST = {"pine": (26.0, 5.0), "spruce": (24.0, 5.5), "birch": (21.0, 4.0)}  # cm, normal, clipped 15-35


# ============================================================================ terrain
class Terrain:
    """Gently undulating heightfield (+-0.5-1 m over tens of metres, micro-relief of hummocks),
    flat clearing at the spawn point, exact z(0, 0) = 0."""

    def __init__(self, rng):
        self.waves = []
        for _ in range(7):
            lam = rng.uniform(18, 70)
            th = rng.uniform(0, np.pi)
            self.waves.append((2 * np.pi / lam * np.cos(th), 2 * np.pi / lam * np.sin(th), rng.uniform(0, 2 * np.pi),
                               rng.uniform(0.12, 0.28)))
        self.bumps = [(rng.uniform(-55, 55), rng.uniform(-55, 55), rng.uniform(3, 9), rng.uniform(-0.45, 0.35))
                      for _ in range(14)]
        self.micro = [(2 * np.pi / l * np.cos(t), 2 * np.pi / l * np.sin(t), p, a) for l, t, p, a in
                      zip(rng.uniform(1.5, 4.0, 12), rng.uniform(0, np.pi, 12), rng.uniform(0, 2 * np.pi, 12),
                          rng.uniform(0.01, 0.03, 12))]
        n = int(round(AREA / GRID)) + 1
        self.xs = np.linspace(-AREA / 2, AREA / 2, n)
        X, Y = np.meshgrid(self.xs, self.xs, indexing="ij")
        self.Z = self.analytic(X, Y)

    def raw(self, x, y):
        z = np.zeros_like(x, dtype=np.float64)
        for kx, ky, p, a in self.waves + self.micro:
            z += a * np.sin(kx * x + ky * y + p)
        for bx, by, r, a in self.bumps:
            z += a * np.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * r * r))
        return z

    def analytic(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        z0 = self.raw(np.zeros(1), np.zeros(1))[0]
        r = np.hypot(x, y)
        t = np.clip((r - CLEAR_FLAT_R) / (CLEAR_BLEND_R - CLEAR_FLAT_R), 0, 1)
        w = t * t * (3 - 2 * t)
        return w * (self.raw(x, y) - z0)

    def z_at(self, x, y):
        """Height of the triangulated mesh (quads split along the (i,j)-(i+1,j+1) diagonal)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(self.xs)
        spacing = float(self.xs[1] - self.xs[0])
        fx = np.clip((x - self.xs[0]) / spacing, 0, n - 1 - 1e-9)
        fy = np.clip((y - self.xs[0]) / spacing, 0, n - 1 - 1e-9)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        u, v = fx - i, fy - j
        z00, z10, z01, z11 = self.Z[i, j], self.Z[i + 1, j], self.Z[i, j + 1], self.Z[i + 1, j + 1]
        upper = u >= v  # triangle (00, 10, 11)
        return np.where(upper, z00 + u * (z10 - z00) + v * (z11 - z10), z00 + v * (z01 - z00) + u * (z11 - z01))

    def mesh(self):
        n = len(self.xs)
        X, Y = np.meshgrid(self.xs, self.xs, indexing="ij")
        V = np.c_[X.ravel(), Y.ravel(), self.Z.ravel()]
        idx = np.arange(n * n).reshape(n, n)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()
        F = np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])  # counter-clockwise seen from +z
        return V, F


class TerrainZ(Terrain):
    """Terrain from the stored grid (forest_fi_stand.json) with the same triangle interpolation."""

    def __init__(self, d):  # noqa: super().__init__ deliberately not called
        self.Z = np.asarray(d["z"], np.float64)
        n = self.Z.shape[0]
        self.xs = np.linspace(-d["area"] / 2, d["area"] / 2, n)


def smooth_field(rng, x, y, scales=(25, 12), n=6):
    f = np.zeros_like(np.asarray(x, float))
    for s in scales:
        for _ in range(n):
            th = rng.uniform(0, np.pi)
            k = 2 * np.pi / (s * rng.uniform(0.7, 1.4))
            f += np.sin(k * (np.cos(th) * x + np.sin(th) * y) + rng.uniform(0, 2 * np.pi)) / math.sqrt(n * len(scales))
    return f


# ============================================================================== stand
def place_trees(rng, terrain, log):
    """Hard-core (2 m) inhibition process with mild density clustering and thinning regularity."""
    fr = np.random.default_rng(rng.integers(1 << 31))
    dens = lambda x, y: np.exp(0.55 * smooth_field(fr, x, y, scales=(30,), n=4))  # noqa: E731
    pts = []
    half = AREA / 2 - 0.6
    tries = 0
    grid = {}

    def near(x, y, r):
        gx, gy = int(math.floor(x / 4)), int(math.floor(y / 4))
        out = []
        for i in range(gx - 1, gx + 2):
            for j in range(gy - 1, gy + 2):
                out += grid.get((i, j), [])
        return [p for p in out if (p[0] - x) ** 2 + (p[1] - y) ** 2 < r * r]
    dmax = 2.2
    while len(pts) < STEMS_TARGET and tries < 400000:
        tries += 1
        x, y = rng.uniform(-half, half, 2)
        if math.hypot(x, y) < TREE_CLEAR_R + 0.45:
            continue
        if rng.random() > dens(np.array([x]), np.array([y]))[0] / dmax:
            continue
        nb = near(x, y, 3.2)
        if any((p[0] - x) ** 2 + (p[1] - y) ** 2 < MIN_SPACING ** 2 for p in nb):
            continue
        if nb and rng.random() > 0.3:  # thinning removes most close pairs
            continue
        pts.append((x, y))
        grid.setdefault((int(math.floor(x / 4)), int(math.floor(y / 4))), []).append((x, y))
    log(f"trees placed: {len(pts)} after {tries} candidates")
    return np.array(pts)


def assign_species(rng, P, terrain, fractions=None):
    fractions = SPECIES_FRAC if fractions is None else fractions
    n = len(P)
    fr = np.random.default_rng(rng.integers(1 << 31))
    zt = terrain.z_at(P[:, 0], P[:, 1])
    s_spruce = smooth_field(fr, P[:, 0], P[:, 1], scales=(22,), n=5) - 1.2 * (zt - zt.mean()) / max(zt.std(), 1e-6) * 0.5 \
        + rng.normal(0, 0.35, n)
    s_birch = smooth_field(fr, P[:, 0], P[:, 1], scales=(15,), n=5) + rng.normal(0, 0.45, n)
    sp = np.array(["pine"] * n, dtype=object)
    n_sp = int(round(fractions["spruce"] * n))
    n_b = int(round(fractions.get("birch", 0) * n))
    order = np.argsort(-s_spruce)
    sp[order[:n_sp]] = "spruce"
    rest = np.flatnonzero(sp == "pine")
    sp[rest[np.argsort(-s_birch[rest])[:n_b]]] = "birch"
    return sp


def tree_dims(rng, species):
    mu, sd = DBH_DIST[species]
    d = float(np.clip(rng.normal(mu, sd), 15.0, 35.0))
    a, b = NASLUND[species]
    h = 1.3 + (d / (a + b * d)) ** 2
    h *= rng.normal(1.0, 0.05)
    return d / 100.0, float(h)


# ======================================================================= materials
def gltf_materials(src):
    out = {}
    ph = os.path.join(src, "polyhaven")
    for asset in sorted(os.listdir(ph)):
        f = os.path.join(ph, asset, f"{asset}_2k.gltf")
        if not os.path.exists(f):
            continue
        g = json.load(open(f))

        def img(t):
            return None if t is None else os.path.join(ph, asset, g["images"][g["textures"][t["index"]]["source"]]["uri"])
        for m in g["materials"]:
            pbr = m.get("pbrMetallicRoughness", {})
            mr = img(pbr.get("metallicRoughnessTexture"))
            out[m["name"]] = dict(diffuse=img(pbr.get("baseColorTexture")), normal=img(m.get("normalTexture")),
                                  orm=mr if mr and "_arm_" in mr else None,
                                  rough_tex=mr if mr and "_rough_" in mr else None)
    return out


MAT_TWEAKS = {  # tints toward Finnish boreal colours / photo-real brightness
    "fir_tree_01_twig": dict(tint=(0.80, 0.92, 0.80)),
    "fir_sapling_medium_twigs": dict(tint=(0.80, 0.92, 0.80)),
    "fir_tree_01_trunk_c": dict(tint=(0.85, 0.85, 0.85)),
    "pine_tree_01_trunk_c": dict(tint=(1.15, 0.95, 0.80)),
}
PROCEDURAL_MATS = {
    "birch_bark": dict(diffuse="birch_bark_diff.png", rough=0.7),
    "birch_bark_base": dict(diffuse="birch_bark_base_diff.png", rough=0.85),
    "birch_branch": dict(color=(0.20, 0.15, 0.13), rough=0.8),
    "birch_leaf": dict(color=(0.20, 0.33, 0.08), rough=0.55),
    "bilberry_stem": dict(color=(0.16, 0.30, 0.08), rough=0.7),
    "bilberry_leaf": dict(color=(0.12, 0.26, 0.06), rough=0.5),
}
GROUND_MATS = {  # name: (polyhaven texture asset, tile m, tint)
    "floor_moss": ("forest_leaves_02", 2.0, (0.70, 0.95, 0.55)),
    "floor_needles": ("forrest_ground_03", 2.5, (0.95, 0.90, 0.85)),
    "floor_bilberry": ("leafy_grass", 2.0, (0.75, 0.95, 0.65)),
}


def birch_textures(tex_dir, seed):
    """Procedural silver-birch bark (white with dark lenticels) and dark fissured base bark."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    W, H = 1024, 2048
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.86 + 0.04 * np.sin(xx / W * 2 * np.pi * 3 + rng.uniform(0, 6)) + rng.normal(0, 0.025, (H, W))
    img = np.stack([base * 0.98, base * 0.96, base * 0.90], -1)
    for _ in range(900):  # horizontal lenticels
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        L, T = rng.integers(8, 70), rng.integers(2, 6)
        dark = rng.uniform(0.15, 0.35)
        xs = (np.arange(cx, cx + L) % W)
        img[cy:cy + T, :][:, xs] = dark
    for _ in range(40):  # darker diamond scars
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        r = rng.integers(15, 45)
        m = (np.abs(xx - cx) / (2 * r) + np.abs(yy - cy) / r) < 1
        img[m] *= 0.25
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(tex_dir, "birch_bark_diff.png"))
    # base: dark grey-brown with vertical cracks
    n = rng.normal(0, 1, (H // 8, W // 8))
    from PIL import ImageFilter
    nb = np.asarray(Image.fromarray(((n - n.min()) / np.ptp(n) * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
                    .filter(ImageFilter.GaussianBlur(3)), np.float32) / 255
    cracks = np.abs(np.sin(xx / W * 2 * np.pi * 14 + 3 * nb)) < 0.12
    v = 0.22 + 0.10 * nb
    v[cracks] *= 0.45
    img = np.stack([v * 1.05, v * 0.98, v * 0.9], -1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(tex_dir, "birch_bark_base_diff.png"))


def copy_texture(src, tex_dir, max_size):
    from PIL import Image
    name = os.path.basename(src)
    dst = os.path.join(tex_dir, name)
    if not os.path.exists(dst):
        im = Image.open(src)
        if max(im.size) > max_size:
            im = im.resize((max_size, max_size), Image.LANCZOS)
        im.save(dst, quality=92) if name.endswith(".jpg") else im.save(dst, optimize=True)
    return name


def define_material(stage, path, spec, tex_rel):
    from pxr import Gf, Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    def tex(name, file, cs):
        sh.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_rel + file))
        sh.GetInput(name).GetAttr().SetColorSpace(cs)
    if spec.get("diffuse"):
        tex("diffuse_texture", spec["diffuse"], "sRGB")
    else:
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["color"]))
    if spec.get("normal"):
        tex("normalmap_texture", spec["normal"], "raw")
    if spec.get("orm"):
        tex("ORM_texture", spec["orm"], "raw")
        sh.CreateInput("enable_ORM_texture", Sdf.ValueTypeNames.Bool).Set(True)
    elif spec.get("rough_tex"):
        tex("reflectionroughness_texture", spec["rough_tex"], "raw")
        sh.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(spec.get("rough", 0.8))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec.get("tint", (1, 1, 1))))
    if spec.get("tile"):
        sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1 / spec["tile"], 1 / spec["tile"]))
    out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for o in (mat.CreateSurfaceOutput("mdl"), mat.CreateDisplacementOutput("mdl"), mat.CreateVolumeOutput("mdl")):
        o.ConnectToSource(out)
    return mat


# ======================================================================= USD helpers
def write_mesh(stage, path, V, F, UV=None, double_sided=False, normals=True):
    from pxr import Sdf, UsdGeom, Vt
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F, np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(bool(double_sided))
    if normals:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(ml.vertex_normals(V, F)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if UV is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(UV, np.float32)))
    lo, hi = V.min(0), V.max(0)
    m.CreateExtentAttr([tuple(map(float, lo)), tuple(map(float, hi))])
    return m


def xform(prim_xf, t, yaw_deg=0.0, scale=None):
    from pxr import Gf
    prim_xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, t)))
    if yaw_deg:
        prim_xf.AddRotateZOp().Set(float(yaw_deg))
    if scale is not None:
        prim_xf.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def world_pts(P, t, yaw_deg, scale):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (np.asarray(P, float) * np.asarray(scale, float)) @ R.T + np.asarray(t, float)


# ===================================================================== the builder
def build_scene(a, work, log, *, name="forest_fi", species_frac=None,
                terrain_factory=Terrain, postprocess=None):
    """Shared stand builder; optional seasonal hooks leave the FOREST_FI defaults intact.

    postprocess(stage, stand, tex_dir, tex_rel) may edit the scene and its stand table
    before either is saved. Existing generated files are archived on a rebuild.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    scene_name = name  # Prototype/material loops below also use the local name variable.
    rng = np.random.default_rng(a.seed)
    pdir = os.path.join(work, "protos")
    PM = json.load(open(os.path.join(pdir, "protos.json")))
    out = os.path.abspath(a.out)
    out_dir = os.path.dirname(out)
    tex_dir = os.path.join(out_dir, "textures", name)
    tex_rel = f"./textures/{name}/"
    if os.path.exists(out) or os.path.isdir(tex_dir):
        from datetime import datetime
        archive = os.path.join(out_dir, "archive", name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(archive)
        if os.path.exists(out):
            shutil.move(out, os.path.join(archive, os.path.basename(out)))
        if os.path.isdir(tex_dir):
            archived_textures = os.path.join(archive, "textures")
            os.makedirs(archived_textures)
            shutil.move(tex_dir, os.path.join(archived_textures, name))
        license_path = os.path.join(out_dir, f"{name}_LICENSES.md")
        if os.path.isfile(license_path):
            shutil.copy2(license_path, archive)
        old_stand = os.path.join(a.report_dir, f"{name}_stand.json")
        if os.path.isfile(old_stand):
            shutil.copy2(old_stand, archive)
        if name in ("forest_snow", "forest_mid_snow", "forest_pine_hills"):
            for filename in ("provenance.json", "needle_alpha_check.json", f"{name}_needle_alpha.json"):
                path = os.path.join(a.report_dir, filename)
                if os.path.isfile(path):
                    shutil.copy2(path, archive)
    os.makedirs(tex_dir)
    os.makedirs(a.report_dir, exist_ok=True)

    terrain = terrain_factory(np.random.default_rng(a.seed + 1))
    # --------------------------------------------------------------- trees
    P = place_trees(np.random.default_rng(a.seed + 2), terrain, log)
    species = assign_species(np.random.default_rng(a.seed + 3), P, terrain, species_frac)
    trng = np.random.default_rng(a.seed + 4)
    crowns = {sp: [k for k, m in PM.items() if m["kind"] in ("crown", "tree") and m.get("species") == sp]
              for sp in SPECIES_FRAC}
    trees = []
    for i, ((x, y), sp) in enumerate(zip(P, species)):
        d, h = tree_dims(trng, sp)
        cands = crowns[sp]
        Hs = np.array([PM[c]["height"] for c in cands])
        ratio = h / Hs
        ok = np.flatnonzero((ratio >= 0.8) & (ratio <= 1.2))
        if len(ok):
            w = np.exp(-np.abs(np.log(ratio[ok])) / 0.12)
            ci = int(ok[trng.choice(len(ok), p=w / w.sum())])
        else:
            ci = int(np.argmin(np.abs(np.log(ratio))))
        crown = cands[ci]
        sz = float(np.clip(h / Hs[ci], 0.8, 1.2))
        sc = float(np.clip(sz * trng.uniform(0.9, 1.1), 0.8, 1.2))
        yaw = float(trng.uniform(0, 360))
        z = float(terrain.z_at(x, y)) - TREE_SINK
        t = dict(id=f"{sp}_{i:04d}", species=sp, x=float(x), y=float(y), z=z, yaw=yaw, crown=crown,
                 crown_scale=[sc, sc, sz], height=PM[crown]["height"] * sz, target_dbh=d, target_height=h)
        if sp == "birch":
            sxy = float(np.clip(d / PM[crown]["dbh"], 0.8, 1.2))
            t["crown_scale"] = [sxy, sxy, sz]
            t["trunk"], t["trunk_scale"] = None, None
            t["dbh"] = PM[crown]["dbh"] * sxy
            t["profile"] = PM[crown]["profile"]
            t["prof_scale"] = [sxy, sz]
            t["crown_base"] = PM[crown]["crown_base"] * sz
        else:
            var = PM[crown]["variant"]
            best = None
            for tv in (80, 120):
                tn = f"{sp}_{var}_trunk_{tv}"
                f = d / PM[tn]["dbh"]
                cost = abs(math.log(f)) if 0.8 <= f <= 1.2 else 1 + abs(math.log(f))
                if best is None or cost < best[0]:
                    best = (cost, tn, float(np.clip(f, 0.8, 1.2)))
            _, tn, f = best
            t["trunk"], t["trunk_scale"] = tn, [f, f, sz]
            t["dbh"] = PM[tn]["dbh"] * f
            t["profile"] = PM[tn]["profile"]
            t["prof_scale"] = [f, sz]
            t["crown_base"] = PM[crown]["crown_base"] * sz
            t["skirts"] = []
            if sp == "spruce":
                r0 = math.hypot(x, y)
                u = trng.random()
                if u > 0.35 and f"{sp}_{var}_skirt_mid" in PM and r0 > 6.5:
                    t["skirts"].append(f"{sp}_{var}_skirt_mid")
                if u > 0.65 and f"{sp}_{var}_skirt_low" in PM and r0 > 8.0:
                    t["skirts"].append(f"{sp}_{var}_skirt_low")
                if t["skirts"]:
                    t["crown_base"] = min(PM[s]["z_range"][0] for s in t["skirts"]) * sz
        trees.append(t)
    log(f"species: { {s: int((species == s).sum()) for s in SPECIES_FRAC} }")

    # trunk collider segments (world frame, vertical cylinders following the stem centre line)
    BOUNDS = [-0.3, 0.5, 1.5, 2.5, 3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0]
    for t in trees:
        prof = np.array(t["profile"])
        fxy, sz = t["prof_scale"]
        top = prof[prof[:, 4] * fxy >= 0.035]
        ztop = float(top[:, 1].max()) if len(top) else float(prof[:, 1].max())
        segs = []
        for z0, z1 in zip(BOUNDS[:-1], BOUNDS[1:]):
            if z0 >= ztop:
                break
            z1 = min(z1, ztop)
            sel = prof[(prof[:, 1] > max(z0, 0)) & (prof[:, 0] < z1)]
            if not len(sel):
                continue
            r = float(np.mean(sel[:, 4])) * fxy
            cxy = sel[:, 2:4].mean(0) * fxy
            c = world_pts([[cxy[0], cxy[1], 0.5 * (z0 + z1) * sz]], [t["x"], t["y"], t["z"]], t["yaw"], [1, 1, 1])[0]
            segs.append(dict(c=c.tolist(), r=r, h=(z1 - z0) * sz))
        t["colliders"] = segs
        t["collider_top"] = t["z"] + ztop * sz

    # --------------------------------------------------------------- understory
    urng = np.random.default_rng(a.seed + 5)
    TP = np.array([[t["x"], t["y"]] for t in trees])
    TR = np.array([t["dbh"] / 2 for t in trees])
    from scipy.spatial import cKDTree
    tkd = cKDTree(TP)
    sp_arr = np.array([t["species"] for t in trees])
    objs = []  # dict(kind, proto, x, y, z, yaw, scale, collider)

    def free(x, y, r_spawn, r_trunk):
        if math.hypot(x, y) < r_spawn or abs(x) > AREA / 2 - 0.5 or abs(y) > AREA / 2 - 0.5:
            return False
        d, i = tkd.query([x, y])
        return d - TR[i] >= r_trunk

    def occupied_ok(x, y, others, rmin):
        return all((o["x"] - x) ** 2 + (o["y"] - y) ** 2 >= rmin * rmin for o in others)

    def add(kind, proto, x, y, yaw, scale, sink=0.0, collider=None, **kw):
        z = float(terrain.z_at(x, y)) - sink
        o = dict(kind=kind, proto=proto, x=float(x), y=float(y), z=z, yaw=float(yaw), scale=[float(s) for s in scale],
                 collider=collider, **kw)
        objs.append(o)
        return o

    # understory spruce (5-9 m), near spruces, in gaps
    usp = [k for k in PM if PM[k]["kind"] == "uspruce"]
    spr = TP[sp_arr == "spruce"]
    n = 0
    while n < 35:
        c = spr[urng.integers(len(spr))]
        x, y = c + urng.normal(0, 4.0, 2)
        if not free(x, y, 7.5, 1.8):
            continue
        if not occupied_ok(x, y, [o for o in objs if o["kind"] == "uspruce"], 2.5):
            continue
        p = usp[urng.integers(len(usp))]
        s = urng.uniform(0.85, 1.15)
        add("uspruce", p, x, y, urng.uniform(0, 360), [s, s, s], sink=0.04, collider="stem", height=PM[p]["height"] * s)
        n += 1
    # saplings 0.5-3 m in regeneration clusters (gaps), 70 % spruce
    saps = {"spruce": [k for k in PM if PM[k]["kind"] == "sapling" and PM[k]["species"] == "spruce"],
            "pine": [k for k in PM if PM[k]["kind"] == "sapling" and PM[k]["species"] == "pine"]}
    n_cl = 0
    while n_cl < 42:
        cx, cy = urng.uniform(-48, 48, 2)
        if not free(cx, cy, 6.0, 1.5):
            continue
        n_cl += 1
        spc = "spruce" if urng.random() < 0.7 else "pine"
        for _ in range(urng.integers(4, 14)):
            x, y = cx + urng.normal(0, 1.6), cy + urng.normal(0, 1.6)
            if not free(x, y, 3.0, 0.6):
                continue
            p = saps[spc][urng.integers(len(saps[spc]))]
            hgt = float(np.clip(urng.lognormal(math.log(1.3), 0.45), 0.5, 3.0))
            s = hgt / PM[p]["height"]
            add("sapling", p, x, y, urng.uniform(0, 360), [s, s, s], sink=0.02,
                collider="stem" if hgt >= 1.0 else None, height=hgt)
    # deadwood: logs, decayed stumps, low thinning stumps
    def seg_clear(x, y, yaw, L, r):
        d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
        for s in np.linspace(-L / 2, L / 2, 9):
            px, py = np.array([x, y]) + s * d
            if not free(px, py, 3.0, r):
                return False
        return True
    n = 0
    while n < 15:
        x, y = urng.uniform(-47, 47, 2)
        yaw = urng.uniform(0, 360)
        sl, sd = urng.uniform(1.0, 1.9), urng.uniform(0.9, 1.4)
        if not seg_clear(x, y, yaw, 3.05 * sl, 0.45):
            continue
        add("log", "log", x, y, yaw, [sl, sd, sd], sink=0.06, collider="pieces6")
        n += 1
    n = 0
    while n < 20:
        x, y = urng.uniform(-48, 48, 2)
        if not free(x, y, 3.5, 1.0):
            continue
        s = urng.uniform(0.7, 1.0)
        add("stump", f"stump_{'ab'[urng.integers(2)]}", x, y, urng.uniform(0, 360), [s, s, s * urng.uniform(0.9, 1.1)],
            sink=0.05, collider="pieces7")
        n += 1
    n = 0
    while n < 120:
        x, y = urng.uniform(-48.5, 48.5, 2)
        if not free(x, y, 3.0, 1.3):
            continue
        sxy = urng.uniform(0.42, 0.6)
        add("cut_stump", f"stump_{'ab'[urng.integers(2)]}", x, y, urng.uniform(0, 360), [sxy, sxy, urng.uniform(0.28, 0.42)],
            sink=0.03, collider="pieces5")
        n += 1
    rocks = [k for k in PM if PM[k]["kind"] == "rock"]
    n = 0
    while n < 45:
        x, y = urng.uniform(-48, 48, 2)
        p = rocks[urng.integers(len(rocks))]
        ext = np.array(PM[p]["bbox"][1]) - np.array(PM[p]["bbox"][0])
        s = urng.uniform(0.65, 1.0) if n < 4 else urng.uniform(0.12, 0.45)
        rr = 0.5 * max(ext[0], ext[1]) * s
        if not free(x, y, 3.0 + rr, rr + 0.4):
            continue
        sz = s * urng.uniform(0.8, 1.1)
        add("rock", p, x, y, urng.uniform(0, 360), [s, s, sz], sink=0.3 * ext[2] * sz, collider="hull")
        n += 1
    # litter: dry branches, exposed pine roots near pine bases
    dbs = [k for k in PM if k.startswith("drybranch_")]
    n = 0
    while n < 150:
        x, y = urng.uniform(-49, 49, 2)
        if not free(x, y, 2.0, 0.35):
            continue
        s = urng.uniform(0.8, 1.6)
        add("drybranch", dbs[urng.integers(len(dbs))], x, y, urng.uniform(0, 360), [s, s, s], sink=0.02)
        n += 1
    pines = np.flatnonzero(sp_arr == "pine")
    for k in urng.choice(pines, 30, replace=False):
        t = trees[k]
        if math.hypot(t["x"], t["y"]) < 6:
            continue
        a_ = urng.uniform(0, 2 * np.pi)
        x, y = t["x"] + 0.5 * math.cos(a_), t["y"] + 0.5 * math.sin(a_)
        s = urng.uniform(0.7, 1.0)
        add("roots", f"roots_{'ab'[urng.integers(2)]}", x, y, urng.uniform(0, 360), [s, s, s], sink=0.03)
    # ground cover
    fr = np.random.default_rng(a.seed + 6)
    gx = np.linspace(-49, 49, 197)
    GX, GY = np.meshgrid(gx, gx, indexing="ij")
    moist = smooth_field(fr, GX, GY, scales=(20, 9), n=5)
    berry = smooth_field(fr, GX, GY, scales=(14, 6), n=5)
    pine_near = np.zeros_like(GX)
    pkd = cKDTree(TP[sp_arr == "pine"])
    d_p, _ = pkd.query(np.c_[GX.ravel(), GY.ravel()])
    pine_near = (d_p < 4.0).reshape(GX.shape)

    def scatter(kind, protos_, n_target, weight, smin, smax, spawn_r=1.5, trunk_r=0.3, sink=0.0):
        w = np.clip(weight, 0, None).ravel()
        w = w / w.sum()
        n = 0
        tries = 0
        while n < n_target and tries < n_target * 30:
            tries += 1
            k = urng.choice(len(w), p=w)
            x, y = GX.ravel()[k] + urng.uniform(-0.25, 0.25), GY.ravel()[k] + urng.uniform(-0.25, 0.25)
            if not free(x, y, spawn_r, trunk_r):
                continue
            s = urng.uniform(smin, smax)
            add(kind, protos_[urng.integers(len(protos_))], x, y, urng.uniform(0, 360), [s, s, s], sink=sink)
            n += 1

    ferns = [k for k in PM if k.startswith("fern_")]
    scatter("fern", ferns, 170, np.exp(1.6 * moist) * (moist > -0.2), 0.8, 1.4)
    scatter("grass", ["grass_a", "grass_b"], 70, np.exp(1.2 * (berry + moist)) * (~pine_near + 0.2), 0.8, 1.3)
    scatter("moss", ["moss_a", "moss_b"], 420, np.exp(1.0 * moist), 0.8, 1.6)
    berry_w = np.exp(2.2 * berry) * (berry > 0.1) * (0.4 + pine_near)
    scatter("bilberry", ["bilberry_a", "bilberry_b"], 1700, berry_w, 0.8, 1.25, trunk_r=0.2)
    counts = {}
    for o in objs:
        counts[o["kind"]] = counts.get(o["kind"], 0) + 1
    log(f"understory: {counts}")

    # =============================================================== USD
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # materials --------------------------------------------------------
    gm = gltf_materials(a.src)
    birch_textures(tex_dir, a.seed)
    mat_specs = {}

    def spec_for(name):
        if name in mat_specs:
            return mat_specs[name]
        if name in PROCEDURAL_MATS:
            s = dict(PROCEDURAL_MATS[name])
        else:
            g = gm[name]
            s = {"rough": 0.85}
            if g["diffuse"]:
                s["diffuse"] = copy_texture(g["diffuse"], tex_dir, 1024)
            if g["normal"]:
                s["normal"] = copy_texture(g["normal"], tex_dir, 1024)
            if g["orm"]:
                s["orm"] = copy_texture(g["orm"], tex_dir, 1024)
            if g["rough_tex"]:
                s["rough_tex"] = copy_texture(g["rough_tex"], tex_dir, 1024)
        s.update(MAT_TWEAKS.get(name, {}))
        mat_specs[name] = s
        return s

    # prototypes ------------------------------------------------------
    protos_root = stage.CreateClassPrim("/World/Prototypes")
    used = sorted({t["crown"] for t in trees} | {t["trunk"] for t in trees if t["trunk"]}
                  | {s for t in trees for s in t.get("skirts", [])} | {o["proto"] for o in objs})
    proto_geo = {}
    total_tris = 0
    hull_cache = {}
    for name in used:
        parts, meta = load_proto(os.path.join(pdir, name + ".npz"))
        pp = Sdf.Path(f"/World/Prototypes/{name}")
        UsdGeom.Xform.Define(stage, pp)
        looks = UsdGeom.Scope.Define(stage, pp.AppendChild("Looks"))
        mats = {}
        for i, p in enumerate(parts):
            if p.mat not in mats:
                mats[p.mat] = define_material(stage, looks.GetPath().AppendChild(p.mat.replace("-", "_")),
                                              spec_for(p.mat), tex_rel)
            m = write_mesh(stage, pp.AppendChild(f"{p.role}_{i}"), p.V, p.F,
                           p.UV if spec_for(p.mat).get("diffuse") else None,
                           double_sided=p.role in ("foliage", "litter", "branch", "sapling"))
            m.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
            UsdShade.MaterialBindingAPI(m).Bind(mats[p.mat])
            total_tris += p.tris
        proto_geo[name] = parts
    log(f"prototypes written: {len(used)}, distinct triangles {total_tris}")

    # terrain ----------------------------------------------------------
    V, F = terrain.mesh()
    tm = write_mesh(stage, "/World/Terrain/ground", V.astype(np.float32), F.astype(np.int32), V[:, :2].astype(np.float32))
    UsdPhysics.CollisionAPI.Apply(tm.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(tm.GetPrim()).CreateApproximationAttr().Set("none")
    cen = V[F].mean(1)
    fr2 = np.random.default_rng(a.seed + 7)
    noise = smooth_field(fr2, cen[:, 0], cen[:, 1], scales=(6, 3), n=4)
    d_con, _ = cKDTree(TP[sp_arr != "birch"]).query(cen[:, :2])
    ber = np.array([o for o in objs if o["kind"] == "bilberry"])
    bkd = cKDTree(np.array([[o["x"], o["y"]] for o in ber]))
    d_b, _ = bkd.query(cen[:, :2])
    cls = np.full(len(F), "floor_moss", dtype=object)
    cls[(d_con < 1.4 + 0.6 * noise) | (noise > 1.4)] = "floor_needles"
    cls[d_b < 0.9] = "floor_bilberry"
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    for gname, (asset, tile, tint) in GROUND_MATS.items():
        tdir = os.path.join(a.src, "polyhaven", asset)
        files = os.listdir(tdir)
        pick = lambda key: next(f for f in files if key in f)  # noqa: E731
        spec = dict(diffuse=copy_texture(os.path.join(tdir, pick("_diff")), tex_dir, 2048),
                    normal=copy_texture(os.path.join(tdir, pick("_nor_gl_")), tex_dir, 2048),
                    orm=copy_texture(os.path.join(tdir, pick("_arm_")), tex_dir, 2048), tile=tile, tint=tint)
        mat = define_material(stage, looks.GetPath().AppendChild(gname), spec, tex_rel)
        sel = np.flatnonzero(cls == gname)
        sub = UsdGeom.Subset.CreateGeomSubset(tm, gname, UsdGeom.Tokens.face, sel.astype(np.int32).tolist(),
                                              "materialBind", UsdGeom.Tokens.partition)
        sub.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
        UsdShade.MaterialBindingAPI(sub).Bind(mat)
    tm.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
    UsdShade.MaterialBindingAPI(tm).Bind(UsdShade.Material(stage.GetPrimAtPath("/World/Looks/floor_moss")))
    floor_cover = {g: round(float((cls == g).mean()), 3) for g in GROUND_MATS}
    log(f"forest floor cover: {floor_cover}")

    # instances --------------------------------------------------------
    def inst(path, proto, t=None, yaw=0.0, scale=None):
        x = UsdGeom.Xform.Define(stage, path)
        prim = x.GetPrim()
        prim.GetReferences().AddInternalReference(Sdf.Path(f"/World/Prototypes/{proto}"))
        prim.SetInstanceable(True)
        if t is not None or scale is not None:
            xform(x, t if t is not None else (0, 0, 0), yaw, scale)
        return x

    UsdGeom.Scope.Define(stage, "/World/Forest")
    UsdGeom.Scope.Define(stage, "/World/Forest/Trees")
    for t in trees:
        root = UsdGeom.Xform.Define(stage, f"/World/Forest/Trees/{t['id']}")
        xform(root, (t["x"], t["y"], t["z"]), t["yaw"])
        root.GetPrim().SetCustomDataByKey("forest_fi", {"species": t["species"], "dbh_m": round(t["dbh"], 3),
                                                       "height_m": round(t["height"], 2),
                                                       "crown_base_m": round(t["crown_base"], 2)})
        inst(root.GetPath().AppendChild("crown"), t["crown"], scale=t["crown_scale"])
        if t["trunk"]:
            inst(root.GetPath().AppendChild("trunk"), t["trunk"], scale=t["trunk_scale"])
        for s in t.get("skirts", []):
            inst(root.GetPath().AppendChild("skirt_" + s.rsplit("_", 1)[1]), s, scale=t["crown_scale"])
    UsdGeom.Scope.Define(stage, "/World/Forest/Understory")
    kinds_seen = {}
    for o in objs:
        k = o["kind"]
        if k not in kinds_seen:
            UsdGeom.Scope.Define(stage, f"/World/Forest/Understory/{k}")
            kinds_seen[k] = 0
        o["id"] = f"{k}_{kinds_seen[k]:04d}"
        kinds_seen[k] += 1
        inst(f"/World/Forest/Understory/{k}/{o['id']}", o["proto"], (o["x"], o["y"], o["z"]), o["yaw"], o["scale"])

    # colliders --------------------------------------------------------
    def invisible_collider(prim):
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(prim)

    UsdGeom.Scope.Define(stage, "/World/Forest/Colliders")
    UsdGeom.Scope.Define(stage, "/World/Forest/Colliders/Trunks")
    n_cyl = 0
    for t in trees:
        for k, s in enumerate(t["colliders"]):
            c = UsdGeom.Cylinder.Define(stage, f"/World/Forest/Colliders/Trunks/{t['id']}_s{k}")
            c.CreateAxisAttr("Z")
            c.CreateRadiusAttr(s["r"])
            c.CreateHeightAttr(s["h"])
            c.AddTranslateOp().Set(Gf.Vec3d(*s["c"]))
            invisible_collider(c.GetPrim())
            n_cyl += 1
    UsdGeom.Scope.Define(stage, "/World/Forest/Colliders/Stems")
    for o in objs:
        if o["collider"] != "stem":
            continue
        parts = proto_geo[o["proto"]]
        meta = PM[o["proto"]]
        if "profile" in meta:  # understory spruce: stem profile (median radius)
            prof = np.array(meta["profile"])
            r = float(np.mean(prof[prof[:, 1] <= 3.0][:, 4])) * o["scale"][0]
            h = float(prof[prof[:, 4] * o["scale"][0] >= 0.012][:, 1].max()) * o["scale"][2]
        else:  # sapling: thin stem up to 60 % of the height
            r = max(0.008, 0.01 * o["scale"][0])
            h = 0.6 * o["height"]
        c = UsdGeom.Cylinder.Define(stage, f"/World/Forest/Colliders/Stems/{o['id']}")
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(h + 0.2)
        c.AddTranslateOp().Set(Gf.Vec3d(o["x"], o["y"], o["z"] + h / 2 - 0.1))
        invisible_collider(c.GetPrim())
        o["collider_geom"] = dict(r=r, h=h)
    UsdGeom.Scope.Define(stage, "/World/Forest/Colliders/Deadwood")
    UsdGeom.Scope.Define(stage, "/World/Forest/Colliders/Rocks")
    n_hull = 0
    for o in objs:
        col = o["collider"]
        if col not in ("pieces5", "pieces6", "pieces7", "hull"):
            continue
        k = 1 if col == "hull" else int(col[-1])
        key = (o["proto"], k)
        if key not in hull_cache:
            Vall = np.concatenate([p.V for p in proto_geo[o["proto"]]])
            if o["kind"] in ("cut_stump",):
                Vall = Vall
            hull_cache[key] = ml.convex_pieces(Vall, k, rng_seed=3)
        base = "/World/Forest/Colliders/Rocks" if col == "hull" else "/World/Forest/Colliders/Deadwood"
        for j, (HP, HF) in enumerate(hull_cache[key]):
            W = world_pts(HP, (o["x"], o["y"], o["z"]), o["yaw"], o["scale"]).astype(np.float32)
            m = write_mesh(stage, f"{base}/{o['id']}_h{j}", W, HF, normals=False)
            invisible_collider(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set("convexHull")
            n_hull += 1
    log(f"colliders: {n_cyl} trunk cylinders, {sum(1 for o in objs if o['collider'] == 'stem')} stem cylinders, "
        f"{n_hull} convex pieces, 1 terrain triangle mesh")

    # lights -------------------------------------------------------------
    UsdGeom.Scope.Define(stage, "/World/Lights")
    # Kit's default RTX camera exposure is fixed (ISO 100, 1/50 s, f/5); under a closed canopy the
    # light reaching the floor is a small fraction of the open-sky light, so both lights are set for a
    # usable 640x400 image at 2-3 m AGL (kit_forest_test: mean ~75-90 / 255, ~1-4 % clipped pixels,
    # the bright misty sky between the stems). The dome is a CC0 overcast pine-forest HDRI (Poly Haven
    # misty_pines), so the horizon beyond the 100 m plot reads as more forest, not empty sky.
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(SUN_INTENSITY)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
    sx = UsdGeom.Xformable(sun)
    sx.AddRotateZOp().Set(215.0)  # sun from the south-west
    sx.AddRotateXOp().Set(55.0)  # 35 deg elevation (light points along -Z rotated by 55 deg)
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    hdri = os.path.join(a.src, "polyhaven", "misty_pines", "misty_pines_2k.hdr")
    shutil.copy(hdri, tex_dir)
    dome.CreateTextureFileAttr().Set(Sdf.AssetPath(tex_rel + os.path.basename(hdri)))
    dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)

    # metadata -----------------------------------------------------------
    area_ha = (AREA * AREA) / 1e4
    summary = dict(seed=a.seed, area_m=[AREA, AREA], stems=len(trees), stems_per_ha=len(trees) / area_ha,
                   species={s: int((sp_arr == s).sum()) for s in SPECIES_FRAC}, understory=counts,
                   distinct_triangles=int(total_tris), prototypes=len(used), trunk_cylinders=n_cyl,
                   convex_pieces=n_hull, floor_cover=floor_cover, spawn=list(SPAWN))
    layer = stage.GetRootLayer()
    layer.documentation = (
        "Finnish managed boreal forest (forest_fi), built by AgiSIM scripts/build_forest_fi.py "
        f"(seed {a.seed}). ~100 x 100 m thinned Scots pine / Norway spruce / silver birch stand on gently "
        "undulating terrain. Vehicle spawn (0, 0, 0.07) yaw 0 in a flat clearing (z = 0 within 3 m, no stem "
        "within 4 m). Trees are instanceable references to /World/Prototypes (class prim). Colliders: invisible "
        "trunk/stem cylinders, convex pieces for logs/stumps/rocks, triangle-mesh terrain; foliage is visual "
        "only. No PhysicsScene (Pegasus creates its own). Assets: Poly Haven CC0, friggog/tree-gen birch, "
        "procedural bilberry.")
    layer.customLayerData = {"forest_fi_summary_json": json.dumps(summary)}
    stand = dict(summary=summary, terrain=dict(grid=float(terrain.xs[1] - terrain.xs[0]), area=float(terrain.xs[-1] - terrain.xs[0]),
                                               flat_r=getattr(terrain, "flat_r", CLEAR_FLAT_R),
                                               blend_r=getattr(terrain, "blend_r", CLEAR_BLEND_R),
                                               z=terrain.Z.astype(np.float32).round(4).tolist()),
                 trees=[{k: v for k, v in t.items() if k != "profile"} for t in trees], objects=objs)
    if postprocess is not None:
        postprocess(stage, stand, tex_dir, tex_rel)
        layer.customLayerData = {f"{scene_name}_summary_json": json.dumps(summary)}
    stage.GetRootLayer().Save()
    log(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB), textures {len(os.listdir(tex_dir))} files "
        f"({sum(os.path.getsize(os.path.join(tex_dir, f)) for f in os.listdir(tex_dir)) / 1e6:.1f} MB)")

    # stand table for the report / GT tools --------------------------------
    stand_path = os.path.join(a.report_dir, f"{scene_name}_stand.json")
    json.dump(stand, open(stand_path, "w"), default=float)
    log(f"stand table -> {stand_path}")
    return summary
