#!/usr/bin/env python3
"""Ground truth and statistics for forest_fi.usd (offline, Isaac Sim's bundled USD/numpy/scipy).

    python3 scripts/forest_fi/forest_gt.py [--usd .../forest_fi.usd] [--out-dir .../forest_fi_01] [--res 0.1]

Writes to --out-dir:
  forest_fi_gt.voxels      AGVOXEL1, every render mesh (terrain, trunks, branches, needles/leaves,
                           understory, rocks, deadwood): what the RTX lidar can see
  forest_fi_trunks.voxels  AGVOXEL1, trunk render meshes + logs/stumps only (mapping metrics)
  *.json sidecars, forest_fi_stats.json, forest_fi_checks.json, figures (*.png)

Voxelization reads the USD itself: every instance prim's prototype meshes (from the /World/Prototypes
class prims) are sampled once in the prototype frame (all vertices + a barycentric grid on triangles
longer than res/5) and transformed by the instance's local-to-world matrix; the voxel key of every
sample is floor(p / res) (same format as usd_to_voxels.py / VoxelScene::load).
"""
import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaacpy import bootstrap  # noqa: E402

ENVS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "extensions", "pegasus.simulator", "pegasus",
                    "simulator", "assets", "Worlds", "isaac_envs")
OUT = os.path.expanduser("~/workspace/agipix_control/super_parity_results/forest_fi_01")
_OFF = 1 << 20


def pack(k):
    import numpy as np
    k = np.asarray(k, np.int64) + _OFF
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def unpack(c):
    import numpy as np
    m = (1 << 21) - 1
    return np.c_[(c >> 42) & m, (c >> 21) & m, c & m].astype(np.int64) - _OFF


def surface_samples(V, F, step):
    """Vertices + barycentric lattice points (spacing <= step) of every triangle longer than step."""
    import numpy as np
    out = [V]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    L = np.maximum.reduce([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1), np.linalg.norm(A - C, axis=1)])
    n = np.ceil(L / step).astype(np.int64)
    big = np.flatnonzero(n > 1)
    for k in np.unique(n[big]):
        sel = big[n[big] == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (i + j) <= k
        u = (i[keep] / k)[None, :, None]
        v = (j[keep] / k)[None, :, None]
        per = max(1, 3_000_000 // u.shape[1])
        for s0 in range(0, len(sel), per):
            ch = sel[s0:s0 + per]
            P = A[ch][:, None] + u * (B[ch] - A[ch])[:, None] + v * (C[ch] - A[ch])[:, None]
            out.append(P.reshape(-1, 3))
    P = np.concatenate(out)
    q = np.unique(np.floor(P / (step * 0.5)).astype(np.int64), axis=0)  # thin to step/2
    return ((q + 0.5) * step * 0.5).astype(np.float32)


def write_vox(path, keys, res, info):
    import numpy as np
    with open(path, "wb") as f:
        f.write(b"AGVOXEL1")
        f.write(np.float64(res).tobytes())
        f.write(np.uint64(len(keys)).tobytes())
        f.write(keys.astype("<i4").tobytes())
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    info = dict(info, voxels=int(len(keys)), res=res, bounds_min_m=(keys.min(0) * res).tolist(),
                bounds_max_m=((keys.max(0) + 1) * res).tolist(), out_sha256=h)
    json.dump(info, open(os.path.splitext(path)[0] + ".json", "w"), indent=1)
    print(f"{len(keys)} voxels -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", default=os.path.join(ENVS, "forest_fi.usd"))
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--name", default="forest_fi", help="Output/stand prefix (e.g. forest_snow)")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--robot-r", type=float, default=0.25)
    ap.add_argument("--isaacsim", default="~/isaacsim")
    a = ap.parse_args()
    if not a.name.replace("_", "").isalnum():
        ap.error("--name must be alphanumeric with optional underscores")
    bootstrap(a.isaacsim)
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics

    res = a.res
    os.makedirs(a.out_dir, exist_ok=True)
    stage = Usd.Stage.Open(os.path.abspath(a.usd))
    xc = UsdGeom.XformCache()
    step = res / 5.0

    # ---------------------------------------------------------- prototype samples
    proto_samples = {}

    def samples_of(name):
        if name in proto_samples:
            return proto_samples[name]
        root = stage.GetPrimAtPath(f"/World/Prototypes/{name}")
        rm = np.asarray(xc.GetLocalToWorldTransform(root), np.float64)
        rinv = np.linalg.inv(rm)
        by_role = {}
        for p in Usd.PrimRange(root, Usd.PrimAllPrimsPredicate):
            if not p.IsActive() or not p.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(p)
            V = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            F = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
            m = np.asarray(xc.GetLocalToWorldTransform(p), np.float64) @ rinv
            V = (np.c_[V, np.ones(len(V))] @ m)[:, :3]
            role = p.GetName().rsplit("_", 1)[0]
            by_role.setdefault(role, []).append(surface_samples(V, F, step))
        proto_samples[name] = {k: np.concatenate(v) for k, v in by_role.items()}
        return proto_samples[name]

    gt_codes, tr_codes = [], []
    per_role = {}
    n_inst = 0
    collider_meshes_in_protos = []
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if p.IsInstance():
            refs = p.GetMetadata("references")
            items = list(refs.prependedItems) + list(refs.explicitItems) + list(refs.appendedItems)
            name = str(items[0].primPath).rsplit("/", 1)[1]
            M = np.asarray(xc.GetLocalToWorldTransform(p), np.float64)
            for role, S in samples_of(name).items():
                W = S.astype(np.float64) @ M[:3, :3] + M[3, :3]
                c = np.unique(pack(np.floor(W / res)))
                gt_codes.append(c)
                per_role[role] = per_role.get(role, 0) + len(c)
                if role in ("trunk", "deadwood"):
                    tr_codes.append(c)
            n_inst += 1
            if n_inst % 200 == 0:
                print(f"voxelized {n_inst} instances ({len(proto_samples)} cached prototypes)", flush=True)
        elif p.IsA(UsdGeom.Mesh) and p.GetPath().pathString.startswith("/World/Terrain"):
            mesh = UsdGeom.Mesh(p)
            V = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
            F = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)
            c = np.unique(pack(np.floor(surface_samples(V, F, step).astype(np.float64) / res)))
            gt_codes.append(c)
            per_role["terrain"] = len(c)
        if len(gt_codes) > 400:
            gt_codes = [np.unique(np.concatenate(gt_codes))]
            tr_codes = [np.unique(np.concatenate(tr_codes))] if tr_codes else []
    # prototype meshes must not carry colliders (foliage visual-only; colliders live in /World/Forest/Colliders)
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Prototypes"), Usd.PrimAllPrimsPredicate):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            collider_meshes_in_protos.append(str(p.GetPath()))
    GT = unpack(np.unique(np.concatenate(gt_codes)))
    TR = unpack(np.unique(np.concatenate(tr_codes)))
    src = dict(stage=os.path.abspath(a.usd), stage_sha256=hashlib.sha256(open(a.usd, "rb").read()).hexdigest(),
               instances=n_inst, sample_step_m=step)
    write_vox(os.path.join(a.out_dir, f"{a.name}_gt.voxels"), GT, res,
              dict(source=src, content="render geometry: terrain, trunks, branches, foliage, understory, rocks, deadwood",
                   voxels_per_role_before_union=per_role))
    write_vox(os.path.join(a.out_dir, f"{a.name}_trunks.voxels"), TR, res,
              dict(source=src, content="trunk render meshes (trees, understory spruce) + logs + stumps"))

    # ---------------------------------------------------------------- statistics
    stand = json.load(open(os.path.join(a.out_dir, f"{a.name}_stand.json")))
    from scene import TerrainZ
    T = TerrainZ(stand["terrain"])
    trees = stand["trees"]
    objs = stand["objects"]
    area_ha = stand["terrain"]["area"] ** 2 / 1e4
    sp = np.array([t["species"] for t in trees])
    dbh = np.array([t["dbh"] for t in trees]) * 100
    hgt = np.array([t["height"] for t in trees])
    cb = np.array([t["crown_base"] for t in trees])
    stats = dict(stems=len(trees), stems_per_ha=len(trees) / area_ha,
                 species={s: int((sp == s).sum()) for s in ("pine", "spruce", "birch")},
                 species_frac={s: round(float((sp == s).mean()), 3) for s in ("pine", "spruce", "birch")},
                 dbh_cm=dict(mean=float(dbh.mean()), sd=float(dbh.std()), min=float(dbh.min()), max=float(dbh.max()),
                             p5=float(np.percentile(dbh, 5)), p95=float(np.percentile(dbh, 95))),
                 height_m=dict(mean=float(hgt.mean()), sd=float(hgt.std()), min=float(hgt.min()), max=float(hgt.max())),
                 basal_area_m2_per_ha=float((np.pi * (dbh / 200) ** 2).sum() / area_ha),
                 crown_base_m={s: dict(mean=float(cb[sp == s].mean()), min=float(cb[sp == s].min()),
                                       max=float(cb[sp == s].max())) for s in ("pine", "spruce", "birch") if (sp == s).any()},
                 spruce_skirts={k: sum(1 for t in trees if len(t.get("skirts", [])) == k) for k in (0, 1, 2)},
                 understory={}, distinct_triangles=stand["summary"]["distinct_triangles"])
    for o in objs:
        stats["understory"][o["kind"]] = stats["understory"].get(o["kind"], 0) + 1
    P = np.array([[t["x"], t["y"]] for t in trees])
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=2)
    nn = d[:, 1]
    # Clark-Evans aggregation index (R < 1 clustered, R > 1 regular)
    R = nn.mean() / (0.5 / math.sqrt(len(P) / (stand["terrain"]["area"] ** 2)))
    stats["nearest_neighbour_m"] = dict(mean=float(nn.mean()), min=float(nn.min()), p5=float(np.percentile(nn, 5)))
    stats["clark_evans_R"] = float(R)

    # height above ground of every voxel, central area (1 m margin)
    half = stand["terrain"]["area"] / 2 - 1.0

    def agl_grid(K, hmin, hmax):
        c = (K + 0.5) * res
        m = (np.abs(c[:, 0]) < half) & (np.abs(c[:, 1]) < half)
        c, K2 = c[m], K[m]
        agl = c[:, 2] - T.z_at(c[:, 0], c[:, 1])
        sel = (agl >= hmin) & (agl < hmax)
        nx = int(round(2 * half / res))
        ix = np.floor((c[sel, 0] + half) / res).astype(int)
        iy = np.floor((c[sel, 1] + half) / res).astype(int)
        iz = np.floor((agl[sel] - hmin) / res).astype(int)
        G = np.zeros((nx, nx, int(round((hmax - hmin) / res))), bool)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < nx) & (iz >= 0) & (iz < G.shape[2])
        G[ix[ok], iy[ok], iz[ok]] = True
        return G

    # vertical profile 0-5 m AGL (exclude the ground layer itself: AGL >= 0.15 m)
    Gp = agl_grid(GT, 0.0, 5.0)
    Gt = agl_grid(TR, 0.0, 5.0)
    zc = (np.arange(Gp.shape[2]) + 0.5) * res
    prof = dict(agl_m=zc.round(2).tolist(), gt_occupied_frac=Gp.mean((0, 1)).round(5).tolist(),
                trunks_occupied_frac=Gt.mean((0, 1)).round(5).tolist())
    stats["vertical_profile"] = prof
    # free space at 2-3 m AGL for a sphere robot of radius robot_r
    from scipy import ndimage
    rr = a.robot_r
    lo, hi = 2.0 - rr - 0.05, 3.0 + rr + 0.05
    r_vox = int(math.ceil(rr / res))
    g = np.arange(-r_vox, r_vox + 1)
    BX, BY, BZ = np.meshgrid(g, g, g, indexing="ij")
    ball = (BX ** 2 + BY ** 2 + BZ ** 2) * res * res <= rr * rr
    free = {}
    for label, K in (("gt", GT), ("trunks", TR)):
        G = agl_grid(K, lo, hi)
        D = ndimage.binary_dilation(G, structure=ball)
        res_h = {}
        for h in (2.0, 2.5, 3.0):
            iz = int(math.floor((h - lo) / res))
            res_h[f"{h:.1f}"] = float(1 - D[:, :, iz].mean())
        izs = [int(math.floor((h - lo) / res)) for h in np.arange(2.0, 3.0001, res)]
        any_ok = ~np.all(D[:, :, izs], axis=2)
        all_ok = ~np.any(D[:, :, izs], axis=2)
        # 2D connectivity of the 2.5 m layer to the spawn column
        lay = ~D[:, :, int(math.floor((2.5 - lo) / res))]
        labs, _ = ndimage.label(lay)
        c0 = int(round(half / res))
        conn = float((labs == labs[c0, c0]).mean()) if labs[c0, c0] else 0.0
        free[label] = dict(flyable_frac_at_agl=res_h, flyable_some_height_2_3=float(any_ok.mean()),
                           flyable_all_heights_2_3=float(all_ok.mean()), connected_to_spawn_at_2_5=conn)
        if label == "gt":
            D25 = D[:, :, int(math.floor((2.5 - lo) / res))]
            Gocc25 = G[:, :, int(math.floor((2.5 - lo) / res))]
    stats["free_space_robot_r"] = rr
    stats["free_space"] = free
    json.dump(stats, open(os.path.join(a.out_dir, f"{a.name}_stats.json"), "w"), indent=1)

    # -------------------------------------------------------------------- checks
    checks = {}
    zs = T.z_at(*np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61)))
    rr0 = np.hypot(*np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61)))
    checks["terrain_z_origin"] = float(T.z_at(np.array([0.0]), np.array([0.0]))[0])
    checks["terrain_max_abs_z_within_3m"] = float(np.abs(zs[rr0 <= 3.0]).max())
    tr_c = (TR + 0.5) * res
    near = np.hypot(tr_c[:, 0], tr_c[:, 1])
    checks["nearest_trunk_or_deadwood_voxel_m"] = float(near[tr_c[:, 2] > -0.5].min())
    stems = [(math.hypot(t["x"], t["y"]) - t["dbh"] / 2, t["id"]) for t in trees]
    checks["nearest_stem_surface_m"] = min(stems)[0]
    checks["nearest_stem_id"] = min(stems)[1]
    gc = (GT + 0.5) * res
    agl = gc[:, 2] - T.z_at(gc[:, 0], gc[:, 1])
    rad = np.hypot(gc[:, 0], gc[:, 1])
    for R_ in (1.0, 2.0, 3.0):
        checks[f"gt_voxels_within_{R_:.0f}m_of_spawn_agl_0.3_3.5"] = int(((rad < R_) & (agl > 0.3) & (agl < 3.5)).sum())
    nearest_obj = min((math.hypot(o["x"], o["y"]), o["kind"]) for o in objs)
    checks["nearest_understory_object"] = dict(dist_m=nearest_obj[0], kind=nearest_obj[1])
    # collider coverage
    cyl_by_tree = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Forest/Colliders/Trunks")):
        if p.IsA(UsdGeom.Cylinder):
            cyl_by_tree.setdefault(p.GetName().rsplit("_s", 1)[0], []).append(p)
        elif p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            tree_id = p.GetCustomDataByKey("tree_id")
            if tree_id:
                cyl_by_tree.setdefault(tree_id, []).append(p)
    missing = [t["id"] for t in trees if t["id"] not in cyl_by_tree]
    # compare collider radius with the render trunk at 1.3 m and 2.5 m AGL (trunk voxels ring)
    ratios = []
    trk = cKDTree(tr_c[:, :2])
    for t in trees[:: max(1, len(trees) // 150)]:
        for h in (1.3, 2.5):
            z = t["z"] + h
            cyl = [s for s in t["colliders"] if abs(s["c"][2] - z) <= s["h"] / 2]
            if not cyl:
                continue
            s = cyl[0]
            idx = trk.query_ball_point(s["c"][:2], s["r"] + 0.35)
            ring = tr_c[idx]
            ring = ring[np.abs(ring[:, 2] - z) < 0.15]
            if len(ring) < 4:
                continue
            r_vox = np.median(np.hypot(ring[:, 0] - s["c"][0], ring[:, 1] - s["c"][1]))
            ratios.append((s["r"], r_vox))
    ratios = np.array(ratios)
    n_col = {"Cylinder": 0, "convexHull": 0, "none": 0}
    foliage_with_collider = []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.CollisionAPI):
            if p.IsA(UsdGeom.Cylinder):
                n_col["Cylinder"] += 1
            elif p.IsA(UsdGeom.Mesh):
                ap_ = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else "none"
                n_col[ap_] = n_col.get(ap_, 0) + 1
            if not p.GetPath().pathString.startswith(("/World/Forest/Colliders", "/World/Terrain")):
                foliage_with_collider.append(str(p.GetPath()))
    checks["colliders"] = dict(by_type=n_col, trees_without_trunk_collider=missing,
                               trees_with_collider=len(cyl_by_tree), trees=len(trees),
                               colliders_outside_collider_scopes=foliage_with_collider,
                               collision_api_inside_prototypes=collider_meshes_in_protos,
                               collider_r_vs_render_trunk_voxel_ring_r=dict(
                                   n=int(len(ratios)), collider_r_mean=float(ratios[:, 0].mean()),
                                   render_ring_r_mean=float(ratios[:, 1].mean()),
                                   median_diff_m=float(np.median(ratios[:, 0] - ratios[:, 1]))) if len(ratios) else None)
    checks["pass"] = bool(checks["terrain_z_origin"] == 0.0 and checks["terrain_max_abs_z_within_3m"] < 1e-9
                          and checks["nearest_stem_surface_m"] >= 4.0 and not missing and not foliage_with_collider
                          and not collider_meshes_in_protos)
    json.dump(checks, open(os.path.join(a.out_dir, f"{a.name}_checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))

    # ------------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    col = {"pine": "#c0602a", "spruce": "#1f5f3a", "birch": "#d9c94a"}
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.2))
    Z = np.array(stand["terrain"]["z"])
    ext = [-50, 50, -50, 50]
    ax[0].imshow(Z.T, origin="lower", extent=ext, cmap="gray", alpha=0.55)
    for t in trees:
        rc = 0.18 * t["height"] * t["crown_scale"][0]
        ax[0].add_patch(Circle((t["x"], t["y"]), rc, color=col[t["species"]], alpha=0.25, lw=0))
        ax[0].add_patch(Circle((t["x"], t["y"]), max(t["dbh"] / 2, 0.12), color=col[t["species"]], lw=0))
    mk = {"sapling": ("^", "#3a8a3a", 9), "uspruce": ("^", "#0b3d20", 30), "log": ("s", "#6b4a2b", 14),
          "stump": ("o", "#6b4a2b", 14), "cut_stump": ("o", "#a0805a", 5), "rock": ("D", "#777777", 10)}
    for k, (m, c, s) in mk.items():
        pts = np.array([[o["x"], o["y"]] for o in objs if o["kind"] == k])
        if len(pts):
            ax[0].scatter(pts[:, 0], pts[:, 1], marker=m, c=c, s=s, label=k, lw=0)
    for k, c in col.items():
        ax[0].scatter([], [], c=c, s=40, label=f"{k} ({(sp == k).sum()})")
    ax[0].plot(0, 0, "r*", ms=16, label="spawn")
    ax[0].add_patch(Circle((0, 0), 4.0, fill=False, ec="r", ls="--"))
    ax[0].set_xlim(-50, 50)
    ax[0].set_ylim(-50, 50)
    ax[0].set_aspect("equal")
    ax[0].legend(loc="upper right", fontsize=7, ncol=2)
    ax[0].set_title(f"{a.name} top view: {len(trees)} stems/ha, terrain {Z.min():.2f}..{Z.max():.2f} m "
                    "(crown discs indicative)")
    n = D25.shape[0]
    img = np.zeros((n, n, 3))
    img[~D25] = (1, 1, 1)
    img[D25 & ~Gocc25] = (1.0, 0.75, 0.6)
    img[Gocc25] = (0.2, 0.2, 0.2)
    ax[1].imshow(img.transpose(1, 0, 2), origin="lower", extent=[-half, half, -half, half], interpolation="nearest")
    ax[1].plot(0, 0, "r*", ms=14)
    ax[1].set_title(f"2.5 m AGL slice: occupied (black), within {rr} m (orange), flyable (white) "
                    f"= {free['gt']['flyable_frac_at_agl']['2.5'] * 100:.1f} %")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f"{a.name}_topview.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.arange(14, 37, 1.0)
    ax[0].hist([dbh[sp == s] for s in col], bins=bins, stacked=True, color=list(col.values()), label=list(col))
    ax[0].set_xlabel("DBH [cm]")
    ax[0].set_ylabel("stems / ha")
    ax[0].legend()
    ax[1].hist([hgt[sp == s] for s in col], bins=np.arange(10, 27, 0.5), stacked=True, color=list(col.values()))
    ax[1].set_xlabel("height [m]")
    ax[2].plot(np.array(prof["gt_occupied_frac"]) * 100, zc, label="all render geometry (lidar)")
    ax[2].plot(np.array(prof["trunks_occupied_frac"]) * 100, zc, label="trunks + deadwood")
    ax[2].axhspan(2.0, 3.0, color="g", alpha=0.1, label="flight band")
    ax[2].set_xlabel("occupied 0.1 m cells [% of area]")
    ax[2].set_ylabel("height above ground [m]")
    ax[2].set_xscale("log")
    ax[2].legend(fontsize=8)
    ax[2].set_title("sub-canopy vertical profile")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, f"{a.name}_stand_stats.png"), dpi=110)
    plt.close(fig)
    print(json.dumps({k: v for k, v in stats.items() if k != "vertical_profile"}, indent=1))


if __name__ == "__main__":
    main()
