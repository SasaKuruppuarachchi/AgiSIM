"""Prototype preparation for the forest_fi scene (stage `protos` of scripts/build_forest_fi.py).

Every prototype is a set of Parts (Z-up metres, base at the origin) saved as WORK/protos/<name>.npz
plus a JSON summary. Roles:
    trunk     main stem (detailed, collider source, in forest_fi_trunks.voxels)
    branch    woody branches / dead branches (visual only)
    foliage   needles, leaves, fern fronds, grass, moss (visual only, lidar-visible)
    deadwood  logs and stumps (convex-piece colliders, in forest_fi_trunks.voxels)
    rock      rocks (convex-hull colliders)
    litter    dry branches, exposed roots (visual only)
    sapling   sapling stems (thin stem collider added by the scene builder)

All randomness is seeded. Poly Haven glTF/bin/jpg/png files are read as data only.
"""
import ast
import json
import os
import subprocess

import numpy as np

import meshlib as ml
from meshlib import Part

HERE = os.path.dirname(os.path.abspath(__file__))

# Prototype tree trunk radial variants (separate trunk prototypes, see build_forest_fi.py):
# trunk xy scale = variant * instance factor, instance factor within +-20 %.
TRUNK_VARIANTS = (0.8, 1.2)

# Pine crown raise: live branches attached below this height are removed (managed Scots pine
# stands self-prune; live crown starts at ~45-55 % of the height).
PINE_CROWN_BASE = {"a": 10.0, "b": 7.5, "c": 8.5}
# Spruce skirts (extra low whorls cloned from the lowest live branches): (z_lo, z_hi) per skirt.
SKIRTS = {"mid": (3.2, None), "low": (1.1, 3.2)}
MIN_SKIRT_Z = 0.4  # lowest needle / branch point of a cloned whorl (m above the stem base)

# foliage decimation ratios (quadric collapse; keeps 80-93 % of the needle surface area)
DECIMATE = {"pine_twig": 0.17, "fir_twig": 0.40, "skirt_twig": 0.35, "sapling_twig": 0.25,
            "usp_twig": 0.25, "trunk": 0.5, "log": 0.35, "stump": 0.4, "roots": 0.3, "rock": 1.0}


def save_proto(out_dir, name, parts, meta):
    os.makedirs(out_dir, exist_ok=True)
    arrs = {}
    for i, p in enumerate(parts):
        arrs[f"p{i}_V"], arrs[f"p{i}_F"], arrs[f"p{i}_UV"] = p.V, p.F, p.UV
    meta = dict(meta, name=name, parts=[{"role": p.role, "mat": p.mat, "tris": p.tris,
                                        "area_m2": round(p.area(), 3)} for p in parts])
    V = np.concatenate([p.V for p in parts])
    meta["bbox"] = [V.min(0).round(3).tolist(), V.max(0).round(3).tolist()]
    meta["tris"] = int(sum(p.tris for p in parts))
    arrs["meta"] = np.array(json.dumps(meta))
    np.savez(os.path.join(out_dir, name + ".npz"), **arrs)
    return meta


def load_proto(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    parts = [Part(pm["role"], pm["mat"], d[f"p{i}_V"], d[f"p{i}_F"], d[f"p{i}_UV"])
             for i, pm in enumerate(meta["parts"])]
    return parts, meta


def part_of(prims, mat_pred, role, mat=None):
    sel = [p for p in prims if mat_pred(p["mat"])]
    if not sel:
        return None
    parts = [Part(role, mat or p["mat"], p["V"], p["F"], p["UV"]) for p in sel]
    return ml.merge_parts(parts) if len(parts) > 1 else parts[0]


def shift(parts, d):
    for p in parts:
        if p is not None:
            p.V = (p.V - np.asarray(d, np.float32)).astype(np.float32)


def stem_centre(V, z0=0.8, z1=1.8):
    s = V[(V[:, 2] > z0) & (V[:, 2] < z1)]
    return 0.5 * (s[:, :2].min(0) + s[:, :2].max(0))


def robust_profile(V, dz=0.25, zmax=None):
    """Trunk slices with the median radius (robust to knots / branch stubs)."""
    V = np.asarray(V, np.float64)
    z1 = V[:, 2].max() if zmax is None else min(zmax, V[:, 2].max())
    rows, z = [], 0.0
    while z < z1 - 1e-6:
        s = V[(V[:, 2] >= z) & (V[:, 2] < z + dz)]
        if len(s) >= 8:
            c = np.median(s[:, :2], 0)
            r = np.linalg.norm(s[:, :2] - c, axis=1)
            rows.append([z, min(z + dz, z1), c[0], c[1], float(np.median(r)), float(np.percentile(r, 90))])
        z += dz
    return np.array(rows)


def cylindrical_uv(V, centre_fn, v_tile=1.0):
    V = np.asarray(V, np.float64)
    c = centre_fn(V[:, 2])
    th = np.arctan2(V[:, 1] - c[:, 1], V[:, 0] - c[:, 0])
    return np.c_[(th + np.pi) / (2 * np.pi), V[:, 2] / v_tile].astype(np.float32)


def centre_fn_from_profile(prof):
    zc = 0.5 * (prof[:, 0] + prof[:, 1])

    def f(z):
        return np.c_[np.interp(z, zc, prof[:, 2]), np.interp(z, zc, prof[:, 3])]
    return f


# --------------------------------------------------------------------- branch logic
def branch_ownership(bark, twig, centre):
    """Per bark component: attach height (vertex closest to the stem axis); per twig component:
    owning bark component (nearest bark vertex to its centroid) and that distance."""
    from scipy.spatial import cKDTree
    nb, vlb, flb = ml.components(bark.F, len(bark.V))
    c = centre(bark.V[:, 2].astype(np.float64))
    dax = np.linalg.norm(bark.V[:, :2] - c, axis=1)
    order = np.lexsort((dax, vlb))
    first = np.r_[0, np.flatnonzero(np.diff(vlb[order])) + 1]
    attach_z = np.zeros(nb)
    attach_v = np.zeros(nb, np.int64)
    attach_z[vlb[order[first]]] = bark.V[order[first], 2]
    attach_v[vlb[order[first]]] = order[first]
    nt, vlt, flt = ml.components(twig.F, len(twig.V))
    cnt = np.bincount(vlt, minlength=nt).astype(np.float64)
    cen = np.stack([np.bincount(vlt, weights=twig.V[:, k], minlength=nt) for k in range(3)], 1) / cnt[:, None]
    d, idx = cKDTree(bark.V).query(cen)
    owner = vlb[idx]
    return dict(nb=nb, vlb=vlb, flb=flb, attach_z=attach_z, attach_v=attach_v, nt=nt, vlt=vlt, flt=flt,
                twig_centroid=cen, owner=owner, owner_d=d)


def crown_raise(bark, twig, centre, cb):
    o = branch_ownership(bark, twig, centre)
    keep_b = o["attach_z"] >= cb
    keep_t = np.where(o["owner_d"] < 0.5, keep_b[o["owner"]], o["twig_centroid"][:, 2] >= cb)
    b2 = Part(bark.role, bark.mat, *ml.compact(bark.V, bark.F, bark.UV, keep_b[o["flb"]]))
    t2 = Part(twig.role, twig.mat, *ml.compact(twig.V, twig.F, twig.UV, keep_t[o["flt"]]))
    return b2, t2, int((~keep_b).sum()), int((~keep_t).sum())


def make_skirt(bark, twig, centre, radius_fn, cb, z_lo, z_hi, rng, whorl_dz=0.36, per_whorl=(5, 7)):
    """Low spruce whorls: clones of the lowest live branches (attached in [cb, cb + 2.5 m]),
    rotated about the stem, scaled up slightly with depth (conical crown) and moved down."""
    o = branch_ownership(bark, twig, centre)
    src = np.flatnonzero((o["attach_z"] >= cb) & (o["attach_z"] < cb + 2.5))
    # only real branches (> 0.6 m long)
    bverts = {i: np.flatnonzero(o["vlb"] == i) for i in src}
    src = [i for i in src if np.ptp(bark.V[bverts[i]][:, :2], 0).max() > 0.6]
    tw_by_owner = {}
    ok = o["owner_d"] < 0.5
    for t in np.flatnonzero(ok):
        tw_by_owner.setdefault(int(o["owner"][t]), []).append(t)
    bf = {i: np.flatnonzero(o["flb"] == i) for i in src}
    order_t = np.argsort(o["flt"], kind="stable")
    bounds_t = np.r_[0, np.cumsum(np.bincount(o["flt"], minlength=o["nt"]))]
    out_b, out_t = [], []
    geo = {}
    skipped = 0
    z_hi = cb if z_hi is None else z_hi
    h = z_hi - whorl_dz * rng.uniform(0.3, 0.8)
    while h >= z_lo:
        n = rng.integers(per_whorl[0], per_whorl[1] + 1)
        th0 = rng.uniform(0, 2 * np.pi)
        for k in range(n):
            i = int(rng.choice(src))
            av = o["attach_v"][i]
            a = bark.V[av].astype(np.float64)
            ca = centre(np.array([a[2]]))[0]
            s = min(1.0 + 0.22 * (cb - h) / max(cb, 1e-3), 1.25) * rng.uniform(0.85, 1.05)
            th = th0 + 2 * np.pi * k / n + rng.uniform(-0.35, 0.35)
            R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
            # direction of the source branch (attach -> branch centroid), for the radial offset
            bv = bark.V[bverts[i]].astype(np.float64)
            dirv = bv.mean(0)[:2] - ca
            dirv = dirv / max(np.linalg.norm(dirv), 1e-6)
            ch = centre(np.array([h]))[0]
            dr = radius_fn(h) - radius_fn(a[2])
            droop = rng.uniform(-0.12, 0.05)  # lower spruce branches droop a little

            def xf(P):
                P = P.astype(np.float64) - np.r_[ca, a[2]]
                P = P * s
                P[:, 2] += droop * np.linalg.norm(P[:, :2], axis=1)
                P[:, :2] += dirv * dr
                return P @ R.T
            if i not in geo:
                Fb = bark.F[bf[i]]
                gb = ml.compact(bark.V, Fb, bark.UV, np.ones(len(Fb), bool))
                tl = tw_by_owner.get(i, [])
                if tl:
                    fidx = np.concatenate([order_t[bounds_t[t]:bounds_t[t + 1]] for t in tl])
                    gt = ml.compact(twig.V, twig.F[fidx], twig.UV, np.ones(len(fidx), bool))
                else:
                    gt = None
                geo[i] = (gb, gt)
            gb, gt = geo[i]
            Vb2 = xf(gb[0])
            Vt2 = xf(gt[0]) if gt is not None else None
            zmin = min(Vb2[:, 2].min(), Vt2[:, 2].min() if Vt2 is not None else 1e9)
            if h + zmin < MIN_SKIRT_Z:  # drooping clone would reach the ground: flatten its droop
                f = (h - MIN_SKIRT_Z) / max(-zmin, 1e-6)
                skipped += 1
                for Q in (Vb2, Vt2):
                    if Q is not None:
                        Q[:, 2] = np.where(Q[:, 2] < 0, Q[:, 2] * f, Q[:, 2])
            Vb2 = (Vb2 + np.r_[ch, h]).astype(np.float32)
            Vt2 = (Vt2 + np.r_[ch, h]).astype(np.float32) if Vt2 is not None else None
            out_b.append(Part("branch", bark.mat, Vb2, gb[1], gb[2]))
            if Vt2 is not None:
                out_t.append(Part("foliage", twig.mat, Vt2, gt[1], gt[2]))
        h -= whorl_dz * rng.uniform(0.8, 1.2)
    if not out_b:
        return None, None
    return ml.merge_parts(out_b), ml.merge_parts(out_t) if out_t else None


# ------------------------------------------------------------------- conifers (PH)
def build_conifers(src, work, log):
    cache = os.path.join(work, "cache")
    out = os.path.join(work, "protos")
    metas = {}
    rng = np.random.default_rng(20261016)
    for asset, species in (("pine_tree_01", "pine"), ("fir_tree_01", "spruce")):
        nodes, mats = ml.load_gltf(os.path.join(src, "polyhaven", asset, f"{asset}_2k.gltf"))
        alpha = ml.load_alpha(os.path.join(src, "polyhaven", asset, "alpha", f"{asset}_twig_alpha_1k.png"))
        for node, prims in sorted(nodes.items()):
            v = node.split("_")[3]  # a / b / c
            trunk = part_of(prims, lambda m: "_trunk_" in m, "trunk")
            bark = part_of(prims, lambda m: m.endswith("_bark"), "branch")
            dead = part_of(prims, lambda m: m.endswith("_dead_branches"), "branch")
            twig = part_of(prims, lambda m: m.endswith("_twig"), "foliage")
            c = stem_centre(trunk.V)
            shift([trunk, bark, dead, twig], [c[0], c[1], 0.0])
            prof = robust_profile(trunk.V)
            cfn = centre_fn_from_profile(prof)
            zc = 0.5 * (prof[:, 0] + prof[:, 1])

            def rfn(z, prof=prof, zc=zc):
                return float(np.interp(z, zc, prof[:, 4]))
            if not np.any(trunk.UV):  # fir trunk_c ships without UVs: tileable bark, cylindrical map
                trunk.UV = cylindrical_uv(trunk.V, cfn, v_tile=1.2)
            name = f"{species}_{v}"
            twig_raw = twig.tris
            twig = ml.alpha_to_geometry(twig, alpha)
            extra = {}
            if species == "pine":
                cb = PINE_CROWN_BASE[v]
                bark, twig, nb_rm, nt_rm = crown_raise(bark, twig, cfn, cb)
                extra = {"crown_raise_m": cb, "removed_branches": nb_rm, "removed_needle_pieces": nt_rm}
            area0 = twig.area()
            skirts = {}
            if species == "spruce":
                cb = float(twig.V[:, 2].min())
                for sk, (zlo, zhi) in SKIRTS.items():
                    zhi_eff = (cb - 0.25) if zhi is None else min(zhi, cb - 0.25)
                    if zhi_eff <= zlo + 0.8:
                        continue
                    sb, st = make_skirt(bark, twig, cfn, rfn, cb, zlo, zhi_eff, rng)
                    if sb is None:
                        continue
                    st = ml.decimate(st, DECIMATE["skirt_twig"], cache, f"{name}_skirt_{sk}")
                    skirts[sk] = (sb, st, zlo, zhi_eff)
            twig = ml.decimate(twig, DECIMATE["pine_twig" if species == "pine" else "fir_twig"], cache, name + "_twig")
            log(f"{name}: needles {twig_raw} -> {twig.tris} tris, area {area0:.0f} -> {twig.area():.0f} m2 {extra}")
            crown_parts = [p for p in (bark, dead, twig) if p is not None and p.tris]
            H = float(max(p.V[:, 2].max() for p in crown_parts + [trunk]))
            live = [p for p in (bark, twig) if p is not None and p.tris]
            crown_base = float(min(np.percentile(p.V[:, 2], 0.5) for p in live))
            m = save_proto(out, name + "_crown", crown_parts,
                           dict(kind="crown", species=species, variant=v, height=H, crown_base=crown_base,
                                source=f"polyhaven:{asset}/{node}", **extra))
            metas[name + "_crown"] = m
            for sk, (sb, st, zlo, zhi) in skirts.items():
                m = save_proto(out, f"{name}_skirt_{sk}", [sb, st],
                               dict(kind="skirt", species=species, variant=v, z_range=[zlo, zhi],
                                    source=f"cloned lowest whorls of polyhaven:{asset}/{node}"))
                metas[f"{name}_skirt_{sk}"] = m
                log(f"{name}_skirt_{sk}: {m['tris']} tris, z {zlo:.1f}-{zhi:.1f} m")
            # trunk prototypes (radial variants)
            tdec = ml.decimate(trunk, DECIMATE["trunk"], cache, name + "_trunk")
            for tv in TRUNK_VARIANTS:
                t = tdec.copy()
                cc = cfn(t.V[:, 2].astype(np.float64))
                t.V[:, :2] = (cc + (t.V[:, :2] - cc) * tv).astype(np.float32)
                p2 = robust_profile(t.V)
                tn = f"{name}_trunk_{int(round(tv * 100))}"
                m = save_proto(out, tn, [t], dict(kind="trunk", species=species, variant=v, radial=tv,
                                                   height=float(t.V[:, 2].max()), dbh=ml.dbh_from_profile(p2),
                                                   profile=p2.round(4).tolist(),
                                                   source=f"polyhaven:{asset}/{node} trunk"))
                metas[tn] = m
                log(f"{tn}: {t.tris} tris, dbh {m['dbh']:.3f} m, top {m['height']:.1f} m")
    return metas


# ----------------------------------------------------------------- understory (PH)
def build_understory(src, work, log):
    cache = os.path.join(work, "cache")
    out = os.path.join(work, "protos")
    metas = {}
    # understory spruce 5-9 m (fir_sapling_medium): trunk = tallest component of the branch mesh
    asset = "fir_sapling_medium"
    nodes, _ = ml.load_gltf(os.path.join(src, "polyhaven", asset, f"{asset}_2k.gltf"))
    alpha = ml.load_alpha(os.path.join(src, "polyhaven", asset, "alpha", f"{asset}_twigs_alpha_1k.png"))
    for node, prims in sorted(nodes.items()):
        v = node.split("_")[3]
        br = part_of(prims, lambda m: m.endswith("_branches"), "branch")
        dead = part_of(prims, lambda m: m.endswith("_branches_dead"), "branch")
        twig = part_of(prims, lambda m: m.endswith("_twigs"), "foliage")
        nc, vl, fl = ml.components(br.F, len(br.V))
        ext = np.array([np.ptp(br.V[vl == i][:, 2]) for i in range(nc)])
        ti = int(np.argmax(ext))
        trunk = Part("trunk", br.mat, *ml.compact(br.V, br.F, br.UV, fl == ti))
        branches = Part("branch", br.mat, *ml.compact(br.V, br.F, br.UV, fl != ti))
        c = stem_centre(trunk.V, 0.2, 1.0)
        shift([trunk, branches, dead, twig], [c[0], c[1], 0.0])
        twig = ml.alpha_to_geometry(twig, alpha)
        a0, n0 = twig.area(), twig.tris
        twig = ml.decimate(twig, DECIMATE["usp_twig"], cache, f"usp_{v}_twig")
        prof = robust_profile(trunk.V, dz=0.5)
        parts = [p for p in (trunk, branches, dead, twig) if p is not None and p.tris]
        m = save_proto(out, f"uspruce_{v}", parts,
                       dict(kind="uspruce", species="spruce", height=float(max(p.V[:, 2].max() for p in parts)),
                            dbh=ml.dbh_from_profile(prof), profile=prof.round(4).tolist(),
                            crown_base=float(np.percentile(twig.V[:, 2], 0.5)), source=f"polyhaven:{asset}/{node}"))
        metas[f"uspruce_{v}"] = m
        log(f"uspruce_{v}: needles {n0} -> {twig.tris}, area {a0:.0f} -> {twig.area():.0f} m2, H {m['height']:.1f}")
    # saplings
    for asset, sp, twig_sfx, bark_sfx in (("pine_sapling_small", "pine", "_twig", "_bark"),
                                          ("fir_sapling", "spruce", "_twigs", "_branches")):
        nodes, _ = ml.load_gltf(os.path.join(src, "polyhaven", asset, f"{asset}_2k.gltf"))
        alpha = ml.load_alpha(os.path.join(src, "polyhaven", asset, "alpha", f"{asset}{twig_sfx}_alpha_1k.png"))
        for node, prims in sorted(nodes.items()):
            v = node.split("_")[-1]
            stem = part_of(prims, lambda m: m.endswith(bark_sfx), "sapling")
            twig = part_of(prims, lambda m: m.endswith(twig_sfx), "foliage")
            low = stem.V[stem.V[:, 2] < 0.15]
            c = 0.5 * (low[:, :2].min(0) + low[:, :2].max(0))
            shift([stem, twig], [c[0], c[1], 0.0])
            twig = ml.alpha_to_geometry(twig, alpha)
            twig = ml.decimate(twig, DECIMATE["sapling_twig"], cache, f"sap_{sp}_{v}_twig")
            m = save_proto(out, f"sap_{sp}_{v}", [stem, twig],
                           dict(kind="sapling", species=sp, height=float(max(stem.V[:, 2].max(), twig.V[:, 2].max())),
                                source=f"polyhaven:{asset}/{node}"))
            metas[f"sap_{sp}_{v}"] = m
            log(f"sap_{sp}_{v}: {m['tris']} tris, H {m['height']:.2f}")
    return metas


def recentre_ground(parts, sink=0.0):
    V = np.concatenate([p.V for p in parts])
    c = 0.5 * (V[:, :2].min(0) + V[:, :2].max(0))
    shift(parts, [c[0], c[1], V[:, 2].min() + sink])


def build_ground(src, work, log):
    cache = os.path.join(work, "cache")
    out = os.path.join(work, "protos")
    metas = {}
    rng = np.random.default_rng(7)
    ph = os.path.join(src, "polyhaven")

    def simple(asset, node_filter, role, name, ratio, kind, alpha=None, subdiv=0, sink=0.0):
        nodes, _ = ml.load_gltf(os.path.join(ph, asset, f"{asset}_2k.gltf"))
        res = []
        for node, prims in sorted(nodes.items()):
            if not node_filter(node):
                continue
            p = ml.merge_parts([Part(role, x["mat"], x["V"], x["F"], x["UV"]) for x in prims]) \
                if len({x["mat"] for x in prims}) == 1 else None
            parts = [p] if p is not None else [Part(role, x["mat"], x["V"], x["F"], x["UV"]) for x in prims]
            if alpha is not None:
                parts = [ml.alpha_to_geometry(q, alpha, subdiv=subdiv) for q in parts]
            parts = [ml.decimate(q, ratio, cache, f"{name}_{node}") for q in parts]
            recentre_ground(parts, sink)
            res.append((node, parts))
        return res

    # deadwood log (3 m long along x) and stumps
    (node, parts), = simple("dead_tree_trunk", lambda n: True, "deadwood", "log", DECIMATE["log"], "log")
    metas["log"] = save_proto(out, "log", parts, dict(kind="log", source="polyhaven:dead_tree_trunk"))
    for k, asset in (("a", "tree_stump_01"), ("b", "tree_stump_02")):
        (node, parts), = simple(asset, lambda n: True, "deadwood", f"stump_{k}", DECIMATE["stump"], "stump")
        # sink so the ground-level skirt of the scan sits in the terrain
        metas[f"stump_{k}"] = save_proto(out, f"stump_{k}", parts, dict(kind="stump", source=f"polyhaven:{asset}"))
    for node, parts in simple("dry_branches_medium_01", lambda n: True, "litter", "branches", 1.0, "litter"):
        k = node.split("_")[-1]
        metas[f"drybranch_{k}"] = save_proto(out, f"drybranch_{k}", parts,
                                             dict(kind="litter", source=f"polyhaven:dry_branches_medium_01/{node}"))
    for node, parts in simple("pine_roots", lambda n: True, "litter", "roots", DECIMATE["roots"], "litter", sink=0.04):
        k = node.split("_")[-1]
        metas[f"roots_{k}"] = save_proto(out, f"roots_{k}", parts, dict(kind="litter", source=f"polyhaven:pine_roots/{node}"))
    for asset in ("rock_moss_set_01", "rock_moss_set_02"):
        for node, parts in simple(asset, lambda n: True, "rock", "rock", 1.0, "rock"):
            k = node.split("_")[-1]
            metas[k] = save_proto(out, k, parts, dict(kind="rock", source=f"polyhaven:{asset}/{node}"))
    # ferns: alpha-card fronds converted to frond-shaped geometry
    alpha = ml.load_alpha(os.path.join(ph, "fern_02", "alpha", "fern_02_alpha_1k.png"))
    for node, parts in simple("fern_02", lambda n: True, "foliage", "fern", 1.0, "fern", alpha=alpha, subdiv=5):
        k = node.split("_")[-1]
        metas[f"fern_{k}"] = save_proto(out, f"fern_{k}", parts, dict(kind="fern", source=f"polyhaven:fern_02/{node}"))
        log(f"fern_{k}: {metas[f'fern_{k}']['tris']} tris after card->geometry")
    # grass patches: tufts of grass_medium_01 scattered in a 1.2 m disk
    nodes, _ = ml.load_gltf(os.path.join(ph, "grass_medium_01", "grass_medium_01_2k.gltf"))
    alpha = ml.load_alpha(os.path.join(ph, "grass_medium_01", "alpha", "grass_medium_01_alpha_1k.png"))
    tufts = []
    for node, prims in sorted(nodes.items()):
        if any(s in node for s in ("large", "mid", "tall")):
            p = Part("foliage", prims[0]["mat"], prims[0]["V"], prims[0]["F"], prims[0]["UV"])
            p = ml.alpha_to_geometry(p, alpha)
            recentre_ground([p], sink=0.01)
            tufts.append(p)
    for k in ("a", "b"):
        pieces = []
        for _ in range(14):
            t = tufts[rng.integers(len(tufts))].copy()
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
            r, a = 0.6 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            s = rng.uniform(0.8, 1.5)
            t.V = t.V * s
            t.V[:, :2] = t.V[:, :2] @ R.T + np.array([r * np.cos(a), r * np.sin(a)], np.float32)
            pieces.append(t)
        metas[f"grass_{k}"] = save_proto(out, f"grass_{k}", [ml.merge_parts(pieces)],
                                         dict(kind="grass", source="polyhaven:grass_medium_01 tufts"))
    # moss clumps: moss_01 cards (cut to shape) packed in a 0.5 m disk
    nodes, _ = ml.load_gltf(os.path.join(ph, "moss_01", "moss_01_2k.gltf"))
    alpha = ml.load_alpha(os.path.join(ph, "moss_01", "alpha", "moss_01_alpha_1k.png"))
    sprigs = []
    for node, prims in sorted(nodes.items()):
        p = Part("foliage", prims[0]["mat"], prims[0]["V"], prims[0]["F"], prims[0]["UV"])
        p = ml.alpha_to_geometry(p, alpha, subdiv=3)
        recentre_ground([p], sink=0.003)
        sprigs.append(p)
    for k in ("a", "b"):
        pieces = []
        for _ in range(500):
            t = sprigs[rng.integers(len(sprigs))].copy()
            th = rng.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
            r, a = 0.28 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            t.V = t.V * rng.uniform(1.5, 2.5)
            t.V[:, :2] = t.V[:, :2] @ R.T + np.array([r * np.cos(a), r * np.sin(a)], np.float32)
            pieces.append(t)
        metas[f"moss_{k}"] = save_proto(out, f"moss_{k}", [ml.merge_parts(pieces)],
                                        dict(kind="moss", source="polyhaven:moss_01 sprigs"))
    return metas


# ----------------------------------------------------------- procedural: bilberry
def build_bilberry(work, log):
    """Bilberry (Vaccinium myrtillus) patch: green zig-zag shoots 15-35 cm with small ovate leaves.
    Procedural (numpy, seeded); geometric leaves, no alpha."""
    out = os.path.join(work, "protos")
    metas = {}
    for k, seed in (("a", 101), ("b", 202)):
        rng = np.random.default_rng(seed)
        stemsV, stemsF, leafV, leafF = [], [], [], []
        for _ in range(130):
            r, a = 0.42 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
            p = np.array([r * np.cos(a), r * np.sin(a), -0.02])
            h = rng.uniform(0.14, 0.34)
            nseg = 8
            d = np.array([rng.normal(0, 0.25), rng.normal(0, 0.25), 1.0])
            pts = [p.copy()]
            for s in range(nseg):
                d2 = d / np.linalg.norm(d) + np.r_[rng.normal(0, 0.25, 2), 0]
                p = p + d2 / np.linalg.norm(d2) * h / nseg
                pts.append(p.copy())
            pts = np.array(pts)
            # stem: 3-sided tube, radius 1.5 mm
            base = sum(len(v) for v in stemsV)
            ring = []
            for q in pts:
                for j in range(3):
                    t = 2 * np.pi * j / 3
                    ring.append(q + 0.0015 * np.r_[np.cos(t), np.sin(t), 0])
            stemsV.append(np.array(ring))
            for s in range(nseg):
                for j in range(3):
                    a0, a1 = base + 3 * s + j, base + 3 * s + (j + 1) % 3
                    b0, b1 = a0 + 3, a1 + 3
                    stemsF += [[a0, a1, b1], [a0, b1, b0]]
            # leaves: ovate, 1.5-3 cm, 2 triangles each (diamond), on the upper 2/3
            for q in pts[2:]:
                for _ in range(rng.integers(2, 5)):
                    L = rng.uniform(0.015, 0.03)
                    w = 0.55 * L
                    dirv = np.r_[rng.normal(0, 1, 2), rng.uniform(0.1, 0.6)]
                    dirv /= np.linalg.norm(dirv)
                    side = np.cross(dirv, [0, 0, 1.0])
                    side /= max(np.linalg.norm(side), 1e-6)
                    b = len(leafV)
                    leafV += [q, q + dirv * 0.4 * L + side * w / 2, q + dirv * L, q + dirv * 0.4 * L - side * w / 2]
                    leafF += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
        SV = np.concatenate(stemsV)
        stem = Part("foliage", "bilberry_stem", SV, np.array(stemsF), np.zeros((len(SV), 2)))
        LV = np.array(leafV)
        leaf = Part("foliage", "bilberry_leaf", LV, np.array(leafF), np.zeros((len(LV), 2)))
        metas[f"bilberry_{k}"] = save_proto(out, f"bilberry_{k}", [stem, leaf],
                                            dict(kind="bilberry", source="procedural (numpy), seed %d" % seed))
        log(f"bilberry_{k}: {metas[f'bilberry_{k}']['tris']} tris")
    return metas


# ------------------------------------------------------------------- birch (tree-gen)
BIRCH = {  # tree-gen silver_birch preset + overrides (managed-stand dimensions: H ~18-21 m, DBH ~20-24 cm)
    "a": dict(seed=11, set={"g_scale": 18, "g_scale_v": 0, "ratio": 0.0065, "length": [1, 0.2, 0.4, 0],
                            "base_size": [0.42, 0.1, 0.02, 0.02], "leaf_scale": 0.1}),
    "b": dict(seed=23, set={"g_scale": 17, "g_scale_v": 0, "ratio": 0.0062, "length": [1, 0.18, 0.4, 0],
                            "base_size": [0.45, 0.1, 0.02, 0.02], "leaf_scale": 0.1}),
}
TREEGEN_COMMON = {"bevel_res": [4, 1, 0, 0], "curve_res": [8, 4, 2, 0]}


def build_birch(src, work, log):
    cache = os.path.join(work, "cache")
    out = os.path.join(work, "protos")
    metas = {}
    tg = os.path.join(src, "tree-gen_src", "tree-gen")
    for k, cfg in BIRCH.items():
        raw = os.path.join(cache, f"birch_{k}_treegen.npz")
        sets = dict(TREEGEN_COMMON, **cfg["set"])
        if not os.path.exists(raw):
            cmd = [ml.blender_bin(), "-b", "--factory-startup", "--disable-autoexec", "-noaudio", "--python-exit-code", "1",
                   "--python", os.path.join(HERE, "blender_treegen.py"), "--", "--treegen", tg, "--preset", "silver_birch",
                   "--seed", str(cfg["seed"]), "--out", raw]
            for kk, vv in sets.items():
                cmd += ["--set", f"{kk}={vv!r}"]
            r = subprocess.run(cmd, capture_output=True, text=True, env=ml.blender_env())
            if r.returncode != 0 or not os.path.exists(raw):
                raise RuntimeError(f"tree-gen failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        d = np.load(raw)
        trunk = Part("trunk", "birch_bark", d["trunk_V"], d["trunk_F"])
        branches = Part("branch", "birch_branch", d["branches_V"], d["branches_F"])
        LV = d["leaves_V"].astype(np.float64).reshape(-1, 13, 3)  # cordate leaf: 13 vertices per leaf
        base, right, tip, left = LV[:, 0], LV[:, 3], LV[:, 6], LV[:, 9]
        mid = 0.5 * (right + left)
        right = mid + 1.25 * (right - mid)  # widen the diamond to keep the leaf area
        left = mid + 1.25 * (left - mid)
        n = len(LV)
        V = np.stack([base, right, tip, left], 1).reshape(-1, 3)
        b = 4 * np.arange(n)[:, None]
        F = np.concatenate([b + [0, 1, 2], b + [0, 2, 3]])
        leaves = Part("foliage", "birch_leaf", V, F)
        prof = robust_profile(trunk.V)
        cfn = centre_fn_from_profile(prof)
        trunk.UV = cylindrical_uv(trunk.V, cfn, v_tile=2.0)
        branches.UV = cylindrical_uv(branches.V, lambda z: np.zeros((len(z), 2)), v_tile=2.0)
        # split the trunk: dark fissured base (< ~2.2 m) vs white upper bark
        zf = trunk.V[trunk.F].mean(1)[:, 2]
        lowF = zf < 2.2
        tlow = Part("trunk", "birch_bark_base", *ml.compact(trunk.V, trunk.F, trunk.UV, lowF))
        thigh = Part("trunk", "birch_bark", *ml.compact(trunk.V, trunk.F, trunk.UV, ~lowF))
        H = float(max(p.V[:, 2].max() for p in (trunk, branches, leaves)))
        meta = json.loads(str(d["meta"]))
        m = save_proto(out, f"birch_{k}", [tlow, thigh, branches, leaves],
                       dict(kind="tree", species="birch", variant=k, height=H, dbh=ml.dbh_from_profile(prof),
                            crown_base=float(np.percentile(leaves.V[:, 2], 0.5)), profile=prof.round(4).tolist(),
                            source=f"friggog/tree-gen silver_birch preset, seed {cfg['seed']}, overrides {sets}",
                            treegen_tris=meta["tris"]))
        metas[f"birch_{k}"] = m
        log(f"birch_{k}: {m['tris']} tris, H {H:.1f}, dbh {m['dbh']:.3f}, crown base {m['crown_base']:.1f}")
    return metas


def build_all(src, work, log=print):
    metas = {}
    metas.update(build_conifers(src, work, log))
    metas.update(build_birch(src, work, log))
    metas.update(build_understory(src, work, log))
    metas.update(build_ground(src, work, log))
    metas.update(build_bilberry(work, log))
    json.dump(metas, open(os.path.join(work, "protos", "protos.json"), "w"), indent=1)
    tot = sum(m["tris"] for m in metas.values())
    log(f"prototypes: {len(metas)}, distinct triangles {tot}")
    return metas


def parse_literal(s):
    return ast.literal_eval(s)
