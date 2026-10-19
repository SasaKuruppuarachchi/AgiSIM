#!/usr/bin/env python3
"""Needle/leaf modelling check: real geometry or alpha-textured cards?

For every foliage primitive, sample points uniformly over the triangle surface (area-weighted),
look up the Poly Haven alpha mask at their UVs and report the opaque fraction. A geometric needle
mesh is ~100 % opaque (the alpha only trims tips); an alpha card is mostly transparent. The RTX
lidar ignores opacity on non-cutout materials, so the transparent part of a card would return
points that do not exist.

    python3 scripts/forest_fi/needle_alpha_check.py ASSETS_SRC_DIR OUT.json
"""
import json
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaacpy import bootstrap  # noqa: E402

bootstrap()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from gltf_stats import accessor  # noqa: E402

CASES = [  # (asset, material, alpha mask file)
    ("pine_tree_01", "pine_tree_01_twig", "pine_tree_01_twig_alpha_1k.png"),
    ("fir_tree_01", "fir_tree_01_twig", "fir_tree_01_twig_alpha_1k.png"),
    ("pine_sapling_small", "pine_sapling_small_twig", "pine_sapling_small_twig_alpha_1k.png"),
    ("fir_sapling", "fir_sapling_twigs", "fir_sapling_twigs_alpha_1k.png"),
    ("fir_sapling_medium", "fir_sapling_medium_twigs", "fir_sapling_medium_twigs_alpha_1k.png"),
    ("fern_02", "fern_02", "fern_02_alpha_1k.png"),
    ("grass_medium_01", "grass_medium_01", "grass_medium_01_alpha_1k.png"),
    ("moss_01", "moss_01", "moss_01_alpha_1k.png"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--protos", help="Also check the processed, decimated conifer NPZ prototypes")
    args = parser.parse_args()
    src, out = args.src, args.out
    rng = np.random.default_rng(0)
    res = {}
    for asset, mat, mask in CASES:
        f = os.path.join(src, "polyhaven", asset, f"{asset}_2k.gltf")
        g = json.load(open(f))
        bins = [np.memmap(os.path.join(os.path.dirname(f), b["uri"]), np.uint8, "r") for b in g["buffers"]]
        a = np.asarray(Image.open(os.path.join(src, "polyhaven", asset, "alpha", mask)).convert("L"), np.float32) / 255.0
        H, W = a.shape
        tot_area = 0.0
        opaque = 0.0
        tris = 0
        for node in g["nodes"]:
            if "mesh" not in node:
                continue
            for p in g["meshes"][node["mesh"]]["primitives"]:
                if g["materials"][p["material"]]["name"] != mat:
                    continue
                V = accessor(g, bins, p["attributes"]["POSITION"]).astype(np.float64)
                T = accessor(g, bins, p["attributes"]["TEXCOORD_0"]).astype(np.float64)
                I = accessor(g, bins, p["indices"]).reshape(-1, 3).astype(np.int64)
                ar = 0.5 * np.linalg.norm(np.cross(V[I[:, 1]] - V[I[:, 0]], V[I[:, 2]] - V[I[:, 0]]), axis=1)
                n = 200000
                t = rng.choice(len(I), n, p=ar / ar.sum())
                r1, r2 = rng.random(n), rng.random(n)
                s = np.sqrt(r1)
                u, v, w = 1 - s, s * (1 - r2), s * r2
                uv = u[:, None] * T[I[t, 0]] + v[:, None] * T[I[t, 1]] + w[:, None] * T[I[t, 2]]
                # glTF UV origin top-left (v down), wrap repeat
                px = np.clip((np.mod(uv[:, 0], 1.0) * W).astype(int), 0, W - 1)
                py = np.clip((np.mod(uv[:, 1], 1.0) * H).astype(int), 0, H - 1)
                op = (a[py, px] >= 0.5).mean()
                tot_area += ar.sum()
                opaque += ar.sum() * op
                tris += len(I)
        res[asset] = {"material": mat, "tris": tris, "surface_m2": round(tot_area, 2),
                      "opaque_fraction": round(opaque / tot_area, 3)}
        print(asset, res[asset], flush=True)
    if args.protos:
        from pathlib import Path
        from protos import load_proto
        masks = {}
        for asset, material, mask in CASES[:5]:
            masks[material] = np.asarray(Image.open(os.path.join(src, "polyhaven", asset, "alpha", mask))
                                        .convert("L"), np.float32) / 255
        processed = {}
        for path in sorted(Path(args.protos).glob("*.npz")):
            if not path.stem.startswith(("pine_", "spruce_", "sap_", "uspruce_")):
                continue
            parts, _ = load_proto(str(path))
            areas, fractions = [], []
            for part in parts:
                if part.mat not in masks:
                    continue
                V, I, T = part.V, part.F, part.UV
                ar = .5 * np.linalg.norm(np.cross(V[I[:, 1]] - V[I[:, 0]], V[I[:, 2]] - V[I[:, 0]]), axis=1)
                if not ar.sum():
                    continue
                t = rng.choice(len(I), 100000, p=ar.astype(float) / ar.sum(dtype=float))
                s, r = np.sqrt(rng.random(len(t))), rng.random(len(t))
                uv = ((1-s)[:, None] * T[I[t, 0]] + (s*(1-r))[:, None] * T[I[t, 1]]
                      + (s*r)[:, None] * T[I[t, 2]])
                a = masks[part.mat]
                H, W = a.shape
                px = np.clip((np.mod(uv[:, 0], 1) * W).astype(int), 0, W-1)
                # NPZ UVs were converted to USD's bottom-left convention.
                py = np.clip((np.mod(1 - uv[:, 1], 1) * H).astype(int), 0, H-1)
                areas.append(float(ar.sum()))
                fractions.append(float((a[py, px] >= .5).mean()))
            if areas:
                processed[path.stem] = dict(surface_m2=sum(areas),
                                            opaque_fraction=float(np.average(fractions, weights=areas)))
        res["processed_conifer_prototypes"] = processed
        print("processed conifer prototypes:", json.dumps(processed), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
