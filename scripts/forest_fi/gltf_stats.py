#!/usr/bin/env python3
"""Per-primitive geometry statistics of a glTF file (numpy only, reads the .bin as data).

    python3 scripts/forest_fi/gltf_stats.py FILE.gltf [...]

Used to decide whether foliage is modelled as real needle geometry or as alpha-textured cards:
needle geometry = many thin triangles (aspect ratio >> 1, area ~mm^2); cards = few broad quads.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _isaacpy import bootstrap  # noqa: E402

bootstrap()
import numpy as np  # noqa: E402

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def accessor(g, bins, i):
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
    return np.frombuffer(buf, dt, count=a["count"] * n, offset=off).reshape(a["count"], n)


def main():
    for f in sys.argv[1:]:
        g = json.load(open(f))
        d = os.path.dirname(f)
        bins = [np.memmap(os.path.join(d, b["uri"]), np.uint8, "r") for b in g["buffers"]]
        print("=====", os.path.basename(f))
        for node in g["nodes"]:
            if "mesh" not in node:
                continue
            for p in g["meshes"][node["mesh"]]["primitives"]:
                V = accessor(g, bins, p["attributes"]["POSITION"]).astype(np.float64)
                I = accessor(g, bins, p["indices"]).reshape(-1, 3).astype(np.int64)
                A, B, C = V[I[:, 0]], V[I[:, 1]], V[I[:, 2]]
                area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
                e = np.stack([np.linalg.norm(B - A, axis=1), np.linalg.norm(C - B, axis=1),
                              np.linalg.norm(A - C, axis=1)], 1)
                lmax = e.max(1)
                # aspect: longest edge / height on it
                h = 2 * area / np.maximum(lmax, 1e-12)
                asp = lmax / np.maximum(h, 1e-9)
                mat = g["materials"][p["material"]]["name"]
                lo, hi = V.min(0), V.max(0)
                print(f"  {node['name']:32s} {mat:32s} tris {len(I):8d} area {area.sum():8.2f} m2 "
                      f"med_tri_area {np.median(area)*1e6:8.2f} mm2 med_lmax {np.median(lmax)*1e3:6.1f} mm "
                      f"med_aspect {np.median(asp):6.1f} bbox {np.round(lo,2)} {np.round(hi,2)}")


if __name__ == "__main__":
    main()
