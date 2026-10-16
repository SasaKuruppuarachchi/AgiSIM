"""Blender helper: quadric (collapse) decimation of a triangle mesh given as numpy arrays.

Run headless with our own script only (no .blend is opened, autoexec disabled):
    blender -b --factory-startup --disable-autoexec -noaudio --python-exit-code 1 \
        --python scripts/forest_fi/blender_decimate.py -- IN.npz OUT.npz RATIO

IN.npz:  V float32 (n,3), F int32 (m,3), UV float32 (n,2) per-vertex texture coordinates.
OUT.npz: V, F, UV (per-vertex again: vertices are split where the decimated loops disagree on UV).
Decimate "collapse" keeps each needle / leaf a separate, thin piece (it never welds loose parts),
so foliage keeps its silhouette and most of its surface area at a fraction of the triangles.
"""
import sys

import bpy
import numpy as np


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    src, dst, ratio = argv[0], argv[1], float(argv[2])
    d = np.load(src)
    V, F, UV = d["V"].astype(np.float32), d["F"].astype(np.int32), d["UV"].astype(np.float32)

    me = bpy.data.meshes.new("m")
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(F), 3, np.int32))
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", UV[F.ravel()].ravel())
    me.update()
    me.validate(verbose=False)
    obj = bpy.data.objects.new("o", me)
    bpy.context.scene.collection.objects.link(obj)
    mod = obj.modifiers.new("dec", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    m2 = ev.to_mesh()
    m2.calc_loop_triangles()
    nv = len(m2.vertices)
    V2 = np.empty(nv * 3, np.float32)
    m2.vertices.foreach_get("co", V2)
    V2 = V2.reshape(-1, 3)
    nt = len(m2.loop_triangles)
    tv = np.empty(nt * 3, np.int32)
    m2.loop_triangles.foreach_get("vertices", tv)
    tl = np.empty(nt * 3, np.int32)
    m2.loop_triangles.foreach_get("loops", tl)
    luv = np.empty(len(m2.loops) * 2, np.float32)
    m2.uv_layers.active.data.foreach_get("uv", luv)
    luv = luv.reshape(-1, 2)
    cuv = luv[tl]  # per triangle corner
    # re-weld: one output vertex per distinct (vertex, uv) pair
    q = np.round(cuv * 65536).astype(np.int64)
    key = np.stack([tv.astype(np.int64), q[:, 0], q[:, 1]], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    first = np.zeros(len(uniq), np.int64)
    first[inv] = np.arange(len(inv))
    V3 = V2[uniq[:, 0]]
    UV3 = cuv[first]
    F3 = inv.reshape(-1, 3).astype(np.int32)
    ev.to_mesh_clear()
    np.savez(dst, V=V3, F=F3, UV=UV3)
    print(f"DECIMATE_OK {len(F)} -> {len(F3)} tris ({len(V)} -> {len(V3)} verts)", flush=True)


main()
