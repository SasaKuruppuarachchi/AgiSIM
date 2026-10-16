"""Blender helper: generate a tree with friggog/tree-gen (Weber-Penn) and export it as numpy arrays.

Run headless with our own script only (no .blend is opened, autoexec disabled):
    blender -b --factory-startup --disable-autoexec -noaudio --python-exit-code 1 \
        --python scripts/forest_fi/blender_treegen.py -- \
        --treegen ASSETS_SRC/tree-gen_src/tree-gen --preset silver_birch --seed 11 \
        [--set key=value ...] --out OUT.npz

Output npz (Z-up, metres, tree base at the origin):
    <part>_V float32 (n,3), <part>_F int32 (m,3) for part in trunk, branches, leaves
    meta: json string (preset, seed, params, stem/leaf counts)
The trunk is the level-0 stem; branches are all other stem levels; leaves are the tree-gen leaf
polygons (real geometry, no alpha). UVs are computed later by the builder (cylindrical for bark).
"""
import argparse
import ast
import importlib.util
import json
import os
import sys

import bpy
import numpy as np


def load_treegen(path):
    """Import the tree-gen package from its source directory under the module name 'treegen'."""
    spec = importlib.util.spec_from_file_location("treegen", os.path.join(path, "__init__.py"),
                                                  submodule_search_locations=[path])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["treegen"] = mod
    spec.loader.exec_module(mod)
    from treegen.parametric import gen  # noqa: E402
    params = {}
    preset_dir = os.path.join(path, "parametric", "tree_params")
    return gen, preset_dir, params


def read_preset(preset_dir, name):
    """Presets are plain `params = {...}` dict literals: parse them as data (ast), do not execute."""
    src = open(os.path.join(preset_dir, name + ".py")).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "params":
            return ast.literal_eval(node.value)
    raise SystemExit(f"no params dict in preset {name}")


def mesh_arrays(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", V)
    F = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", F)
    M = np.asarray(obj.matrix_world, np.float64)
    V = V.reshape(-1, 3).astype(np.float64)
    V = V @ M[:3, :3].T + M[:3, 3]
    ev.to_mesh_clear()
    return V.astype(np.float32), F.reshape(-1, 3)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--treegen", required=True)
    ap.add_argument("--preset", default="silver_birch")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], help="override key=python_literal")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    gen, preset_dir, _ = load_treegen(os.path.abspath(a.treegen))
    params = read_preset(preset_dir, a.preset)
    for kv in a.set:
        k, v = kv.split("=", 1)
        params[k] = ast.literal_eval(v)

    # empty scene
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    tree = gen.construct(params, seed=a.seed, generate_leaves=True)

    parts = {"trunk": [], "branches": [], "leaves": []}
    for child in tree.children:
        if child.type == "CURVE":
            part = "trunk" if child.name.startswith("Trunk") else "branches"
        elif child.type == "MESH" and child.name.startswith("Leaves"):
            part = "leaves"
        else:
            continue
        parts[part].append(mesh_arrays(child))
    out = {}
    counts = {}
    for k, lst in parts.items():
        if not lst:
            out[k + "_V"] = np.zeros((0, 3), np.float32)
            out[k + "_F"] = np.zeros((0, 3), np.int32)
            counts[k] = 0
            continue
        Vs, Fs, off = [], [], 0
        for V, F in lst:
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        out[k + "_V"] = np.concatenate(Vs)
        out[k + "_F"] = np.concatenate(Fs)
        counts[k] = int(len(out[k + "_F"]))
    out["meta"] = np.array(json.dumps({"preset": a.preset, "seed": a.seed, "params": params,
                                       "tris": counts, "blender": bpy.app.version_string}))
    np.savez_compressed(a.out, **out)
    print("TREEGEN_OK", a.out, counts, flush=True)


main()
