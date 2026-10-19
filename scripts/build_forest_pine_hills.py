#!/usr/bin/env python3
"""Build FOREST_PINE_HILLS from verified CC0 data and the shared forest cache.

No downloaded code or Blender files are opened. A rebuild archives the previous
USD once; a further rebuild refuses rather than accumulating large archives.
Renderer/material comparisons should use session overrides, not USD rebuilds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "forest_fi"))
from _isaacpy import bootstrap
from build_forest_snow import verify_manifest, write_licenses


def log(message):
    print(f"[forest_pine_hills {time.strftime('%H:%M:%S')}] {message}", flush=True)


def scene_counts(stage):
    from collections import Counter
    from pxr import Usd, UsdGeom, UsdPhysics
    prototypes = stage.GetPrimAtPath("/World/Prototypes")
    meshes = [UsdGeom.Mesh(p) for p in Usd.PrimRange(prototypes, Usd.PrimAllPrimsPredicate)
              if p.IsActive() and p.IsA(UsdGeom.Mesh)]
    collision_types = Counter()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_types[(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
                            if prim.IsA(UsdGeom.Mesh) else prim.GetTypeName()] += 1
    prototype_triangles = {}
    for proto in stage.GetPrototypes():
        prototype_triangles[str(proto.GetPath())] = sum(
            len(UsdGeom.Mesh(p).GetFaceVertexIndicesAttr().Get()) // 3
            for p in Usd.PrimRange(proto) if p.IsA(UsdGeom.Mesh))
    return dict(prototypes=len(prototypes.GetAllChildren()), native_usd_prototypes=len(stage.GetPrototypes()),
                trunk_cylinders=sum(p.IsA(UsdGeom.Cylinder) for p in Usd.PrimRange(
                    stage.GetPrimAtPath("/World/Forest/Colliders/Trunks"))),
                distinct_prototype_triangles=sum(len(m.GetFaceVertexIndicesAttr().Get()) // 3 for m in meshes),
                active_instances=sum(p.IsInstance() for p in stage.Traverse()),
                instanced_triangles=sum(prototype_triangles[str(p.GetPrototype().GetPath())]
                                        for p in stage.Traverse() if p.IsInstance()),
                ground_triangles=len(UsdGeom.Mesh.Get(stage, "/World/Terrain/ground").GetFaceVertexIndicesAttr().Get()) // 3,
                collision_prims=sum(collision_types.values()), collision_types=dict(collision_types))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--isaacsim", default="~/isaacsim")
    parser.add_argument("--src", type=Path, default=Path.home() / "workspace/agipix_control/assets_src/forest_fi")
    parser.add_argument("--pine-src", type=Path, default=ROOT / "assets_src/forest_pine_hills")
    parser.add_argument("--work", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "extensions/pegasus.simulator/pegasus/simulator/assets/Worlds/isaac_envs/forest_pine_hills.usd")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts/forest_pine_hills/build")
    args = parser.parse_args()
    if args.out.exists() and list((args.out.parent / "archive").glob("forest_pine_hills_*")):
        parser.error("The one superseded PINE_HILLS build is already archived. Reuse the current stage/session for trials; no files deleted.")
    bootstrap(args.isaacsim)
    import scene
    from forest_pine_hills.terrain import PineHillsTerrain
    from forest_pine_hills.scatter import author_scatter
    from forest_pine_hills.materials import author_materials
    from forest_pine_hills.lighting import build_lighting
    from forest_snow.needles import trim_conifer_needles

    manifests = [verify_manifest(args.src), verify_manifest(args.pine_src)]
    log(f"verified {sum(len(a['files']) for m in manifests for a in m.values())} source files against API MD5s")
    work = args.work or args.src / "work"
    if not (work / "protos/protos.json").is_file():
        parser.error("Missing shared FOREST_FI prototype cache; supply --work")
    sky = manifests[1]["syferfontein_18d_clear_puresky"]["files"][0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    def compose(stage, stand, tex_dir, tex_rel):
        author_scatter(stage, stand, log)
        stand["summary"]["needle_alpha_cleanup"] = trim_conifer_needles(stage, args.src, log)
        stand["summary"]["materials"] = author_materials(stage, stand, tex_dir, tex_rel, args.src)
        stand["summary"]["lighting"] = build_lighting(stage, args.pine_src / sky["file"], tex_dir, tex_rel,
                                                       expected_md5=sky["md5"], sun_azimuth_deg=45.)
        stand["summary"]["counts"] = scene_counts(stage)
        counts = stand["summary"]["counts"]
        stand["summary"].update(distinct_triangles=counts["distinct_prototype_triangles"],
                                prototypes=counts["prototypes"],
                                trunk_cylinders=counts["trunk_cylinders"],
                                convex_pieces=counts["collision_types"].get("convexHull", 0))
        # The postprocessed layered material replaces these shared-builder
        # triangle subsets; preserve their values only as build provenance.
        stand["summary"]["initial_floor_cover"] = stand["summary"].pop("floor_cover", {})
        stage.GetRootLayer().documentation = (
            "FOREST_PINE_HILLS: 100 x 100 m CC0 pine forest, hilly matching triangle terrain and flat spawn at (0,0,0). "
            "Built by scripts/build_forest_pine_hills.py using shared FOREST_FI prototypes. "
            "Z is relative to spawn, not terrain AGL. No physics scene, sensors, or runtime callbacks in this asset. "
            "See forest_pine_hills_LICENSES.md and artifacts/forest_pine_hills for provenance and validation.")

    summary = scene.build_scene(args, str(work), log, name="forest_pine_hills",
        species_frac={"pine": .82, "spruce": .18, "birch": 0}, terrain_factory=PineHillsTerrain, postprocess=compose)
    write_licenses(args.out, manifests, name="forest_pine_hills", surface_note=(
        "The final scene uses tiled forest_leaves_02, forrest_ground_03 and rock_moss_set_01 maps, reused scanned conifers/deadwood, "
        "and a sun-separated syferfontein_18d_clear_puresky dome. Kloppenheim_06 was evaluated and rejected for its obscured 4.6 degree sun. "
        "Terrain, root placement, crooked-stem derivatives and shader code are generated locally; no reference-image pixels are used as textures."))
    (args.report_dir / "forest_pine_hills_needle_alpha.json").write_text(json.dumps(summary["needle_alpha_cleanup"], indent=2))
    hashes = {}
    for path in sorted((work / "protos").glob("*.npz")):
        if not path.name.startswith("birch"):
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    with args.out.open("rb") as stream:
        usd_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(seed=args.seed, prototype_cache=str(work), prototype_sha256=hashes,
                      source_manifests=manifests, usd=str(args.out), usd_bytes=args.out.stat().st_size,
                      usd_sha256=usd_sha256, summary=summary)
    (args.report_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
