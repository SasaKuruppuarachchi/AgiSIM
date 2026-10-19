#!/usr/bin/env python3
"""Build FOREST_SNOW using FOREST_FI's stand, mesh and prototype pipeline.

Fetch new CC0 maps with scripts/forest_snow/fetch_assets.py first. By default,
reuse the existing verified FOREST_FI conifer prototype cache (no Blender run).
--rebuild-protos rebuilds only CC0/procedural prototypes with our existing
factory-startup/disable-autoexec Blender conversion; tree-gen is never invoked.

    python3 scripts/build_forest_snow.py
    python3 scripts/forest_fi/forest_gt.py --name forest_snow --usd .../forest_snow.usd \
        --out-dir artifacts/forest_snow/build
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


def log(message):
    print(f"[forest_snow {time.strftime('%H:%M:%S')}] {message}", flush=True)


def verify_manifest(source):
    manifest = json.loads((source / "polyhaven_manifest.json").read_text())
    for asset in manifest.values():
        if not asset["license"].replace("-", " ").startswith("CC0"):
            raise ValueError("Non-CC0 source in manifest")
        for entry in asset["files"]:
            path = (source / entry["file"]).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError("Source path escapes asset directory")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "md5").hexdigest()
            if digest != entry["md5"]:
                raise ValueError(f"MD5 mismatch: {path}")
    return manifest


def write_licenses(out, manifests, *, name="forest_snow", surface_note=None):
    lines = [f"# {name.upper()} asset sources and licenses", "",
             "All third-party content used by this scene is Poly Haven CC0 1.0:",
             "https://polyhaven.com/license. MD5 values below are checked against the saved Poly Haven API manifests on each build.",
             "Downloaded files are treated as data: no downloaded scripts or .blend files are opened or executed.",
             "Optional prototype conversion uses only our scripts and Blender --factory-startup --disable-autoexec.",
             "", "The shared FOREST_FI cache supplies alpha-trimmed, decimated conifers, deadwood and rocks.",
             "No tree-gen/birch geometry is included. Terrain, placements, bilberry stems and snow caps are procedural AgiSIM geometry.",
             surface_note or "The original summer ground maps/misty_pines are retained pipeline inputs; the final ground and dome use snow_02 and snowy_forest.",
             "Source MD5s refer to the original downloads,",
             "not the resized runtime texture copies. Build provenance includes SHA-256 hashes of reused prototype NPZ files.", ""]
    combined = {}
    for manifest in manifests:
        combined.update(manifest)
    for asset_name, asset in sorted(combined.items()):
        lines += [f"## {asset_name}", "", f"Source: {asset['page']} — CC0 1.0", "",
                  "Authors: " + json.dumps(asset.get("authors", {}), ensure_ascii=False), "",
                  "| File | MD5 |", "|---|---|"]
        lines += [f"| [{Path(f['file']).name}]({f['url']}) | `{f['md5']}` |" for f in asset["files"]]
        lines.append("")
    (out.parent / f"{name}_LICENSES.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--isaacsim", default="~/isaacsim")
    parser.add_argument("--src", type=Path, default=Path.home() / "workspace/agipix_control/assets_src/forest_fi")
    parser.add_argument("--snow-src", type=Path, default=ROOT / "assets_src/forest_snow")
    parser.add_argument("--work", type=Path, help="Prototype cache: --src/work for reuse, --snow-src/work for --rebuild-protos")
    parser.add_argument("--rebuild-protos", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "extensions/pegasus.simulator/pegasus/simulator/assets/Worlds/isaac_envs/forest_snow.usd")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts/forest_snow/build")
    args = parser.parse_args()
    bootstrap(args.isaacsim)
    import numpy as np
    import scene
    from forest_snow.scene import SnowTerrain, winterize

    manifests = [verify_manifest(args.src), verify_manifest(args.snow_src)]
    log(f"verified {sum(len(v['files']) for m in manifests for v in m.values())} source files against API MD5s")
    work = args.work or (args.snow_src / "work" if args.rebuild_protos else args.src / "work")
    if args.rebuild_protos:
        if work.resolve().is_relative_to(args.src.resolve()):
            parser.error("--rebuild-protos requires a separate snow cache outside --src; the FOREST_FI cache is read-only")
        import protos
        (work / "protos").mkdir(parents=True, exist_ok=True)
        (work / "cache").mkdir(exist_ok=True)
        metas = {}
        for build in (protos.build_conifers, protos.build_understory, protos.build_ground):
            metas.update(build(str(args.src), str(work), log))
        metas.update(protos.build_bilberry(str(work), log))
        (work / "protos/protos.json").write_text(json.dumps(metas, indent=1))
    if not (work / "protos/protos.json").is_file():
        parser.error("Missing prototype cache. Supply --work or use --rebuild-protos with a writable --work.")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = scene.build_scene(args, str(work), log, name="forest_snow",
                      species_frac={"pine": .65, "spruce": .35, "birch": 0}, terrain_factory=SnowTerrain,
                      postprocess=lambda stage, stand, tex_dir, tex_rel: winterize(
                          stage, stand, tex_dir, tex_rel, args.snow_src, args.src, log))
    write_licenses(args.out, manifests)
    (args.report_dir / "forest_snow_needle_alpha.json").write_text(
        json.dumps(summary["needle_alpha_cleanup"], indent=2))
    prototype_hashes = {}
    for path in sorted((work / "protos").glob("*.npz")):
        if not path.name.startswith("birch"):
            with path.open("rb") as stream:
                prototype_hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(seed=args.seed, prototype_cache=str(work), prototype_sha256=prototype_hashes,
                      source_manifests=manifests, usd=str(args.out), usd_bytes=args.out.stat().st_size)
    (args.report_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
