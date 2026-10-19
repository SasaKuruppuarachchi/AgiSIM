#!/usr/bin/env python3
"""Build a mossy spruce forest with patchy melting snow from verified CC0 assets.

    python3 scripts/build_forest_mid_snow.py --snow-cover .23

Snow cover is a build-time ground-area fraction, not a runtime simulation flag.
Reuses the read-only FOREST_FI prototype cache; no Blender or downloads run here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "forest_fi"))
from _isaacpy import bootstrap
from build_forest_snow import verify_manifest, write_licenses


def log(message):
    print(f"[forest_mid_snow {time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snow-cover", type=float, default=.23, help="Fraction of ground covered in melting snow, 0..1 (default .23)")
    parser.add_argument("--ground-resolution", type=int, choices=(2048, 4096, 8192), default=8192)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--isaacsim", default="~/isaacsim")
    parser.add_argument("--src", type=Path, default=Path.home() / "workspace/agipix_control/assets_src/forest_fi")
    parser.add_argument("--snow-src", type=Path, default=ROOT / "assets_src/forest_snow")
    parser.add_argument("--work", type=Path, help="Read-only prototype cache; default --src/work")
    parser.add_argument("--out", type=Path, default=ROOT / "extensions/pegasus.simulator/pegasus/simulator/assets/Worlds/isaac_envs/forest_mid_snow.usd")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "artifacts/forest_mid_snow/build")
    args = parser.parse_args()
    if not math.isfinite(args.snow_cover) or not 0 <= args.snow_cover <= 1:
        parser.error("--snow-cover must be a finite fraction between 0 and 1")
    bootstrap(args.isaacsim)
    import scene
    from forest_mid_snow.scene import MidSnowTerrain, compose
    manifests = [verify_manifest(args.src), verify_manifest(args.snow_src)]
    log(f"verified {sum(len(a['files']) for m in manifests for a in m.values())} original files against API MD5s")
    work = args.work or args.src / "work"
    if not (work / "protos/protos.json").is_file():
        parser.error("Missing shared prototype cache. Build FOREST_FI prototypes first or supply --work.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    summary = scene.build_scene(args, str(work), log, name="forest_mid_snow",
        species_frac={"pine": .08, "spruce": .92, "birch": 0}, terrain_factory=MidSnowTerrain,
        postprocess=lambda stage, stand, tex_dir, tex_rel: compose(stage, stand, tex_dir, tex_rel,
            args.snow_src, args.src, log, snow_fraction=args.snow_cover, resolution=args.ground_resolution))
    write_licenses(args.out, manifests, name="forest_mid_snow",
        surface_note="The ground atlas mixes forest_leaves_02 moss, forrest_ground_03 litter and snow_02. A single visible dome uses snowy_forest with a documented small linear ambient radiance offset; statistics and hashes are in the stand summary. These are derivatives of CC0 data, not of the reference photograph.")
    (args.report_dir / "forest_mid_snow_needle_alpha.json").write_text(json.dumps(summary["needle_alpha_cleanup"], indent=2))
    hashes = {}
    for path in sorted((work / "protos").glob("*.npz")):
        if not path.name.startswith("birch"):
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = dict(seed=args.seed, snow_cover=args.snow_cover, ground_resolution=args.ground_resolution,
        prototype_cache=str(work), prototype_sha256=hashes, source_manifests=manifests,
        usd=str(args.out), usd_bytes=args.out.stat().st_size)
    (args.report_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
