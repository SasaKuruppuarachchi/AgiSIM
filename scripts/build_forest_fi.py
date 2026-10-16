#!/usr/bin/env python3
"""Build the sim-ready Finnish managed boreal forest environment (forest_fi.usd).

A ~100 x 100 m thinned Scots pine / Norway spruce / silver birch stand on gently undulating
terrain, for sub-canopy lidar mapping flights (quadrotor at 2-3 m AGL, Livox Mid-360 pitched 45 deg
nose-down, RealSense camera). The vehicle spawns at (0, 0, 0.07), yaw 0, in a small clearing.

Stages (all seeded, reproducible):
  fetch   download the Poly Haven 'pine_forest' assets (CC0) into ASSETS_SRC
          (scripts/forest_fi/fetch_polyhaven.py; tree-gen is cloned separately, see LICENSES.md)
  protos  prototype preparation (scripts/forest_fi/protos.py): Poly Haven glTF -> Z-up parts,
          needle/frond alpha-card cleanup, pine crown raise, spruce low-whorl skirts, Blender
          quadric decimation of needles (scripts/forest_fi/blender_decimate.py), silver birch
          from friggog/tree-gen (scripts/forest_fi/blender_treegen.py), procedural bilberry.
  scene   terrain, stand, understory, colliders, materials, lights -> USD (this file)
  all     protos + scene (default)

Runs on the USD/numpy/scipy libraries bundled with Isaac Sim (no Kit startup) and, for the protos
stage, a portable Blender 4.2 LTS (~/tools/blender-4.2/blender or $BLENDER), run headless with
--factory-startup --disable-autoexec and our own scripts only:
    python3 scripts/build_forest_fi.py [all|protos|scene] [--seed 2026] [--isaacsim ~/isaacsim]
        [--src ~/workspace/agipix_control/assets_src/forest_fi] [--out .../isaac_envs/forest_fi.usd]
        [--report-dir ~/workspace/agipix_control/super_parity_results/forest_fi_01]

Outputs: forest_fi.usd (usdc) + textures/forest_fi/ next to it; the stand table
(forest_fi_stand.json: every tree, collider, understory object) in --report-dir.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "forest_fi"))
from _isaacpy import bootstrap  # noqa: E402

ENVS_DIR = os.path.join(HERE, "..", "extensions", "pegasus.simulator", "pegasus", "simulator", "assets", "Worlds",
                        "isaac_envs")
DEFAULT_SRC = os.path.expanduser("~/workspace/agipix_control/assets_src/forest_fi")
DEFAULT_REPORT = os.path.expanduser("~/workspace/agipix_control/super_parity_results/forest_fi_01")


def log(msg):
    print(f"[forest_fi {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all", choices=("all", "protos", "scene"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--isaacsim", default="~/isaacsim")
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--work", default=None, help="intermediate files (default SRC/work)")
    ap.add_argument("--out", default=os.path.join(ENVS_DIR, "forest_fi.usd"))
    ap.add_argument("--report-dir", default=DEFAULT_REPORT)
    a = ap.parse_args()
    bootstrap(a.isaacsim)
    work = a.work or os.path.join(a.src, "work")
    os.makedirs(work, exist_ok=True)
    if a.stage in ("all", "protos"):
        import protos
        protos.build_all(a.src, work, log)
    if a.stage in ("all", "scene"):
        import scene
        scene.build_scene(a, work, log)


if __name__ == "__main__":
    main()
