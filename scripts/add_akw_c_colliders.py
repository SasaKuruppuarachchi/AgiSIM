#!/usr/bin/env python3
"""Give the AKW_C stage (akw_lvl1_climb.usd) static colliders, in place.

Every mesh under /World/akw gets an exact triangle-mesh collider (CollisionAPI +
MeshCollisionAPI approximation "none"), the same as scripts/build_akw_2floor.py does for
AKW_2F (add_static_colliders). Without it the stage had no colliders except the ground
plane: the drone passed through every wall the lidar sees. Idempotent; re-running changes
nothing. build_akw_2floor.py (which uses this stage as its --base) re-applies the same
setting to the meshes it copies from it.

Runs on the USD libraries bundled with Isaac Sim (no Kit startup needed):
    python3 scripts/add_akw_c_colliders.py [--stage .../isaac_envs/akw_lvl1_climb.usd] [--isaacsim ~/isaacsim]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_akw_2floor import ENVS_DIR, _bootstrap_usd, add_static_colliders  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=os.path.join(ENVS_DIR, "akw_lvl1_climb.usd"))
    parser.add_argument("--root", default="/World/akw")
    parser.add_argument("--isaacsim", default=os.path.expanduser("~/isaacsim"))
    args = parser.parse_args()
    _bootstrap_usd(args.isaacsim)

    from pxr import Usd

    stage = Usd.Stage.Open(args.stage)
    layer = stage.GetRootLayer()
    count = add_static_colliders(stage, args.root)
    if layer.dirty:
        layer.Save()
        print(f"colliders: {count} triangle-mesh; wrote {os.path.abspath(args.stage)}")
    else:
        print(f"colliders: {count} triangle-mesh; already up to date")


if __name__ == "__main__":
    main()
