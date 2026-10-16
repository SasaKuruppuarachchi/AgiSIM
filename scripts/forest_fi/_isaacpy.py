"""Re-exec a script under Isaac Sim's bundled Python 3.12 with pxr (USD + PhysX schemas), numpy,
scipy, matplotlib and PIL on the path -- no Kit startup (same idea as _bootstrap_usd in
scripts/build_akw_2floor.py)."""
import glob
import os
import sys

FLAG = "_FOREST_FI_BOOTSTRAPPED"


def bootstrap(isaacsim="~/isaacsim"):
    if os.environ.get(FLAG):
        return
    isaacsim = os.path.expanduser(isaacsim)
    ext = os.path.join(isaacsim, "extscache")
    libs = [glob.glob(os.path.join(ext, p))[0] for p in ("omni.usd.libs-*", "omni.usd.schema.physx-*")]
    pydirs = [glob.glob(os.path.join(ext, "omni.kit.pip_archive-*", "pip_prebundle"))[0],
              os.path.join(isaacsim, "exts", "omni.pip.compute", "pip_prebundle"),
              os.path.join(isaacsim, "exts", "omni.isaac.core_archive", "pip_prebundle")]
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, **{FLAG: "1"}, MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([here] + libs + pydirs),
               LD_LIBRARY_PATH=os.pathsep.join([os.path.join(p, "bin") for p in libs]
                                               + [os.environ.get("LD_LIBRARY_PATH", "")]))
    python = os.path.join(isaacsim, "kit", "python", "bin", "python3")
    os.execve(python, [python, "-s", os.path.abspath(sys.argv[0])] + sys.argv[1:], env)
