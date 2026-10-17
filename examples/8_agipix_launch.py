#!/usr/bin/env python3
"""Clean host bootstrap for the canonical Agipix Isaac Sim example.

This wrapper deliberately imports only the Python standard library. It clears
host ROS/Python library overlays, sources the Isaac-bundled ROS environment,
then starts the unchanged ``8_agipix.py`` example in Isaac's Python runtime.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shlex
import sys


AGISIM_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCRIPT = AGISIM_ROOT / "examples" / "8_agipix.py"
PEGASUS_EXTENSION = AGISIM_ROOT / "extensions" / "pegasus.simulator"
PEGASUS_PARAMS = PEGASUS_EXTENSION / "pegasus" / "simulator" / "params.py"
PEGASUS_ASSETS = PEGASUS_EXTENSION / "pegasus" / "simulator" / "assets"
ISAAC_ROS_ENV_SCRIPT = AGISIM_ROOT / "scripts" / "isaac_ros_env.sh"
DEFAULT_ISAACSIM_PATH = Path("/home/sasa/isaacsim")
DEFAULT_PX4_DIR = Path("/home/sasa/PX4-Autopilot")
DEFAULT_ASSET_ROOT = Path("/home/sasa/isaac_sim_assets/Assets/Isaac/6.0")

PX4_DEFAULTS = {
    "PX4_UXRCE_DDS_NS": "",
    "PX4_UXRCE_DDS_PORT": "8888",
    "PX4_PARAM_EKF2_HGT_REF": "3",
    "PX4_PARAM_EKF2_EV_CTRL": "15",
    "PX4_PARAM_EKF2_GPS_CTRL": "0",
    "PX4_PARAM_EKF2_MAG_TYPE": "6",
    "PX4_PARAM_MIS_TAKEOFF_ALT": "1.3",
    "PX4_PARAM_SIM_BAT_DRAIN": "0",
    "AGIPIX_CPU_THREADS": "8",
    "AGIPIX_PHYSICS_THREADS": "0",
}

CLEAN_ENV_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "AMENT_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "ROS_VERSION",
    "ROS_PYTHON_VERSION",
    "ROS_DISTRO",
)


def _flat_environment_assets() -> dict[str, Path]:
    """Read Pegasus' flat-world table without importing Isaac/Pegasus modules."""
    if not PEGASUS_PARAMS.is_file():
        raise RuntimeError(f"Pegasus environment table not found: {PEGASUS_PARAMS}")
    tree = ast.parse(PEGASUS_PARAMS.read_text(encoding="utf-8"), filename=str(PEGASUS_PARAMS))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == "FLAT_ENVIRONMENTS"
                   for target in targets):
            continue
        mapping = node.value
        if not isinstance(mapping, ast.Dict):
            raise RuntimeError("Pegasus FLAT_ENVIRONMENTS is not a literal mapping")
        result: dict[str, Path] = {}
        for key, value in zip(mapping.keys, mapping.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise RuntimeError("Pegasus FLAT_ENVIRONMENTS contains a nonliteral key")
            if (not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add) or
                    not isinstance(value.left, ast.Name) or value.left.id != "WORLDS_ASSETS" or
                    not isinstance(value.right, ast.Constant) or
                    not isinstance(value.right.value, str)):
                raise RuntimeError(
                    f"Unsupported Pegasus asset expression for environment {key.value!r}"
                )
            result[key.value] = PEGASUS_ASSETS / "Worlds" / value.right.value.lstrip("/")
        if not result:
            raise RuntimeError("Pegasus FLAT_ENVIRONMENTS is empty")
        return result
    raise RuntimeError("Pegasus params.py has no FLAT_ENVIRONMENTS mapping")


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("physics dt must be a finite positive number of seconds") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("physics dt must be a finite positive number of seconds")
    return parsed


def _nonnegative_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("vehicle id must be a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("vehicle id must be a nonnegative integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    environments = _flat_environment_assets()
    parser = argparse.ArgumentParser(
        description="Launch the canonical Agipix Isaac Sim example with a clean host ROS environment.",
        epilog="Additional Isaac Kit arguments are passed through after the standard options.",
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true",
                         help="Run without the Isaac Sim window (default).")
    display.add_argument("--gui", dest="headless", action="store_false",
                         help="Show the Isaac Sim window.")
    parser.set_defaults(headless=True)
    parser.add_argument("--chase-cam", action=argparse.BooleanOptionalAction, default=True,
                        help="Follow travel direction in the GUI viewport (default: on; ignored headless). "
                             "Tune view/filter defaults in agipix_chase_camera.ChaseCameraSettings.")
    parser.add_argument("--namespace", default="drone", help="Vehicle namespace prefix (default: drone).")
    parser.add_argument("--id", dest="vehicle_id", type=_nonnegative_id, default=0,
                        help="Vehicle ID appended to namespace (default: 0).")
    parser.add_argument("--environment", choices=tuple(environments), default="AKW_C",
                        help="Pegasus flat environment (default: AKW_C).")
    parser.add_argument("--px4-dir", type=Path, default=DEFAULT_PX4_DIR,
                        help="PX4 source/build directory (default: /home/sasa/PX4-Autopilot).")
    parser.add_argument("--physics-dt", type=_positive_seconds, default=0.004,
                        help="Physics timestep in seconds (default: 0.004, 250 Hz).")
    parser.add_argument("--lidar-timestamps", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable per-point lidar timestamps (default: enabled).")
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--preflight", action="store_true",
                         help="Check local paths and print launch settings without starting Isaac Sim.")
    dry_run.add_argument("--print-command", action="store_true",
                         help="Print the clean-environment launch command without executing it.")
    return parser


def _isaac_paths() -> tuple[Path, Path]:
    isaac_root = Path(os.environ.get("ISAACSIM_PATH", str(DEFAULT_ISAACSIM_PATH))).expanduser()
    return isaac_root, isaac_root / "python.sh"


def _canonical_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    argv = [
        str(CANONICAL_SCRIPT),
        "--environment", args.environment,
        "--namespace", args.namespace,
        "--id", str(args.vehicle_id),
        "--px4-dir", str(args.px4_dir.expanduser().resolve()),
        "--physics-dt", format(args.physics_dt, ".12g"),
    ]
    if args.headless:
        argv.append("--headless")
    if not args.chase_cam:
        argv.append("--no-chase-cam")
    if args.lidar_timestamps:
        argv.append("--lidar-timestamps")
    # Match the successful Flight 26 Isaac renderer/asset-root settings.
    argv.extend([
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/autoEnable=false",
        f"--/persistent/isaac/asset_root/default={DEFAULT_ASSET_ROOT}",
    ])
    argv.extend(passthrough)
    return argv


def _launch_plan(args: argparse.Namespace, passthrough: list[str]) -> tuple[list[str], dict[str, str], dict[str, object]]:
    environments = _flat_environment_assets()
    environment_asset = environments[args.environment]
    isaac_root, isaac_python = _isaac_paths()
    px4_dir = args.px4_dir.expanduser().resolve()
    canonical_args = _canonical_args(args, passthrough)
    shell_script = (
        'source "$ISAAC_ROS_ENV_SCRIPT" && '
        'export PYTHONPATH="$PEGASUS_SIMULATOR_PATH" && '
        'exec "$ISAACSIM_PATH/python.sh" "$CANONICAL_SCRIPT" "$@"'
    )
    argv = ["bash", "-c", shell_script, "agipix-launch", *canonical_args[1:]]
    env = os.environ.copy()
    for key in CLEAN_ENV_KEYS:
        env.pop(key, None)
    env.update(PX4_DEFAULTS)
    env.update({
        "ISAACSIM_PATH": str(isaac_root),
        "ISAAC_ROS_ENV_SCRIPT": str(ISAAC_ROS_ENV_SCRIPT),
        "PEGASUS_SIMULATOR_PATH": str(PEGASUS_EXTENSION),
        "CANONICAL_SCRIPT": str(CANONICAL_SCRIPT),
        "ROS_DOMAIN_ID": "5",
        "ROS_LOCALHOST_ONLY": "0",
        "ISAACSIM_ROS_DISTRO": "humble",
        "ISAACSIM_ROS_RMW": "rmw_cyclonedds_cpp",
    })
    info: dict[str, object] = {
        "environment": args.environment,
        "environment_asset": str(environment_asset),
        "environment_asset_exists": environment_asset.is_file(),
        "vehicle": f"{args.namespace}{args.vehicle_id}",
        "namespace": args.namespace,
        "vehicle_id": args.vehicle_id,
        "headless": args.headless,
        "chase_cam": not args.headless and args.chase_cam,
        "physics_dt": args.physics_dt,
        "lidar_timestamps": args.lidar_timestamps,
        "ros_domain_id": 5,
        "flight_loop": "native C++ on the separate ROS 2 mission stack; no Python flight loop",
        "isaac_python": str(isaac_python),
        "px4_dir": str(px4_dir),
        "canonical_script": str(CANONICAL_SCRIPT),
        "isaac_ros_env_script": str(ISAAC_ROS_ENV_SCRIPT),
        "pegasus_python_path": str(PEGASUS_EXTENSION),
        "px4_defaults": PX4_DEFAULTS,
        "kit_args": canonical_args[canonical_args.index("--/renderer/multiGpu/enabled=false"):],
    }
    return argv, env, info


def _preflight_errors(info: dict[str, object]) -> list[str]:
    _, isaac_python = _isaac_paths()
    required_files = (CANONICAL_SCRIPT, ISAAC_ROS_ENV_SCRIPT, isaac_python,
                      Path(str(info["environment_asset"])),
                      Path(str(info["px4_dir"])) / "build/px4_sitl_default/bin/px4")
    errors = [f"required path missing: {path}" for path in required_files if not path.exists()]
    isaac_root, _ = _isaac_paths()
    rmw_library = isaac_root / "exts/isaacsim.ros2.core/humble/lib/librmw_cyclonedds_cpp.so"
    if not rmw_library.is_file():
        errors.append(f"Isaac-bundled ROS middleware library missing: {rmw_library}")
    if not PEGASUS_EXTENSION.is_dir():
        errors.append(f"Pegasus extension missing: {PEGASUS_EXTENSION}")
    if not str(info["namespace"]).strip():
        errors.append("namespace must not be empty")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if not args.namespace.strip():
        parser.error("--namespace must not be empty")
    command, env, info = _launch_plan(args, passthrough)

    if args.preflight:
        errors = _preflight_errors(info)
        print(json.dumps({"ok": not errors, "errors": errors, **info}, indent=2))
        return 0 if not errors else 2
    if args.print_command:
        print(json.dumps(info, indent=2))
        print("\nLaunch command (cleaned environment):")
        print(shlex.join(command))
        print("Environment overrides:")
        for key in sorted(set(PX4_DEFAULTS) | {
            "ISAACSIM_PATH", "ISAAC_ROS_ENV_SCRIPT", "PEGASUS_SIMULATOR_PATH",
            "CANONICAL_SCRIPT", "ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY",
            "ISAACSIM_ROS_DISTRO", "ISAACSIM_ROS_RMW",
        }):
            print(f"{key}={env.get(key, '')}")
        print("Unset before sourcing Isaac ROS:")
        print(" ".join(CLEAN_ENV_KEYS))
        return 0

    errors = _preflight_errors(info)
    if errors:
        parser.error("preflight failed: " + "; ".join(errors))
    os.execvpe(command[0], command, env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
