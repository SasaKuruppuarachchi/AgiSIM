#!/usr/bin/env python3
"""Bounded snowy-forest visual/lidar check using the canonical Agipix launcher.

  python3 examples/validate_forest_snow.py --output /tmp/snow-headless
  python3 examples/validate_forest_snow.py --gui --splat-mode --output /tmp/snow-gui
  python3 examples/validate_forest_snow.py --environment FOREST_MID_SNOW --gui --splat-mode --output /tmp/mid-snow-gui

The owned drone is held at 2 m and moved within the spawn clearing, without
arming PX4. This is a sensor/visual check, not a PX4 flight test or an RTF
benchmark. Use bench_agipix_splat_rtf.py for comparable performance results.
Point-cloud figures contain actual ROS RTX returns, never GT voxel samples.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from bench_agipix_splat_rtf import ensure_idle, load_script
from agipix_owned_process import OwnedProcessGroup


HERE = Path(__file__).resolve().parent
ENVIRONMENTS = frozenset({"FOREST_SNOW", "FOREST_MID_SNOW", "FOREST_PINE_HILLS"})


def default_stand(environment="FOREST_SNOW"):
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unsupported forest validation environment: {environment}")
    name = environment.lower()
    return HERE.parent / "artifacts" / name / "build" / f"{name}_stand.json"


DEFAULT_STAND = default_stand()


def ground_point_key(environment):
    return "ground_snow_points" if environment == "FOREST_SNOW" else "ground_surface_points"


def validation_pose(seconds):
    """A static observation followed by a slow path wholly inside a 1.25 m disc."""
    phase = max(0.0, seconds - 5.0) * .3
    # Smooth onset, returning through the origin; no collision with the 4 m stem clearing.
    ramp = min(1.0, max(0.0, seconds - 5.0) / 2.0)
    ramp = ramp * ramp * (3 - 2 * ramp)
    return [1.2 * ramp * math.sin(phase), .6 * ramp * math.sin(2 * phase), 2.0]


def pointcloud_xyz(message, stride=1):
    """Respect endian, field offsets, row padding and point padding in ROS clouds."""
    import numpy as np

    fields = {field.name: field for field in message.fields}
    if not all(name in fields and fields[name].datatype == 7 for name in ("x", "y", "z")):
        raise ValueError("Expected PointCloud2 float32 x/y/z fields")
    dtype = np.dtype(dict(names=["x", "y", "z"],
                         formats=[">f4" if message.is_bigendian else "<f4"] * 3,
                         offsets=[fields[name].offset for name in ("x", "y", "z")],
                         itemsize=message.point_step))
    rows = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                      strides=(message.row_step, message.point_step))
    xyz = np.stack([rows[name].ravel() for name in ("x", "y", "z")], axis=1)
    finite = xyz[np.isfinite(xyz).all(axis=1)]
    return finite[::stride].copy(), len(finite)


def terrain_heights(terrain, xy):
    """Same diagonal triangle interpolation as forest_fi.scene.Terrain.z_at."""
    import numpy as np

    grid = np.asarray(terrain["z"], dtype=float)
    step = float(terrain["grid"])
    f = np.clip((xy + float(terrain["area"]) / 2) / step, 0, len(grid) - 1 - 1e-9)
    ij = np.floor(f).astype(int)
    uv = f - ij
    i, j, u, v = ij[:, 0], ij[:, 1], uv[:, 0], uv[:, 1]
    z00, z10, z01, z11 = grid[i, j], grid[i + 1, j], grid[i, j + 1], grid[i + 1, j + 1]
    return np.where(u >= v, z00 + u * (z10 - z00) + v * (z11 - z10),
                    z00 + v * (z01 - z00) + u * (z11 - z01))


def classify_cloud(world, stand, environment="FOREST_SNOW"):
    """Conservative geometric evidence; trunk labels are approximate, not RTX semantic IDs."""
    import numpy as np
    from scipy.spatial import cKDTree

    agl = world[:, 2] - terrain_heights(stand["terrain"], world[:, :2])
    half = float(stand["terrain"]["area"]) / 2
    inside = np.max(np.abs(world[:, :2]), axis=1) < half - .1
    ground = inside & (np.abs(agl) < .12)
    trees = stand["trees"]
    distance, index = cKDTree([[tree["x"], tree["y"]] for tree in trees]).query(world[:, :2])
    radii = np.asarray([tree["dbh"] / 2 for tree in trees])[index]
    crown = np.asarray([tree["crown_base"] for tree in trees])[index]
    trunks = inside & (agl > .3) & (agl < np.minimum(crown - .2, 3.5)) & (np.abs(distance - radii) < .18)
    labels = np.zeros(len(world), dtype=np.uint8)
    labels[ground] = 1
    labels[trunks] = 2
    return labels, agl, {
        ground_point_key(environment): int(ground.sum()), "approximate_low_trunk_points": int(trunks.sum()),
        "other_points": int(np.count_nonzero(labels == 0)),
        "ground_vertical_tolerance_m": .12, "trunk_radial_tolerance_m": .18,
        "classification": "Stored terrain triangles and approximate DBH cylinders below crowns; no semantic IDs.",
        "snow_check_scope": ("Snow-covered ground only; crown cap returns are not individually identified."
                             if environment == "FOREST_SNOW" else
                             "Mixed ground surface only; snow, moss and soil returns are not semantically classified. "
                             "Crown cap returns are not individually identified."),
    }


def plot_cloud(path, world, agl, poses, environment="FOREST_SNOW", vehicle_name="drone0"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    take = np.arange(0, len(world), max(1, math.ceil(len(world) / 160000)))
    xyz, height = world[take], agl[take]
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        fig.patch.set_facecolor("#081622")
        for axis in axes:
            axis.set_facecolor("#081622")
            axis.grid(alpha=.12)
        colors = np.clip(height, 0, 6)
        cloud = axes[0].scatter(xyz[:, 0], xyz[:, 1], c=colors, s=.35, alpha=.75,
                                cmap="icefire" if "icefire" in plt.colormaps() else "viridis", vmin=0, vmax=6)
        axes[0].plot(poses[:, 0], poses[:, 1], color="#ffba6a", linewidth=1.7, label="owned drone path")
        axes[0].scatter([0], [0], color="white", marker="+", s=55)
        axes[0].set(xlabel="World X (m)", ylabel="World Y (m)", title="Top view · actual RTX Mid-360 returns")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].scatter(xyz[:, 0], xyz[:, 2], c=colors, s=.35, alpha=.65, cmap=cloud.cmap, vmin=0, vmax=6)
        axes[1].plot(poses[:, 0], poses[:, 2], color="#ffba6a", linewidth=1.5)
        floor = "snow floor" if environment == "FOREST_SNOW" else "mixed forest floor"
        axes[1].set(xlabel="World X (m)", ylabel="World Z (m)", title=f"Elevation · {floor} and tree stems")
        axes[1].set_ylim(float(np.percentile(xyz[:, 2], .2)) - .3, min(12, float(np.percentile(xyz[:, 2], 99.8)) + .5))
        fig.colorbar(cloud, ax=axes, shrink=.7, label="Height above ground surface (m)")
        fig.suptitle(f"{environment} | ROS /{vehicle_name}/livox/lidar", fontsize=17, fontweight="bold")
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)


def launch():
    launcher = load_script("snow_launcher", HERE / "8_agipix_launch.py")
    parser = launcher._build_parser()
    parser.description = __doc__
    parser.set_defaults(environment="FOREST_SNOW")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory, never overwritten.")
    parser.add_argument("--stand", type=Path,
                        help="Stand metadata; defaults to artifacts/<environment>/build/<environment>_stand.json.")
    parser.add_argument("--validation-seconds", type=float, default=15)
    parser.add_argument("--timeout", type=float, default=300, help="Wall-clock timeout including startup.")
    parser.add_argument("--cloud-stride", type=int, default=4, help="Retain every Nth finite lidar return (default 4).")
    parser.add_argument("--establishing-view", action=argparse.BooleanOptionalAction, default=None,
                        help="Additional wide render (default enabled with --gui).")
    parser.add_argument("--pine-render-trials", action="store_true",
                        help="Pine-only low-resolution material/render comparisons; no full intermediate captures.")
    parser.add_argument("--asset-override", type=Path,
                        help="Pine-only small diagnostic USD overlay, leaving the built asset unchanged.")
    args, extra = parser.parse_known_args()
    if args.environment not in ENVIRONMENTS:
        parser.error("This scene validation requires a supported generated forest")
    if args.pine_render_trials and args.environment != "FOREST_PINE_HILLS":
        parser.error("--pine-render-trials requires FOREST_PINE_HILLS")
    if args.asset_override and (args.environment != "FOREST_PINE_HILLS" or not args.asset_override.is_file()):
        parser.error("--asset-override requires FOREST_PINE_HILLS and an existing USD")
    if args.save_splat or args.splat_depth or args.splat_out:
        parser.error("Use --splat-mode without dataset saving; this check saves its own bounded image samples")
    if args.preflight or args.print_command:
        parser.error("Use 8_agipix_launch.py directly for --preflight or --print-command")
    if not math.isfinite(args.validation_seconds) or args.validation_seconds < 8:
        parser.error("--validation-seconds must be finite and >= 8")
    if not math.isfinite(args.timeout) or args.timeout < 30 or args.cloud_stride < 1:
        parser.error("--timeout must be finite and >= 30; --cloud-stride must be positive")
    stand = (args.stand or default_stand(args.environment)).expanduser().resolve()
    if not stand.is_file():
        parser.error(f"Stand metadata is required for lidar checks: {stand}")
    command, env, info = launcher._launch_plan(args, extra)
    errors = launcher._preflight_errors(info)
    if errors:
        parser.error("preflight failed: " + "; ".join(errors))
    with Path("/tmp/agipix-sim-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle()
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        files = [Path(__file__), HERE / "8_agipix.py", HERE / "8_agipix_launch.py",
                 HERE / "agipix_chase_camera.py", HERE / "utils/drone_location_pub.py", stand,
                 Path(info["environment_asset"]), *HERE.glob("agipix_splat_*.py")]
        if args.environment == "FOREST_PINE_HILLS":
            files += sorted((HERE.parent / "scripts/forest_pine_hills").glob("*.py"))
            files += [HERE.parent / "scripts/forest_pine_hills/PineHills.mdl",
                      HERE / "pine_hills_lidar_probe.py", HERE / "pine_hills_render_trials.py",
                      Path(info["environment_asset"]).parent / "textures/forest_pine_hills/PineHills.mdl"]
            if args.asset_override:
                # Record the override itself; diagnostic sublayer/MDL composition
                # audits live beside each overlay. Final validation has no overlay.
                files.append(args.asset_override.resolve())
        config = dict(output=str(output), seconds=args.validation_seconds, cloud_stride=args.cloud_stride,
                      stand=str(stand), timeout=args.timeout, pine_render_trials=args.pine_render_trials,
                      asset_override=str(args.asset_override.resolve()) if args.asset_override else None,
                      establishing=(not args.headless if args.establishing_view is None else args.establishing_view),
                      launch=info, command=command,
                      source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
        (output / "config.json").write_text(json.dumps(config, indent=2))
        env.update(CANONICAL_SCRIPT=str(Path(__file__).resolve()), AGIPIX_SNOW_CHECK=json.dumps(config),
                   ROS_DOMAIN_ID="195")
        started = time.monotonic()
        with (output / "simulator.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            owned = OwnedProcessGroup(process)
            try:
                code = process.wait(timeout=args.timeout)
            finally:
                owned.stop()
        log_text = (output / "simulator.log").read_text()
        loaded = f"AGIPIX_ENVIRONMENT_LOADED={args.environment}" in log_text
        passed = f"{args.environment}_VALIDATION_PASSED=" in log_text
        if code == 0 and not (loaded and passed and (output / "report.json").is_file()):
            code = 1
        (output / "launch_result.json").write_text(json.dumps(dict(exit_code=code,
            wall_seconds=time.monotonic() - started, environment_loaded=loaded, validation_passed=passed,
            report_exists=(output / "report.json").is_file()), indent=2))
        print(f"{args.environment}_CHECK_EXIT={code} output={output}", flush=True)
        raise SystemExit(code)


def run_child(config):
    output = Path(config["output"])
    (output / "isaac_pid").write_text(str(os.getpid()))
    module = load_script("snow_target", HERE / "8_agipix.py")
    code = 0
    observer = None
    app = None
    lidar_probe = None
    try:
        import numpy as np
        import rclpy
        from PIL import Image
        from nav_msgs.msg import Odometry
        from pxr import Gf, UsdGeom, UsdShade
        from rclpy.qos import qos_profile_sensor_data
        from scipy.spatial.transform import Rotation, Slerp
        from sensor_msgs.msg import PointCloud2

        args = module.parse_cli_args()
        if config.get("asset_override"):
            module.FLAT_ENVIRONMENTS[args.environment] = config["asset_override"]
        app = module.AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
            lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
            gui=not args.headless, chase_cam=args.chase_cam, splat_mode=args.splat_mode, leds=args.leds)
        assert app.environment in ENVIRONMENTS
        ground = app.stage.GetPrimAtPath("/World/layout/Terrain/ground")
        material, _ = UsdShade.MaterialBindingAPI(ground).ComputeBoundMaterial()
        ground_material = str(material.GetPath()) if material else None
        if not ground or not material:
            raise RuntimeError("Expected bound ground mesh at /World/layout/Terrain/ground")
        expected_material = {"FOREST_SNOW": "snow", "FOREST_MID_SNOW": "mid_snow_ground",
                             "FOREST_PINE_HILLS": "pine_hills_ground_needles"}[app.environment]
        assert ground_material == f"/World/layout/Looks/{expected_material}", ground_material
        establishing = None
        if config["establishing"] or config.get("pine_render_trials"):
            import omni.replicator.core as rep
            camera = UsdGeom.Camera.Define(app.stage, "/World/ForestSnowEvidenceCamera")
            camera.CreateFocalLengthAttr(20)
            camera.CreateHorizontalApertureAttr(36)
            camera.CreateVerticalApertureAttr(20.25)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 300))
            camera.CreateFStopAttr(0)
            if app.environment == "FOREST_PINE_HILLS":
                terrain = json.loads(Path(config["stand"]).read_text())["terrain"]
                z = float(terrain_heights(terrain, np.array([[-6.5, -11.]]))[0])
                target_z = float(terrain_heights(terrain, np.array([[3.2, 6.]]))[0])
                eye, target = (-6.5, -11, z + 1.45), (3.2, 6, target_z + 1.3)
                camera.CreateFocalLengthAttr(28)
            elif app.environment == "FOREST_MID_SNOW":
                terrain = json.loads(Path(config["stand"]).read_text())["terrain"]
                z = float(terrain_heights(terrain, np.array([[18., -20.]]))[0])
                eye, target = (18, -20, z + 1.65), (45, -15, z + 2.5)
            else:
                eye, target = (-3, -2.5, 3.8), (8, 5, 3.5)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),
                Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())
            product = rep.create.render_product(str(camera.GetPath()), (580, 326) if config.get("pine_render_trials") else (1600, 900))
            establishing = rep.annotators.get("rgb")
            establishing.attach(product)
            if app._render_scope is not None:
                app._render_scope.fix_camera_exposure(app.stage, [str(camera.GetPath())])

        if config.get("pine_render_trials"):
            from pine_hills_render_trials import run_trials
            run_trials(module, app, establishing, config, eye, target)
            return

        if app.environment == "FOREST_PINE_HILLS":
            from pine_hills_lidar_probe import LidarScalarProbe
            lidar_probe = LidarScalarProbe(app)

        recorder = app._splat_recorder
        images, frame_counts = {}, {}
        if recorder is not None:
            original_capture = recorder._capture
            def capture(data):
                previous = recorder._last_stamp
                original_capture(data)
                if previous == recorder._last_stamp:
                    return
                for key, values in data["renderProducts"].items():
                    name = recorder._by_product[key].name
                    frame_counts[name] = frame_counts.get(name, 0) + 1
                    images[name] = np.asarray(values["rgb"]["data"])[..., :3].copy()
            recorder._capture = capture

        clouds, cloud_stamps, cloud_sizes = [], [], []
        pose_stamps, poses, quaternions = [], [], []
        observer = rclpy.create_node("forest_snow_validation")
        def odometry(message):
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if pose_stamps and stamp <= pose_stamps[-1]:
                return
            assert message.header.frame_id == f"{app.vehicle_name}/world"
            p, q = message.pose.pose.position, message.pose.pose.orientation
            pose_stamps.append(stamp)
            poses.append([p.x, p.y, p.z])
            quaternions.append([q.x, q.y, q.z, q.w])
        def cloud(message):
            if lidar_probe is not None:
                lidar_probe.observe_cloud(message)
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if stamp < 2_000_000_000:
                return
            assert message.header.frame_id == f"{app.vehicle_name}/lidar_link"
            xyz, count = pointcloud_xyz(message, config["cloud_stride"])
            clouds.append(xyz)
            cloud_stamps.append(stamp)
            cloud_sizes.append(count)
        subscriptions = [  # Keep subscription handles alive until the observer is destroyed.
            observer.create_subscription(Odometry, f"/{app.vehicle_name}/physics_gt/odom", odometry, qos_profile_sensor_data),
            observer.create_subscription(PointCloud2, f"/{app.vehicle_name}/livox/lidar", cloud, qos_profile_sensor_data),
        ]
        original_update = module.simulation_app.update
        captures, saved, timing = [], set(), {}
        screenshot_times = (4.0, max(6.0, config["seconds"] - 2))
        def update():
            t = module.SimulationManager.get_simulation_time()
            body = app.drone._body_rigid_prim
            if body is not None:
                body.set_world_poses(positions=[validation_pose(t)], orientations=[[1, 0, 0, 0]])
                body.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
            if t >= 3 and "wall_start" not in timing:
                timing.update(wall_start=time.monotonic(), sim_start=t)
            original_update()
            if lidar_probe is not None:
                lidar_probe.sample()
            for _ in range(20):
                rclpy.spin_once(observer, timeout_sec=0)
            t = module.SimulationManager.get_simulation_time()
            for index, threshold in enumerate(screenshot_times):
                if t < threshold or index in saved:
                    continue
                if recorder is not None and set(images) != {"FL", "FR", "RL", "RR"}:
                    continue
                suffix = "static" if index == 0 else "moving"
                if recorder is not None:
                    from agipix_splat_preview import make_preview
                    for name, rgb in images.items():
                        Image.fromarray(rgb).save(output / f"fisheye_{name}_{suffix}.png")
                    Image.fromarray(make_preview(images)).save(output / f"panorama_{suffix}.png")
                if establishing is not None:
                    rgba = np.asarray(establishing.get_data())
                    if not rgba.size:
                        continue
                    Image.fromarray(rgba[..., :3]).save(output / f"forest_{suffix}.png")
                if not args.headless:
                    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                    viewport = get_active_viewport()
                    timing[f"viewport_camera_{suffix}"] = str(viewport.camera_path)
                    captures.append(capture_viewport_to_file(viewport, str(output / f"chase_{suffix}.png")))
                saved.add(index)
            if t >= config["seconds"]:
                timing.update(wall_end=time.monotonic(), sim_end=t)
                app.stop_sim = True
        module.simulation_app.update = update
        render_scope_metadata = app._render_scope.metadata() if app._render_scope is not None else None
        render_previous = dict(app._render_scope.previous) if app._render_scope is not None else {}
        camera_exposures = {}
        if app._render_scope is not None:
            for path in app._render_scope.camera_paths:
                camera = UsdGeom.Camera.Get(app.stage, path)
                camera_exposures[path] = dict(iso=camera.GetExposureIsoAttr().Get(),
                    time=camera.GetExposureTimeAttr().Get(), f_stop=camera.GetExposureFStopAttr().Get(),
                    auto=camera.GetPrim().GetAttribute("omni:rtx:autoExposure:enabled").Get())
        app.run()
        restored = {key: module.carb.settings.get_settings().get(key) == value
                    for key, value in render_previous.items()}
        for _ in range(20):
            rclpy.spin_once(observer, timeout_sec=.005)
        topics = sorted(observer.get_topic_names_and_types())
        observer.destroy_node()
        observer = None
        if not clouds or len(pose_stamps) < 2:
            raise RuntimeError("Missing ROS lidar or original physics odometry")
        pose_times = np.asarray(pose_stamps, dtype=np.float64) / 1e9
        scan_times = np.asarray(cloud_stamps, dtype=np.float64) / 1e9
        sample_times = np.clip(scan_times, pose_times[0], pose_times[-1])
        p = np.asarray(poses)
        scan_positions = np.column_stack([np.interp(sample_times, pose_times, p[:, axis]) for axis in range(3)])
        scan_rotations = Slerp(pose_times, Rotation.from_quat(quaternions))(sample_times)
        lidar_rotation = Rotation.from_quat([*app.lidar_ori[1:], app.lidar_ori[0]])
        world_clouds = [rotation.apply(lidar_rotation.apply(xyz) + app.lidar_trans) + position
                       for xyz, position, rotation in zip(clouds, scan_positions, scan_rotations)]
        world = np.concatenate(world_clouds).astype(np.float32)
        stand = json.loads(Path(config["stand"]).read_text())
        labels, agl, checks = classify_cloud(world, stand, app.environment)
        offsets = np.r_[0, np.cumsum([len(xyz) for xyz in clouds])]
        # Static phase avoids motion-distortion ambiguity in the geometric pass/fail counts.
        static_mask = np.repeat(scan_times < 4.9, np.diff(offsets))
        static_checks = {ground_point_key(app.environment): int(np.count_nonzero((labels == 1) & static_mask)),
                         "approximate_low_trunk_points": int(np.count_nonzero((labels == 2) & static_mask))}
        np.savez_compressed(output / "pointcloud.npz", world_xyz=world, lidar_xyz=np.concatenate(clouds),
            cloud_offsets=offsets, cloud_stamps_ns=cloud_stamps, finite_points_per_cloud=cloud_sizes,
            geometry_labels=labels, height_above_ground_m=agl.astype(np.float32),
            pose_stamps_ns=pose_stamps, poses_world=p, quaternions_xyzw=quaternions,
            scan_positions_world=scan_positions, scan_quaternions_xyzw=scan_rotations.as_quat())
        plot_cloud(output / "lidar_cloud.png", world, agl, p, app.environment, app.vehicle_name)
        timing["validation_rtf"] = (timing["sim_end"] - timing["sim_start"]) / (timing["wall_end"] - timing["wall_start"])
        report = dict(environment=app.environment, environment_loaded=True, physics_dt=args.physics_dt,
            environment_render_scope=render_scope_metadata,
            camera_exposure_before_run=camera_exposures, render_scope_restored=restored,
            render_settings_note="The following values are read after app.run restores this environment's scope.",
            render_settings={key: module.carb.settings.get_settings().get(key) for key in (
                "/rtx/scenedb/skipMostLights", "/rtx/directLighting/enabled", "/rtx/indirectDiffuse/enabled",
                "/rtx/domeLight/upperLowerStrategy", "/rtx/rendermode", "/rtx/post/histogram/enabled",
                "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/exposureTime", "/rtx/post/tonemap/fNumber")},
            motion="Scripted unarmed body poses inside spawn clearing; not a PX4 flight test.",
            lidar_scalar_probe=lidar_probe.summary() if lidar_probe is not None else None,
            lidar_messages=len(clouds), finite_returns=sum(cloud_sizes), retained_points=len(world),
            cloud_stride=config["cloud_stride"], geometry_label_values={"other": 0,
                ("snow_ground" if app.environment == "FOREST_SNOW" else "ground_surface"): 1, "approximate_trunk": 2},
            lidar_transform="Mount + interpolated original ROS physics pose at each scan header; no per-point deskew.",
            pose_time_clamp_max_s=float(np.max(np.abs(scan_times - sample_times))),
            ground_material=ground_material, lidar_geometry_checks=checks, static_phase_checks=static_checks,
            splat_frames=frame_counts, screenshots=[path.name for path in sorted(output.glob("*.png"))],
            timing=timing, performance_note="Evidence capture/extra render/ROS observer overhead; use the serial RTF benchmark.",
            topics=topics, source_sha256=config["source_sha256"])
        (output / "report.json").write_text(json.dumps(report, indent=2))
        assert len(clouds) >= 30 and static_checks[ground_point_key(app.environment)] >= 100, static_checks
        assert all(restored.values()), restored
        assert static_checks["approximate_low_trunk_points"] >= 20, static_checks
        assert len(saved) == len(screenshot_times), saved
        if recorder is not None:
            assert set(frame_counts) == {"FL", "FR", "RL", "RR"} and min(frame_counts.values()) >= 30, frame_counts
        if not args.headless:
            assert all((output / f"chase_{suffix}.png").is_file() for suffix in ("static", "moving"))
            if args.chase_cam:
                assert all("ChaseCamera" in timing[f"viewport_camera_{suffix}"] for suffix in ("static", "moving")), timing
        print(f"{app.environment}_VALIDATION_PASSED=" + json.dumps(dict(lidar_messages=len(clouds),
              static_phase_checks=static_checks, splat_frames=frame_counts)), flush=True)
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        if lidar_probe is not None:
            lidar_probe.close()
        if observer is not None:
            observer.destroy_node()
        # Errors during evidence setup occur before app.run() owns cleanup.
        if code and app is not None:
            for backend in app._owned_backends:
                try:
                    backend.stop()
                except Exception:
                    traceback.print_exc()
        module.restore_thread_settings()
        module.simulation_app.close(exit_code=code)


if __name__ == "__main__":
    run_child(json.loads(os.environ["AGIPIX_SNOW_CHECK"])) if "AGIPIX_SNOW_CHECK" in os.environ else launch()
