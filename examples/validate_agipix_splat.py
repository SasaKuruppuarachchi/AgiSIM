#!/usr/bin/env python3
"""Bounded, opt-in simulator validation through the normal clean launcher.

Example: python3 examples/validate_agipix_splat.py --splat-mode --save-splat --environment FOREST_FI
         --splat-out /tmp/forest_splat_validation --validation-seconds 30

Moves ONLY this run's drone via the rigid-body API (level hover, then translation
and yaw), adds a tiny magenta reprojection target, subscribes to the preview, and
writes validation.json plus sample images into the dataset. No flight commands.
"""

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch():
    launcher = load_script("agipix_launcher", Path(__file__).with_name("8_agipix_launch.py"))
    parser = launcher._build_parser()
    parser.add_argument("--validation-seconds", type=float, default=30.0)
    args, extra = parser.parse_known_args()
    if not args.splat_mode or not args.save_splat:
        parser.error("dataset validation requires --splat-mode --save-splat")
    if not math.isfinite(args.validation_seconds) or args.validation_seconds <= 0:
        parser.error("--validation-seconds must be finite and positive")
    for command in (["pgrep", "-x", "px4"], ["pgrep", "-f", "[8]_agipix"], ["pgrep", "-x", "kit"]):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            parser.error(f"Existing simulator/PX4 process detected: {result.stdout.strip()}; leaving it untouched")
    command, env, _ = launcher._launch_plan(args, extra)
    env["CANONICAL_SCRIPT"] = str(Path(__file__).resolve())
    env["AGIPIX_SPLAT_VALIDATION_CHILD"] = "1"
    env["AGIPIX_SPLAT_VALIDATION_SECONDS"] = str(args.validation_seconds)
    os.execvpe(command[0], command, env)


def analyse(output, target, previews, own_module_pixels, segmentation_samples, geometry_checks):
    import numpy as np
    from PIL import Image
    from agipix_splat_preview import make_preview
    from agipix_splat_geometry import CV_FROM_GL, project_equidistant

    transforms = json.loads((output / "transforms.json").read_text())
    frames = transforms["frames"]
    errors = []
    horizon_errors = []
    # Sample hover and moving frames throughout the recorded interval.
    front_frames = [f for f in frames if f["camera_id"] in ("FL", "FR") and f["timestamp"] > 1.0]
    selected = []
    for camera in ("FL", "FR"):
        per_camera = [f for f in front_frames if f["camera_id"] == camera]
        selected.extend(per_camera[::max(1, len(per_camera) // 20)])
    for frame in selected:
        world_cv = np.array(frame["transform_matrix"]) @ CV_FROM_GL
        horizon_errors.append(abs(float(world_cv[2, 0])))
        point = np.linalg.inv(world_cv) @ [*target, 1.0]
        expected = project_equidistant(point[:3])
        if point[2] <= 0 or not (35 < expected[0] < 1245 and 35 < expected[1] < 765):
            continue
        image = np.asarray(Image.open(output / frame["file_path"])).astype(np.int16)
        x, y = np.round(expected).astype(int)
        crop = image[y-30:y+31, x-30:x+31]
        mask = (crop[..., 0] - crop[..., 1] > 60) & (crop[..., 2] - crop[..., 1] > 60)
        v, u = np.nonzero(mask)
        if len(u) < 4:
            continue
        measured = np.array([x - 30 + u.mean() + 0.5, y - 30 + v.mean() + 0.5])
        errors.append({"camera": frame["camera_id"], "timestamp": frame["timestamp"],
                       "expected_px": expected.tolist(), "observed_px": measured.tolist(),
                       "error_px": float(np.linalg.norm(expected - measured))})
    samples = output / "validation"
    samples.mkdir(exist_ok=True)
    sample_time = frames[len(frames) // 2]["timestamp_ns"]
    sample_frames = [f for f in frames if f["timestamp_ns"] == sample_time]
    images = {}
    for frame in sample_frames:
        image = Image.open(output / frame["file_path"])
        image.save(samples / f"{frame['camera_id']}.png")
        images[frame["camera_id"]] = np.asarray(image)
    Image.fromarray(make_preview(images)).save(samples / "preview.png")
    stamps = [p[0] for p in previews]
    walls = [p[1] for p in previews]
    summary = json.loads((output / "summary.json").read_text())
    summary.update(preview_messages=len(previews),
                   preview_sim_hz=(len(stamps)-1) * 1e9 / (stamps[-1]-stamps[0]) if len(stamps)>1 else 0,
                   preview_wall_hz=(len(walls)-1) / (walls[-1]-walls[0]) if len(walls)>1 else 0,
                   reprojection_target_world_m=target, reprojection_checks=errors,
                   reprojection_max_px=max((e["error_px"] for e in errors), default=None),
                   reprojection_mean_px=float(np.mean([e["error_px"] for e in errors])) if errors else None,
                   max_image_right_vertical_component=max(horizon_errors, default=None),
                   own_module_pixels=own_module_pixels, segmentation_samples=segmentation_samples,
                   geometry_reprojection_max_px=max((e["error_px"] for e in geometry_checks), default=None))
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SPLAT_VALIDATION=" + json.dumps({k: v for k, v in summary.items() if k != "reprojection_checks"}), flush=True)
    assert {e["camera"] for e in errors} == {"FL", "FR"}, "Target was not detected in both front cameras"
    # The centroid of a finite, rasterized sphere is an approximate point
    # measurement. RGB adds shading/reconstruction; the instance mask is tighter.
    assert summary["reprojection_max_px"] < 5.0, summary["reprojection_max_px"]
    assert summary["geometry_reprojection_max_px"] < 3.0, summary["geometry_reprojection_max_px"]
    assert all(count == 0 for count in own_module_pixels.values()), own_module_pixels
    assert all(count > 0 for count in segmentation_samples.values()), "Missing module visibility observations"
    assert summary["max_image_right_vertical_component"] < 0.002
    assert len(previews) >= summary["frames_per_camera"] - 2


def run_validation():
    module = load_script("agipix_validation_target", Path(__file__).with_name("8_agipix.py"))
    exit_code = 0
    try:
        import numpy as np
        import omni.replicator.core as rep
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        args = module.parse_cli_args()
        app = module.AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
                              lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
                              gui=not args.headless, chase_cam=args.chase_cam,
                              splat_mode=args.splat_mode, splat_depth=args.splat_depth, splat_out=args.splat_out,
                              save_splat=args.save_splat, leds=args.leds)
        assert not app.stage.GetPrimAtPath(app.drone.prim_path + "/body/RealSense_Camera")
        target = [1.6, 0.0, 1.2]
        sphere = UsdGeom.Sphere.Define(app.stage, "/World/SplatValidationTarget")
        sphere.CreateRadiusAttr(0.025)
        sphere.AddTranslateOp().Set(Gf.Vec3d(*target))
        material = UsdShade.Material.Define(app.stage, "/World/SplatValidationMaterial")
        shader = UsdShade.Shader.Define(app.stage, "/World/SplatValidationMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 1))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 1))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)

        from agipix_splat_geometry import project_equidistant, rendered_camera_to_opencv

        own_module_pixels = {s.mount.name: 0 for s in app._splat_cameras}
        segmentation_samples = {s.mount.name: 0 for s in app._splat_cameras}
        geometry_checks = []
        for sensor in app._splat_cameras:
            prim = app.stage.GetPrimAtPath(sensor.prim_path + "/Module")
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert not prim.HasAPI(UsdPhysics.MassAPI)
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        # Diagnostic instance masks share the writer's exact RGB/matrix tick.
        recorder = app._splat_recorder
        recorder.detach()
        recorder.annotators.append(rep.annotators.get("instance_id_segmentation", init_params={"colorize": False}))
        recorder.attach([s.render_product for s in app._splat_cameras])
        original_capture = recorder._capture
        def checked_capture(data):
            original_capture(data)
            for product, values in data["renderProducts"].items():
                mount = recorder._by_product[product]
                segmentation = values["instance_id_segmentation"]
                mask = np.asarray(segmentation["data"])
                if mask.size != 1280 * 800:
                    continue
                segmentation_samples[mount.name] += 1
                for label, path in segmentation["idToLabels"].items():
                    if str(path).startswith(values["camera"] + "/Module"):
                        count = int(np.count_nonzero(mask == int(label)))
                        own_module_pixels[mount.name] = max(own_module_pixels[mount.name], count)
                    if str(path) == "/World/SplatValidationTarget":
                        v, u = np.nonzero(mask == int(label))
                        if len(u) < 4 or recorder._last_stamp < 1e9:
                            continue
                        world_cv = rendered_camera_to_opencv(values["camera_params"]["cameraViewTransform"])
                        expected = project_equidistant((np.linalg.inv(world_cv) @ [*target, 1])[:3])
                        measured = np.array([u.mean() + 0.5, v.mean() + 0.5])
                        geometry_checks.append({"camera": mount.name, "timestamp_ns": recorder._last_stamp,
                                                "expected_px": expected.tolist(), "observed_px": measured.tolist(),
                                                "error_px": float(np.linalg.norm(expected - measured))})
        recorder._capture = checked_capture
        previews = []
        from agipix_splat_preview import PREVIEW_WIDTH, PREVIEW_HEIGHT
        def on_preview(message):
            assert message.encoding == "rgb8" and (message.width, message.height) == (PREVIEW_WIDTH, PREVIEW_HEIGHT)
            assert message.header.frame_id == f"{app.vehicle_name}/splat_panorama"
            previews.append((message.header.stamp.sec * 10**9 + message.header.stamp.nanosec, time.monotonic()))
        subscription = app.node.create_subscription(Image, f"{app.vehicle_name}/splat/preview", on_preview, qos_profile_sensor_data)
        original_update = module.simulation_app.update
        duration = float(os.environ["AGIPIX_SPLAT_VALIDATION_SECONDS"])
        def bounded_update():
            t = module.SimulationManager.get_simulation_time()
            body = app.drone._body_rigid_prim
            if body is not None:
                moving = max(0.0, t - 5.0)
                yaw = math.radians(18.0) * math.sin(1.1 * moving)
                position = [0.12 * math.sin(0.9 * moving), 0.12 * math.sin(0.7 * moving), 1.2]
                body.set_world_poses(positions=[position], orientations=[[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])
                body.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
            original_update()
            rclpy.spin_once(app.node, timeout_sec=0)
            if app._splat_recorder._last_stamp >= round(duration * 1e9):
                app.stop_sim = True
        module.simulation_app.update = bounded_update
        app.run()
        rclpy.spin_once(app.node, timeout_sec=0.05)
        print("SPLAT_TOPICS=" + json.dumps(sorted(app.node.get_topic_names_and_types())), flush=True)
        (app._splat_recorder._dataset.output_dir / "preview_receipts.json").write_text(json.dumps(previews))
        (app._splat_recorder._dataset.output_dir / "geometry_checks.json").write_text(json.dumps(geometry_checks))
        analyse(app._splat_recorder._dataset.output_dir, target, previews, own_module_pixels, segmentation_samples, geometry_checks)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        module.restore_thread_settings()
        module.simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    run_validation() if os.environ.get("AGIPIX_SPLAT_VALIDATION_CHILD") == "1" else launch()
