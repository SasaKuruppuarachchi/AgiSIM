#!/usr/bin/env python3
"""Serial LED sensor/visual checks; use the RTF benchmark for performance.

Run with --splat-mode, optionally --leds/--gui/--night/--tilt. Writes all four
camera samples, instance-mask LED counts and lidar clouds to a new --output.
Only this validation's unarmed drone is moved. Scene darkening is test-only.
"""

import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import traceback

from bench_agipix_splat_rtf import ensure_idle, load_script


def launch():
    launcher = load_script("led_launcher", Path(__file__).with_name("8_agipix_launch.py"))
    parser = launcher._build_parser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seconds", type=float, default=12)
    parser.add_argument("--night", action="store_true")
    parser.add_argument("--tilt", action="store_true")
    parser.add_argument("--light-panels", action="store_true", help="Add matte targets 25 cm outside each LED to check illumination.")
    parser.add_argument("--light-intensity", type=float, help="Diagnostic override for the LED source intensity.")
    args, extra = parser.parse_known_args()
    if not args.splat_mode or args.save_splat or args.splat_depth or args.splat_out:
        parser.error("Use --splat-mode without dataset saving for this check")
    if not math.isfinite(args.validation_seconds) or args.validation_seconds < 5:
        parser.error("--validation-seconds must be finite and >= 5")
    if args.light_intensity is not None and (not math.isfinite(args.light_intensity) or args.light_intensity < 0):
        parser.error("--light-intensity must be finite and nonnegative")
    with Path("/tmp/agipix-sim-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle()
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        command, env, _ = launcher._launch_plan(args, extra)
        config = dict(output=str(output), seconds=args.validation_seconds, night=args.night, tilt=args.tilt,
                      light_panels=args.light_panels, light_intensity=args.light_intensity)
        env.update(CANONICAL_SCRIPT=str(Path(__file__).resolve()), AGIPIX_LED_CHECK=json.dumps(config),
                   ROS_DOMAIN_ID="194")
        with (output / "simulator.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"LED_CHECK_EXIT={result.returncode} output={output}", flush=True)
        raise SystemExit(result.returncode)


def run_child(config):
    module = load_script("led_target", Path(__file__).with_name("8_agipix.py"))
    code = 0
    try:
        import numpy as np
        import omni.replicator.core as rep
        import rclpy
        from PIL import Image
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2
        from scipy.spatial.transform import Rotation
        from agipix_leds import BULB_RADIUS, mounts
        from agipix_splat_preview import make_preview

        args = module.parse_cli_args()
        app = module.AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
                              lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
                              gui=not args.headless, chase_cam=args.chase_cam, splat_mode=True, leds=args.leds)
        output = Path(config["output"])
        root = app.drone.prim_path + "/body/NavigationLights"
        assert bool(app.stage.GetPrimAtPath(root)) == args.leds
        if args.leds:
            for prim in Usd.PrimRange(app.stage.GetPrimAtPath(root)):
                assert not any(prim.HasAPI(api) for api in (UsdPhysics.CollisionAPI, UsdPhysics.MassAPI, UsdPhysics.RigidBodyAPI))
                if prim.IsA(UsdLux.SphereLight) and config["light_intensity"] is not None:
                    UsdLux.SphereLight(prim).CreateIntensityAttr(config["light_intensity"])
        if config["night"]:
            for prim in app.stage.Traverse():
                if prim.HasAPI(UsdLux.LightAPI) and not str(prim.GetPath()).startswith(root):
                    UsdLux.LightAPI(prim).CreateIntensityAttr(0)
            # A little ambient illumination keeps the airframe visible at night.
            ambient = UsdLux.DomeLight.Define(app.stage, "/World/LEDValidationAmbient")
            ambient.CreateIntensityAttr(0.2)

        if config["light_panels"]:
            material = UsdShade.Material.Define(app.stage, "/World/LEDPanelMaterial")
            shader = UsdShade.Shader.Define(app.stage, "/World/LEDPanelMaterial/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.5))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            for name, position, direction, _ in mounts():
                panel = UsdGeom.Cube.Define(app.stage, "/World/LEDPanel_" + name)
                panel.CreateSizeAttr(1)
                centre = np.array(position) + np.array(direction) * .25 + [0, 0, 1.2]
                panel.AddTranslateOp().Set(Gf.Vec3d(*centre))
                panel.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*direction)).GetQuat()))
                panel.AddScaleOp().Set(Gf.Vec3f(.16, .16, .005))
                UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(material)

        camera = UsdGeom.Camera.Define(app.stage, "/World/LEDInspectionCamera")
        camera.CreateFocalLengthAttr(35)
        camera.CreateHorizontalApertureAttr(36)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 1000))
        camera.CreateFStopAttr(0)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(.65, -.75, 1.7),
                                                           Gf.Vec3d(0, 0, 1.2), Gf.Vec3d(0, 0, 1)).GetInverse())
        product = rep.create.render_product(str(camera.GetPath()), (1280, 720))
        inspection = rep.annotators.get("rgb")
        inspection.attach(product)
        recorder = app._splat_recorder
        recorder.detach()
        recorder.annotators.append(rep.annotators.get("instance_id_segmentation", init_params={"colorize": False}))
        recorder.attach([s.render_product for s in app._splat_cameras])
        counts = {s.mount.name: 0 for s in app._splat_cameras}
        samples = {name: 0 for name in counts}
        images = {}
        panel_rgb = {}
        original_capture = recorder._capture
        def capture(data):
            original_capture(data)
            for key, values in data["renderProducts"].items():
                name = recorder._by_product[key].name
                seg = values["instance_id_segmentation"]
                mask = np.asarray(seg["data"])
                if mask.size != 1280 * 800:
                    continue
                samples[name] += 1
                for label, path in seg["idToLabels"].items():
                    if str(path).startswith(root):
                        counts[name] = max(counts[name], int(np.count_nonzero(mask == int(label))))
                    if str(path).startswith("/World/LEDPanel_"):
                        selected = mask.reshape(800, 1280) == int(label)
                        if selected.any():
                            rgb = np.asarray(values["rgb"]["data"])[..., :3]
                            panel_rgb[name + ":" + str(path)] = rgb[selected].mean(axis=0).tolist()
                images[name] = np.asarray(values["rgb"]["data"])[..., :3].copy()
        recorder._capture = capture
        lidar_rotation = Rotation.from_quat([*app.lidar_ori[1:], app.lidar_ori[0]]).as_matrix()
        led_positions = np.array([p for _, p, _, _ in mounts()])
        clouds, stamps = [], []
        near_counts, led_hits = [], []
        nearest = np.full(4, np.inf)
        def cloud(message):
            stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
            if stamp < 2e9:
                return
            assert message.header.frame_id == f"{app.vehicle_name}/lidar_link", message.header.frame_id
            fields = {f.name: f for f in message.fields}
            assert all(fields[name].datatype == 7 for name in ("x", "y", "z"))
            dtype = np.dtype(dict(names=["x", "y", "z"], formats=[">f4" if message.is_bigendian else "<f4"] * 3,
                                 offsets=[fields[n].offset for n in ("x", "y", "z")], itemsize=message.point_step))
            rows = np.ndarray((message.height, message.width), dtype=dtype, buffer=message.data,
                              strides=(message.row_step, message.point_step))
            xyz = np.stack([rows[n].ravel() for n in ("x", "y", "z")], axis=1)
            xyz = xyz[np.isfinite(xyz).all(axis=1)]
            body_points = xyz @ lidar_rotation.T + app.lidar_trans
            if len(xyz):
                distances = np.linalg.norm(body_points[:, None, :] - led_positions[None, :, :], axis=2)
                np.minimum(nearest, distances.min(axis=0), out=nearest)
                led_hits.append(int(np.count_nonzero(distances.min(axis=1) < BULB_RADIUS + .003)))
                near_counts.append(int(np.count_nonzero(np.linalg.norm(body_points, axis=1) < .4)))
            stamps.append(stamp)
            clouds.append(xyz)
        observer = rclpy.create_node("led_validation")
        subscription = observer.create_subscription(PointCloud2, f"/{app.vehicle_name}/livox/lidar", cloud, qos_profile_sensor_data)
        original_update = module.simulation_app.update
        inspection_saved = False
        viewport_capture = None
        light_debug = {}
        def update():
            nonlocal inspection_saved, viewport_capture
            t = module.SimulationManager.get_simulation_time()
            body = app.drone._body_rigid_prim
            if body is not None:
                angles = [20 * math.sin(t), 15 * math.sin(.7*t), 25 * math.sin(.4*t)] if config["tilt"] else [0, 0, 0]
                q = Rotation.from_euler("xyz", angles, degrees=True).as_quat()
                body.set_world_poses(positions=[[0, 0, 1.2]], orientations=[[q[3], *q[:3]]])
                body.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
            original_update()
            for _ in range(4):
                rclpy.spin_once(observer, timeout_sec=0)
            t = module.SimulationManager.get_simulation_time()
            if t > 4 and not inspection_saved:
                settings = module.carb.settings.get_settings()
                light_debug["skip_most_lights"] = settings.get("/rtx/scenedb/skipMostLights")
                for setting in ("/rtx/directLighting/enabled", "/rtx/rendermode"):
                    light_debug[setting] = settings.get(setting)
                for prim in app.stage.Traverse():
                    if str(prim.GetPath()).startswith(root) and prim.IsA(UsdLux.SphereLight):
                        light_debug[str(prim.GetPath())] = str(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                rgba = np.asarray(inspection.get_data())
                if rgba.size:
                    Image.fromarray(rgba[..., :3]).save(output / "inspection.png")
                    for name, rgb in images.items():
                        Image.fromarray(rgb).save(output / f"{name}.png")
                    Image.fromarray(make_preview(images)).save(output / "preview.png")
                    inspection_saved = True
                    if not args.headless:
                        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                        viewport_capture = capture_viewport_to_file(get_active_viewport(), str(output / "chase.png"))
            if recorder._last_stamp >= round(config["seconds"] * 1e9):
                app.stop_sim = True
        module.simulation_app.update = update
        app.run()
        np.savez_compressed(output / "lidar.npz", stamps=np.array(stamps), **{f"cloud_{i}": c for i, c in enumerate(clouds)})
        report = dict(leds=args.leds, night=config["night"], tilt=config["tilt"], led_pixels=counts, panel_rgb=panel_rgb,
                      light_debug=light_debug,
                      segmentation_frames=samples, lidar_messages=len(stamps), lidar_near_led_hits=sum(led_hits),
                      nearest_lidar_point_to_each_led_m=nearest.tolist(), near_body_points=sum(near_counts),
                      topics=sorted(app.node.get_topic_names_and_types()))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print("LED_VALIDATION=" + json.dumps(report), flush=True)
        assert inspection_saved and len(stamps) > 10
        assert all(n > 0 for n in samples.values()) and not any(counts.values()), counts
        assert not any(led_hits), report
        observer.destroy_node()
    except BaseException:
        code = 1
        traceback.print_exc()
    finally:
        module.restore_thread_settings()
        module.simulation_app.close(exit_code=code)


if __name__ == "__main__":
    run_child(json.loads(os.environ["AGIPIX_LED_CHECK"])) if "AGIPIX_LED_CHECK" in os.environ else launch()
