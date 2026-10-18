#!/usr/bin/env python3
"""Serial, unarmed RTF benchmark using the canonical Agipix launcher.

Example (16 runs, each 5 s warm-up + 60 s measured simulation time):
  python3 examples/bench_agipix_splat_rtf.py --output /tmp/splat-rtf-baseline

Reports native PhysX timing, Python capture subcomponents, GPU profiler timing,
GPU/CPU-thread telemetry, frame accounting and observed lidar/preview cadence.
The app-update residual is NOT a pure render measurement: it also includes
native Kit/ROS work and synchronization. GPU timing overlaps CPU wall time.
Use --profile for a separate diagnostic run; profiler overhead is not a baseline.
No arming or flight commands. Only this benchmark's drone pose is manipulated.
"""

import argparse
from collections import Counter, defaultdict
import cProfile
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import threading
import time
import traceback


HERE = Path(__file__).resolve().parent
MODES = {"off": (), "preview": ("--splat-mode",),
         "save": ("--splat-mode", "--save-splat"),
         "depth": ("--splat-mode", "--save-splat", "--splat-depth")}


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_idle():
    for command in (["pgrep", "-x", "px4"], ["pgrep", "-f", "[8]_agipix"], ["pgrep", "-x", "kit"]):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            raise RuntimeError(f"Existing simulator/PX4: {result.stdout.strip()}; leaving it untouched")


class SystemSampler(threading.Thread):
    """One Hz external telemetry; never uses the simulator's Python thread."""

    def __init__(self, pid, destination):
        super().__init__(daemon=True)
        self.pid, self.destination = pid, destination
        self.stop_event = threading.Event()

    def run(self):
        ticks = os.sysconf("SC_CLK_TCK")
        with self.destination.open("w") as stream:
            while not self.stop_event.is_set():
                started = time.monotonic()
                pid_file = self.destination.parent / "isaac_pid"
                try:
                    pid = int(pid_file.read_text())
                except (OSError, ValueError):
                    pid = self.pid  # Child startup may overlap this sample.
                record = {"wall": started, "pid": pid, "threads": {}}
                for path in Path(f"/proc/{pid}/task").glob("*/stat"):
                    try:
                        raw = path.read_text()
                        end = raw.rfind(")")
                        fields = raw[end + 2:].split()
                        record["threads"][path.parent.name] = {
                            "name": raw[raw.find("(") + 1:end],
                            "cpu_s": (int(fields[11]) + int(fields[12])) / ticks,
                        }
                    except (OSError, ValueError):
                        pass
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                    record["gpu"] = [[float(x.strip()) for x in line.split(",")]
                                     for line in result.stdout.splitlines()]
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    record["gpu_error"] = str(error)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                self.stop_event.wait(max(0.01, 1.0 - (time.monotonic() - started)))


def system_summary(path, start, end):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records = [r for r in records if start <= r["wall"] <= end]
    if not records:
        return {"samples": 0}
    gpu = [r["gpu"][0] for r in records if r.get("gpu")]
    threads = {}
    for r in records:
        for tid, value in r["threads"].items():
            if tid not in threads:
                threads[tid] = {"name": value["name"], "start_cpu": value["cpu_s"],
                                "start_wall": r["wall"]}
            threads[tid].update(end_cpu=value["cpu_s"], end_wall=r["wall"])
    cpu = []
    for tid, t in threads.items():
        span = t["end_wall"] - t["start_wall"]
        cpu.append({"tid": int(tid), "name": t["name"],
                    "cpu_s": t["end_cpu"] - t["start_cpu"],
                    "cpu_percent_one_core": 100 * (t["end_cpu"] - t["start_cpu"]) / span if span else 0})
    return {"samples": len(records),
            "gpu_utilization_mean_percent": sum(g[0] for g in gpu) / len(gpu) if gpu else None,
            "gpu_memory_peak_mib": max((g[2] for g in gpu), default=None),
            "gpu_power_mean_w": sum(g[3] for g in gpu) / len(gpu) if gpu else None,
            "threads": sorted(cpu, key=lambda t: t["cpu_s"], reverse=True)}


def launch():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; never overwritten.")
    parser.add_argument("--environments", nargs="+", choices=("FOREST_FI", "AKW_C"), default=["FOREST_FI", "AKW_C"])
    parser.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--profile", action="store_true", help="Diagnostic cProfile + native CPU trace (adds overhead).")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--leds", action="store_true", help="Include the optional navigation lights.")
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds < 1 or args.warmup < 0 or args.cpu_threads < 1:
        parser.error("repeats/cpu-threads must be positive, seconds >= 1 and warmup nonnegative")
    if any(not math.isfinite(v) or abs(v * 10 - round(v * 10)) > 1e-6 for v in (args.seconds, args.warmup)):
        parser.error("seconds and warmup must be finite multiples of 0.1 s")
    lock = Path("/tmp/agipix-sim-benchmark.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ensure_idle()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    launcher = load_script("agipix_bench_launcher", HERE / "8_agipix_launch.py")
    reports = []
    for environment in args.environments:
        for mode in args.modes:
            for repeat in range(1, args.repeats + 1):
                ensure_idle()
                case = output / f"{environment}-{mode}-{repeat}"
                case.mkdir()
                flags = ["--headless", "--environment", environment, *MODES[mode]]
                if args.leds:
                    flags.append("--leds")
                if mode in ("save", "depth"):
                    flags += ["--splat-out", str(case / "dataset")]
                parsed, extra = launcher._build_parser().parse_known_args(flags)
                command, env, _ = launcher._launch_plan(parsed, extra)
                config = dict(case=str(case), mode=mode, environment=environment, seconds=args.seconds,
                              warmup=args.warmup, profile=args.profile, repeat=repeat, leds=args.leds,
                              cpu_threads=args.cpu_threads,
                              physics_threads=int(env.get("AGIPIX_PHYSICS_THREADS", "0")))
                config["benchmark_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                config["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in [HERE / "8_agipix.py", HERE / "utils/drone_location_pub.py",
                                                     HERE / "agipix_leds.py", *HERE.glob("agipix_splat_*.py")]}
                (case / "config.json").write_text(json.dumps(config, indent=2))
                env.update(CANONICAL_SCRIPT=str(Path(__file__).resolve()), AGIPIX_RTF_CHILD=json.dumps(config),
                           ROS_DOMAIN_ID="193", AGIPIX_CPU_THREADS=str(args.cpu_threads))
                print(f"BENCH_START={case.name}", flush=True)
                with (case / "simulator.log").open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    sampler = SystemSampler(process.pid, case / "system.jsonl")
                    sampler.start()
                    try:
                        code = process.wait(timeout=max(180, (args.seconds + args.warmup) * 12))
                    except BaseException:
                        # Signal only the process group this benchmark created.
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=25)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=10)
                        raise
                    finally:
                        sampler.stop_event.set()
                        sampler.join(timeout=5)
                if code != 0 or not (case / "report.json").is_file():
                    raise RuntimeError(f"Benchmark failed ({code}): {case / 'simulator.log'}")
                report = json.loads((case / "report.json").read_text())
                report["system"] = system_summary(case / "system.jsonl", report["wall_start"], report["wall_end"])
                (case / "report.json").write_text(json.dumps(report, indent=2))
                reports.append(report)
                (output / "reports.json").write_text(json.dumps(reports, indent=2))
                print("BENCH_RESULT=" + json.dumps({k: report[k] for k in
                      ("environment", "mode", "repeat", "rtf_mean", "rtf_p5", "frames_per_camera", "lidar_hz")}), flush=True)


def run_child(config):
    (Path(config["case"]) / "isaac_pid").write_text(str(os.getpid()))
    module = load_script("agipix_benchmark_target", HERE / "8_agipix.py")
    exit_code = 0
    case = Path(config["case"])
    try:
        import numpy as np
        import rclpy
        import omni.physics.core
        import omni.hydratexture
        from carb.eventdispatcher import get_eventdispatcher
        from omni.hydra.engine.stats import HydraEngineStats
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image, PointCloud2

        args = module.parse_cli_args()
        warmup, stop_time = config["warmup"], config["warmup"] + config["seconds"]
        warm_ns, stop_ns = round(warmup * 1e9), round(stop_time * 1e9)
        samples = defaultdict(list)
        camera_stamps = defaultdict(list)
        render_frames = defaultdict(set)
        render_observers = []
        received = defaultdict(list)
        queue_depths, updates, gpu_ms = [], [], []
        callback_state = {"measuring": False}
        def measuring():
            return callback_state["measuring"]
        def timed(name, function):
            def wrapper(*a, **kw):
                start = time.perf_counter()
                try:
                    return function(*a, **kw)
                finally:
                    if measuring():
                        samples[name].append(time.perf_counter() - start)
            return wrapper

        if args.splat_mode:
            from agipix_splat_camera import SplatRecorder
            SplatRecorder._capture = timed("writer", SplatRecorder._capture)
        if args.save_splat:
            from agipix_splat_dataset import SplatDataset
            for method, name in (("submit", "dataset_submit"), ("_write_batch", "disk_batch"),
                                 ("_write_camera_files", "disk_camera"), ("_write_manifest", "manifest")):
                setattr(SplatDataset, method, timed(name, getattr(SplatDataset, method)))
        app = module.AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
                              lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
                              gui=not args.headless, chase_cam=args.chase_cam, splat_mode=args.splat_mode, leds=args.leds,
                              splat_depth=args.splat_depth, splat_out=args.splat_out, save_splat=args.save_splat)
        assert args.physics_dt == .004 and args.lidar_timestamps
        recorder = app._splat_recorder
        if recorder is not None:
            for sensor in app._splat_cameras:
                texture = sensor.render_product.hydra_texture
                def rendered(event, texture=texture, name=sensor.mount.name):
                    if measuring():
                        render_frames[name].add(texture.get_frame_info(event["result_handle"])["frame_number"])
                render_observers.append(get_eventdispatcher().observe_event(
                    observer_name=f"bench_splat_{sensor.mount.name}",
                    event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                    filter=texture.get_event_key(), on_event=rendered))
            recorder._panorama.stitch = timed("panorama", recorder._panorama.stitch)
            app.node.publish_splat_preview = timed("ros_publish", app.node.publish_splat_preview)
            app.node._encode_splat_jpeg = timed("jpeg", app.node._encode_splat_jpeg)
            for name, publisher in (("ros_raw_transport", app.node._splat_preview_publisher),
                                    ("ros_jpeg_transport", app.node._splat_preview_compressed_publisher)):
                publisher.publish = timed(name, publisher.publish)
            for name, message_type in (("ros_raw_data", Image), ("ros_jpeg_data", CompressedImage)):
                prop = message_type.data
                message_type.data = property(prop.fget, timed(name, prop.fset))
            capture = recorder._capture
            def counted_capture(data):
                previous = recorder._last_stamp
                capture(data)
                if recorder._last_stamp != previous:
                    for mount in recorder._by_product.values():
                        camera_stamps[mount.name].append(recorder._last_stamp)
                    if recorder._dataset is not None and measuring():
                        queue_depths.append(recorder._dataset._queue.qsize())
            recorder._capture = counted_capture

        observer = rclpy.create_node("agipix_rtf_observer")
        subscriptions = []
        def receive(name):
            def callback(message):
                stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                received[name].append((stamp, time.monotonic()))
                if name == "lidar" and not any(f.name == "timestamp" for f in message.fields):
                    callback_state["lidar_fields"] = [f.name for f in message.fields]
                if name in ("preview", "compressed"):
                    assert message.header.frame_id == f"{app.vehicle_name}/splat_panorama"
                if name == "preview":
                    assert message.encoding == "rgb8" and message.step == message.width * 3
                    callback_state["preview_resolution"] = [message.width, message.height]
                if name == "compressed":
                    assert message.format == "rgb8; jpeg compressed bgr8"
            return callback
        subscriptions.append(observer.create_subscription(PointCloud2, f"/{app.vehicle_name}/livox/lidar", receive("lidar"), qos_profile_sensor_data))
        if recorder is not None:
            subscriptions.append(observer.create_subscription(Image, f"/{app.vehicle_name}/splat/preview", receive("preview"), qos_profile_sensor_data))
            subscriptions.append(observer.create_subscription(CompressedImage, f"/{app.vehicle_name}/splat/preview/compressed", receive("compressed"), qos_profile_sensor_data))

        def physics_stats(stats):
            if measuring():
                samples["physics"].extend(s.ms / 1000 for s in stats if s.zone_name == "PhysX Update")
        physics_subscription = omni.physics.core.get_physics_benchmarks_interface().subscribe_profile_stats_events(physics_stats)
        hydra_stats = HydraEngineStats()
        profiler = cProfile.Profile() if config["profile"] else None
        native_profiler = None
        original_update = module.simulation_app.update
        wall_start = wall_end = None
        sim_start = None
        bins, bin_wall, bin_sim = [], None, None

        def bounded_update():
            nonlocal wall_start, wall_end, sim_start, bin_wall, bin_sim, native_profiler
            t = module.SimulationManager.get_simulation_time()
            body = app.drone._body_rigid_prim
            if body is not None:
                yaw = math.radians(15) * math.sin(.2 * t)
                body.set_world_poses(positions=[[.3 * math.sin(.3*t), .2 * math.sin(.2*t), 1.2]],
                                     orientations=[[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])
                body.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
            if wall_start is None and t >= warmup - 1e-8:
                wall_start = bin_wall = time.monotonic()
                sim_start = bin_sim = t
                callback_state["measuring"] = True
                if profiler:
                    profiler.enable()
                    try:
                        import carb.profiler
                        native_profiler = carb.profiler.acquire_profiler_interface(plugin_name="carb.profiler-mux.plugin")
                        callback_state["old_capture_mask"] = native_profiler.get_capture_mask()
                        native_profiler.set_capture_mask(0x0FFFFFFFFFFFFFFFF)
                        settings = module.carb.settings.get_settings()
                        settings.set_string("/plugins/carb.profiler-cpu.plugin/filePath", str(case / "native_trace.gz"))
                        settings.set_bool("/plugins/carb.profiler-cpu.plugin/compressProfile", True)
                        settings.set_bool("/plugins/carb.profiler-cpu.plugin/saveProfile", True)
                    except Exception as error:
                        callback_state["native_profiler_error"] = str(error)
                print("BENCH_WARMUP_COMPLETE", flush=True)
            start = time.perf_counter()
            original_update()
            duration = time.perf_counter() - start
            t = module.SimulationManager.get_simulation_time()
            if measuring():
                updates.append(duration)
                devices = hydra_stats.get_gpu_profiler_result()
                if devices:
                    gpu_ms.append(max(sum(n.get("duration", 0) for n in device if n.get("indent", 0) == 0)
                                      for device in devices))
                now = time.monotonic()
                if t - bin_sim >= 1 - 1e-8:
                    bins.append((t - bin_sim) / (now - bin_wall))
                    bin_sim, bin_wall = t, now
            # Drain the three 10 Hz observer streams without changing production callbacks.
            for _ in range(8):
                rclpy.spin_once(observer, timeout_sec=0)
            end_stamp = recorder._last_stamp / 1e9 if recorder is not None else t
            if end_stamp >= stop_time - 1e-8:
                wall_end = time.monotonic()
                app.stop_sim = True
                callback_state["measuring"] = False
                if profiler:
                    profiler.disable()
                if native_profiler:
                    module.carb.settings.get_settings().set_bool("/plugins/carb.profiler-cpu.plugin/saveProfile", False)
                    native_profiler.set_capture_mask(callback_state["old_capture_mask"])

        module.simulation_app.update = bounded_update
        app.run()
        close_finished = time.monotonic()
        for _ in range(20):
            rclpy.spin_once(observer, timeout_sec=.01)
        physics_subscription = None
        render_observers.clear()
        topics = sorted(observer.get_topic_names_and_types())
        observer.destroy_node()
        if profiler:
            profiler.dump_stats(str(case / "python.pstats"))
            with (case / "python_profile.txt").open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(80)
        def distribution(values):
            return {"calls": len(values), "total_s": float(sum(values)),
                    "mean_ms": float(np.mean(values) * 1000) if values else None,
                    "p95_ms": float(np.percentile(values, 95) * 1000) if values else None}
        def in_window(stamp):
            return warm_ns + 1000 < stamp <= stop_ns + 1000
        measured = {name: [s for s in stamps if in_window(s)] for name, stamps in camera_stamps.items()}
        observed = {name: [s for s, _ in stamps if in_window(s)] for name, stamps in received.items()}
        expected = round(config["seconds"] * 10)
        report = {**config, "wall_start": wall_start, "wall_end": wall_end,
                  "sim_start": sim_start, "rtf_mean": config["seconds"] / (wall_end - wall_start),
                  "rtf_including_close": config["seconds"] / (close_finished - wall_start),
                  "close_drain_s": close_finished - wall_end,
                  "rtf_p5": float(np.percentile(bins, 5)), "rtf_1s_windows": bins,
                  "app_update": distribution(updates),
                  "timings": {name: distribution(values) for name, values in samples.items()},
                  "gpu_profiler_ms_mean": float(np.mean(gpu_ms)) if gpu_ms else None,
                  "gpu_profiler_ms_p95": float(np.percentile(gpu_ms, 95)) if gpu_ms else None,
                  "frames_per_camera": {name: len(stamps) for name, stamps in measured.items()},
                  "frames_total_including_warmup": {name: len(stamps) for name, stamps in camera_stamps.items()},
                  "unique_render_frames": {name: len(frames) for name, frames in render_frames.items()},
                  "observed_messages": {name: len(stamps) for name, stamps in observed.items()},
                  "lidar_hz": len(observed.get("lidar", [])) / config["seconds"],
                  "disk_queue_peak": max(queue_depths, default=0),
                  "physics_dt": args.physics_dt, "callback_state": callback_state}
        report["topics"] = topics
        report["update_residual_s"] = sum(updates) - sum(samples["physics"]) - sum(samples["writer"])
        if recorder is not None and recorder._dataset is not None:
            manifest = json.loads((recorder._dataset.output_dir / "transforms.json").read_text())
            report["written_frames"] = dict(Counter(f["camera_id"] for f in manifest["frames"] if in_window(f["timestamp_ns"])))
            report["dataset_summary"] = json.loads((recorder._dataset.output_dir / "summary.json").read_text())
            # The writer records pressure immediately on enqueue, before the
            # capture callback can spend time stitching/publishing. Includes warm-up.
            report["disk_queue_peak_including_warmup"] = report["dataset_summary"]["queue_peak_batches"]
            assert report["written_frames"] == report["frames_per_camera"], report
        (case / "report.json").write_text(json.dumps(report, indent=2))
        assert abs(len(observed.get("lidar", [])) - expected) <= 1, report["observed_messages"]
        assert "lidar_fields" not in callback_state, callback_state
        for name, stamps in observed.items():
            assert len(set(stamps)) == len(stamps), f"Duplicate {name} stamps"
            assert all(abs(delta - 1e8) <= args.physics_dt * 1e9 + 2 for delta in np.diff(stamps)), f"{name} cadence"
        if recorder is not None:
            assert set(measured) == {"FL", "FR", "RL", "RR"}
            assert all(len(stamps) == expected for stamps in measured.values()), report["frames_per_camera"]
            assert all(len(set(stamps)) == expected for stamps in measured.values())
            assert all(abs(delta - 1e8) <= args.physics_dt * 1e9 + 2
                       for stamps in measured.values() for delta in np.diff(stamps)), "Camera cadence"
            assert len(observed["preview"]) == expected, report["observed_messages"]
            assert len(observed["compressed"]) == expected, report["observed_messages"]
        print("BENCH_CHILD_PASSED", flush=True)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        module.restore_thread_settings()
        module.simulation_app.close(exit_code=exit_code)


if __name__ == "__main__":
    if os.environ.get("AGIPIX_RTF_CHILD"):
        run_child(json.loads(os.environ["AGIPIX_RTF_CHILD"]))
    else:
        launch()
