#!/usr/bin/env python3
"""Optional real PX4 hover/move/land check, aided by ideal simulation GT odometry.

python3 examples/validate_forest_snow_px4.py --output /tmp/snow-px4-01 [--gui]
Select --environment FOREST_SNOW, FOREST_MID_SNOW or FOREST_PINE_HILLS.

This validates PX4 control and the forest clearing, NOT autonomy localization.
The isolated test feeds MAVLink ODOMETRY from original Pegasus State. No pose
teleporting, direct motor commands, parameter changes or forced arming occur.
Only the exact PX4 child owned by this simulation may receive flight commands.
"""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import traceback

from bench_agipix_splat_rtf import ensure_idle, load_script
from validate_forest_snow import ENVIRONMENTS
from agipix_owned_process import OwnedProcessGroup

HERE = Path(__file__).resolve().parent


def json_finite(value):
    """Keep MAVLink's NaN sentinel on the wire, but use valid JSON null in reports."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def launch():
    launcher = load_script("snow_px4_launcher", HERE / "8_agipix_launch.py")
    parser = launcher._build_parser()
    parser.description = __doc__
    parser.set_defaults(environment="FOREST_SNOW")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-seconds", type=float, default=70)
    args, extra = parser.parse_known_args()
    if args.environment not in ENVIRONMENTS or args.vehicle_id != 0:
        parser.error("This bounded test requires a supported generated forest and vehicle id 0")
    if args.save_splat or args.splat_depth or args.splat_out or args.preflight or args.print_command:
        parser.error("Dataset saving and launcher dry-run flags are not part of this flight check")
    if not math.isfinite(args.validation_seconds) or not 45 <= args.validation_seconds <= 120:
        parser.error("--validation-seconds must be between 45 and 120")
    command, env, info = launcher._launch_plan(args, extra)
    errors = launcher._preflight_errors(info)
    if errors:
        parser.error("; ".join(errors))
    with Path("/tmp/agipix-sim-benchmark.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 14540))  # Refuse a port already owned by another SDK client.
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        config = dict(output=str(output), seconds=args.validation_seconds, launch=info,
            source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__), HERE / "8_agipix.py", HERE / "8_agipix_launch.py", Path(info["environment_asset"]))})
        (output / "config.json").write_text(json.dumps(config, indent=2))
        env.update(CANONICAL_SCRIPT=str(Path(__file__).resolve()), AGIPIX_SNOW_PX4=json.dumps(config),
                   ROS_DOMAIN_ID="196", MAVLINK20="1")
        with (output / "simulator.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            owned = OwnedProcessGroup(process)
            try:
                code = process.wait(timeout=max(300, args.validation_seconds * 6))
            finally:
                owned.stop()
        print(f"{args.environment}_PX4_EXIT={code} output={output}", flush=True)
        raise SystemExit(code)


def run_child(config):
    os.environ["MAVLINK20"] = "1"
    output = Path(config["output"])
    (output / "isaac_pid").write_text(str(os.getpid()))
    module = load_script("snow_px4_target", HERE / "8_agipix.py")
    app, link, code = None, None, 1
    report = dict(environment=config["launch"]["environment"],
                  scope="Real PX4 control with ideal GT MAVLink external odometry; autonomy localization not tested.",
                  success=False, events=[], commands=[], acknowledgements=[], samples=[], statuses=[])
    try:
        import numpy as np
        from pymavlink import mavutil
        args = module.parse_cli_args()
        app = module.AgipixApp(namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
            lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
            gui=not args.headless, chase_cam=args.chase_cam, splat_mode=args.splat_mode, leds=args.leds)
        backend = next(b for b in app._owned_backends if isinstance(b, module.PX4MavlinkBackend))
        link = mavutil.mavlink_connection("udpin:127.0.0.1:14540", source_system=247, source_component=191,
                                         dialect="common")
        link.port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        mav = mavutil.mavlink
        latest, received_at, captures = {}, {}, []
        phase, phase_time, settled_since = "WAIT_READY", 0.0, None
        owner = None
        target = np.array([0.0, 0.0, 2.0])
        arm_sent = mode_sent = land_sent = None
        last_sample = last_heartbeat = -1.0
        ever_armed = hover_ok = move_ok = False
        abort = None
        covariance = [0.0] * 21
        for index in (0, 6, 11, 15, 18, 20):
            covariance[index] = .0025
        def owned_process():
            tool = backend.px4_tool
            process = tool.px4_process if tool else None
            if process is None or process.poll() is not None:
                raise RuntimeError("The simulation has no live owned PX4 child")
            proc = Path("/proc") / str(process.pid)
            if (proc / "exe").resolve() != Path(args.px4_dir, "build/px4_sitl_default/bin/px4").resolve():
                raise RuntimeError("Owned PX4 executable identity mismatch")
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != os.getpid():
                raise RuntimeError("PX4 is not the direct child of this validation")
            pids = subprocess.run(["pgrep", "-x", "px4"], capture_output=True, text=True, check=False).stdout.split()
            if pids != [str(process.pid)]:
                raise RuntimeError("Another PX4 appeared; refusing all flight commands")
            return {"pid": process.pid, "start_ticks": int(fields[19]), "cwd": str((proc / "cwd").resolve())}
        def command(cmd, *values):
            if owned_process() != owner or link.clients != {("127.0.0.1", 14580)}:
                raise RuntimeError("Flight command destination is not the verified owned PX4")
            params = list(values) + [0.0] * (7 - len(values))
            link.mav.command_long_send(1, 1, cmd, 0, *params)
            report["commands"].append(dict(time=sim_time(), command=cmd, parameters=params))
        def sim_time():
            return module.SimulationManager.get_simulation_time()
        def enter(name, t):
            nonlocal phase, phase_time, settled_since
            phase, phase_time, settled_since = name, t, None
            report["events"].append(dict(time=t, phase=name))
            print(f"SNOW_PX4_PHASE={name} sim_time={t:.3f}", flush=True)
        def landed():
            return (latest.get("EXTENDED_SYS_STATE") is not None and
                    latest["EXTENDED_SYS_STATE"].landed_state == mav.MAV_LANDED_STATE_ON_GROUND)
        original_update = module.simulation_app.update
        def update():
            nonlocal owner, last_sample, last_heartbeat, arm_sent, mode_sent, land_sent
            nonlocal ever_armed, hover_ok, move_ok, settled_since, abort
            original_update()
            t = sim_time()
            if owner is None:
                owner = owned_process()
                report["owned_px4"] = owner
            for _ in range(300):
                message = link.recv_match(blocking=False)
                if message is None:
                    break
                if link.clients != {("127.0.0.1", 14580)}:
                    raise RuntimeError("Unexpected UDP source on the isolated PX4 link")
                if message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
                    continue
                kind = message.get_type()
                latest[kind], received_at[kind] = message, t
                if kind == "TIMESYNC" and message.tc1 == 0:
                    link.mav.timesync_send(round(t * 1e9), message.ts1)
                if kind in ("STATUSTEXT", "ESTIMATOR_STATUS", "SYS_STATUS"):
                    report["statuses"].append(dict(time=t, **message.to_dict()))
                if kind == "COMMAND_ACK":
                    report["acknowledgements"].append(dict(time=t, **message.to_dict()))
            state = app.drone._state
            position = np.asarray(state.position, dtype=float)
            if not np.isfinite(position).all():
                raise RuntimeError("Nonfinite original physics position")
            heartbeat = latest.get("HEARTBEAT")
            armed = bool(heartbeat and heartbeat.base_mode & mav.MAV_MODE_FLAG_SAFETY_ARMED)
            ever_armed |= armed
            if link.clients:
                if t - last_heartbeat >= 1:
                    link.mav.heartbeat_send(mav.MAV_TYPE_GCS, mav.MAV_AUTOPILOT_INVALID, 0, 0, mav.MAV_STATE_ACTIVE)
                    last_heartbeat = t
                q = state.get_attitude_ned_frd()
                p, velocity, omega = state.get_position_ned(), state.get_linear_velocity_ned(), state.get_angular_velocity_frd()
                link.mav.odometry_send(round(t * 1e6), mav.MAV_FRAME_LOCAL_NED, mav.MAV_FRAME_LOCAL_NED,
                    *p, [q[3], *q[:3]], *velocity, *omega, covariance, covariance, 0, mav.MAV_ESTIMATOR_TYPE_VISION, 100)
                if phase not in ("LAND", "DONE"):
                    link.mav.set_position_target_local_ned_send(round(t * 1000), 1, 1, mav.MAV_FRAME_LOCAL_NED,
                        0b100111111000, target[1], target[0], -target[2], 0, 0, 0, 0, 0, 0, math.pi / 2, 0)
            if ever_armed and (np.linalg.norm(position[:2]) > 2.5 or position[2] > 4 or position[2] < -.2):
                abort = abort or "Physical flight boundary exceeded (radius 2.5 m, height -0.2..4 m)"
                if phase != "LAND":
                    enter("LAND", t)
            if t > config["seconds"] - 12 and phase not in ("LAND", "DONE"):
                abort = abort or "Validation time limit reached before the hover/move sequence completed"
                enter("LAND", t)
            error = float(np.linalg.norm(position - target))
            speed = float(np.linalg.norm(state.linear_velocity))
            if phase == "WAIT_READY":
                estimator = latest.get("ESTIMATOR_STATUS")
                required = 1 | 2 | 4 | 8 | 32  # attitude, horizontal/vertical velocity, relative XY and Z position
                ready = estimator is not None and estimator.flags & required == required
                ready = ready and t - received_at.get("ESTIMATOR_STATUS", -100) < 3 and heartbeat is not None
                if t >= 12 and ready:
                    command(mav.MAV_CMD_DO_SET_MODE, mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 6, 0)
                    mode_sent = t
                    enter("ARM", t)
                elif t > 28:
                    abort = "PX4 estimator did not become ready with ideal GT odometry; no arm command sent"
                    app.stop_sim = True
            elif phase == "ARM":
                offboard = heartbeat is not None and ((heartbeat.custom_mode >> 16) & 255) == 6
                if offboard and arm_sent is None:
                    command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1, 0)  # normal checks; no force magic
                    arm_sent = t
                if armed:
                    enter("HOVER", t)
                elif t - phase_time > 8:
                    abort = "Offboard/normal arm rejected or unacknowledged"
                    app.stop_sim = True
            elif phase in ("HOVER", "MOVE"):
                if phase == "MOVE":
                    target[0] = min(1.0, (t - phase_time) * .25)
                arrived = error < .22 and speed < .25 and (phase == "HOVER" or target[0] >= 1)
                settled_since = (t if settled_since is None else settled_since) if arrived else None
                if settled_since is not None and t - settled_since >= 3:
                    if not args.headless:
                        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
                        captures.append(capture_viewport_to_file(get_active_viewport(), str(output / f"px4_{phase.lower()}.png")))
                    if phase == "HOVER":
                        hover_ok = True
                        enter("MOVE", t)
                    else:
                        move_ok = True
                        enter("LAND", t)
            elif phase == "LAND":
                if land_sent is None:
                    command(mav.MAV_CMD_NAV_LAND, 0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))
                    land_sent = t
                if landed() and not armed and ever_armed:
                    enter("DONE", t)
                    app.stop_sim = True
                elif t - phase_time > 12:
                    abort = abort or "PX4 landing did not finish within 12 simulated seconds"
                    app.stop_sim = True
            if t - last_sample >= .1:
                local = latest.get("LOCAL_POSITION_NED")
                report["samples"].append(dict(time=t, phase=phase, position_enu=position.tolist(), target_enu=target.tolist(),
                    error_m=error, speed_m_s=speed, armed=armed, estimator_position_ned=([local.x, local.y, local.z] if local else None)))
                last_sample = t
            if t >= config["seconds"]:
                abort = abort or "Hard simulation time limit"
                app.stop_sim = True
        module.simulation_app.update = update
        app.run()
        report.update(success=hover_ok and move_ok and phase == "DONE" and abort is None, abort_reason=abort,
            hover_passed=hover_ok, move_passed=move_ok, final_phase=phase, ever_armed=ever_armed,
            odometry_aiding="Original Pegasus world pose/velocity to MAVLink ODOMETRY NED, 30 Hz, covariance .0025; no autonomy localization.",
            limits=dict(radius_m=2.5, max_height_m=4, target_height_m=2, move_m=1), source_sha256=config["source_sha256"])
        code = 0 if report["success"] else 1
        print(f"{app.environment}_PX4_RESULT=" + json.dumps({k: report[k] for k in ("success", "abort_reason", "final_phase")}), flush=True)
    except BaseException as error:
        report["exception"] = str(error)
        traceback.print_exc()
    finally:
        (output / "report.json").write_text(json.dumps(json_finite(report), indent=2, allow_nan=False))
        if link is not None:
            link.close()
        if app is not None:
            for backend in app._owned_backends:
                backend.stop()
        module.restore_thread_settings()
        module.simulation_app.close(exit_code=code)


if __name__ == "__main__":
    run_child(json.loads(os.environ["AGIPIX_SNOW_PX4"])) if "AGIPIX_SNOW_PX4" in os.environ else launch()
