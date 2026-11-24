#!/usr/bin/env python
"""
Agipix PX4 + ROS2 standalone example for Isaac Sim 6.0.
"""

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from isaacsim import SimulationApp
from scipy.spatial.transform import Rotation


# SimulationApp must be created immediately after import in Isaac Sim standalone scripts.
_bootstrap_parser = argparse.ArgumentParser(add_help=False)
_bootstrap_parser.add_argument("--headless", action="store_true")
_bootstrap_parser.add_argument("--environment", default="AKW_C")
_bootstrap_parser.add_argument("--splat-mode", action="store_true")
_bootstrap_args, _ = _bootstrap_parser.parse_known_args()
# Gaussian-splat environments are neither rendered nor seen by RTX lidar while motion BVH is on.
SPLAT_ENVIRONMENTS = ("HV22_FOREST",)
simulation_app = SimulationApp({
    "headless": _bootstrap_args.headless,
    "enable_motion_bvh": _bootstrap_args.environment not in SPLAT_ENVIRONMENTS,
    "limit_cpu_threads": int(os.environ.get("AGIPIX_CPU_THREADS", "8")),
    # Spatial FXAA avoids temporal DLSS trails in moving calibration/dataset images.
    **({"anti_aliasing": 2} if _bootstrap_args.splat_mode else {}),
})


import carb

# A single CPU-physics vehicle runs faster without PhysX worker dispatch.
# Restore the persistent preference before application shutdown.
_thread_settings = carb.settings.get_settings()
if _bootstrap_args.splat_mode:
    _thread_settings.set_bool("/rtx/post/motionblur/enabled", False)
_previous_physics_threads = _thread_settings.get("/persistent/physics/numThreads")
_environment_render_scope = None
_thread_settings.set_int(
    "/persistent/physics/numThreads", int(os.environ.get("AGIPIX_PHYSICS_THREADS", "0"))
)


def restore_thread_settings():
    if _environment_render_scope is not None:
        _environment_render_scope.restore()
    if _previous_physics_threads is not None:
        _thread_settings.set_int("/persistent/physics/numThreads", _previous_physics_threads)


import omni
import omni.timeline
import omni.usd
import isaacsim.storage.native as nucleus

from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
from isaacsim.core.experimental.utils.app import enable_extension
from isaacsim.sensors.experimental.physics import IMU, IMUSensor
from pxr import Sdf, Usd

# Ensure ROS2 bridge extension is enabled.
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import FLAT_ENVIRONMENTS, ROBOTS

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
sys.path.append(os.path.dirname(__file__))
from drone_location_pub import DroneLocationPublisher


DEFAULT_PHYSICS_HZ = 250.0
DEFAULT_PHYSICS_DT = 1.0 / DEFAULT_PHYSICS_HZ
DEFAULT_PUBLISH_HZ = 100.0
DEFAULT_RENDER_HZ = 30.0


class AgipixApp:
    def __init__(self, namespace="drone", vehicle_id=0, physics_dt=DEFAULT_PHYSICS_DT,
                 lidar_timestamps=False, environment="AKW_C", px4_dir=None,
                 gui=False, chase_cam=True, splat_mode=False, splat_depth=False, splat_out=None,
                 save_splat=False, leds=False, livox_imu=False, record_video=None, flow=False):
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be a finite, positive number of seconds")
        self.namespace = namespace
        self.id = vehicle_id
        self.vehicle_name = f"{self.namespace}{self.id}"
        self.lidar_timestamps = lidar_timestamps
        if environment not in FLAT_ENVIRONMENTS:
            raise ValueError("Unknown environment: " + environment)
        self.environment = environment
        self.px4_dir = px4_dir
        self.gui = gui
        self._chase_camera = None
        self.splat_mode = splat_mode
        self._splat_recorder = None
        if not splat_mode and (save_splat or splat_depth or splat_out is not None):
            raise ValueError("save_splat, splat_depth and splat_out require splat_mode")
        if not save_splat and (splat_depth or splat_out is not None):
            raise ValueError("splat_depth and splat_out require save_splat")
        if record_video is not None and not splat_mode:
            raise ValueError("record_video requires splat_mode")
        self._video = None

        self.timeline = omni.timeline.get_timeline_interface()
        self.assets_root_path = nucleus.get_assets_root_path()

        # Existing publication scheduling uses the physics frequency in Hz.
        self.phy_dt = 1.0 / physics_dt
        self.pub_dt = DEFAULT_PUBLISH_HZ
        self.rendering_dt = DEFAULT_RENDER_HZ

        self.lidar_trans = [0.0795, 0.0, 0.0323]
        self.lidar_ori = [0.9238795, 0.0, 0.3826834, 0.0]

        self.stop_sim = False
        self.sim_elapsed_time = None
        self.real_elapsed_time = None
        self.physics_stp_cnt = 0
        self.isaac_imu = None
        self._cb_physics = None
        self._owned_backends = []
        self._render_scope = None
        self._livox_imu_enabled = livox_imu
        self._livox_imu = None
        self._flow_enabled = flow
        self._flow = None
        self._px4_backend = None
        if flow:
            # PX4 reads PX4_PARAM_* at boot; export before the backend autolaunches PX4.
            from agipix_flow import apply_px4_param_env

            carb.log_info("ARK Flow MR PX4 params: " + str(apply_px4_param_env()))

        self._setup_world()
        self._spawn_vehicle()
        if leds:
            from agipix_leds import add_navigation_lights

            add_navigation_lights(omni.usd.get_context().get_stage(), self.drone.prim_path + "/body")
        self._setup_publishers_and_sensors()
        if self.splat_mode:
            from agipix_splat_camera import SplatRecorder, add_camera_hardware

            hardware = add_camera_hardware(self.stage, self.drone.prim_path + "/body")
            if hardware:
                carb.log_info("Camera hardware meshes: " + ", ".join(hardware))
            if record_video is not None:
                from agipix_chase_camera import settings_for_environment
                from agipix_video import VideoRecorder

                state = self.drone._state
                self._video = VideoRecorder(
                    self.stage, self.drone.prim_path, SimulationManager.get_simulation_time(),
                    settings_for_environment(self.environment), record_video, self.environment,
                    state_fn=lambda: (state.position, state.linear_velocity),
                )
            self.node.enable_splat_preview()
            self._splat_recorder = SplatRecorder(
                self._splat_cameras, self.node, self.environment, splat_out, splat_depth, save=save_splat,
                video=self._video,
            )
        if self.gui and chase_cam:
            from agipix_chase_camera import ChaseCamera, settings_for_environment

            self._chase_camera = ChaseCamera(
                self.stage, self.drone.prim_path, SimulationManager.get_simulation_time(),
                settings_for_environment(self.environment),
            )

        if self._render_scope is not None:
            self._render_scope.fix_camera_exposure(self.stage)

        # Register physics step callback with SimulationManager
        self._cb_physics = SimulationManager.register_callback(
            self.physics_step, SimulationEvent.PHYSICS_POST_STEP
        )

    def _setup_world(self):
        self.pg = PegasusInterface()
        self.pg.set_world_settings(
            physics_dt=1.0 / self.phy_dt,
            stage_units_in_meters=1.0,
            rendering_dt=1.0 / self.rendering_dt,
        )
        self.pg.initialize_world()
        # Compose the scene before vehicle/RTX sensor creation. The asynchronous
        # GUI loader can otherwise race sensor initialization in standalone runs.
        self.pg.load_asset(FLAT_ENVIRONMENTS[self.environment], "/World/layout")
        # This environment's renderer choices are reversible and have no frame
        # callback. The common shutdown hook also covers construction failures.
        global _environment_render_scope
        if _environment_render_scope is not None:
            _environment_render_scope.restore()
        if self.environment == "FOREST_PINE_HILLS":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
            from forest_pine_hills.runtime import ScopedRenderSettings
            self._render_scope = ScopedRenderSettings(self.environment)
            _environment_render_scope = self._render_scope
            self._render_scope.apply()
        print("AGIPIX_ENVIRONMENT_LOADED=" + self.environment, flush=True)

    def _build_vehicle_config(self):
        config_multirotor = MultirotorConfig()

        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": self.id,
                "px4_autolaunch": True,
                "px4_dir": self.px4_dir or self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
                "simulation_time_provider": SimulationManager.get_simulation_time,
                "simulation_fault_handler": self.stop_on_backend_fault,
            }
        )

        self._px4_backend = PX4MavlinkBackend(mavlink_config)
        config_multirotor.backends = [
            self._px4_backend,
            ROS2Backend(
                vehicle_id=self.id,
                config={
                    "namespace": self.namespace,
                    "pub_sensors": False,
                    "pub_graphical_sensors": True,
                    "pub_lidar_laserscan": False,
                    "pub_lidar_timestamps": self.lidar_timestamps,
                    "pub_state": True,
                    "sub_control": False,
                },
            ),
        ]
        self._owned_backends = list(config_multirotor.backends)

        if self.splat_mode:
            from agipix_splat_camera import SplatCamera
            from agipix_splat_geometry import MOUNTS

            self._splat_cameras = [SplatCamera(mount) for mount in MOUNTS]
        config_multirotor.graphical_sensors = list(self._splat_cameras) if self.splat_mode else [
            MonocularCamera(
                "RealSense_Camera",
                config={
                    "depth": True,
                    "f_stop": 0.0,
                    "focus_distance": 0.6,
                    "position": np.array([0.13, 0.0, -0.022]),
                    "orientation": np.array([180.0, -180.0, 0.0]),
                    "intrinsics": np.array([
                        [606.3120727539062, 0.0, 314.6913146972656],
                        [0.0, 605.92626953125, 252.1909942626953],
                        [0.0, 0.0, 1.0],
                    ]),
                    "resolution": (640, 400),
                    "frequency": 30,
                },
            ),
        ]
        config_multirotor.graphical_sensors += [
            Lidar(
                "livox",
                config={
                    "position": np.array(self.lidar_trans),
                    "orientation": np.array(self.lidar_ori),
                    "sensor_configuration": "Mid_360",
                    "frame_id": "lidar_link",
                    "show_render": False,
                    "frequency": 10,
                    # Per-point time lives in basic GMO; FULL auxiliary channels are unnecessary.
                    "aux_output_level": "NONE" if self.lidar_timestamps else None,
                },
            ),
        ]

        return config_multirotor

    def _spawn_vehicle(self):
        config_multirotor = self._build_vehicle_config()

        self.drone = Multirotor(
            f"/World/{self.vehicle_name}",
            ROBOTS["Agipix v2"],
            self.id,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self._resolve_vehicle_material_assets()

    def _resolve_vehicle_material_assets(self):
        """Resolve the bundled Agipix model's legacy materials against this installation."""
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.drone.prim_path)):
            for attribute in prim.GetAttributes():
                if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue
                # Read authored values without resolving obsolete URLs first.
                value = next(
                    (spec.default for spec in attribute.GetPropertyStack() if spec.HasInfo("default")),
                    None,
                )
                if not isinstance(value, Sdf.AssetPath):
                    continue
                path = value.path
                if "/Isaac/5.1/Isaac/Materials/" in path:
                    relative = "Isaac/Materials/" + path.split("/Isaac/5.1/Isaac/Materials/", 1)[1]
                elif path.startswith("omniverse://localhost/NVIDIA/Materials/"):
                    relative = "Isaac/Materials/" + path.split("/NVIDIA/Materials/", 1)[1]
                else:
                    continue
                attribute.Set(Sdf.AssetPath(self.assets_root_path.rstrip("/") + "/" + relative))

    def _setup_publishers_and_sensors(self):
        self.node = DroneLocationPublisher(
            namespace=self.namespace,
            vehicle_id=self.id,
            lidar_trans=self.lidar_trans,
            lidar_ori=self.lidar_ori,
        )

        simulation_app.update()
        self.create_imu_sensor()
        if self._livox_imu_enabled:
            self.create_livox_imu()
        if self._flow_enabled:
            self.create_flow_sensor()

        self.stage = omni.usd.get_context().get_stage()
        self.drone_prim = self.stage.GetPrimAtPath(self.drone._stage_prefix + "/body")

    def create_livox_imu(self):
        """Emulated Livox Mid-360 internal IMU (agipix_livox_imu.py): computed analytically from the
        vehicle state every physics step at the manual's position inside the Mid-360, in the lidar's
        axes, published at 200 Hz of sim time. No Isaac IMU prim (its IMUSensor drops omega x v)."""
        import dataclasses
        from agipix_livox_imu import LivoxImuEmulator, LivoxImuSettings, imu_position_in_body

        # AGIPIX_LIVOX_IMU_VIBRATION: scale of the coloured prop vibration fitted to the real in-flight
        # spectrum above 8 Hz (default 1.0 = real; 0 = clean sensor).
        settings = dataclasses.replace(LivoxImuSettings(),
                                       vibration_scale=float(os.environ.get("AGIPIX_LIVOX_IMU_VIBRATION", "1.0")))
        position = imu_position_in_body(self.lidar_trans, self.lidar_ori, settings.position_in_lidar)
        self.node.enable_livox_imu()
        self._livox_imu = LivoxImuEmulator(self.node.publish_livox_imu, self.lidar_ori, position, settings)
        carb.log_info("Livox IMU emulation at body " + str(position.round(4).tolist()))

    def create_flow_sensor(self):
        """Emulated ARK Flow MR (agipix_flow.py): PAA3905 flow + AFBR-S50 ToF from the vehicle state and
        PhysX ray casts that skip the vehicle's own colliders, sent to PX4 on the HIL link as
        HIL_OPTICAL_FLOW / DISTANCE_SENSOR. AGIPIX_FLOW_LOG=<prefix> writes GT/sensor CSVs."""
        from omni.physx import get_physx_scene_query_interface
        from agipix_flow import FlowRangeEmulator, settings_from_env

        query = get_physx_scene_query_interface()
        own = self.drone.prim_path + "/"

        def raycast(origin, direction, max_distance):
            origin = carb.Float3(*map(float, origin))
            direction = carb.Float3(*map(float, direction))
            hit = query.raycast_closest(origin, direction, float(max_distance), False)
            if not hit or not hit.get("hit"):
                return None
            if not (hit.get("rigidBody", "").startswith(own) or hit.get("collision", "").startswith(own)):
                return hit["distance"]
            # The closest hit is this vehicle: walk all hits (slower, Python callback per hit).
            counts = self._flow.counts
            counts["ray_fallback"] = counts.get("ray_fallback", 0) + 1
            best = [None]

            def report(h):
                if not (h.rigid_body.startswith(own) or h.collision.startswith(own)):
                    if best[0] is None or h.distance < best[0]:
                        best[0] = h.distance
                return True

            query.raycast_all(origin, direction, float(max_distance), report, False)
            return best[0]

        backend = self._px4_backend

        def send_flow(m):
            time_us = int(SimulationManager.get_simulation_time() * 1e6)
            backend.send_mavlink(lambda mav: mav.hil_optical_flow_send(
                time_us, 0, m["integration_time_us"], m["integrated_x"], m["integrated_y"],
                m["integrated_xgyro"], m["integrated_ygyro"], m["integrated_zgyro"], 2500, m["quality"],
                0, m["distance"]))

        def send_range(m):
            time_ms = int(SimulationManager.get_simulation_time() * 1e3) & 0xFFFFFFFF
            backend.send_mavlink(lambda mav: mav.distance_sensor_send(
                time_ms, m["min_distance"], m["max_distance"], m["current_distance"], 0, 0, 25,
                m["covariance"], m["fov"], m["fov"], [0.0, 0.0, 0.0, 0.0], m["signal_quality"]))

        settings = settings_from_env()
        self._flow = FlowRangeEmulator(raycast, send_flow, send_range, settings,
                                       log_path=os.environ.get("AGIPIX_FLOW_LOG") or None)
        carb.log_warn("ARK Flow MR emulation on: flow at body " + str(settings.flow_position)
                      + ", range at " + str(settings.range_position) + ", texture " + str(settings.texture)
                      + ", featureless " + str(settings.featureless_boxes))

    def create_imu_sensor(self):
        self.isaac_imu = IMUSensor(
            IMU.create(
                self.drone.prim_path + "/body/Imu",
                translations=[[0.0, 0.0, 0.0]],
                orientations=[[0.0, -1.0, 0.0, 0.0]],
                linear_acceleration_filter_size=10,
                angular_velocity_filter_size=10,
                orientation_filter_size=10,
            )
        )
        carb.log_info("IMU sensor created successfully.")

    def physics_step(self, dt: float, context=None):
        current_sim_time = SimulationManager.get_simulation_time()
        current_time = time.time()
        self.node.publish_clock(current_sim_time)
        if self._livox_imu is not None:
            # Props spinning (any rotor thrust) gates the optional in-flight vibration noise.
            spinning = sum(abs(float(f)) for f in self.drone.forces) > 0.5
            st = self.drone._state
            self._livox_imu.step(current_sim_time, Rotation.from_quat(st.attitude).as_matrix(),
                                 st.linear_velocity, st.angular_velocity, armed=spinning)
        if self._flow is not None:
            st = self.drone._state
            started = time.perf_counter()
            self._flow.step(current_sim_time, Rotation.from_quat(st.attitude).as_matrix(), st.position,
                            st.linear_velocity, st.angular_velocity)
            self._flow.cost_s += time.perf_counter() - started

        if self.physics_stp_cnt == 0:
            if self.sim_elapsed_time is None:
                self.sim_elapsed_time = current_sim_time
                self.real_elapsed_time = current_time
            else:
                sim_dt = current_sim_time - self.sim_elapsed_time
                real_dt = current_time - self.real_elapsed_time
                self.sim_elapsed_time = current_sim_time
                self.real_elapsed_time = current_time

                self.node.publish_rtf(real_dt, sim_dt)

                state = self.drone._state
                self.node.publish_gt(state, current_sim_time)
                self.node.publish_gt_imu(current_sim_time, state)
                self.node.publish_gt_forces(self.drone.forces, self.drone.rolling_moment)

                if self.isaac_imu is not None:
                    self.node.publish_self_imu(self.isaac_imu.get_data())

        if self.physics_stp_cnt >= int(self.phy_dt / self.pub_dt) - 1:
            self.physics_stp_cnt = 0
        else:
            self.physics_stp_cnt += 1

    def stop_on_backend_fault(self, reason):
        self.stop_sim = True
        self.timeline.pause()
        carb.log_error('Stopping owned Agipix simulation: ' + reason)

    def run(self):
        try:
            self.timeline.play()
            if self._splat_recorder is not None:
                while simulation_app.is_running() and not self.stop_sim:
                    simulation_app.update()
                    self._splat_recorder.poll()
                    sim_time = SimulationManager.get_simulation_time()
                    if self._chase_camera is not None:
                        self._chase_camera.update(self.drone._state.position, sim_time)
                    if self._video is not None:
                        self._video.update(self.drone._state.position, sim_time)
                    self.node.publish_timeline_state(self.timeline, sim_time)
            elif self._chase_camera is None:
                # Preserve the headless loop: no camera import, prim or per-step work.
                while simulation_app.is_running() and not self.stop_sim:
                    simulation_app.update()
                    self.node.publish_timeline_state(self.timeline, SimulationManager.get_simulation_time())
            else:
                while simulation_app.is_running() and not self.stop_sim:
                    simulation_app.update()
                    sim_time = SimulationManager.get_simulation_time()
                    self._chase_camera.update(self.drone._state.position, sim_time)
                    self.node.publish_timeline_state(self.timeline, sim_time)
        finally:
            cleanup_errors = []
            if self._render_scope is not None:
                try:
                    self._render_scope.restore()
                except Exception as error:
                    cleanup_errors.append(error)
            if self._splat_recorder is not None:
                try:
                    self._splat_recorder.close()
                except Exception as error:
                    cleanup_errors.append(error)
            try:
                self.timeline.stop()
            except Exception as error:
                cleanup_errors.append(error)
            # Timeline callbacks may fail or be deferred. Explicitly close every
            # owned backend independently before releasing vehicle physics views.
            for backend in self._owned_backends:
                try:
                    backend.stop()
                except Exception as error:
                    cleanup_errors.append(error)
            try:
                self.drone.stop()
            except Exception as error:
                cleanup_errors.append(error)
            if self._flow is not None:
                try:
                    self._flow.close()
                    steps = max(1, self._flow.steps)
                    carb.log_warn("ARK Flow MR emulation: " + str(self._flow.counts) + " discarded "
                                  + str(self._flow.discard_reasons) + ", mean cost per physics step "
                                  + f"{1e6 * self._flow.cost_s / steps:.0f} us over {steps} steps")
                except Exception as error:
                    cleanup_errors.append(error)
            if self._cb_physics is not None:
                try:
                    SimulationManager.deregister_callback(self._cb_physics)
                except Exception as error:
                    cleanup_errors.append(error)
            carb.log_warn("Agipix Simulation App is closing.")
            if cleanup_errors:
                raise RuntimeError("Agipix cleanup failed") from cleanup_errors[0]


def positive_physics_dt(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("physics dt must be a finite, positive number of seconds")
    return value


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run the Agipix standalone simulation example.")
    parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window.")
    parser.add_argument("--leds", action="store_true",
                        help="Enable white front/red rear navigation lights on the guard corners (default: off).")
    parser.add_argument("--livox-imu", action="store_true",
                        help="Publish an emulated Livox Mid-360 IMU on /<ns>/livox/imu (200 Hz, lidar axes, g units; default: off).")
    parser.add_argument("--flow", action="store_true",
                        help="Emulate the ARK Flow MR (PAA3905 flow + AFBR-S50 ToF) into PX4 over the HIL link and "
                             "export its EKF2/SENS_FLOW parameter set (agipix_flow.py; default: off).")
    parser.add_argument("--chase-cam", action=argparse.BooleanOptionalAction, default=True,
                        help="Follow travel direction in the GUI viewport (default: on; ignored headless). "
                             "Tune view/filter defaults in agipix_chase_camera.ChaseCameraSettings.")
    parser.add_argument("--splat-mode", action="store_true",
                        help="Replace RealSense with the four OV9782-W fisheyes (AGIPIX_CAMERA_MOUNTS: landing_gear "
                             "(default) or setting_h) and a stitched 10 Hz panorama preview.")
    parser.add_argument("--save-splat", action="store_true",
                        help="Write the full-resolution splat dataset to disk (default: off; requires --splat-mode).")
    parser.add_argument("--splat-depth", action="store_true",
                        help="Also save float32 optical-axis depth in metres (.npy); requires --save-splat.")
    parser.add_argument("--splat-out", type=Path,
                        help="New dataset directory; requires --save-splat "
                             "(default: ~/agipix_splat/<environment>_<timestamp>).")
    parser.add_argument("--record-video", type=Path, metavar="DIR",
                        help="Record a 10 Hz chase-camera view plus the four onboard cameras into a new DIR "
                             "(requires --splat-mode; compose with agipix_video_compose.py).")
    parser.add_argument("--environment", choices=tuple(FLAT_ENVIRONMENTS), default="AKW_C",
                        help="Installed environment to load (default: AKW_C).")
    parser.add_argument("--px4-dir", help="PX4 source/build directory (defaults to Pegasus configuration).")
    parser.add_argument("--lidar-timestamps", action="store_true",
                        help="Publish Isaac Sim 6 per-point timestamps; requires a timestamp-aware lidar parser.")
    parser.add_argument(
        "--namespace",
        default="drone",
        help="Base namespace prefix used for ROS topics and frame IDs.",
    )
    parser.add_argument(
        "--physics-dt",
        type=positive_physics_dt,
        default=DEFAULT_PHYSICS_DT,
        help="Physics timestep in seconds (default: 0.004, or 250 Hz).",
    )
    parser.add_argument(
        "--id",
        dest="vehicle_id",
        type=int,
        default=0,
        help="Vehicle identifier appended to the namespace.",
    )
    args, _ = parser.parse_known_args()
    if not args.splat_mode and (args.save_splat or args.splat_depth or args.splat_out is not None):
        parser.error("--save-splat, --splat-depth and --splat-out require --splat-mode")
    if not args.save_splat and (args.splat_depth or args.splat_out is not None):
        parser.error("--splat-depth and --splat-out require --save-splat")
    if args.record_video is not None and not args.splat_mode:
        parser.error("--record-video requires --splat-mode")
    return args


def main():
    try:
        args = parse_cli_args()
        app = AgipixApp(
            namespace=args.namespace, vehicle_id=args.vehicle_id, physics_dt=args.physics_dt,
            lidar_timestamps=args.lidar_timestamps, environment=args.environment, px4_dir=args.px4_dir,
            gui=not args.headless, chase_cam=args.chase_cam,
            splat_mode=args.splat_mode, splat_depth=args.splat_depth, splat_out=args.splat_out,
            save_splat=args.save_splat, leds=args.leds, livox_imu=args.livox_imu,
            record_video=args.record_video, flow=args.flow,
        )
        app.run()
    finally:
        try:
            restore_thread_settings()
        finally:
            if _bootstrap_args.splat_mode and sys.exc_info()[0] is not None:
                # SimulationApp closes the process; preserve capture/disk failures.
                import traceback
                traceback.print_exc()
                simulation_app.close(exit_code=1)
            else:
                simulation_app.close()


if __name__ == "__main__":
    main()
