"""GUI-only world-frame chase camera; no sensors, render products or ROS publishers.

Tune ChaseCameraSettings below to change the view. Angles are in degrees, lengths
in metres and time constants in simulation seconds. Only travel direction affects
yaw; vehicle roll, pitch and subsequent body yaw are deliberately never read.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ChaseCameraSettings:
    distance: float = 4.0
    height: float = 2.0
    pitch_degrees: float = 25.0  # Downward from the world horizontal.
    horizontal_fov_degrees: float = 95.0
    speed_threshold: float = 0.3
    velocity_time_constant: float = 2.5
    position_time_constant: float = 0.25
    yaw_rate_degrees: float = 60.0


# Indoor corridors and rooms: a 4 m boom puts the camera behind walls, so sit closer and lower.
INDOOR_SETTINGS = ChaseCameraSettings(distance=2.0, height=0.9, pitch_degrees=20.0)
OUTDOOR_ENVIRONMENTS = frozenset({"FOREST_FI"})


def settings_for_environment(environment):
    """Outdoor environments keep the wide default boom; everything else is indoor."""
    return ChaseCameraSettings() if environment in OUTDOOR_ENVIRONMENTS else INDOOR_SETTINGS


class ChaseCameraMotion:
    """Render-rate follow/filter math, usable without importing Isaac Sim."""

    def __init__(self, position, initial_yaw, simulation_time, settings=None):
        self.settings = settings or ChaseCameraSettings()
        self.initial_yaw = initial_yaw
        self.reset(position, simulation_time)

    def reset(self, position, simulation_time):
        self.yaw = self.initial_yaw
        self.velocity = (0.0, 0.0)
        self._previous_position = tuple(position)
        self._previous_time = simulation_time
        self.position = self._desired_position(position)

    def _desired_position(self, position):
        return (
            position[0] - self.settings.distance * math.cos(self.yaw),
            position[1] - self.settings.distance * math.sin(self.yaw),
            position[2] + self.settings.height,
        )

    @property
    def forward(self):
        pitch = math.radians(self.settings.pitch_degrees)
        return (math.cos(pitch) * math.cos(self.yaw),
                math.cos(pitch) * math.sin(self.yaw), -math.sin(pitch))

    def update(self, position, simulation_time):
        dt = simulation_time - self._previous_time
        if dt < 0.0:  # Timeline stop/restart: discard the old velocity history.
            self.reset(position, simulation_time)
            return True
        if dt == 0.0:  # Pausing must not accumulate wall-clock time or rotate the view.
            return False

        alpha = -math.expm1(-dt / self.settings.velocity_time_constant)
        velocity = tuple((position[i] - self._previous_position[i]) / dt for i in range(2))
        self.velocity = tuple(v + alpha * (sample - v) for v, sample in zip(self.velocity, velocity))
        self._previous_position = tuple(position)
        self._previous_time = simulation_time

        if math.hypot(*self.velocity) >= self.settings.speed_threshold:
            desired_yaw = math.atan2(self.velocity[1], self.velocity[0])
            # Shortest signed turn, including crossings of +/- pi.
            error = math.atan2(math.sin(desired_yaw - self.yaw), math.cos(desired_yaw - self.yaw))
            max_turn = math.radians(self.settings.yaw_rate_degrees) * dt
            self.yaw += max(-max_turn, min(max_turn, error))
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        alpha = -math.expm1(-dt / self.settings.position_time_constant)
        self.position = tuple(p + alpha * (target - p)
                              for p, target in zip(self.position, self._desired_position(position)))
        return True


class ChaseCamera:
    """Own a standalone USD camera and select it in the existing GUI viewport."""

    PRIM_PATH = "/World/ChaseCamera"

    def __init__(self, stage, vehicle_prim_path, simulation_time, settings=None):
        # Kept here so headless runs never import viewport/camera support.
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, Usd, UsdGeom

        self._gf = Gf
        settings = settings or ChaseCameraSettings()
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("Chase camera requires an active GUI viewport")

        # Pegasus' State is still zero-initialized before the first physics step.
        # Read the authored spawn pose once, including its actual initial heading.
        vehicle = UsdGeom.Xformable(stage.GetPrimAtPath(vehicle_prim_path))
        world = vehicle.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        heading = world.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        self.motion = ChaseCameraMotion(world.ExtractTranslation(), math.atan2(heading[1], heading[0]),
                                        simulation_time, settings)

        camera = UsdGeom.Camera.Define(stage, self.PRIM_PATH)
        camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        aperture = 20.955
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(aperture / (2.0 * math.tan(math.radians(settings.horizontal_fov_degrees) / 2.0)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10000.0))
        camera.CreateFStopAttr(0.0)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(True)  # Pose is in world coordinates, even if /World is transformed.
        self._transform = xform.AddTransformOp()
        self._write_pose()
        viewport.camera_path = self.PRIM_PATH

    def _write_pose(self):
        gf = self._gf
        eye = gf.Vec3d(*self.motion.position)
        # USD cameras look down local -Z with +Y up. A fixed world up vector
        # keeps the horizon level; aim along the travel yaw with constant pitch.
        target = eye + gf.Vec3d(*self.motion.forward)
        self._transform.Set(gf.Matrix4d().SetLookAt(eye, target, gf.Vec3d(0.0, 0.0, 1.0)).GetInverse())

    def update(self, position, simulation_time):
        if self.motion.update(position, simulation_time):
            self._write_pose()
