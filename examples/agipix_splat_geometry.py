"""Setting H calibration and dataset coordinate conversions (no Isaac imports).

Body/world: X forward/east, Y left/north, Z up. OpenCV camera: X right,
Y down, Z forward. USD/nerfstudio camera: X right, Y up, Z backward.
All transforms use column vectors and metres.
"""

from dataclasses import dataclass
import math

import numpy as np


WIDTH, HEIGHT = 1280, 800
FOCAL_PX = 577.5
CX, CY = 640.0, 400.0
RATE_HZ = 10.0
CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])
WORLD_AXES_FROM_CV = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
INTRINSICS = dict(camera_model="OPENCV_FISHEYE", fl_x=FOCAL_PX, fl_y=FOCAL_PX,
                  cx=CX, cy=CY, k1=0.0, k2=0.0, k3=0.0, k4=0.0, w=WIDTH, h=HEIGHT)


@dataclass(frozen=True)
class CameraMount:
    name: str
    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float = -10.0  # Handover elevation: negative means DOWN.


MOUNTS = (
    CameraMount("FL", (0.17, 0.18, 0.01157), 15.0),
    CameraMount("FR", (0.17, -0.18, 0.01157), -15.0),
    CameraMount("RL", (-0.17, 0.19, 0.01157), 110.0),
    CameraMount("RR", (-0.17, -0.19, 0.01157), -110.0),
)


def body_to_camera_cv(mount):
    """Return T_body_cam, mapping OpenCV camera points into the body frame.

    Isaac's world-axis intrinsic ZYX angles are (yaw, -elevation, 0).
    Thus the handover's -10 degree elevation is +10 degrees about Y.
    """
    yaw, pitch = math.radians(mount.yaw_degrees), math.radians(-mount.pitch_degrees)
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    rotation_zyx = np.array([[c * cp, -s, c * sp], [s * cp, c, s * sp], [-sp, 0.0, cp]])
    transform = np.eye(4)
    transform[:3, :3] = rotation_zyx @ WORLD_AXES_FROM_CV
    transform[:3, 3] = mount.position
    return transform


def opencv_to_nerfstudio(world_from_cv):
    """Flip camera Y/Z only; keep the ENU world frame and metric scale intact."""
    return np.asarray(world_from_cv) @ CV_FROM_GL


def rendered_camera_to_opencv(view_transform):
    """Replicator's USD row-vector world-to-view matrix -> column-vector T_world_cv."""
    return np.linalg.inv(np.asarray(view_transform, dtype=np.float64).reshape(4, 4).T) @ CV_FROM_GL


def project_equidistant(points_cv):
    """Project front-facing OpenCV points using KB4 with k1..k4 = 0."""
    points = np.asarray(points_cv, dtype=np.float64)
    radius = np.linalg.norm(points[..., :2], axis=-1)
    theta = np.arctan2(radius, points[..., 2])
    scale = np.divide(FOCAL_PX * theta, radius, out=np.zeros_like(radius), where=radius > 1e-15)
    return points[..., :2] * scale[..., None] + [CX, CY]


def module_corners_cv():
    """26 x 26 x 15 mm module, entirely 3..18 mm behind the lens plane."""
    return np.array([(x, y, z) for x in (-0.013, 0.013) for y in (-0.013, 0.013)
                     for z in (-0.018, -0.003)])
