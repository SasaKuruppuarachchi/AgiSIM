"""Isaac Sim 6 setting-H cameras and synchronized capture, opt-in from AgipixApp."""

import math
from pathlib import Path

import carb
import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import RtxCamera
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from pxr import Gf, UsdGeom, Vt

from agipix_splat_preview import SplatPanorama
from agipix_splat_geometry import (
    CX, CY, FOCAL_PX, HEIGHT, MOUNT_SET, MOUNTS, NEAR_CLIP_M, RATE_HZ, WIDTH,
    body_to_camera_cv, opencv_to_nerfstudio, rendered_camera_to_opencv,
)


HARDWARE_PARTS = Path(__file__).resolve().parents[1] / "assets/cad/camera_mounts/landing_gear_mounts/parts"


def _read_binary_stl(path):
    """Binary STL (mm) -> unique vertices (m) and triangle indices."""
    data = Path(path).read_bytes()
    count = int(np.frombuffer(data, "<u4", 1, 80)[0])
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data, record, count, 84)["vertices"].reshape(-1, 3) / 1000.0
    vertices, indices = np.unique(np.round(triangles, 7), axis=0, return_inverse=True)
    return vertices.astype(np.float32), indices.reshape(-1).astype(np.int32)


def add_camera_hardware(stage, body_path):
    """Render-only landing-gear mounts and OV9782-W modules under the vehicle body.

    The STLs are already in the drone body frame (mm). No collision, rigid body or mass,
    so flight dynamics are unchanged. Returns the created prim paths.
    """
    if MOUNT_SET != "landing_gear":
        return []
    parts = [(f"Mount_{m.name}", HARDWARE_PARTS / f"cam_mount_{m.name}_bodyframe.stl", (0.035, 0.035, 0.04))
             for m in MOUNTS]
    parts.append(("OV9782W_x4", HARDWARE_PARTS / "ov9782w_x4_bodyframe.stl", (0.10, 0.11, 0.12)))
    root = body_path + "/CameraHardware"
    UsdGeom.Xform.Define(stage, root)
    paths = []
    for name, stl, colour in parts:
        vertices, indices = _read_binary_stl(stl)
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*colour)])
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(0).tolist()), Gf.Vec3f(*vertices.max(0).tolist())])
        paths.append(mesh.GetPath().pathString)
    return paths


class SplatCamera(GraphicalSensor):
    """GraphicalSensor lifecycle without Pegasus' per-camera ROS forwarding."""

    def __init__(self, mount):
        super().__init__(sensor_type="SplatCamera", update_rate=RATE_HZ)
        self.mount = mount

    def initialize(self, vehicle):
        super().initialize(vehicle)
        self.prim_path = f"{vehicle.prim_path}/body/SplatCamera_{self.mount.name}"
        attributes = {"omni:lensdistortion:model": "opencvFisheye"}
        for name, value in dict(fx=FOCAL_PX, fy=FOCAL_PX, cx=CX, cy=CY,
                                k1=0.0, k2=0.0, k3=0.0, k4=0.0,
                                imageSize=Gf.Vec2i(WIDTH, HEIGHT)).items():
            attributes[f"omni:lensdistortion:opencvFisheye:{name}"] = value
        self.sensor = RtxCamera(self.prim_path, tick_rate=RATE_HZ,
                                schemas=["OmniLensDistortionOpenCvFisheyeAPI"], attributes=attributes)
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera(stage.GetPrimAtPath(self.prim_path))
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * HEIGHT / WIDTH)
        camera.CreateFocalLengthAttr(20.955 * FOCAL_PX / WIDTH)
        camera.CreateClippingRangeAttr(Gf.Vec2f(NEAR_CLIP_M, 1000.0))
        camera.CreateFStopAttr(0.0)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(False)
        local_gl = opencv_to_nerfstudio(body_to_camera_cv(self.mount))
        xform.AddTransformOp().Set(Gf.Matrix4d(*local_gl.T.ravel().tolist()))

        if MOUNT_SET == "setting_h":
            # Placeholder module; the landing-gear set uses the real meshes (add_camera_hardware).
            # USD/OpenGL +Z is behind the lens. No CollisionAPI/RigidBodyAPI/MassAPI.
            module = UsdGeom.Cube.Define(stage, self.prim_path + "/Module")
            module.CreateSizeAttr(1.0)
            module.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.12, 0.18)])
            module.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0105))
            module.AddScaleOp().Set(Gf.Vec3f(0.026, 0.026, 0.015))
        self.render_product = rep.create.render_product(self.prim_path, (WIDTH, HEIGHT),
                                                        name=f"Splat_{self.mount.name}")

    def update(self, state, dt):
        # The synchronized writer handles acquisition. Returning None prevents
        # ROS2Backend from creating any per-camera image/info/depth publishers.
        return None


class SplatRecorder(rep.Writer):
    """Publish a synchronized panorama; optionally record originals and poses.

    When saving, CameraParams comes from the SAME render products as RGB,
    including Kit's render delay. Recover the render-time body ground truth by
    removing the calibrated mount; validate all four agree and compose the fixed
    mounts for export. Reading drone._state would pair images with newer poses.
    Preview-only mode needs neither camera-pose annotations nor a disk worker.
    """

    _video = None
    _video_product = None

    def __init__(self, sensors, node, environment, output_dir=None, depth=False, save=False, video=None):
        if not save and (depth or output_dir is not None):
            raise ValueError("Depth and output directory require save=True")
        self.version = "1.0.0"
        self.data_structure = "renderProduct"
        self.annotators = [rep.annotators.get("rgb")]
        if save:
            self.annotators.append(rep.annotators.get("camera_params"))
        if depth:
            self.annotators.append(rep.annotators.get("distance_to_camera"))
        self._sensors = sensors
        self._preview_node = node
        self._depth = depth
        self._last_stamp = -1
        self._error = None
        self._closed = False
        self._sim_interface = SimulationManager._simulation_manager_interface
        self._panorama = SplatPanorama()
        self._dataset = None
        if save:
            from agipix_splat_dataset import SplatDataset, default_output_directory

            self._dataset = SplatDataset(output_dir or default_output_directory(environment), environment, depth)
        self._by_product = {sensor.render_product.path.rsplit("/", 1)[-1]: sensor.mount for sensor in sensors}
        # Optional chase-view video (agipix_video.VideoRecorder): one extra RGB product on the
        # same 10 Hz tick, delivered in the same write() as the four cameras.
        self._video = video
        self._video_product = video.render_product.path.rsplit("/", 1)[-1] if video is not None else None
        # Range -> optical-axis Z for the .npy depth product. Pixel centres use +0.5.
        if depth:
            y, x = np.mgrid[:HEIGHT, :WIDTH]
            self._depth_cosine = np.cos(np.hypot(x + 0.5 - CX, y + 0.5 - CY) / FOCAL_PX).astype(np.float32)
        try:
            self.attach([sensor.render_product for sensor in sensors]
                        + ([video.render_product] if video is not None else []))
        except Exception:
            if self._dataset is not None:
                self._dataset.close()
            raise
        if self._dataset is not None:
            print("AGIPIX_SPLAT_OUTPUT=" + str(self._dataset.output_dir), flush=True)

    def write_metadata(self):
        # Suppress the base writer's metadata I/O. SplatDataset owns it when saving.
        self._is_metadata_written = True

    def write(self, data):
        if self._closed or self._error is not None:
            return
        try:
            self._capture(data)
        except Exception as error:
            # Kit may log/swallow writer exceptions. The app's poll() re-raises
            # this on the main loop so capture cannot silently stop or corrupt.
            self._error = error
            if self._dataset is not None:
                self._dataset.capture_error = str(error)
            carb.log_error(f"Splat capture failed: {error}")

    def _capture(self, data):
        if not SimulationManager.is_simulating():
            return
        reference = tuple(map(int, data["reference_time"]))
        stamp_seconds = self._sim_interface.get_simulation_time_at_time(reference)
        if not math.isfinite(stamp_seconds) or stamp_seconds <= 0:
            return
        stamp = round(stamp_seconds * 1e9)
        if stamp == self._last_stamp:
            return
        if stamp < self._last_stamp and self._dataset is not None:
            raise RuntimeError("Timeline restarted during splat capture; start a new recording directory")

        images, depths, bodies = {}, {}, []
        video_frame = None
        for product, values in data["renderProducts"].items():
            if product == self._video_product:
                video_frame = np.asarray(values["rgb"]["data"])
                continue
            mount = self._by_product[product]
            # Pack once in native code. A strided RGBA slice makes both dataset
            # copying and panorama resizing repack every pixel much more slowly.
            images[mount.name] = cv2.cvtColor(np.asarray(values["rgb"]["data"]), cv2.COLOR_RGBA2RGB)
            if self._dataset is not None:
                params = values["camera_params"]
                world_cv = rendered_camera_to_opencv(params["cameraViewTransform"])
                bodies.append(world_cv @ np.linalg.inv(body_to_camera_cv(mount)))
            if self._depth:
                depths[mount.name] = np.asarray(values["distance_to_camera"]["data"]) * self._depth_cosine
        if set(images) != {m.name for m in MOUNTS}:
            raise RuntimeError("Incomplete synchronized splat camera batch")
        if any(image.shape != (HEIGHT, WIDTH, 3) for image in images.values()):
            raise RuntimeError("Invalid splat RGB resolution")
        if self._dataset is not None:
            if not all(np.allclose(body, bodies[0], atol=2e-4, rtol=0) for body in bodies):
                raise RuntimeError("Splat camera render poses disagree on the rigid body's pose")
            self._dataset.submit(stamp, reference, bodies[0], images, depths if self._depth else None)
        panorama = self._panorama.stitch(images)
        self._preview_node.publish_splat_preview(panorama, stamp)
        if self._video is not None and video_frame is not None and video_frame.size:
            self._video.submit(stamp, video_frame, images, panorama)
        self._last_stamp = stamp

    def poll(self):
        if self._error is not None:
            raise RuntimeError("Splat capture failed") from self._error
        if self._video is not None:
            self._video.check_error()
        if self._dataset is not None:
            self._dataset.check_error()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.detach()
        finally:
            try:
                if self._dataset is not None:
                    self._dataset.close()
            finally:
                if self._video is not None:
                    self._video.close()
        self.poll()
