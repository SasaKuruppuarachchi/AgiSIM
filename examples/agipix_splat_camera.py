"""Isaac Sim 6 setting-H cameras and synchronized capture, opt-in from AgipixApp."""

import math

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import RtxCamera
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from pxr import Gf, UsdGeom

from agipix_splat_preview import SplatPanorama
from agipix_splat_geometry import (
    CX, CY, FOCAL_PX, HEIGHT, MOUNTS, RATE_HZ, WIDTH,
    body_to_camera_cv, opencv_to_nerfstudio, rendered_camera_to_opencv,
)


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
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.001, 1000.0))
        camera.CreateFStopAttr(0.0)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.SetResetXformStack(False)
        local_gl = opencv_to_nerfstudio(body_to_camera_cv(self.mount))
        xform.AddTransformOp().Set(Gf.Matrix4d(*local_gl.T.ravel().tolist()))

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

    def __init__(self, sensors, node, environment, output_dir=None, depth=False, save=False):
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
        # Range -> optical-axis Z for the .npy depth product. Pixel centres use +0.5.
        if depth:
            y, x = np.mgrid[:HEIGHT, :WIDTH]
            self._depth_cosine = np.cos(np.hypot(x + 0.5 - CX, y + 0.5 - CY) / FOCAL_PX).astype(np.float32)
        try:
            self.attach([sensor.render_product for sensor in sensors])
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
        for product, values in data["renderProducts"].items():
            mount = self._by_product[product]
            images[mount.name] = np.asarray(values["rgb"]["data"])[..., :3]
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
        self._preview_node.publish_splat_preview(self._panorama.stitch(images), stamp)
        self._last_stamp = stamp

    def poll(self):
        if self._error is not None:
            raise RuntimeError("Splat capture failed") from self._error
        if self._dataset is not None:
            self._dataset.check_error()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.detach()
        finally:
            if self._dataset is not None:
                self._dataset.close()
        self.poll()
