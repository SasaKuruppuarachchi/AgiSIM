"""Headless flight video for splat-mode runs (opt-in: 8_agipix.py --splat-mode --record-video DIR).

A world-frame chase camera (agipix_chase_camera motion and settings) is rendered as an RTX
camera on the same 10 Hz tick as the four fisheyes. SplatRecorder hands each synchronized
batch here, so every stored stamp has the chase view, the four onboard images and the
stitched panorama. JPEG encoding runs on worker threads; the render thread only converts
and resizes. Compose the MP4 afterwards with agipix_video_compose.py.

Output layout: chase/<stamp>.jpg (1280x720), cams/<CAM>/<stamp>.jpg (640x400),
pano/<stamp>.jpg (1024x256), meta.json (settings, written at start) and frames.csv (stamp,
GT position/velocity; one flushed line per batch, so it survives a hard kill of Kit).
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np

from agipix_splat_geometry import MOUNT_SET, MOUNTS, RATE_HZ

VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720
TILE_SIZE = (640, 400)
JPEG_QUALITY = 92
MAX_PENDING = 48  # Batches in flight before the render thread waits for disk.


class VideoRecorder:
    PRIM_PATH = "/World/VideoChaseCamera"

    def __init__(self, stage, vehicle_prim_path, simulation_time, settings, output_dir, environment,
                 state_fn=None):
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import RtxCamera

        from agipix_chase_camera import ChaseCamera

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for sub in ["chase", "pano"] + [f"cams/{m.name}" for m in MOUNTS]:
            (self.output_dir / sub).mkdir(parents=True)
        self.sensor = RtxCamera(self.PRIM_PATH, tick_rate=RATE_HZ)
        self.chase = ChaseCamera(stage, vehicle_prim_path, simulation_time, settings,
                                 prim_path=self.PRIM_PATH, bind_viewport=False)
        from pxr import UsdGeom

        camera = UsdGeom.Camera(stage.GetPrimAtPath(self.PRIM_PATH))
        camera.CreateVerticalApertureAttr(20.955 * VIDEO_HEIGHT / VIDEO_WIDTH)
        self.render_product = rep.create.render_product(self.PRIM_PATH, (VIDEO_WIDTH, VIDEO_HEIGHT),
                                                        name="Video_Chase")
        self._state_fn = state_fn
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="agipix_video")
        self._pending = deque()
        self._closed = False
        self._meta = dict(environment=environment, mount_set=MOUNT_SET, rate_hz=RATE_HZ,
                          chase=dict(vars(settings)) if settings is not None else None,
                          mounts={m.name: dict(position_m=list(m.position), yaw_deg=m.yaw_degrees,
                                               pitch_deg=m.pitch_degrees) for m in MOUNTS})
        (self.output_dir / "meta.json").write_text(json.dumps(self._meta, indent=1))
        self._csv = open(self.output_dir / "frames.csv", "w", buffering=1)
        self._csv.write("stamp_ns,x,y,z,vx,vy,vz\n")
        print("AGIPIX_VIDEO_OUTPUT=" + str(self.output_dir), flush=True)

    def update(self, position, simulation_time):
        self.chase.update(position, simulation_time)

    def submit(self, stamp, chase_rgba, images, panorama):
        """Called from SplatRecorder.write() with one synchronized 10 Hz batch."""
        if self._closed:
            return
        self.check_error()
        chase = cv2.cvtColor(np.asarray(chase_rgba)[..., :3], cv2.COLOR_RGB2BGR)
        tiles = {name: cv2.cvtColor(cv2.resize(image, TILE_SIZE, interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_RGB2BGR) for name, image in images.items()}
        pano = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
        values = ["", "", "", "", "", ""]
        if self._state_fn is not None:
            position, velocity = self._state_fn()
            values = [f"{v:.4f}" for v in (*np.asarray(position, float), *np.asarray(velocity, float))]
        self._csv.write(",".join([str(int(stamp))] + values) + "\n")
        while len(self._pending) >= MAX_PENDING:
            self._pending.popleft().result()
        self._pending.append(self._pool.submit(self._write, stamp, chase, tiles, pano))

    def _write(self, stamp, chase, tiles, pano):
        name = f"{int(stamp):019d}.jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        jobs = [(self.output_dir / "chase" / name, chase), (self.output_dir / "pano" / name, pano)]
        jobs += [(self.output_dir / "cams" / cam / name, tile) for cam, tile in tiles.items()]
        for path, image in jobs:
            if not cv2.imwrite(str(path), image, params):
                raise OSError(f"Could not write {path}")

    def check_error(self):
        while self._pending and self._pending[0].done():
            self._pending.popleft().result()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._pool.shutdown(wait=True)
            self._csv.close()
