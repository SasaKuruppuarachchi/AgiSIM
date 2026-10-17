"""Asynchronous, bounded dataset writer. This module has no Isaac or ROS imports."""

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from PIL import Image
import yaml

from agipix_splat_geometry import INTRINSICS, MOUNTS, RATE_HZ, body_to_camera_cv, opencv_to_nerfstudio


def default_output_directory(environment):
    return Path.home() / "agipix_splat" / f"{environment}_{datetime.now():%Y%m%d_%H%M%S_%f}"


class SplatDataset:
    """Copy render data at submission; do PNG/NPY/metadata I/O on one worker.

    A full queue or disk error is fatal and reported to the simulation loop.
    Frames are never silently dropped and the render thread never waits for disk.
    """

    def __init__(self, output_dir, environment, depth=False, queue_size=8):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.depth = depth
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._disk_failed = False
        self.capture_error = None
        self._closed = False
        self._frames = []
        self._batches = 0
        self._first_stamp = self._last_stamp = None
        self._first_wall = self._last_wall = None
        self._queue_peak = 0
        for mount in MOUNTS:
            (self.output_dir / "images" / mount.name).mkdir(parents=True)
            if depth:
                (self.output_dir / "depth" / mount.name).mkdir(parents=True)
        calibration = {
            "environment": environment, "intrinsics": INTRINSICS, "rate_hz": RATE_HZ,
            "world_frame": "ENU, metres; X east/forward, Y north/left, Z up",
            "camera_frame": "OpenCV: X right, Y down, Z forward",
            "transforms_json_camera_frame": "OpenGL: X right, Y up, Z backward",
            "conversion": "T_world_gl = T_world_cv @ diag(1, -1, -1, 1); no world-axis change",
            "pose_source": "render-synchronized CameraParams; rigid mount inverted to recover T_world_body, "
                           "then composed with each fixed T_body_cam; never the latest physics state",
            "timestamp_source": "writer reference_time mapped by SimulationManager.get_simulation_time_at_time",
            "image_processing": "spatial FXAA; temporal DLSS and motion blur disabled in splat mode",
            "depth": "float32 NPY, optical-axis Z in metres; +inf means no return" if depth else None,
            "cameras": {m.name: {"T_body_cam": body_to_camera_cv(m).tolist(),
                                  "yaw_degrees": m.yaw_degrees, "elevation_degrees": m.pitch_degrees}
                        for m in MOUNTS},
        }
        (self.output_dir / "cameras.yaml").write_text(yaml.safe_dump(calibration, sort_keys=False))
        (self.output_dir / "README.txt").write_text(
            "AgiSIM setting H / OV9782 W synthetic dataset\n\n"
            "RGB: images/{FL,FR,RL,RR}/<frame>_<simulation_nanoseconds>.png, 1280x800.\n"
            "Optional depth: float32 .npy optical-axis Z in METRES, +inf for no return.\n"
            "Use a depth-unit scale of 1.0 when loading these arrays.\n\n"
            "transforms.json: nerfstudio OPENCV_FISHEYE, camera-to-world matrices with\n"
            "OpenGL camera axes (right, up, backward). No world-axis or scale change.\n"
            "frames.csv: OpenCV camera-to-world T_world_cam (right, down, forward),\n"
            "plus the shared ground-truth T_world_body and render reference timestamp.\n"
            "Invert T_world_cam for tools such as COLMAP that expect world-to-camera.\n"
            "cameras.yaml contains intrinsics and fixed body-to-camera calibration.\n"
            "T_world_gl = T_world_cv @ diag(1,-1,-1,1).\n\n"
            "Images and camera poses come from a single synchronized Replicator writer\n"
            "payload. The body pose is recovered from the rendered camera transform,\n"
            "checked against all four rigid mounts, then composed with those mounts.\n"
            "No latest-state or ROS pose lookup is used.\n\n"
            "transforms.json is checkpointed each simulated second and finalized on\n"
            "shutdown. frames.jsonl is an incremental recovery journal. summary.json\n"
            "reports capture throughput, RTF, queue pressure and any capture error.\n"
        )
        self._write_manifest()
        self._encoders = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agipix-splat-png")
        self._thread = threading.Thread(target=self._worker, name="agipix-splat-disk", daemon=False)
        self._thread.start()

    def check_error(self):
        if self._error is not None:
            raise RuntimeError(f"Splat dataset failed: {self.output_dir}") from self._error

    def submit(self, timestamp_ns, reference_time, world_body, images, depths=None):
        self.check_error()
        if self._closed:
            raise RuntimeError("Splat dataset is closed")
        # Replicator owns/reuses its buffers. Take ownership before returning to Kit.
        batch = (int(timestamp_ns), tuple(reference_time), np.array(world_body, copy=True),
                 {name: np.array(image, dtype=np.uint8, copy=True) for name, image in images.items()},
                 None if depths is None else {name: np.array(d, dtype=np.float32, copy=True)
                                               for name, d in depths.items()})
        try:
            self._queue.put_nowait(batch)
        except queue.Full as error:
            self._error = RuntimeError("Splat disk queue is full; stopping capture instead of dropping frames")
            raise self._error from error
        self._queue_peak = max(self._queue_peak, self._queue.qsize())
        now = time.monotonic()
        if self._first_stamp is None:
            self._first_stamp, self._first_wall = timestamp_ns, now
        self._last_stamp, self._last_wall = timestamp_ns, now

    def _write_manifest(self):
        path = self.output_dir / "transforms.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**INTRINSICS, "frames": self._frames}, indent=2) + "\n")
        temporary.replace(path)

    def _write_batch(self, batch, csv_writer, journal):
        stamp, reference, world_body, images, depths = batch
        stem = f"{self._batches:06d}_{stamp:019d}"
        # Encode the four independent PNGs concurrently. A single PNG encoder
        # cannot reliably keep up with 40 full-resolution images per sim second.
        writes = {m.name: self._encoders.submit(self._write_camera_files, m.name, stem, images[m.name],
                                                None if depths is None else depths[m.name]) for m in MOUNTS}
        for mount in MOUNTS:
            name = mount.name
            relative = f"images/{name}/{stem}.png"
            writes[name].result()  # Worker waits, never the simulation thread.
            world_cv = world_body @ body_to_camera_cv(mount)
            frame = {"file_path": relative, "camera_id": name, "timestamp": stamp / 1e9,
                     "timestamp_ns": stamp, "transform_matrix": opencv_to_nerfstudio(world_cv).tolist()}
            depth_path = ""
            if depths is not None:
                depth_path = f"depth/{name}/{stem}.npy"
                frame["depth_file_path"] = depth_path
            csv_writer.writerow([self._batches, name, stamp, f"{stamp / 1e9:.9f}", *reference,
                                 relative, depth_path, *world_cv.ravel(), *world_body.ravel()])
            journal.write(json.dumps(frame) + "\n")
            self._frames.append(frame)
        self._batches += 1
        if self._batches % 10 == 0:
            self._write_manifest()

    def _write_camera_files(self, name, stem, image, depth):
        Image.fromarray(image).save(self.output_dir / "images" / name / f"{stem}.png", compress_level=1)
        if depth is not None:
            np.save(self.output_dir / "depth" / name / f"{stem}.npy", depth, allow_pickle=False)

    def _worker(self):
        try:
            with (self.output_dir / "frames.csv").open("w", newline="") as csv_file, \
                    (self.output_dir / "frames.jsonl").open("w") as journal:
                writer = csv.writer(csv_file)
                writer.writerow(["frame", "camera", "timestamp_ns", "timestamp_s", "reference_numerator",
                                 "reference_denominator", "rgb_path", "depth_path"] +
                                [f"T_world_cam_{i}{j}" for i in range(4) for j in range(4)] +
                                [f"T_world_body_{i}{j}" for i in range(4) for j in range(4)])
                while True:
                    batch = self._queue.get()
                    try:
                        if batch is None:
                            break
                        if not self._disk_failed:
                            self._write_batch(batch, writer, journal)
                            csv_file.flush()
                            journal.flush()
                    except Exception as error:
                        self._error = error
                        self._disk_failed = True
                    finally:
                        self._queue.task_done()
        except Exception as error:
            self._error = error

    def close(self):
        if self._closed:
            self.check_error()
            return
        self._closed = True
        # Only shutdown waits for the disk queue. Poll so an early worker failure
        # (e.g. opening the CSV) cannot leave shutdown waiting on a dead worker.
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._encoders.shutdown(wait=True)
        self._write_manifest()
        span = ((self._last_stamp - self._first_stamp) / 1e9) if self._first_stamp is not None else 0.0
        wall = (self._last_wall - self._first_wall) if self._first_wall is not None else 0.0
        summary = {"frames_total": len(self._frames), "frames_per_camera": self._batches,
                   "first_timestamp_ns": self._first_stamp, "last_timestamp_ns": self._last_stamp,
                   "sim_span_s": span, "capture_wall_span_s": wall,
                   "camera_rate_sim_hz": (self._batches - 1) / span if span > 0 else 0,
                   "camera_rate_wall_hz": (self._batches - 1) / wall if wall > 0 else 0,
                   "rtf": span / wall if wall > 0 else 0, "queue_peak_batches": self._queue_peak,
                   "error": str(self._error) if self._error is not None else self.capture_error}
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self.check_error()
