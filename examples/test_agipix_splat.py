"""Calibration, dataset serialization and failure handling; no Isaac imports."""

import csv
import json
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np
from PIL import Image
import yaml

from agipix_splat_dataset import SplatDataset
from agipix_splat_geometry import (
    CV_FROM_GL, CX, CY, FOCAL_PX, HEIGHT, MOUNTS, WIDTH,
    body_to_camera_cv, module_corners_cv, opencv_to_nerfstudio,
    project_equidistant, rendered_camera_to_opencv,
)
from agipix_splat_preview import (
    BOTTOM_ELEVATION_DEG, PREVIEW_HEIGHT, PREVIEW_WIDTH, TOP_ELEVATION_DEG, SplatPanorama,
    encode_preview_jpeg,
)


class SplatGeometryTest(unittest.TestCase):
    def test_all_extrinsics_match_handover_yaml_including_downward_optical_axis(self):
        handover = Path(__file__).resolve().parents[1] / "assets/agipixv2_setting_H_handover/log_work/2026-10-17_agipixv2_setting_H_camera_mount_handover.md"
        calibration = yaml.safe_load(handover.read_text().split("```yaml\n")[1].split("```")[0])
        for mount in MOUNTS:
            with self.subTest(camera=mount.name):
                transform = body_to_camera_cv(mount)
                np.testing.assert_allclose(transform[:3, :3], calibration[mount.name]["R_body_cam"], atol=1e-6)
                np.testing.assert_allclose(transform[:3, 3], calibration[mount.name]["t_body_cam_m"], atol=1e-10)
                self.assertLess(transform[2, 2], 0)
                self.assertLess(transform[2, 1], 0)  # Image down points down; sky at image top.
                self.assertAlmostEqual(transform[2, 0], 0)  # Image right is horizontal.
                self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1)

    def test_nerfstudio_and_opencv_project_the_same_world_point(self):
        world_body = np.array([[0, -1, 0, 3], [1, 0, 0, 4], [0, 0, 1, 2], [0, 0, 0, 1]], dtype=float)
        for mount in MOUNTS:
            world_cv = world_body @ body_to_camera_cv(mount)
            world_point = world_cv @ [0.9, -0.3, 2.0, 1.0]
            world_gl = opencv_to_nerfstudio(world_cv)
            cv_point = np.linalg.inv(world_cv) @ world_point
            gl_point = np.linalg.inv(world_gl) @ world_point
            self.assertLess(gl_point[2], 0)
            np.testing.assert_allclose(project_equidistant(cv_point[:3]),
                                       project_equidistant((CV_FROM_GL @ gl_point)[:3]), atol=1e-10)
            # The renderer supplies row-vector WORLD-TO-view, not camera-to-world.
            view = np.linalg.inv(world_gl).T
            np.testing.assert_allclose(rendered_camera_to_opencv(view), world_cv, atol=1e-12)

    def test_equidistant_projection_matches_opencv_kb4(self):
        points = np.array([[0, 0, 1], [1, 0, 1], [-2, 0.5, 1], [0.2, -0.3, 1], [2, 1.1, 1]], dtype=float)
        k = np.array([[FOCAL_PX, 0, CX], [0, FOCAL_PX, CY], [0, 0, 1]])
        projected, _ = cv2.fisheye.projectPoints(points.reshape(1, -1, 3), np.zeros(3), np.zeros(3), k, np.zeros(4))
        np.testing.assert_allclose(project_equidistant(points), projected.reshape(-1, 2), atol=1e-10)

    def test_module_is_behind_its_lens_and_outside_every_pixel_ray(self):
        corners = module_corners_cv()
        np.testing.assert_allclose(np.ptp(corners, axis=0), [0.026, 0.026, 0.015])
        self.assertAlmostEqual(corners[:, 2].max(), -0.003)
        self.assertAlmostEqual(corners[:, 2].min(), -0.018)
        # Even the image corner's equidistant ray is <90 degrees off axis.
        self.assertLess(np.hypot(CX, CY) / FOCAL_PX, np.pi / 2)


class SplatPanoramaTest(unittest.TestCase):
    def test_jpeg_preserves_preview_dimensions_and_rgb_channel_order(self):
        image = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), np.uint8)
        image[:, :320] = [240, 10, 20]
        image[:, 320:640] = [10, 240, 20]
        image[:, 640:] = [10, 20, 240]
        encoded = encode_preview_jpeg(image)
        self.assertIsInstance(encoded, bytes)
        self.assertLess(len(encoded), image.nbytes // 10)
        decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        self.assertEqual(decoded.shape, image.shape)
        # Sample inside blocks: JPEG chroma subsampling is lossy near edges.
        np.testing.assert_allclose(decoded[128, [160, 480, 800]], image[128, [160, 480, 800]], atol=3)

    def test_overlap_preserves_brightness_and_blind_directions_stay_black(self):
        panorama = SplatPanorama()
        images = {m.name: np.full((HEIGHT, WIDTH, 3), 173, np.uint8) for m in MOUNTS}
        preview = panorama.stitch(images)
        self.assertEqual(preview.shape, (PREVIEW_HEIGHT, PREVIEW_WIDTH, 3))
        self.assertEqual(PREVIEW_WIDTH / PREVIEW_HEIGHT, 4)
        self.assertLess(preview.size, images["FL"].size)
        self.assertTrue(np.any(panorama.coverage >= 2))
        self.assertTrue(np.any(panorama.coverage == 0))
        np.testing.assert_array_equal(preview[panorama.coverage > 0], 173)
        np.testing.assert_array_equal(preview[panorama.coverage == 0], 0)

    def test_camera_order_and_front_overlap_are_spatial_not_tiles(self):
        panorama = SplatPanorama()
        colors = {"FL": [255, 0, 0], "FR": [0, 0, 255], "RL": [255, 255, 0], "RR": [0, 255, 0]}
        images = {name: np.broadcast_to(np.uint8(color), (HEIGHT, WIDTH, 3)).copy()
                  for name, color in colors.items()}
        preview = panorama.stitch(images)
        horizon = round(PREVIEW_HEIGHT * TOP_ELEVATION_DEG / (TOP_ELEVATION_DEG - BOTTOM_ELEVATION_DEG) - 0.5)
        np.testing.assert_array_equal(preview[horizon, PREVIEW_WIDTH // 4], colors["RL"])
        np.testing.assert_array_equal(preview[horizon, PREVIEW_WIDTH * 3 // 4], colors["RR"])
        front = preview[horizon, PREVIEW_WIDTH // 2]
        self.assertGreater(front[0], 100)
        self.assertGreater(front[2], 100)
        self.assertEqual(front[1], 0)
        np.testing.assert_array_equal(preview[horizon, [0, -1]], 0)  # Actual rear blind spot.

    def test_all_fisheyes_align_to_the_same_angular_scene_including_overlaps(self):
        # A synthetic distant sphere colored by direction, rendered independently
        # through OpenCV's inverse fisheye model. Its panorama is known analytically.
        y, x = np.mgrid[:HEIGHT, :WIDTH]
        pixels = np.stack((x + 0.5, y + 0.5), axis=-1).reshape(-1, 1, 2)
        intrinsic = np.array([[FOCAL_PX, 0, CX], [0, FOCAL_PX, CY], [0, 0, 1.]])
        xy = cv2.fisheye.undistortPoints(pixels, intrinsic, np.zeros(4)).reshape(HEIGHT, WIDTH, 2)
        camera_rays = np.concatenate((xy, np.ones((HEIGHT, WIDTH, 1))), axis=-1)
        camera_rays /= np.linalg.norm(camera_rays, axis=-1, keepdims=True)
        images = {}
        for mount in MOUNTS:
            body_rays = camera_rays @ body_to_camera_cv(mount)[:3, :3].T
            images[mount.name] = np.rint(127.5 * (body_rays + 1)).astype(np.uint8)
        panorama = SplatPanorama()
        preview = panorama.stitch(images)
        azimuth = np.deg2rad(180 - 360 * (np.arange(PREVIEW_WIDTH) + 0.5) / PREVIEW_WIDTH)
        elevation = np.deg2rad(TOP_ELEVATION_DEG - (TOP_ELEVATION_DEG - BOTTOM_ELEVATION_DEG)
                              * (np.arange(PREVIEW_HEIGHT) + 0.5) / PREVIEW_HEIGHT)
        expected = np.empty_like(preview, dtype=float)
        expected[..., 0] = np.cos(elevation)[:, None] * np.cos(azimuth)
        expected[..., 1] = np.cos(elevation)[:, None] * np.sin(azimuth)
        expected[..., 2] = np.sin(elevation)[:, None]
        expected = 127.5 * (expected + 1)
        np.testing.assert_allclose(preview[panorama.coverage > 0], expected[panorama.coverage > 0], atol=2)


class SplatDatasetTest(unittest.TestCase):
    def test_images_depth_csv_and_nerfstudio_share_the_pose_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            dataset = SplatDataset(output, "FOREST_FI", depth=True)
            images = {mount.name: np.full((HEIGHT, WIDTH, 3), 40 + i * 50, np.uint8)
                      for i, mount in enumerate(MOUNTS)}
            depths = {m.name: np.full((HEIGHT, WIDTH), 1.234, np.float32) for m in MOUNTS}
            body = np.eye(4)
            body[:3, 3] = [3, 4, 5]
            dataset.submit(1234000000, (617, 500), body, images, depths)
            for image in images.values():
                image[:] = 0  # Submission must own its copy of recycled render buffers.
            dataset.close()
            frames = json.loads((output / "transforms.json").read_text())["frames"]
            self.assertEqual(len(frames), 4)
            rows = list(csv.DictReader((output / "frames.csv").open()))
            for i, (mount, frame, row) in enumerate(zip(MOUNTS, frames, rows)):
                self.assertEqual(frame["timestamp_ns"], 1234000000)
                self.assertEqual(int(row["timestamp_ns"]), frame["timestamp_ns"])
                np.testing.assert_array_equal(np.asarray(Image.open(output / frame["file_path"])), 40 + i * 50)
                np.testing.assert_allclose(np.load(output / frame["depth_file_path"]), 1.234)
                csv_pose = np.array([float(row[f"T_world_cam_{r}{c}"]) for r in range(4) for c in range(4)]).reshape(4, 4)
                np.testing.assert_allclose(csv_pose, body @ body_to_camera_cv(mount))
                np.testing.assert_allclose(frame["transform_matrix"], opencv_to_nerfstudio(csv_pose))
            self.assertEqual(json.loads((output / "summary.json").read_text())["error"], None)

    def test_disk_failure_is_reported_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = SplatDataset(Path(directory) / "dataset", "AKW_C")
            def fail(*args):
                raise OSError("simulated disk full")
            dataset._write_batch = fail
            dataset.submit(100000000, (1, 10), np.eye(4), {m.name: np.zeros((2, 2, 3), np.uint8) for m in MOUNTS})
            with self.assertRaisesRegex(RuntimeError, "Splat dataset failed"):
                dataset.close()

    def test_queue_overload_fails_without_blocking_and_drains_accepted_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            dataset = SplatDataset(output, "AKW_C", queue_size=1)
            started, release = threading.Event(), threading.Event()
            write_batch = dataset._write_batch
            def slow_write(*args):
                started.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test writer was not released")
                write_batch(*args)
            dataset._write_batch = slow_write
            images = {m.name: np.zeros((2, 2, 3), np.uint8) for m in MOUNTS}
            try:
                dataset.submit(100000000, (1, 10), np.eye(4), images)
                self.assertTrue(started.wait(timeout=2))
                dataset.submit(200000000, (2, 10), np.eye(4), images)
                with self.assertRaisesRegex(RuntimeError, "queue is full"):
                    dataset.submit(300000000, (3, 10), np.eye(4), images)
            finally:
                release.set()
                with self.assertRaisesRegex(RuntimeError, "Splat dataset failed"):
                    dataset.close()
            self.assertEqual(len(json.loads((output / "transforms.json").read_text())["frames"]), 8)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                SplatDataset(directory, "AKW_C")


if __name__ == "__main__":
    unittest.main()
