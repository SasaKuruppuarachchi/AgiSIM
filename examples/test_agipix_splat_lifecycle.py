"""Capture lifecycle, recycled buffers and preview subscriber changes without Kit.

Extract the production methods with AST so these checks neither load Isaac Sim
nor initialize ROS. The stubs model callbacks and publishers; live simulator and
DDS behavior is verified separately by the integration benchmark/validator.
"""
from array import array
import ast
import json
import math
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
from agipix_splat_dataset import SplatDataset
from agipix_splat_geometry import (HEIGHT, WIDTH, MOUNTS, body_to_camera_cv,
                                   opencv_to_nerfstudio, rendered_camera_to_opencv)


def extracted(source, name, globals_):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), globals_)
    return globals_[name]


class Simulation:
    active = True
    @classmethod
    def is_simulating(cls):
        return cls.active


Recorder = extracted(ROOT / 'examples/agipix_splat_camera.py', 'SplatRecorder', {
    'rep': SimpleNamespace(Writer=object), 'SimulationManager': Simulation,
    'np': np, 'cv2': cv2, 'math': math, 'HEIGHT': HEIGHT, 'WIDTH': WIDTH, 'MOUNTS': MOUNTS,
    'body_to_camera_cv': body_to_camera_cv, 'rendered_camera_to_opencv': rendered_camera_to_opencv,
    'carb': SimpleNamespace(log_error=lambda message: None),
})


def make_recorder(dataset=None):
    Simulation.active = True
    recorder = Recorder.__new__(Recorder)
    recorder._dataset = dataset
    recorder._depth = False
    recorder._closed = False
    recorder._error = None
    recorder._last_stamp = -1
    recorder._by_product = {m.name: m for m in MOUNTS}
    recorder._sim_interface = SimpleNamespace(get_simulation_time_at_time=lambda ref: ref[0] / ref[1])
    recorder.published = []
    recorder._preview_node = SimpleNamespace(publish_splat_preview=lambda image, stamp: recorder.published.append(stamp))
    recorder._panorama = SimpleNamespace(stitch=lambda images: np.zeros((2, 2, 3), np.uint8))
    return recorder


def payload(tenth):
    products = {}
    for i, mount in enumerate(MOUNTS):
        products[mount.name] = {
            'rgb': {'data': np.full((HEIGHT, WIDTH, 4), 30+i, np.uint8)},
            'camera_params': {'cameraViewTransform': np.linalg.inv(opencv_to_nerfstudio(body_to_camera_cv(mount))).T},
        }
    return {'reference_time': (tenth, 10), 'renderProducts': products}


def test_pause_duplicate_stop_and_preview_restart():
    recorder = make_recorder()
    data = payload(1)
    recorder.write(data)
    assert recorder.published == [100_000_000]
    recorder.write(data)  # Duplicate render while paused.
    assert recorder.published == [100_000_000]
    Simulation.active = False
    recorder.write(payload(2))  # Stop callback cannot produce a new frame.
    assert recorder.published == [100_000_000]
    Simulation.active = True
    recorder.write(payload(2))
    recorder.write(payload(0))  # Zero-time initialization is ignored.
    recorder.write(payload(1))  # Preview can follow a new timeline.
    assert recorder.published == [100_000_000, 200_000_000, 100_000_000]
    recorder.poll()


def test_save_restart_fails_loudly_and_retains_accepted_batch(tmp_path):
    dataset = SplatDataset(tmp_path / 'recording', 'AKW_C')
    recorder = make_recorder(dataset)
    try:
        recorder.write(payload(2))
        recorder.write(payload(1))
        with pytest.raises(RuntimeError, match='Splat capture failed'):
            recorder.poll()
        assert 'Timeline restarted' in dataset.capture_error
    finally:
        dataset.close()
    manifest = json.loads((dataset.output_dir / 'transforms.json').read_text())
    assert len(manifest['frames']) == 4
    assert {f['timestamp_ns'] for f in manifest['frames']} == {200_000_000}


def test_camera_rigid_pose_mismatch_stops_capture(tmp_path):
    dataset = SplatDataset(tmp_path / 'recording', 'AKW_C')
    recorder = make_recorder(dataset)
    data = payload(1)
    data['renderProducts']['RR']['camera_params']['cameraViewTransform'][3, 0] += 0.1
    try:
        recorder.write(data)
        with pytest.raises(RuntimeError, match='Splat capture failed'):
            recorder.poll()
        assert 'render poses disagree' in dataset.capture_error
    finally:
        dataset.close()
    assert json.loads((dataset.output_dir / 'summary.json').read_text())['frames_total'] == 0


def test_recycled_noncontiguous_rgba_depth_and_pose_are_owned(tmp_path):
    dataset = SplatDataset(tmp_path / 'recording', 'AKW_C', depth=True)
    started, release = threading.Event(), threading.Event()
    original = dataset._write_batch
    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)
    dataset._write_batch = blocked
    rng = np.random.default_rng(20261018)
    rgba = {m.name: rng.integers(0, 256, (7, 11, 4), dtype=np.uint8) for m in MOUNTS}
    expected_rgb = {name: image[..., :3].copy() for name, image in rgba.items()}
    rgb = {name: image[..., :3] for name, image in rgba.items()}
    depth = {m.name: np.full((7, 11), 1.23, np.float32) for m in MOUNTS}
    body = np.eye(4)
    try:
        dataset.submit(100_000_000, (1,10), body, rgb, depth)
        assert started.wait(2)
        for image in rgba.values(): image[:] = 0
        for image in depth.values(): image[:] = 0
        body[:] = 0
    finally:
        release.set()
        dataset.close()
    frames = json.loads((dataset.output_dir/'transforms.json').read_text())['frames']
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(Image.open(dataset.output_dir/frame['file_path']), expected_rgb[frame['camera_id']])
        np.testing.assert_allclose(np.load(dataset.output_dir/frame['depth_file_path']), 1.23)
        np.testing.assert_allclose(frame['transform_matrix'], opencv_to_nerfstudio(body_to_camera_cv(MOUNTS[i])))


def test_foxglove_connect_disconnect_reconnect_preserves_header_and_stops_encoding():
    tree = ast.parse((ROOT/'examples/utils/drone_location_pub.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DroneLocationPublisher')
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'publish_splat_preview')
    scope = {"array": array}
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<production-publisher>', 'exec'), scope)
    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0), frame_id='')
    raw, compressed, encoded = [], [], []
    counts = iter([0, 1, 0, 2])
    def encode(image):
        encoded.append(image.copy())
        return b'jpeg'
    publisher = SimpleNamespace(vehicle_name='drone0', _splat_image_type=Message,
        _splat_compressed_image_type=Message, _encode_splat_jpeg=encode,
        _splat_preview_publisher=SimpleNamespace(publish=raw.append),
        _splat_preview_compressed_publisher=SimpleNamespace(publish=compressed.append,
            get_subscription_count=lambda: next(counts)))
    image = np.full((2, 3, 3), 73, np.uint8)
    for stamp in [100_000_000, 200_000_000, 300_000_000, 400_000_000]:
        scope['publish_splat_preview'](publisher, image, stamp)
    image[:] = 0
    assert len(raw) == 4 and len(compressed) == len(encoded) == 2
    assert [m.header.stamp.nanosec for m in compressed] == [200_000_000, 400_000_000]
    assert all(m.header.frame_id == 'drone0/splat_panorama' for m in raw+compressed)
    assert all(bytes(m.data) == bytes([73])*18 for m in raw)
    assert all(bytes(m.data) == b'jpeg' for m in compressed)
    assert compressed[0].header is raw[1].header


def test_capture_preserves_arbitrary_rgb_and_ignores_alpha():
    """Exercise RGB extraction itself, including conversion changes, on all pixels."""
    class OwnedSubmission:
        def submit(self, stamp, reference, body, images, depths):
            # SplatDataset's actual deferred ownership is tested separately above.
            self.images = {name: image.copy() for name, image in images.items()}
        def check_error(self):
            pass

    dataset = OwnedSubmission()
    recorder = make_recorder(dataset)
    data = payload(1)
    rng = np.random.default_rng(5775)
    expected = {}
    for mount in MOUNTS:
        rgba = rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
        data['renderProducts'][mount.name]['rgb']['data'] = rgba
        expected[mount.name] = rgba[..., :3].copy()
    recorder.write(data)
    recorder.poll()
    for mount in MOUNTS:
        rgba = data['renderProducts'][mount.name]['rgb']['data']
        rgba[..., 3] = 0  # Alpha must never modulate RGB during conversion.
        np.testing.assert_array_equal(dataset.images[mount.name], expected[mount.name])
        rgba[:] = 0  # Renderer buffer recycling cannot alter accepted data.
        np.testing.assert_array_equal(dataset.images[mount.name], expected[mount.name])
    assert recorder.published == [100_000_000]
