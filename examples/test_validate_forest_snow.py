"""Geometry and ROS binary parsing checks without Kit, a GPU, ROS or PX4."""

import math
import json
from types import SimpleNamespace

import numpy as np
import pytest

from validate_forest_snow import DEFAULT_STAND, classify_cloud, default_stand, pointcloud_xyz, terrain_heights, validation_pose
from validate_forest_snow_px4 import json_finite


@pytest.mark.parametrize("endian", ("<", ">"))
def test_pointcloud_handles_padding_offsets_endian_and_nonfinite(endian):
    # Two rows with padding, and deliberately shuffled fields within each point.
    buffer = bytearray(2 * 60)
    dtype = np.dtype(dict(names=["x", "y", "z"], formats=[endian + "f4"] * 3,
                         offsets=[8, 0, 12], itemsize=24))
    rows = np.ndarray((2, 2), dtype=dtype, buffer=buffer, strides=(60, 24))
    rows["x"] = [[1, 4], [7, np.nan]]
    rows["y"] = [[2, 5], [8, 11]]
    rows["z"] = [[3, 6], [9, 12]]
    message = SimpleNamespace(data=buffer, height=2, width=2, point_step=24, row_step=60,
        is_bigendian=endian == ">", fields=[SimpleNamespace(name=n, datatype=7, offset=o)
                                           for n, o in (("x", 8), ("y", 0), ("z", 12))])
    xyz, finite = pointcloud_xyz(message, stride=2)
    np.testing.assert_array_equal(xyz, [[1, 2, 3], [7, 8, 9]])
    assert finite == 3


def test_terrain_interpolation_uses_triangle_diagonal():
    terrain = {"z": [[0, 2], [4, 10]], "grid": 2, "area": 2}
    # Non-planar quad: triangle interpolation differs from bilinear interpolation.
    xy = np.array([[-.5, 0], [0, -.5], [0, 0], [-1, -1], [1, 1]])
    np.testing.assert_allclose(terrain_heights(terrain, xy), [3, 3.5, 5, 0, 10], atol=2e-8)


def test_cloud_classification_distinguishes_snow_ground_low_stem_and_crown():
    stand = {"terrain": {"z": [[0, 0], [0, 0]], "grid": 10, "area": 10},
             "trees": [{"x": 2, "y": 1, "dbh": .3, "crown_base": 4}]}
    world = np.array([[0, 0, .02], [2.15, 1, 1.3], [2.15, 1, 5], [0, 0, 2], [6, 0, 0]])
    labels, agl, counts = classify_cloud(world, stand)
    np.testing.assert_array_equal(labels, [1, 2, 0, 0, 0])
    np.testing.assert_array_equal(agl, world[:, 2])
    assert counts["ground_snow_points"] == 1
    assert counts["approximate_low_trunk_points"] == 1


def test_mixed_ground_does_not_claim_semantic_snow_returns():
    stand = {"terrain": {"z": [[0, 0], [0, 0]], "grid": 10, "area": 10},
             "trees": [{"x": 2, "y": 1, "dbh": .3, "crown_base": 4}]}
    labels, _, counts = classify_cloud(np.array([[0, 0, .02], [2.15, 1, 1.3]]), stand, "FOREST_MID_SNOW")
    np.testing.assert_array_equal(labels, [1, 2])
    assert counts["ground_surface_points"] == 1
    assert "ground_snow_points" not in counts
    assert "not semantically classified" in counts["snow_check_scope"]


def test_stand_defaults_follow_environment_and_preserve_original_default():
    assert default_stand() == DEFAULT_STAND
    assert str(default_stand()).endswith("artifacts/forest_snow/build/forest_snow_stand.json")
    assert str(default_stand("FOREST_MID_SNOW")).endswith("artifacts/forest_mid_snow/build/forest_mid_snow_stand.json")
    with pytest.raises(ValueError):
        default_stand("AKW_C")


def test_px4_json_replaces_nonfinite_sentinels_without_mutating_command():
    original = {"commands": [{"parameters": [0, float("nan"), float("inf"), -float("inf")]}], "success": True}
    clean = json_finite(original)
    assert json.loads(json.dumps(clean, allow_nan=False)) == {
        "commands": [{"parameters": [0, None, None, None]}], "success": True}
    assert math.isnan(original["commands"][0]["parameters"][1])


def test_scripted_path_stays_within_spawn_clearing_and_starts_static():
    for seconds in np.linspace(0, 120, 1000):
        x, y, z = validation_pose(seconds)
        assert math.hypot(x, y) <= 1.25
        assert z == 2
        if seconds <= 5:
            assert x == y == 0
