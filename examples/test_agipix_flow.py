"""ARK Flow MR emulation: PX4 sign conventions, quantisation, validity rules and range rates (no Isaac)."""
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from agipix_flow import (FlowRangeEmulator, FlowRangeSettings, PIXART_COUNT_RAD, PX4_FLOW_PARAMS,
                         PX4_FLOW_PARAMS_SIM_ONLY, apply_px4_param_env)

DT = 0.004


def flat_ground(z0=0.0):
    def cast(origin, direction, max_distance):
        if direction[2] >= -1e-9 or origin[2] <= z0:
            return None
        d = (origin[2] - z0) / -direction[2]
        return d if d <= max_distance else None
    return cast


def flat_ground_closed(z0=0.0):
    """A floor that a ray starting below it never hits (like a one-sided PhysX mesh)."""
    return flat_ground(z0)


def fly(settings, R, v_w, omega, height, seconds=1.0, cast=None, p_xy=(0.0, 0.0)):
    flows, ranges = [], []
    e = FlowRangeEmulator(cast or flat_ground(), flows.append, ranges.append, settings)
    p = np.array([p_xy[0], p_xy[1], height])
    for k in range(int(round(seconds / DT)) + 1):
        e.step(k * DT, R, p + np.asarray(v_w) * k * DT, v_w, omega)
    return e, flows, ranges


def px4_body_velocity(msg, rng):
    """PX4 VehicleOpticalFlow vehicle_optical_flow_vel: FRD body velocity from one sample."""
    dt = msg["integration_time_us"] * 1e-6
    comp_x = -msg["integrated_x"] + msg["integrated_xgyro"]
    comp_y = -msg["integrated_y"] + msg["integrated_ygyro"]
    return np.array([-rng * comp_y / dt, rng * comp_x / dt])


class FlowSignTest(unittest.TestCase):
    def test_translation_matches_px4_velocity_formula(self):
        s = FlowRangeSettings(flow_noise_counts=0.0, board_gyro_bias=(0, 0, 0), board_gyro_noise_density=0.0)
        yaw = math.radians(30)
        R = Rotation.from_euler("z", yaw).as_matrix()
        v_w = np.array([0.6, -0.3, 0.0])                    # ENU
        _, flows, _ = fly(s, R, v_w, np.zeros(3), 2.0, seconds=2.0)
        h = 2.0 + s.flow_position[2]
        vel = np.mean([px4_body_velocity(m, h) for m in flows[5:]], axis=0)
        v_flu = R.T @ v_w
        np.testing.assert_allclose(vel, [v_flu[0], -v_flu[1]], atol=0.01)   # FRD

    def test_pure_rotation_is_compensated_by_delta_angle(self):
        s = FlowRangeSettings(flow_noise_counts=0.0, board_gyro_bias=(0, 0, 0), board_gyro_noise_density=0.0,
                              flow_position=(0.0, 0.0, 0.0))
        omega = np.array([0.4, -0.2, 0.0])                   # FLU body rate, level attitude held
        _, flows, _ = fly(s, np.eye(3), np.zeros(3), omega, 1.5)
        span = sum(m["integration_time_us"] for m in flows) * 1e-6
        # Positive pixel_flow for a right-handed rotation: x with p, y with q (FRD q = -FLU wy = +0.2).
        self.assertAlmostEqual(sum(m["integrated_x"] for m in flows) / span, 0.4, delta=0.01)
        self.assertAlmostEqual(sum(m["integrated_y"] for m in flows) / span, 0.2, delta=0.01)
        vel = np.mean([px4_body_velocity(f, 1.5) for f in flows[3:]], axis=0)
        np.testing.assert_allclose(vel, [0.0, 0.0], atol=0.01)

    def test_quantisation_carries_residual(self):
        s = FlowRangeSettings(flow_noise_counts=0.0, board_gyro_bias=(0, 0, 0), board_gyro_noise_density=0.0)
        v_w = np.array([0.05, 0.0, 0.0])                     # well below one count per frame at 5 m
        _, flows, _ = fly(s, np.eye(3), v_w, np.zeros(3), 5.0, seconds=4.0)
        total = sum(m["integrated_y"] for m in flows)
        expected = 0.05 / (5.0 + s.flow_position[2]) * sum(m["integration_time_us"] for m in flows) * 1e-6
        self.assertAlmostEqual(total, expected, delta=PIXART_COUNT_RAD)
        for m in flows:
            self.assertAlmostEqual(m["integrated_y"] / PIXART_COUNT_RAD, round(m["integrated_y"] / PIXART_COUNT_RAD), 6)

    def test_rates_and_fields(self):
        _, flows, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 1.0, seconds=2.0)
        self.assertAlmostEqual(len(flows) / 2.0, 125.0, delta=1.0)
        self.assertAlmostEqual(len(ranges) / 2.0, 25.0, delta=1.0)
        self.assertEqual(flows[0]["distance"], -1.0)
        self.assertEqual({r["min_distance"] for r in ranges}, {8})
        self.assertEqual({r["max_distance"] for r in ranges}, {5000})


class ValidityTest(unittest.TestCase):
    def test_on_the_feet_flow_discarded_range_below_min(self):
        s = FlowRangeSettings()
        e, flows, ranges = fly(s, np.eye(3), np.zeros(3), np.zeros(3), 0.0914)
        self.assertEqual(e.discard_reasons, {})               # published (PX4 must see it before arming)
        self.assertTrue(flows and all(m["quality"] == 10 and m["integrated_x"] == 0.0 for m in flows))
        self.assertTrue(all(r["current_distance"] < 8 and r["signal_quality"] == 100 for r in ranges))

    def test_sensor_inside_the_floor_reads_too_close(self):
        # Sim landing gear: body origin 0.066 m above the floor, sensor points 11-15 mm below it.
        e, flows, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 0.066,
                               cast=flat_ground_closed())
        self.assertTrue(flows and all(m["quality"] == 10 for m in flows))
        self.assertEqual(e.counts["flow_too_close"], len(flows))
        self.assertTrue(all(r["current_distance"] == 0 and r["signal_quality"] == 100 for r in ranges))

    def test_no_surface(self):
        e, flows, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 2.0,
                               cast=lambda *a: None)
        self.assertEqual(flows, [])
        self.assertTrue(all(r["current_distance"] == 0 and r["signal_quality"] == 0 for r in ranges))

    def test_beyond_rate_limit_discarded(self):
        e, flows, _ = fly(FlowRangeSettings(), np.eye(3), np.array([5.0, 0, 0]), np.zeros(3), 0.5)
        self.assertEqual(flows, [])                          # 5 m/s at 0.42 m ~ 12 rad/s > 7.4
        self.assertIn("rate_limit", e.discard_reasons)

    def test_featureless_box_and_texture(self):
        s = FlowRangeSettings(featureless_boxes=((-1, -1, 1, 1),))
        _, flows, _ = fly(s, np.eye(3), np.zeros(3), np.zeros(3), 2.0)
        self.assertEqual(flows, [])
        _, flows, _ = fly(s, np.eye(3), np.zeros(3), np.zeros(3), 2.0, p_xy=(3.0, 0.0))
        self.assertTrue(flows and all(m["quality"] == s.flow_quality_nominal for m in flows))
        _, flows, _ = fly(FlowRangeSettings(texture=0.5), np.eye(3), np.zeros(3), np.zeros(3), 2.0)
        self.assertEqual({m["quality"] for m in flows}, {80})


class RangeTest(unittest.TestCase):
    def test_tilt_and_noise(self):
        R = Rotation.from_euler("x", 20, degrees=True).as_matrix()
        s = FlowRangeSettings()
        _, _, ranges = fly(s, R, np.zeros(3), np.zeros(3), 3.0, seconds=8.0)
        z = 3.0 + (R @ np.array(s.range_position))[2]
        cm = np.array([r["current_distance"] for r in ranges])
        self.assertAlmostEqual(cm.mean() / 100.0, z / math.cos(math.radians(20)), delta=0.01)
        self.assertAlmostEqual(cm.std() / 100.0, s.range_noise_m + s.range_noise_rel * 3.2, delta=0.006)

    def test_long_range_mode_switch(self):
        _, _, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 15.0, seconds=5.0)
        # The first valid reading above 10 m switches to the 10 Hz long-range mode.
        self.assertAlmostEqual(len(ranges), 1 + 50, delta=2)
        _, _, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 60.0, seconds=1.0)
        self.assertTrue(all(r["current_distance"] > 5000 and r["signal_quality"] == 100 for r in ranges))
        _, _, ranges = fly(FlowRangeSettings(), np.eye(3), np.zeros(3), np.zeros(3), 90.0, seconds=1.0)
        self.assertTrue(all(r["current_distance"] == 0 and r["signal_quality"] == 0 for r in ranges))


class ParamTest(unittest.TestCase):
    def test_env_defaults_do_not_override(self):
        env = {"PX4_PARAM_EKF2_MIN_RNG": "0.1"}
        applied = apply_px4_param_env(env)
        self.assertEqual(applied["EKF2_MIN_RNG"], "0.1")
        self.assertEqual(env["PX4_PARAM_EKF2_OF_POS_Z"], "0.0767")
        self.assertEqual(len(applied), len(PX4_FLOW_PARAMS) + len(PX4_FLOW_PARAMS_SIM_ONLY))

    def test_positions_are_frd_of_the_flu_mounts(self):
        s = FlowRangeSettings()
        frd = lambda p: (p[0], -p[1], -p[2])
        np.testing.assert_allclose([float(PX4_FLOW_PARAMS["EKF2_OF_POS_" + a]) for a in "XYZ"], frd(s.flow_position))
        np.testing.assert_allclose([float(PX4_FLOW_PARAMS["EKF2_RNG_POS_" + a]) for a in "XYZ"], frd(s.range_position))


if __name__ == "__main__":
    unittest.main()
