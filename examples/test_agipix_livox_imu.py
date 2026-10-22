"""Livox Mid-360 IMU emulation: placement, units, noise and the 200 Hz sampling grid."""
import math
import unittest

import numpy as np

from agipix_livox_imu import LivoxImuEmulator, LivoxImuSettings, imu_position_in_body, quat_wxyz_to_rot

LIDAR_TRANS = [0.0795, 0.0, 0.0323]
LIDAR_ORI = [0.9238795, 0.0, 0.3826834, 0.0]  # 45 deg about +y (8_agipix.py)


class LivoxImuTest(unittest.TestCase):
    def test_position_inside_mid360(self):
        p = imu_position_in_body(LIDAR_TRANS, LIDAR_ORI, LivoxImuSettings().position_in_lidar)
        R = quat_wxyz_to_rot(LIDAR_ORI)
        np.testing.assert_allclose(R.T @ (p - np.array(LIDAR_TRANS)), [0.011, 0.02329, -0.04412], atol=1e-9)
        self.assertLess(np.linalg.norm(p - np.array(LIDAR_TRANS)), 0.06)

    def test_units_bias_and_noise(self):
        s = LivoxImuSettings(seed=1, vibration_scale=0.0)
        e = LivoxImuEmulator(lambda *a: None, LIDAR_ORI, [0, 0, 0], s)
        # Gravity reaction in lidar axes, 45 deg tilt: about (-0.707, 0, 0.707) g like the real bag.
        acc = np.array([-math.sqrt(0.5), 0.0, math.sqrt(0.5)]) * s.gravity
        samples = np.array([np.concatenate(e.sample(acc, [0, 0, 0])) for _ in range(4000)])
        np.testing.assert_allclose(samples[:, :3].mean(0), [-0.7071, 0, 0.7071], atol=2e-3)
        np.testing.assert_allclose(samples[:, 3:].mean(0), s.gyro_bias, atol=5e-4)
        np.testing.assert_allclose(samples[:, :3].std(0), s.accel_noise_std_g, rtol=0.1)
        np.testing.assert_allclose(samples[:, 3:].std(0), s.gyro_noise_std, rtol=0.1)

    def test_200hz_on_250hz_physics(self):
        stamps = []
        e = LivoxImuEmulator(lambda t, a, g: stamps.append(t), LIDAR_ORI, [0, 0, 0])
        for k in range(2500):  # 10 s of 4 ms physics steps
            e.step(k * 0.004, np.eye(3), [0, 0, 0], [0, 0, 0])
        self.assertAlmostEqual(len(stamps) / 10.0, 200.0, delta=1.0)
        self.assertLessEqual(max(np.diff(stamps)), 0.008 + 1e-9)

    def test_vibration_matches_real_spectrum_above_8hz(self):
        import json
        from agipix_livox_imu import VIBRATION_FILE
        ref = json.loads(VIBRATION_FILE.read_text())["rms_above_8Hz"]
        e = LivoxImuEmulator(lambda *a: None, LIDAR_ORI, [0, 0, 0], LivoxImuSettings(seed=3))
        v = np.array([np.concatenate(e.vibration()) for _ in range(40000)])
        np.testing.assert_allclose(v.std(0), [ref[k] for k in ("gx", "gy", "gz", "ax_g", "ay_g", "az_g")], rtol=0.1)
        # Coloured: almost no energy below 5 Hz (the sim supplies the drone's own motion there).
        spec = np.abs(np.fft.rfft(v[:, 1])) ** 2
        f = np.fft.rfftfreq(len(v), 1 / 200.0)
        self.assertLess(spec[(f > 0) & (f < 5)].sum() / spec[f > 0].sum(), 0.05)

    def test_specific_force_physics(self):
        r = np.array([0.05, 0.02, -0.01])
        e = LivoxImuEmulator(lambda *a: None, [1, 0, 0, 0], r)  # IMU axes = body axes
        # At rest, level: +g up.
        f, w = e.specific_force(np.eye(3), np.zeros(3), np.zeros(3), 0.004)
        np.testing.assert_allclose(f, [0, 0, 9.80665], atol=1e-9)
        # Level coordinated turn: constant speed v along body x, yaw rate omega; the inertial velocity
        # rotates, so a = omega x v (centripetal), plus omega x (omega x r) at the IMU offset.
        omega, v, dt = 1.5, 3.0, 0.004
        e.prev = (0.0, np.array([v, 0, 0]), np.array([0, 0, omega]))
        th = omega * dt
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        f, w = e.specific_force(Rz, Rz @ np.array([v, 0, 0]), np.array([0, 0, omega]), dt)
        expect = np.cross([0, 0, omega], [v, 0, 0]) + np.cross([0, 0, omega], np.cross([0, 0, omega], r)) + [0, 0, 9.80665]
        np.testing.assert_allclose(f, expect, atol=0.02)  # small-angle: R^T (omega x v) ~ omega x v
        np.testing.assert_allclose(w, [0, 0, omega], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
