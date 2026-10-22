"""Livox Mid-360 internal IMU emulation for AgiSIM (/<ns>/livox/imu).

Matches the real sensor in the AgiPix benchmark bags (agi_log_20260821_172847_forrest_test_1):
200 Hz sensor_msgs/Imu, axes of the lidar frame (tilted 45 deg with the Mid-360), linear
acceleration in g (the Livox driver's unit; LIOM's mapping_m_livox_imu.yaml scales by gnorm),
angular velocity in rad/s, identity orientation and zero covariances. The IMU sits inside the
Mid-360 at the Livox manual position (11.0, 23.29, -44.12) mm in the lidar frame.

The reading is computed analytically from the vehicle's rigid-body state every physics step, like
Pegasus' PX4 IMU: specific force f = R_bl^T [R_wb^T (a - g) + alpha x r + omega x (omega x r)] and
omega_imu = R_bl^T omega, with a and alpha from finite differences of the inertial velocity and the
body rate, r the IMU position in the body and R_bl the lidar mount. (Isaac's experimental
IMUSensor, used before 2026-10-22, dropped most of the rotating-frame term: -0.78 omega x v, up to
5-6 m/s^2 in turns; liom_stage/livox_tune/RESULTS.md.) White noise and
the gyro bias are the bag's props-off values. While the props spin, structural/prop vibration is
added as coloured noise: FIR filters (agipix_livox_vibration.json) fitted to the real in-flight IMU
spectrum above 8 Hz (gyro about 0.02-0.03 rad/s, accel about 0.015-0.03 g). Below 8 Hz the real
signal is the drone's own attitude motion (2-3 Hz peak), which the sim produces physically.
vibration_scale scales it (1.0 = real, 0 = clean sensor).
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

VIBRATION_FILE = Path(__file__).with_name("agipix_livox_vibration.json")
VIB_AXES = ("gx", "gy", "gz", "ax_g", "ay_g", "az_g")


@dataclass(frozen=True)
class LivoxImuSettings:
    rate_hz: float = 200.0
    position_in_lidar: tuple = (0.011, 0.02329, -0.04412)       # m, Livox Mid-360 manual
    gyro_noise_std: tuple = (0.0056, 0.0028, 0.0013)            # rad/s, real bag, props off
    accel_noise_std_g: tuple = (0.0045, 0.0033, 0.0118)         # g, real bag, props off
    gyro_bias: tuple = (0.0102, -0.0026, 0.0038)                # rad/s, real bag, props off
    accel_bias_g: tuple = (0.0, 0.0, 0.0)
    vibration_scale: float = 1.0  # coloured prop vibration while the props spin; 1.0 = real spectrum, 0 = off
    gravity: float = 9.80665       # m/s^2 per g
    seed: int = 7


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def imu_position_in_body(lidar_trans, lidar_quat_wxyz, position_in_lidar):
    """Body-frame position of the Mid-360 IMU: lidar mount + R_body_lidar * p_lidar_imu."""
    return np.asarray(lidar_trans, float) + quat_wxyz_to_rot(lidar_quat_wxyz) @ np.asarray(position_in_lidar, float)


class LivoxImuEmulator:
    """Analytic Mid-360 IMU from the rigid-body state, published at rate_hz of sim time."""

    def __init__(self, publish, lidar_quat_wxyz, position_in_body, settings=None):
        self.s = settings or LivoxImuSettings()
        self.publish = publish  # publish(stamp_s, accel_g(3), gyro_rad_s(3))
        self.period = 1.0 / self.s.rate_hz
        self.R_bl = quat_wxyz_to_rot(lidar_quat_wxyz)   # lidar (= IMU axes) in body
        self.r = np.asarray(position_in_body, float)    # IMU position in body (FLU)
        self.next_t = None
        self.prev = None  # (t, v_world, omega_body)
        self.rng = np.random.default_rng(self.s.seed)
        taps = json.loads(VIBRATION_FILE.read_text())["taps"]
        self.vib_taps = np.array([taps[a] for a in VIB_AXES])          # 6 x ntap, at rate_hz
        self.vib_hist = np.zeros_like(self.vib_taps)                   # white input history

    def vibration(self):
        """One sample of coloured vibration (gyro rad/s, accel g) from the fitted FIR filters."""
        self.vib_hist = np.roll(self.vib_hist, 1, axis=1)
        self.vib_hist[:, 0] = self.rng.standard_normal(6)
        v = (self.vib_taps * self.vib_hist).sum(axis=1)
        return v[:3], v[3:]

    def specific_force(self, R_wb, v_world, omega, dt):
        """True specific force (m/s^2) and rate at the IMU, in the IMU (lidar) axes."""
        if self.prev is None or dt <= 0:
            a_w, alpha = np.zeros(3), np.zeros(3)
        else:
            a_w = (v_world - self.prev[1]) / dt
            alpha = (omega - self.prev[2]) / dt
        g_w = np.array([0.0, 0.0, -self.s.gravity])
        f_body = R_wb.T @ (a_w - g_w) + np.cross(alpha, self.r) + np.cross(omega, np.cross(omega, self.r))
        return self.R_bl.T @ f_body, self.R_bl.T @ omega

    def sample(self, accel_mps2, gyro, armed=False):
        """True reading (m/s^2, rad/s, IMU axes) -> Livox units with bias and noise."""
        s = self.s
        a = np.asarray(accel_mps2, float) / s.gravity + np.asarray(s.accel_bias_g)
        g = np.asarray(gyro, float) + np.asarray(s.gyro_bias)
        a = a + self.rng.normal(0.0, s.accel_noise_std_g)
        g = g + self.rng.normal(0.0, s.gyro_noise_std)
        if s.vibration_scale > 0 and armed:
            vg, va = self.vibration()
            g = g + s.vibration_scale * vg
            a = a + s.vibration_scale * va
        return a, g

    def step(self, sim_time, R_wb, v_world, omega_body, armed=False):
        """Call every physics step with the vehicle state (world-from-body rotation, inertial
        velocity of the body origin, FLU body rate); publishes on the rate_hz grid."""
        v_world = np.asarray(v_world, float); omega_body = np.asarray(omega_body, float)
        dt = sim_time - self.prev[0] if self.prev is not None else 0.0
        f, w = self.specific_force(np.asarray(R_wb, float), v_world, omega_body, dt)
        self.prev = (sim_time, v_world, omega_body)
        if self.next_t is None:
            self.next_t = sim_time
        if sim_time + 1e-9 < self.next_t:
            return False
        a, g = self.sample(f, w, armed)
        self.publish(sim_time, a, g)
        self.next_t += self.period
        if self.next_t <= sim_time:
            self.next_t = sim_time + self.period
        return True
