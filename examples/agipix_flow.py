"""ARK Flow MR emulation for AgiSIM (--flow): PAA3905 optical flow + AFBR-S50LX85D ToF rangefinder.

The real unit is a DroneCAN node: it runs PX4's paa3905, afbrs50 and iim42653 drivers and sends
com.hex.equipment.flow.Measurement (flow integral + its own gyro integral, x/y only) and
uavcan.equipment.range_sensor.Measurement. This module produces the same quantities from the sim
state and a PhysX ray cast and sends them to PX4 SITL on the HIL link as HIL_OPTICAL_FLOW and
DISTANCE_SENSOR, which simulator_mavlink turns into sensor_optical_flow and distance_sensor, the
topics the UAVCAN bridges publish on the real autopilot. Everything downstream (VehicleOpticalFlow,
EKF2) is unchanged PX4.

Conventions (PX4 msg/SensorOpticalFlow.msg, ekf2 optical_flow_fusion.cpp predictFlow): body FRD,
pixel_flow positive for a right-handed rotation of the sensor, so for a down-facing sensor at range D
(along its axis) with sensor-point velocity v (FRD) and body rate (p, q, r)
    pixel_flow_x = integral(p - v_y / D) dt,   pixel_flow_y = integral(q + v_x / D) dt,
and delta_angle = integral(p, q, r) dt (gyro compensation is done in PX4 from delta_angle, as on the
real node). distance is sent as unknown (-1): on the real drone the flow message carries no
distance and VehicleOpticalFlow takes it from the downward distance_sensor.

Modelled
- Flow: 125 Hz frames (PAA3905 bright mode is 126 fps), counts of 1 / (11.914 CPI/m * 39.37 in/m)
  rad with carried quantisation residual, white count noise growing as quality drops, a small board
  gyro bias. Closer than 80 mm (out of focus; standing on the feet is ~15 mm on the real drone) frames
  are still sent, with quality 10 and no motion: the IR LED keeps the shutter short there, so the
  driver's discard rule (SQUAL < 25 AND long shutter) does not fire. This also matters for PX4: the
  sensors module starts VehicleOpticalFlow only while disarmed and only once sensor_optical_flow has
  been published, so a sensor that is silent on the ground is never used in flight. Frames are
  discarded (nothing is sent, as the driver does) when there is no surface within 80 m (dark), the
  image rate exceeds 7.4 rad/s (datasheet limit; tracking lost), or SQUAL falls under the driver's
  bright-mode discard threshold (25) through a texture proxy: AGIPIX_FLOW_TEXTURE (0..1, default 1)
  scales SQUAL, AGIPIX_FLOW_FEATURELESS world XY boxes "xmin,ymin,xmax,ymax;..." set it to 10.
- Range: 2 deg beam as 5 rays (centre + 4 at 1 deg), median of the hits; noise
  sigma = 5 mm + 0.25 % of range, 1 mm driver resolution (1 cm on the MAVLink path). Rate follows the
  ARK board defaults (SENS_AFBR_S_RATE 25 Hz, SENS_AFBR_L_RATE 10 Hz, switch at 8 +/- 2 m, >= 1 s
  apart). Below 0.08 m the reading is still sent (PX4 rejects it with UAVCAN_RNG_MIN / min_distance,
  and EKF2 substitutes EKF2_MIN_RNG on the ground); no echo within 1.5 x 50 m gives range 0 with
  unknown quality (the driver's rule), which PX4 rejects as out of range.

Not modelled: lens blur/defocus near 80 mm, real SQUAL from scene texture or illumination (the IR
LED is assumed to keep it lit), low-light frame rate (50 fps super-low-light mode), rolling shutter,
motion blur below 7.4 rad/s, flow scale error, specular/transparent or rain/dust returns, sunlight
(the AFBR is rated to 200 klux), multipath at edges beyond the 5-ray median, CAN latency and
jitter (samples are stamped at the end of their integration), board IMU misalignment, vibration.
"""
from dataclasses import dataclass, field
import math
import os

import numpy as np

PIXART_COUNT_RAD = 1.0 / (11.914 * 39.3701)   # PX4 paa3905 driver SCALE (rad per count)
MAV_SENSOR_ROTATION_PITCH_270 = 25           # downward facing
MAV_DISTANCE_SENSOR_LASER = 0
FLU_TO_FRD = np.array([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class FlowRangeSettings:
    # Positions in the sim body frame (FLU, m) = CAD body frame (camera-mount frame check 0.0 mm;
    # the vehicle's centre of mass and Pegasus' IMU are at the origin). CAD 2026-11-21 with the
    # user's measurements: landed, the real ground is at z = -0.0914 (ToF face 10 mm above it).
    flow_position: tuple = (-0.0150, -0.0115, -0.0767)    # PAA3905 centre
    range_position: tuple = (-0.0128, 0.0099, -0.0814)    # AFBR-S50 lowest face, module centre
    # Flow (PAA3905 via the ARK node)
    flow_rate_hz: float = 125.0
    flow_noise_counts: float = 0.35          # per frame, at nominal quality
    flow_quality_nominal: int = 160          # SQUAL on a well textured floor
    flow_quality_discard: int = 25           # driver: bright mode, SQUAL < 0x19 -> discarded
    flow_quality_featureless: int = 10
    flow_quality_too_close: int = 10         # below 80 mm: published, blurred, no motion detected
    flow_min_distance: float = 0.08          # datasheet: 80 mm .. infinity
    flow_max_distance: float = 80.0          # ray length; beyond it there is nothing to track
    flow_max_rate: float = 7.4               # rad/s, datasheet
    board_gyro_bias: tuple = (0.002, -0.0015, 0.001)       # rad/s, FRD, IIM-42653 residual
    board_gyro_noise_density: float = 6.6e-5               # rad/s/sqrt(Hz) (0.0038 dps/sqrt(Hz))
    texture: float = 1.0
    featureless_boxes: tuple = ()             # ((xmin, ymin, xmax, ymax), ...) world ENU
    # Range (AFBR-S50LX85D)
    range_min: float = 0.08
    range_max: float = 50.0
    range_beam_deg: float = 2.0
    range_reported_fov_deg: float = 6.0       # what the PX4 afbrs50 driver reports for the LX85D
    range_short_hz: float = 25.0
    range_long_hz: float = 10.0
    range_switch_m: float = 8.0
    range_hysteresis_m: float = 2.0
    range_switch_min_s: float = 1.0
    range_noise_m: float = 0.005
    range_noise_rel: float = 0.0025
    # Rays start this far up the sensor axis (inside the vehicle, whose colliders are skipped) and the
    # lift is subtracted, so a sensor point that ends up inside the floor still measures (<= 0 m).
    # The sim landing gear rests the body origin 0.066 m above the floor, the real drone 0.0914 m,
    # which puts both sensor points ~11-15 mm below the sim floor while landed.
    ray_lift: float = 0.1
    seed: int = 11


def settings_from_env(base=None):
    """AGIPIX_FLOW_TEXTURE (0..1) and AGIPIX_FLOW_FEATURELESS ("xmin,ymin,xmax,ymax;...") overrides."""
    import dataclasses
    s = base or FlowRangeSettings()
    texture = float(os.environ.get("AGIPIX_FLOW_TEXTURE", s.texture))
    boxes = []
    for part in os.environ.get("AGIPIX_FLOW_FEATURELESS", "").split(";"):
        if part.strip():
            values = [float(v) for v in part.split(",")]
            if len(values) != 4:
                raise ValueError("AGIPIX_FLOW_FEATURELESS boxes need xmin,ymin,xmax,ymax")
            boxes.append(tuple(values))
    return dataclasses.replace(s, texture=texture, featureless_boxes=tuple(boxes) or s.featureless_boxes)


# PX4 parameters for this sensor. Positions are PX4 body FRD relative to the IMU/centre of mass
# (the sim's body origin); the same numbers apply on the drone if EKF2_IMU_POS_* is measured from
# the same origin. Sim-only entries are listed separately (see PX4_FLOW_PARAMS_SIM_ONLY).
PX4_FLOW_PARAMS = {
    "EKF2_OF_CTRL": "1",
    "EKF2_OF_GYR_SRC": "0",          # auto: the node's own gyro integral (x, y), PX4 gyro for z
    "EKF2_OF_POS_X": "-0.015",
    "EKF2_OF_POS_Y": "0.0115",
    "EKF2_OF_POS_Z": "0.0767",
    "EKF2_OF_QMIN": "1",
    "EKF2_OF_QMIN_GND": "0",
    "EKF2_RNG_CTRL": "1",            # conditional height; terrain (HAGL) always
    "EKF2_RNG_POS_X": "-0.0128",
    "EKF2_RNG_POS_Y": "-0.0099",
    "EKF2_RNG_POS_Z": "0.0814",
    "EKF2_RNG_A_HMAX": "5.0",
    "EKF2_RNG_A_VMAX": "1.0",
    "EKF2_RNG_QLTY_T": "1.0",
    # IMU height above the floor on the feet = the range after EKF2's lever-arm correction
    # (0.010 m reading + 0.0814 m offset; real drone, measured), substituted while landed.
    "EKF2_MIN_RNG": "0.091",
    "SENS_FLOW_ROT": "0",            # board X = body X (ARK silkscreen, FRD)
    "SENS_FLOW_MINHGT": "0.08",
    "SENS_FLOW_MAXHGT": "50.0",      # flow is focus-free; usable HAGL is set by the ToF range
    "SENS_FLOW_MAXR": "7.4",
}
# Transport delays: the sim stamps samples at the end of their integration with no CAN latency,
# so 0 is exact here. On the drone keep the PX4 defaults (20 / 5 ms) until measured from a log.
PX4_FLOW_PARAMS_SIM_ONLY = {"EKF2_OF_DELAY": "0", "EKF2_RNG_DELAY": "0"}
# Drone only (DroneCAN): UAVCAN_SUB_FLOW 1, UAVCAN_SUB_RNG 1, UAVCAN_RNG_MIN 0.08, UAVCAN_RNG_MAX 50.


def apply_px4_param_env(environ=None):
    """Export the parameter set as PX4_PARAM_* (read by the SITL rcS) unless already set."""
    environ = os.environ if environ is None else environ
    applied = {}
    for name, value in {**PX4_FLOW_PARAMS, **PX4_FLOW_PARAMS_SIM_ONLY}.items():
        key = "PX4_PARAM_" + name
        environ.setdefault(key, value)
        applied[name] = environ[key]
    return applied


class FlowRangeEmulator:
    """Call step() every physics step with the rigid-body state; it calls send_flow / send_range.

    raycast(origin_w, dir_w, max_dist) -> distance (m) of the nearest non-vehicle hit, or None.
    send_flow(dict) and send_range(dict) receive the MAVLink field values (see below).
    """

    def __init__(self, raycast, send_flow, send_range, settings=None, log_path=None):
        self.s = settings or FlowRangeSettings()
        self.raycast = raycast
        self.send_flow = send_flow
        self.send_range = send_range
        self.rng = np.random.default_rng(self.s.seed)
        self.r_flow = np.asarray(self.s.flow_position, float)
        self.r_range = np.asarray(self.s.range_position, float)
        half = math.radians(self.s.range_beam_deg / 2.0)
        self.range_dirs_b = [np.array([0.0, 0.0, -1.0])] + [
            np.array([sx * math.sin(half), sy * math.sin(half), -math.cos(half)])
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        self.t_prev = None
        self.flow_period = 1.0 / self.s.flow_rate_hz
        self.next_flow_t = None
        self.next_range_t = None
        self.range_rate = self.s.range_short_hz
        self.last_range_switch_t = -1e9
        self._reset_frame()
        self.residual = np.zeros(2)
        self.counts = {"flow_sent": 0, "flow_discarded": 0, "range_sent": 0}
        self.discard_reasons = {}
        self.log = FlowLog(log_path) if log_path else None
        self.steps = 0
        self.ground_d = None    # last flow-ray distance, persistent across frames (chooses the cast)
        self.cost_s = 0.0   # accumulated by the caller (wall time spent in step())

    def _reset_frame(self):
        self.acc_flow = np.zeros(2)
        self.acc_gyro = np.zeros(3)
        self.acc_dt = 0.0
        self.max_rate = 0.0
        self.min_d = math.inf
        self.last_d = None
        self.hit_xy = None

    # ---- geometry -----------------------------------------------------------------------------
    def flow_rates(self, R_wb, v_w, omega_b, distance):
        """True PX4-convention image rate (rad/s, x/y) and FRD body rate for a down-facing sensor."""
        v_sensor_b = R_wb.T @ np.asarray(v_w, float) + np.cross(omega_b, self.r_flow)
        v = v_sensor_b * FLU_TO_FRD
        w = np.asarray(omega_b, float) * FLU_TO_FRD
        return np.array([w[0] - v[1] / distance, w[1] + v[0] / distance]), w

    def _cast(self, origin, direction, max_distance):
        # In flight cast from the sensor point (below the vehicle's collider, so the closest hit is the
        # ground: one cheap query). Near the ground or after a miss use the lifted cast, which also
        # measures a sensor point that is inside the floor.
        if self.ground_d is not None and self.ground_d > 0.05:
            d = self.raycast(origin, direction, max_distance)
            if d is not None:
                return d
        lift = self.s.ray_lift
        d = self.raycast(np.asarray(origin) - np.asarray(direction) * lift, direction, max_distance + lift)
        return None if d is None else d - lift

    def quality_at(self, hit_xy):
        q = self.s.flow_quality_nominal * max(0.0, min(1.0, self.s.texture))
        if hit_xy is not None:
            for xmin, ymin, xmax, ymax in self.s.featureless_boxes:
                if xmin <= hit_xy[0] <= xmax and ymin <= hit_xy[1] <= ymax:
                    q = min(q, self.s.flow_quality_featureless)
        return int(round(q))

    # ---- main loop ----------------------------------------------------------------------------
    def step(self, t, R_wb, p_w, v_w, omega_b):
        R_wb = np.asarray(R_wb, float); p_w = np.asarray(p_w, float); omega_b = np.asarray(omega_b, float)
        self.steps += 1
        if self.t_prev is None:
            self.t_prev = t
            self.next_flow_t = t + self.flow_period
            self.next_range_t = t
            return
        dt = t - self.t_prev
        self.t_prev = t
        if dt <= 0.0:
            return

        down_w = R_wb @ np.array([0.0, 0.0, -1.0])
        origin = p_w + R_wb @ self.r_flow
        d = self._cast(origin, down_w, self.s.flow_max_distance)
        w_frd = omega_b * FLU_TO_FRD
        if d is not None and d > 1e-3:
            rate, _ = self.flow_rates(R_wb, v_w, omega_b, d)
            self.acc_flow += rate * dt
            self.max_rate = max(self.max_rate, float(np.linalg.norm(rate)))
            self.min_d = min(self.min_d, d)
            self.hit_xy = (origin + down_w * d)[:2]
        else:
            self.min_d = min(self.min_d, d if d is not None else math.inf)
        self.last_d = d
        self.ground_d = d
        noise = self.s.board_gyro_noise_density / math.sqrt(dt) * self.rng.standard_normal(3)
        self.acc_gyro += (w_frd + np.asarray(self.s.board_gyro_bias) + noise) * dt
        self.acc_dt += dt

        if t >= self.next_flow_t - 1e-9:
            self.next_flow_t += self.flow_period
            if self.next_flow_t < t:
                self.next_flow_t = t + self.flow_period
            self._emit_flow(t)
        if t >= self.next_range_t - 1e-9:
            self._emit_range(t, R_wb, p_w)
            self.next_range_t = t + 1.0 / self.range_rate
        if self.log is not None:
            self.log.gt(t, p_w, v_w, R_wb, omega_b, d)
            if self.log.due_flush(t):
                self.log.flush(dict(t=t, steps=self.steps, cost_us_per_step=1e6 * self.cost_s / max(1, self.steps),
                                    counts=self.counts, discarded=self.discard_reasons))

    def _emit_flow(self, t):
        reason = None
        if not math.isfinite(self.min_d) or self.last_d is None:
            reason = "no_surface"
        elif self.min_d < self.s.flow_min_distance:
            self._emit_too_close(t)
            return
        elif self.max_rate > self.s.flow_max_rate:
            reason = "rate_limit"
        quality = self.quality_at(self.hit_xy) if reason is None else 0
        if reason is None and quality < self.s.flow_quality_discard:
            reason = "low_quality"
        if reason is not None:
            self.counts["flow_discarded"] += 1
            self.discard_reasons[reason] = self.discard_reasons.get(reason, 0) + 1
            self.residual[:] = 0.0
            if self.log is not None:
                self.log.flow(t, self.acc_dt, None, None, 0, reason)
            self._reset_frame()
            return
        sigma = self.s.flow_noise_counts * self.s.flow_quality_nominal / max(quality, 1)
        counts_f = self.acc_flow / PIXART_COUNT_RAD + self.residual + sigma * self.rng.standard_normal(2)
        counts = np.round(counts_f)
        self.residual = counts_f - counts
        flow = counts * PIXART_COUNT_RAD
        msg = dict(integration_time_us=int(round(self.acc_dt * 1e6)),
                   integrated_x=float(flow[0]), integrated_y=float(flow[1]),
                   integrated_xgyro=float(self.acc_gyro[0]), integrated_ygyro=float(self.acc_gyro[1]),
                   integrated_zgyro=float(self.acc_gyro[2]), quality=int(min(255, quality)), distance=-1.0)
        self.send_flow(msg)
        self.counts["flow_sent"] += 1
        if self.log is not None:
            self.log.flow(t, self.acc_dt, self.acc_flow.copy(), flow, quality, "ok", self.acc_gyro.copy())
        self._reset_frame()

    def _emit_too_close(self, t):
        self.residual[:] = 0.0
        msg = dict(integration_time_us=int(round(self.acc_dt * 1e6)), integrated_x=0.0, integrated_y=0.0,
                   integrated_xgyro=float(self.acc_gyro[0]), integrated_ygyro=float(self.acc_gyro[1]),
                   integrated_zgyro=float(self.acc_gyro[2]), quality=int(self.s.flow_quality_too_close), distance=-1.0)
        self.send_flow(msg)
        self.counts["flow_sent"] += 1
        self.counts["flow_too_close"] = self.counts.get("flow_too_close", 0) + 1
        if self.log is not None:
            self.log.flow(t, self.acc_dt, self.acc_flow.copy(), (0.0, 0.0), self.s.flow_quality_too_close,
                          "too_close", self.acc_gyro.copy())
        self._reset_frame()

    def _emit_range(self, t, R_wb, p_w):
        origin = p_w + R_wb @ self.r_range
        hits = []
        for d_b in self.range_dirs_b:
            d = self._cast(origin, R_wb @ d_b, self.s.range_max * 1.5)
            if d is not None:
                hits.append(d)
        true_d = None
        if len(hits) >= 3:
            true_d = float(np.median(hits))
        elif hits:
            true_d = float(min(hits))
        if true_d is None:
            measured, valid = 0.0, False            # driver: no usable echo -> 0 m, quality 0
        else:
            sigma = self.s.range_noise_m + self.s.range_noise_rel * true_d
            measured = max(0.0, round(true_d + sigma * self.rng.standard_normal(), 3))
            valid = True
            if measured > self.s.range_max * 1.5:
                measured, valid = 0.0, False
        self._update_range_mode(t, measured if valid else None)
        sigma_cm = 100.0 * (self.s.range_noise_m + self.s.range_noise_rel * (true_d or 0.0))
        msg = dict(min_distance=int(round(self.s.range_min * 100)), max_distance=int(round(self.s.range_max * 100)),
                   current_distance=int(min(65535, round(measured * 100))),
                   covariance=int(min(254, round(sigma_cm ** 2))),
                   fov=math.radians(self.s.range_reported_fov_deg),
                   # MAVLink 1..100 -> uORB 0..100; 0 -> unknown (-1), what the real DroneCAN path gives
                   # for any non-VALID_RANGE reading.
                   signal_quality=100 if valid else 0)
        self.send_range(msg)
        self.counts["range_sent"] += 1
        if self.log is not None:
            self.log.range(t, true_d, measured, valid, self.range_rate)

    def _update_range_mode(self, t, distance):
        if distance is None or distance <= 0 or t - self.last_range_switch_t < self.s.range_switch_min_s:
            return
        if distance >= self.s.range_switch_m + self.s.range_hysteresis_m and self.range_rate != self.s.range_long_hz:
            self.range_rate, self.last_range_switch_t = self.s.range_long_hz, t
        elif distance <= self.s.range_switch_m - self.s.range_hysteresis_m and self.range_rate != self.s.range_short_hz:
            self.range_rate, self.last_range_switch_t = self.s.range_short_hz, t

    def close(self):
        if self.log is not None:
            self.log.close()


class FlowLog:
    """Ground-truth and sensor CSVs for offline evaluation (AGIPIX_FLOW_LOG=<prefix>)."""

    def __init__(self, prefix, gt_rate_hz=50.0):
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        self.f_gt = open(prefix + "_gt.csv", "w", buffering=1 << 16)
        self.f_flow = open(prefix + "_flow.csv", "w", buffering=1 << 16)
        self.f_rng = open(prefix + "_range.csv", "w", buffering=1 << 16)
        self.f_gt.write("t,px,py,pz,vx,vy,vz,qx,qy,qz,qw,wx,wy,wz,d_flow\n")
        self.f_flow.write("t,dt,true_x,true_y,meas_x,meas_y,quality,status,gyro_x,gyro_y,gyro_z\n")
        self.f_rng.write("t,true_d,measured,valid,rate_hz\n")
        self.gt_period = 1.0 / gt_rate_hz
        self.next_gt = -1.0
        self.prefix = prefix
        self.next_flush = 0.0

    def due_flush(self, t, period=2.0):
        # Kit may end the process without running Python cleanup: keep the files current.
        if t < self.next_flush:
            return False
        self.next_flush = t + period
        return True

    def flush(self, status):
        import json
        for f in (self.f_gt, self.f_flow, self.f_rng):
            f.flush()
        with open(self.prefix + "_status.json", "w") as f:
            json.dump(status, f)

    def gt(self, t, p, v, R, w, d):
        if t < self.next_gt:
            return
        self.next_gt = t + self.gt_period
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()
        vals = [t, *p, *v, *q, *w, d if d is not None else float("nan")]
        self.f_gt.write(",".join(f"{x:.6f}" for x in vals) + "\n")

    def flow(self, t, dt, true, meas, quality, status, gyro=None):
        nan = float("nan")
        true = true if true is not None else (nan, nan)
        meas = meas if meas is not None else (nan, nan)
        gyro = gyro if gyro is not None else (nan, nan, nan)
        self.f_flow.write(f"{t:.6f},{dt:.6f},{true[0]:.6f},{true[1]:.6f},{meas[0]:.6f},{meas[1]:.6f},"
                          f"{quality},{status},{gyro[0]:.6f},{gyro[1]:.6f},{gyro[2]:.6f}\n")

    def range(self, t, true_d, measured, valid, rate):
        td = true_d if true_d is not None else float("nan")
        self.f_rng.write(f"{t:.6f},{td:.4f},{measured:.4f},{int(valid)},{rate:.0f}\n")

    def close(self):
        for f in (self.f_gt, self.f_flow, self.f_rng):
            f.close()
