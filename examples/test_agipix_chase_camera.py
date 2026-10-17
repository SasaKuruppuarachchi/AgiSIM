"""Travel-heading behavior independent of the Isaac Sim runtime."""

import math
import unittest

from agipix_chase_camera import ChaseCameraMotion, ChaseCameraSettings, INDOOR_SETTINGS, settings_for_environment


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


class ChaseCameraMotionTest(unittest.TestCase):
    def test_indoor_environments_use_the_closer_boom(self):
        self.assertEqual(settings_for_environment("FOREST_FI"), ChaseCameraSettings())
        for env in ("AKW_C", "AKW_2F", "Hospital", "Office"):
            self.assertIs(settings_for_environment(env), INDOOR_SETTINGS)
        self.assertLess(INDOOR_SETTINGS.distance, ChaseCameraSettings().distance)

    def test_spawn_heading_and_hover_takeoff_keep_level_fixed_pitch(self):
        motion = ChaseCameraMotion((10, 20, 1), math.pi / 2, 0)
        self.assertAlmostEqual(motion.position[0], 10)
        self.assertAlmostEqual(motion.position[1], 16)
        self.assertAlmostEqual(motion.position[2], 3)
        for step in range(1, 301):
            motion.update((10, 20, 1 + step / 30), step / 30)
        self.assertEqual(motion.yaw, math.pi / 2)
        self.assertEqual(motion.velocity, (0, 0))
        self.assertAlmostEqual(math.asin(-motion.forward[2]), math.radians(25))
        self.assertAlmostEqual(math.hypot(*motion.forward), 1)

    def test_slow_drift_does_not_turn_the_camera(self):
        motion = ChaseCameraMotion((0, 0, 0), 0.7, 0)
        for step in range(1, 601):
            motion.update((-0.2 * step / 30, 0, 0), step / 30)
        self.assertEqual(motion.yaw, 0.7)

    def test_travel_heading_converges_and_every_turn_is_slew_limited(self):
        motion = ChaseCameraMotion((0, 0, 0), 0, 0)
        for step in range(1, 301):
            old_yaw = motion.yaw
            motion.update((0, 2 * step / 30, 0), step / 30)
            self.assertLessEqual(abs(angle_difference(motion.yaw, old_yaw)), math.radians(60) / 30 + 1e-12)
        self.assertAlmostEqual(motion.yaw, math.pi / 2)
        self.assertAlmostEqual(motion.velocity[1], 2 * (1 - math.exp(-10 / 2.5)))
        self.assertAlmostEqual(motion.position[0], 0, places=6)
        self.assertLess(motion.position[1], 20 - 4)

    def test_yaw_wrap_uses_short_turn(self):
        settings = ChaseCameraSettings(velocity_time_constant=0.01)
        motion = ChaseCameraMotion((0, 0, 0), math.radians(179), 0, settings)
        target = math.radians(-179)
        motion.update((math.cos(target) / 30, math.sin(target) / 30, 0), 1 / 30)
        self.assertAlmostEqual(angle_difference(motion.yaw, target), 0)

    def test_stop_freezes_last_heading_after_velocity_decays(self):
        motion = ChaseCameraMotion((0, 0, 0), 0, 0)
        for step in range(1, 301):
            motion.update((0, 2 * step / 30, 0), step / 30)
        last_yaw = motion.yaw
        for step in range(301, 901):
            motion.update((0, 20, 0), step / 30)
        self.assertLess(math.hypot(*motion.velocity), motion.settings.speed_threshold)
        self.assertEqual(motion.yaw, last_yaw)

    def test_velocity_filter_uses_sim_time_at_different_render_rates(self):
        motions = []
        for hz in (15, 30, 120):
            motion = ChaseCameraMotion((0, 0, 0), 0, 0)
            for step in range(1, hz * 5 + 1):
                t = step / hz
                motion.update((t, 2 * t, 0), t)
            motions.append(motion)
        for motion in motions[1:]:
            self.assertAlmostEqual(motion.velocity[0], motions[0].velocity[0])
            self.assertAlmostEqual(motion.velocity[1], motions[0].velocity[1])
            self.assertAlmostEqual(motion.yaw, motions[0].yaw)

    def test_position_smoothing_and_custom_view_settings(self):
        settings = ChaseCameraSettings(distance=6, height=3, pitch_degrees=20,
                                       position_time_constant=0.5)
        motion = ChaseCameraMotion((0, 0, 0), 0, 0, settings)
        motion.update((0, 0, 2), 0.5)
        self.assertAlmostEqual(motion.position[0], -6)
        self.assertAlmostEqual(motion.position[2], 3 + 2 * (1 - math.exp(-1)))
        self.assertAlmostEqual(motion.forward[2], -math.sin(math.radians(20)))

    def test_pause_and_timeline_restart(self):
        motion = ChaseCameraMotion((0, 0, 0), 0.4, 0)
        motion.update((0, 5, 0), 3)
        before = (motion.position, motion.yaw, motion.velocity)
        self.assertFalse(motion.update((0, 5, 0), 3))
        self.assertEqual(before, (motion.position, motion.yaw, motion.velocity))
        self.assertTrue(motion.update((0, 0, 0), 0))
        self.assertEqual(motion.velocity, (0, 0))
        self.assertEqual(motion.yaw, 0.4)
        motion.update((0, 0, 0), 1 / 30)
        self.assertEqual(motion.velocity, (0, 0))


if __name__ == "__main__":
    unittest.main()
