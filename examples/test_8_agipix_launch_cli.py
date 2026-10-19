"""Pure CLI tests for the clean Isaac bootstrap; they never import or start Isaac Sim."""

import json
import shlex
import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("8_agipix_launch.py")


class HospitalFlight25CliTest(unittest.TestCase):
    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_help_is_pure_and_exposes_requested_overrides(self) -> None:
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--headless", "--gui", "--namespace", "--id", "--environment",
                       "--px4-dir", "--physics-dt", "--preflight", "--print-command",
                       "--chase-cam", "--no-chase-cam", "--splat-mode", "--save-splat", "--splat-depth", "--splat-out", "--leds"):
            self.assertIn(option, result.stdout)
        self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_default_preflight_matches_validated_gt_simulator(self) -> None:
        result = self.run_cli("--preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertTrue(plan["ok"], plan["errors"])
        self.assertEqual(plan["environment"], "AKW_C")
        self.assertTrue(plan["environment_asset"].endswith("akw_lvl1_climb.usd"))
        self.assertEqual(plan["vehicle"], "drone0")
        self.assertTrue(plan["headless"])
        self.assertFalse(plan["chase_cam"])
        self.assertEqual(plan["physics_dt"], 0.004)
        self.assertTrue(plan["lidar_timestamps"])
        self.assertEqual(plan["ros_domain_id"], 5)
        self.assertEqual(plan["px4_defaults"]["PX4_PARAM_EKF2_EV_CTRL"], "15")
        self.assertIn("no Python flight loop", plan["flight_loop"])
        self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_hospital_resolves_to_local_pegasus_usd_without_loading_simulator(self) -> None:
        result = self.run_cli("--preflight", "--environment", "Hospital", "--gui",
                              "--namespace", "drone", "--id", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertTrue(plan["environment_asset_exists"])
        self.assertTrue(plan["environment_asset"].endswith("Worlds/isaac_envs/hospital.usd"))
        self.assertFalse(plan["headless"])
        self.assertTrue(plan["chase_cam"])
        self.assertEqual(plan["vehicle"], "drone2")
        self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_invalid_physics_timestep_is_rejected_before_launch(self) -> None:
        result = self.run_cli("--physics-dt", "0", "--preflight")
        self.assertEqual(result.returncode, 2)
        self.assertIn("physics dt must be a finite positive number", result.stderr)

    def test_forest_is_available_in_headless_and_gui_plans(self) -> None:
        cases = [(env, flags) for env in ("FOREST_SNOW", "FOREST_MID_SNOW", "FOREST_PINE_HILLS")
                 for flags in (("--headless",), ("--gui", "--splat-mode"))]
        for env, flags in cases:
            with self.subTest(environment=env, flags=flags):
                result = self.run_cli("--environment", env, *flags, "--print-command")
                self.assertEqual(result.returncode, 0, result.stderr)
                plan_text, command_text = result.stdout.split("\nLaunch command (cleaned environment):\n")
                plan = json.loads(plan_text)
                command = shlex.split(command_text.splitlines()[0])
                self.assertEqual(plan["environment"], env)
                self.assertTrue(plan["environment_asset"].endswith(f"isaac_envs/{env.lower()}.usd"))
                self.assertTrue(plan["environment_asset_exists"])
                self.assertEqual(command[command.index("--environment") + 1], env)
                self.assertEqual(plan["chase_cam"], "--gui" in flags)
                self.assertEqual(plan["physics_dt"], 0.004)
                self.assertTrue(plan["lidar_timestamps"])
                self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_chase_camera_switches_reach_the_canonical_script(self) -> None:
        for options, enabled, headless, disabled_flag in (
            ((), False, True, False),
            (("--gui",), True, False, False),
            (("--gui", "--no-chase-cam"), False, False, True),
            (("--gui", "--no-chase-cam", "--chase-cam"), True, False, False),
            (("--headless", "--chase-cam"), False, True, False),
        ):
            with self.subTest(options=options):
                result = self.run_cli(*options, "--print-command")
                self.assertEqual(result.returncode, 0, result.stderr)
                plan_text, command_text = result.stdout.split("\nLaunch command (cleaned environment):\n")
                plan = json.loads(plan_text)
                command = shlex.split(command_text.splitlines()[0])
                self.assertEqual(plan["chase_cam"], enabled)
                self.assertEqual("--headless" in command, headless)
                self.assertEqual("--no-chase-cam" in command, disabled_flag)

    def test_splat_mode_defaults_off_and_does_not_change_the_command(self) -> None:
        result = self.run_cli("--print-command")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("splat", result.stdout)

    def test_leds_default_off_and_reach_canonical_script_when_enabled(self) -> None:
        for flags in ((), ("--leds",), ("--gui", "--leds"), ("--splat-mode", "--leds")):
            with self.subTest(flags=flags):
                result = self.run_cli(*flags, "--print-command")
                self.assertEqual(result.returncode, 0, result.stderr)
                plan_text, command_text = result.stdout.split("\nLaunch command (cleaned environment):\n")
                plan = json.loads(plan_text)
                command = shlex.split(command_text.splitlines()[0])
                enabled = "--leds" in flags
                self.assertEqual(plan.get("leds", False), enabled)
                self.assertEqual("--leds" in command, enabled)
                if not enabled:
                    self.assertNotIn("leds", plan)
                self.assertEqual(plan["physics_dt"], 0.004)
                self.assertTrue(plan["lidar_timestamps"])
                self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_splat_options_reach_the_canonical_script_without_starting_isaac(self) -> None:
        result = self.run_cli("--splat-mode", "--save-splat", "--splat-depth", "--splat-out", "/tmp/splat data",
                              "--environment", "FOREST_FI", "--print-command")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan_text, command_text = result.stdout.split("\nLaunch command (cleaned environment):\n")
        plan = json.loads(plan_text)
        command = shlex.split(command_text.splitlines()[0])
        self.assertTrue(plan["splat_mode"])
        self.assertTrue(plan["save_splat"])
        self.assertTrue(plan["splat_depth"])
        self.assertEqual(plan["splat_out"], "/tmp/splat data")
        self.assertIn("--splat-mode", command)
        self.assertIn("--save-splat", command)
        self.assertIn("--splat-depth", command)
        self.assertEqual(command[command.index("--splat-out") + 1], "/tmp/splat data")
        self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_splat_preview_defaults_to_no_dataset(self) -> None:
        result = self.run_cli("--splat-mode", "--print-command")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan_text, command_text = result.stdout.split("\nLaunch command (cleaned environment):\n")
        plan = json.loads(plan_text)
        command = shlex.split(command_text.splitlines()[0])
        self.assertFalse(plan["save_splat"])
        self.assertFalse(plan["splat_depth"])
        self.assertIsNone(plan["splat_out"])
        self.assertNotIn("--save-splat", command)
        self.assertNotIn("--splat-out", command)

    def test_splat_saved_default_directory_and_depth(self) -> None:
        result = self.run_cli("--splat-mode", "--save-splat", "--preflight")
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["splat_depth"])
        self.assertEqual(plan["splat_out"], "~/agipix_splat/AKW_C_<timestamp>")

    def test_splat_dependent_options_require_the_mode(self) -> None:
        for options in (("--save-splat",), ("--splat-depth",), ("--splat-out", "/tmp/data")):
            with self.subTest(options=options):
                result = self.run_cli(*options, "--preflight")
                self.assertEqual(result.returncode, 2)
                self.assertIn("require --splat-mode", result.stderr)

    def test_dataset_options_require_explicit_save(self) -> None:
        for options in (("--splat-depth",), ("--splat-out", "/tmp/data")):
            with self.subTest(options=options):
                result = self.run_cli("--splat-mode", *options, "--preflight")
                self.assertEqual(result.returncode, 2)
                self.assertIn("require --save-splat", result.stderr)


if __name__ == "__main__":
    unittest.main()
