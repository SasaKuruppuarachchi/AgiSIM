"""Pure CLI tests for the clean Isaac bootstrap; they never import or start Isaac Sim."""

import json
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
                       "--px4-dir", "--physics-dt", "--preflight", "--print-command"):
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
        self.assertEqual(plan["vehicle"], "drone2")
        self.assertNotIn("AGIPIX_ENVIRONMENT_LOADED", result.stdout)

    def test_invalid_physics_timestep_is_rejected_before_launch(self) -> None:
        result = self.run_cli("--physics-dt", "0", "--preflight")
        self.assertEqual(result.returncode, 2)
        self.assertIn("physics dt must be a finite positive number", result.stderr)


if __name__ == "__main__":
    unittest.main()
