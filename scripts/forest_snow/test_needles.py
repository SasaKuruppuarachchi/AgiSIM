"""Offline checks for snow-only conifer alpha geometry trimming."""

import os
import sys
import unittest

import numpy as np

sys.path[:0] = [os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "..", "forest_fi")]
from needles import AlphaClipper, _opacity


class NeedleTrimmingTest(unittest.TestCase):
    def test_uv_origin_matches_usd_bottom_left(self):
        alpha = np.zeros((32, 32))
        alpha[:16] = 1
        clipper = AlphaClipper(alpha)
        np.testing.assert_array_equal(clipper.opaque(np.array([[.3, .8], [.3, .2]])), [True, False])

    def test_mask_is_cut_into_real_geometry_without_moving_support(self):
        alpha = np.zeros((64, 64))
        alpha[:, 16:48] = 1
        clipper = AlphaClipper(alpha)
        V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        F = np.array([[0, 1, 2]], np.int32)
        U = np.array([[.05, .05], [.95, .05], [.05, .95]], np.float32)
        before = _opacity(V, F, U, clipper)
        SV, SF, SU = clipper.trim(V, F, U)
        after = _opacity(SV, SF, SU, clipper)
        self.assertGreater(after["opaque_fraction"], .95)
        self.assertLess(after["surface_m2"], before["surface_m2"])
        self.assertGreater(after["surface_m2"], before["surface_m2"] * .25)
        self.assertTrue(np.all(SV >= -1e-7))
        self.assertTrue(np.all(SV[:, :2].sum(axis=1) <= 1 + 1e-7))
        self.assertTrue(np.all(SV[:, 2] == 0))
        np.testing.assert_allclose(SU, .05 + SV[:, :2] * .9, atol=1e-7)
        np.testing.assert_array_equal(V, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    def test_opaque_geometry_keeps_area_and_transparent_geometry_disappears(self):
        V = np.eye(3, dtype=np.float32)
        F = np.array([[0, 1, 2]], np.int32)
        U = np.array([[.1, .1], [.9, .1], [.1, .9]], np.float32)
        full = AlphaClipper(np.ones((32, 32)))
        SV, SF, SU = full.trim(V, F, U)
        self.assertEqual(len(SF), 1)
        self.assertAlmostEqual(_opacity(SV, SF, SU, full)["surface_m2"], np.sqrt(3) / 2)
        empty = AlphaClipper(np.zeros((32, 32)))
        self.assertEqual(len(empty.trim(V, F, U)[1]), 0)


if __name__ == "__main__":
    unittest.main()
