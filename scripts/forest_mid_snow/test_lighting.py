"""HDR lighting preparation tests without Kit, GPU, or downloaded test inputs."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lighting import radiance_stats, write_ambient_hdri


class LightingTest(unittest.TestCase):
    def test_linear_hdr_roundtrip_channels_highlights_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source.hdr", Path(temp) / "derived.hdr"
            rgb = np.full((16, 32, 3), [.04, .12, .20], np.float32)
            rgb[3, 7] = [120., 50., 10.]
            cv2.imwrite(str(source), rgb[..., ::-1])
            original = source.read_bytes()
            expected_md5 = hashlib.md5(original).hexdigest()
            stats = write_ambient_hdri(source, output, ambient=.25, ambient_color=(1, .8, .6),
                                       expected_md5=expected_md5, log=lambda _: None)
            decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)[..., ::-1]
            np.testing.assert_allclose(decoded[0, 0], [.29, .32, .35], atol=.003)
            self.assertGreater(float(decoded.max()), 100, "HDR highlights must not be clipped to1")
            self.assertEqual(source.read_bytes(), original)
            self.assertLess(stats["max_relative_rgbe_quantization_error"], .008)
            with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
                write_ambient_hdri(source, output, expected_md5="bad", log=lambda _: None)
            self.assertFalse((Path(temp) / "archive").exists())

    def test_constant_radiance_has_correct_spherical_and_irradiance_means(self):
        stats = radiance_stats(np.full((16, 32, 3), 2., np.float32))
        for key in ("solid_angle_mean", "upper_hemisphere_mean", "horizon_mean", "upward_irradiance_over_pi"):
            self.assertAlmostEqual(stats[key], 2., places=6)


if __name__ == "__main__":
    unittest.main()
