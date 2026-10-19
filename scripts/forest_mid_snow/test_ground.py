"""Offline checks for coherent snow/moss floor generation; no Kit or GPU."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("mid_snow_ground", Path(__file__).with_name("ground.py"))
ground = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ground
SPEC.loader.exec_module(ground)


class GroundTest(unittest.TestCase):
    def test_coverage_coherence_tree_exclusion_and_determinism(self):
        kwargs = dict(area=30., size=301, snow_fraction=.23, seed=34,
                      tree_xy=[(x, y) for x in range(-12, 13, 4) for y in range(-12, 13, 4)])
        a = ground.make_snow_mask(**kwargs)
        b = ground.make_snow_mask(**kwargs)
        np.testing.assert_array_equal(a.values, b.values)
        self.assertAlmostEqual(float(a.values.mean()), .23, places=5)
        self.assertTrue(np.all(a.sample(np.array(kwargs["tree_xy"])[:, 0],
                                        np.array(kwargs["tree_xy"])[:, 1]) < .02))
        # Adjacent pixels should usually share material, while both types are
        # abundant: this rejects scattered salt-and-pepper / per-face patches.
        binary = a.values > .5
        boundary = (binary[1:] != binary[:-1]).mean()
        self.assertLess(float(boundary), .06)
        self.assertGreater(float(boundary), .002)

    def test_world_axes_and_linear_colour_normal_math(self):
        values = np.array([[0., .25], [.75, 1.]], np.float32)
        mask = ground.SnowMask(values, 2., values, values)
        np.testing.assert_allclose(mask.sample([-1, 1, -1, 1], [1, 1, -1, -1]), [0, .25, .75, 1])
        # A half mixture of black and white is ~0.735 sRGB, not 0.5.
        self.assertAlmostEqual(float(ground.linear_to_srgb(.5)), .7353569, places=6)
        v = ground.normalized(np.array([[.2, -.1, .9], [0, 0, 1]], np.float32))
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1, atol=1e-6)

    def test_coverage_endpoints_and_root_relaxation(self):
        kwargs = dict(area=10., size=101, tree_xy=[(x, y) for x in range(-4, 5, 2)
                                                 for y in range(-4, 5, 2)])
        for fraction in (0., .97, 1.):
            mask = ground.make_snow_mask(snow_fraction=fraction, **kwargs)
            self.assertAlmostEqual(float(mask.values.mean()), fraction, places=5)
            self.assertTrue(np.all((mask.values >= 0) & (mask.values <= 1)))

    def test_baked_maps_preserve_world_orientation_and_unit_normals(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            maps = {}
            for name, color in [("moss", (40, 80, 20)), ("needles", (50, 30, 10)), ("snow", (180, 180, 180))]:
                maps[name] = {}
                for kind, value in [("diffuse", color), ("normal", (128, 128, 255)), ("roughness", 190)]:
                    path = root / f"{name}_{kind}.png"
                    Image.new("L" if kind == "roughness" else "RGB", (16, 16), value).save(path)
                    maps[name][kind] = path
            values = np.zeros((65, 65), np.float32)
            values[:32] = 1  # snow only at positive world Y
            mask = ground.SnowMask(values, 10, np.zeros_like(values), np.ones_like(values))
            stats = ground.bake_atlas(root, mask, maps, resolution=128, stripe=31, log=lambda _: None)
            diffuse = np.asarray(Image.open(root / stats["files"]["diffuse"]))
            self.assertGreater(float(diffuse[:32].mean()), float(diffuse[-32:].mean()) * 2)
            normal = np.asarray(Image.open(root / stats["files"]["normal"]), np.float32) / 127.5 - 1
            np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1, atol=.01)
            self.assertAlmostEqual(stats["snow_fraction"], .4921875, places=5)

    def test_usd_binding_preserves_ground_collision_and_points(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "forest_fi"))
        from scene import write_mesh
        stage = Usd.Stage.CreateInMemory()
        vertices = np.array([[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]], np.float32)
        mesh = write_mesh(stage, "/World/Terrain/ground", vertices,
                          np.array([[0, 1, 2], [0, 2, 3]], np.int32))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, "summer", "face", [0, 1], "materialBind")
        stand = dict(summary={"seed": 8}, terrain={"area": 10, "z": [[0, 0], [0, 0]]}, trees=[])
        stats = dict(files={"diffuse": "d.png", "normal": "n.png", "roughness": "r.png"},
                     snow_fraction=.23)
        with patch.object(ground, "_verified_maps", return_value={}), patch.object(ground, "bake_atlas", return_value=stats):
            result = ground.apply_patchy_ground(stage, stand, "/tmp", "./textures/", "/tmp", "/tmp",
                                                log=lambda _: None, resolution=128)
        np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), vertices)
        self.assertTrue(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), "none")
        np.testing.assert_array_equal(np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()),
                                      [[0, 0], [1, 0], [1, 1], [0, 1]])
        for prim in (mesh.GetPrim(), subset.GetPrim()):
            bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            self.assertEqual(bound.GetPath(), result.material.GetPath())
        self.assertEqual(stand["terrain"]["z"], [[0, 0], [0, 0]])


if __name__ == "__main__":
    unittest.main()
