"""Geometry checks without Isaac; USD construction checks when pxr is available."""

import math
import unittest

import numpy as np

from agipix_leds import BULB_RADIUS, MOTOR_X, MOTOR_Y, RING_RADIUS, add_navigation_lights, mounts
from agipix_splat_geometry import MOUNTS, body_to_camera_cv, project_equidistant


class NavigationLightsTest(unittest.TestCase):
    def test_bulbs_are_on_outward_ring_diagonals(self):
        for name, position, direction, color in mounts():
            sx, sy = (1 if position[0] > 0 else -1), (1 if position[1] > 0 else -1)
            offset = np.array(position[:2]) - [sx * MOTOR_X, sy * MOTOR_Y]
            self.assertAlmostEqual(np.linalg.norm(offset) - BULB_RADIUS, RING_RADIUS)
            np.testing.assert_allclose(offset / np.linalg.norm(offset), direction[:2])
            self.assertEqual(color, (1, 1, 1) if name.startswith("F") else (1, 0, 0))

    def test_entire_bulb_surfaces_clear_every_fisheye(self):
        # Dense deterministic surface sample includes the bulb silhouettes;
        # this supplements live instance segmentation rather than replacing it.
        theta = np.linspace(0, 2 * math.pi, 361)
        phi = np.linspace(0, math.pi, 181)
        tt, pp = np.meshgrid(theta, phi)
        surface = np.stack([np.cos(tt)*np.sin(pp), np.sin(tt)*np.sin(pp), np.cos(pp)], axis=-1)
        surface = surface.reshape(-1, 3) * BULB_RADIUS
        for camera in MOUNTS:
            t = body_to_camera_cv(camera)
            for name, position, _, _ in mounts():
                q = (surface + position - t[:3, 3]) @ t[:3, :3]
                uv = project_equidistant(q)
                seen = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1280) & (uv[:, 1] >= 0) & (uv[:, 1] < 800)
                self.assertFalse(seen.any(), (camera.name, name))

    def test_usd_rigid_attachment_and_no_physics(self):
        try:
            from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        except ImportError:
            self.skipTest("Requires the installed Isaac USD Python libraries")
        stage = Usd.Stage.CreateInMemory()
        body = UsdGeom.Xform.Define(stage, "/World/drone0/body")
        body.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        body.AddRotateXYZOp().Set(Gf.Vec3f(20, -15, 70))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        root = add_navigation_lights(stage, str(body.GetPath()))
        transform = body.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for name, position, direction, _ in mounts():
            bulb = UsdGeom.Sphere(stage.GetPrimAtPath(root + "/" + name + "/Bulb"))
            world = bulb.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            np.testing.assert_allclose(world.ExtractTranslation(), transform.Transform(Gf.Vec3d(*position)))
            light = UsdLux.SphereLight(stage.GetPrimAtPath(root + "/" + name + "/Light"))
            np.testing.assert_allclose(world.TransformDir(Gf.Vec3d(0, 0, -1)),
                                       transform.TransformDir(Gf.Vec3d(*direction)), atol=1e-6)
            self.assertTrue(light.GetNormalizeAttr().Get())
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for api in (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
                self.assertFalse(prim.HasAPI(api), prim.GetPath())


if __name__ == "__main__":
    unittest.main()
