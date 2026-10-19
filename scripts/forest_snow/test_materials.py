"""Offline snow geometry checks (Isaac Python/pxr; no Kit or simulator)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from materials import snow_caps, winterize_prototypes


def plane(x0=0.0, x1=0.7, step=0.025, z=0.0):
    xs = np.arange(x0, x1 + step * 0.1, step)
    ys = np.arange(0, 0.7 + step * 0.1, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
    ind = np.arange(X.size).reshape(X.shape)
    a, b = ind[:-1, :-1].ravel(), ind[1:, :-1].ravel()
    c, d = ind[:-1, 1:].ravel(), ind[1:, 1:].ravel()
    return V, np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])


class SnowCapsTest(unittest.TestCase):
    def test_upward_support_closed_thickness_and_no_input_mutation(self):
        V, F = plane()
        before_v, before_f = V.copy(), F.copy()
        S, T = snow_caps(V, F, cell=0.1, thickness=0.08)
        self.assertGreater(len(T), 0)
        self.assertLessEqual(float(S[:, 2].max()), 0.08)
        self.assertGreater(float(S[:, 2].max()), 0.06)
        self.assertGreaterEqual(float(S[:, 2].min()), -0.00601)
        edges = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2), "cap must have closed side walls")
        np.testing.assert_array_equal(V, before_v)
        np.testing.assert_array_equal(F, before_f)
        # Every snow vertex sits directly above an actual source triangle centre.
        centers = V[F].mean(axis=1)
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(centers[:, :2]).query(S[:, :2])
        self.assertLess(float(distance.max()), 1e-6)

    def test_no_cap_on_downward_solid_or_unsupported_sparse_triangles(self):
        V, F = plane()
        self.assertEqual(len(snow_caps(V, F[:, ::-1])[1]), 0)
        self.assertEqual(len(snow_caps(V, F[:1])[1]), 0)
        # Double-sided needles have an upper surface regardless of mesh winding.
        self.assertGreater(len(snow_caps(V, F[:, ::-1], foliage=True)[1]), 0)

    def test_empty_gaps_and_separate_height_levels_remain_open(self):
        A, AF = plane(0.0, 0.7)
        B, BF = plane(1.3, 2.0, z=0.0)
        C, CF = plane(0.0, 0.7, z=1.0)
        V = np.concatenate([A, B, C])
        F = np.concatenate([AF, BF + len(A), CF + len(A) + len(B)])
        S, T = snow_caps(V, F, cell=0.1)
        self.assertGreater(len(T), 0)
        self.assertLess(float(np.ptp(S[T, 0], axis=1).max()), 0.3)
        self.assertLess(float(np.ptp(S[T, 2], axis=1).max()), 0.2)
        self.assertFalse(np.any((S[:, 0] > 0.7) & (S[:, 0] < 1.3)))

    def test_usd_binding_instances_and_no_physics_on_caps(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "forest_fi"))
        from scene import write_mesh
        stage = Usd.Stage.CreateInMemory()
        stage.CreateClassPrim("/World/Prototypes")
        proto = UsdGeom.Xform.Define(stage, "/World/Prototypes/pine_test_crown")
        V, F = plane()
        source = write_mesh(stage, proto.GetPath().AppendChild("foliage_0"), V, F)
        green = UsdShade.Material.Define(stage, "/World/Looks/pine_twig")
        UsdShade.MaterialBindingAPI.Apply(source.GetPrim()).Bind(green)
        snow = UsdShade.Material.Define(stage, "/World/Looks/snow")
        instance = UsdGeom.Xform.Define(stage, "/World/Forest/test").GetPrim()
        instance.GetReferences().AddInternalReference(proto.GetPath())
        instance.SetInstanceable(True)
        stats = winterize_prototypes(stage, snow, lambda _: None)
        self.assertEqual(stats["snow_meshes"], 1)
        cap = stage.GetPrimAtPath("/World/Prototypes/pine_test_crown/snow_0")
        self.assertTrue(cap)
        self.assertFalse(cap.HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(cap.HasAPI(UsdPhysics.RigidBodyAPI))
        bound, _ = UsdShade.MaterialBindingAPI(cap).ComputeBoundMaterial()
        self.assertEqual(bound.GetPath(), snow.GetPath())
        self.assertTrue(stage.GetPrimAtPath("/World/Forest/test/snow_0").IsInstanceProxy())
        self.assertTrue(UsdGeom.PrimvarsAPI(cap).GetPrimvar("st"))


if __name__ == "__main__":
    unittest.main()
