"""Offline USD crown/root invariants; synthetic geometry only, no Kit or GPU."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "forest_fi"))
from scene import TerrainZ, write_mesh
from forest_mid_snow.scene import open_spruce_crowns, seat_tree_roots
from forest_snow.scene import trunk_mesh


def needle_pieces(count):
    """Each separate piece has four identifiable vertices and two joined faces."""
    vertices, faces = [], []
    for index in range(count):
        x, z = index * .2, (.3, 9., 16.)[index % 3]
        vertices.extend([[x, 0, z], [x + .08, 0, z], [x + .08, .05, z], [x, .05, z]])
        faces.extend([[4 * index, 4 * index + 1, 4 * index + 2],
                      [4 * index, 4 * index + 2, 4 * index + 3]])
    vertices = np.asarray(vertices, np.float32)
    uv = np.c_[np.arange(len(vertices)) / max(len(vertices), 1), vertices[:, 2] / 20].astype(np.float32)
    return vertices, np.asarray(faces, np.int32), uv


def world_vertices(mesh):
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    return (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]


class CrownOpeningTest(unittest.TestCase):
    def test_retains_whole_needle_pieces_and_preserves_wood_uv_binding_and_instances(self):
        stage = Usd.Stage.CreateInMemory()
        stage.CreateClassPrim("/World/Prototypes")
        path = "/World/Prototypes/spruce_test_crown"
        UsdGeom.Xform.Define(stage, path)
        vertices, faces, uv = needle_pieces(96)
        foliage = write_mesh(stage, path + "/foliage_0", vertices, faces, uv)
        branch = write_mesh(stage, path + "/branch_0", vertices[:4], faces[:2], uv[:4])
        pine = write_mesh(stage, "/World/Prototypes/pine_test_crown/foliage_0", vertices, faces, uv)
        material = UsdShade.Material.Define(stage, "/World/Looks/needles")
        UsdShade.MaterialBindingAPI.Apply(foliage.GetPrim()).Bind(material)
        instance = UsdGeom.Xform.Define(stage, "/World/Forest/Trees/test/crown").GetPrim()
        instance.GetReferences().AddInternalReference(path)
        instance.SetInstanceable(True)
        branch_before = {attr.GetName(): str(attr.Get()) for attr in branch.GetPrim().GetAttributes()}
        stats = open_spruce_crowns(stage)[str(foliage.GetPath())]
        after_v = np.asarray(foliage.GetPointsAttr().Get())
        after_f = np.asarray(foliage.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        after_uv = np.asarray(UsdGeom.PrimvarsAPI(foliage).GetPrimvar("st").Get())
        # Recover original vertex IDs from geometry, independently of compact().
        lookup = {tuple(point): index for index, point in enumerate(vertices)}
        original_ids = np.array([lookup[tuple(point)] for point in after_v])
        original_faces = original_ids[after_f]
        component_ids = original_faces // 4
        self.assertTrue(np.all(component_ids == component_ids[:, :1]), "no face may join separate pieces")
        kept_components, counts = np.unique(component_ids[:, 0], return_counts=True)
        self.assertTrue(np.all(counts == 2), "each selected two-face piece must survive whole")
        expected_faces = faces[np.isin(faces[:, 0] // 4, kept_components)]
        np.testing.assert_array_equal(original_faces, expected_faces)
        np.testing.assert_array_equal(after_uv, uv[original_ids])
        self.assertGreater(stats["after"], 0)
        self.assertLess(stats["after"], stats["before"])
        self.assertEqual(stats["retained_components"], len(kept_components))
        self.assertEqual(branch_before, {attr.GetName(): str(attr.Get()) for attr in branch.GetPrim().GetAttributes()})
        np.testing.assert_array_equal(pine.GetPointsAttr().Get(), vertices)
        np.testing.assert_array_equal(pine.GetFaceVertexIndicesAttr().Get(), faces.ravel())
        bound, _ = UsdShade.MaterialBindingAPI(foliage).ComputeBoundMaterial()
        self.assertEqual(bound.GetPath(), material.GetPath())
        self.assertFalse(foliage.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        proxy = stage.GetPrimAtPath("/World/Forest/Trees/test/crown/foliage_0")
        self.assertTrue(proxy.IsInstanceProxy())
        np.testing.assert_array_equal(UsdGeom.Mesh(proxy).GetFaceVertexIndicesAttr().Get(), after_f.ravel())

    def test_sparse_low_foliage_can_be_completely_removed_without_losing_branches(self):
        stage = Usd.Stage.CreateInMemory()
        stage.CreateClassPrim("/World/Prototypes")
        path = "/World/Prototypes/spruce_sparse"
        vertices, faces, uv = needle_pieces(1)
        foliage = write_mesh(stage, path + "/foliage_0", vertices, faces, uv)
        branch = write_mesh(stage, path + "/branch_0", vertices, faces, uv)
        stats = open_spruce_crowns(stage)[str(foliage.GetPath())]
        self.assertEqual(stats["after"], 0)
        self.assertTrue(not foliage.GetPrim().IsActive() or len(foliage.GetFaceVertexCountsAttr().Get()) == 0)
        self.assertTrue(branch.GetPrim().IsActive())
        np.testing.assert_array_equal(branch.GetFaceVertexIndicesAttr().Get(), faces.ravel())


class RootSeatingTest(unittest.TestCase):
    def test_render_and_collision_shift_together_without_changing_terrain_or_clearing(self):
        stage = Usd.Stage.CreateInMemory()
        stage.CreateClassPrim("/World/Prototypes")
        name = "spruce_test_trunk_80"
        angle = np.arange(12) * np.pi / 6
        radius = np.tile([.24, .4, .65], 4)
        ring = np.c_[radius * np.cos(angle), radius * np.sin(angle), np.full(12, .02)]
        vertices = np.vstack([[0, 0, 0], ring, ring * [0.35, 0.35, 0] + [0, 0, 2]])
        faces = np.array([[0, 1 + i, 1 + (i + 1) % 12] for i in range(12)] +
                         [[1 + i, 13 + i, 13 + (i + 1) % 12] for i in range(12)], np.int32)
        trunk = write_mesh(stage, f"/World/Prototypes/{name}/trunk_0", vertices, faces)
        xs = np.arange(-6, 6.01, .5)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        grid = .12 * np.maximum(X - 3.8, 0)
        terrain = dict(area=12, grid=.5, flat_r=3.8, z=grid.tolist())
        ground_v, ground_f = TerrainZ(terrain).mesh()
        ground = write_mesh(stage, "/World/Terrain/ground", ground_v, ground_f)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr("none")
        trees, before_world, before_proxy = [], [], []
        for index, (x, y, z) in enumerate(((4.7, .3, .45), (-4.7, -.3, -.2))):
            tree = dict(id=f"spruce_{index}", trunk=name, x=x, y=y, z=z, yaw=37., trunk_scale=[1.2, .85, 1.1],
                height=9., crown_base=4., collider_top=8., colliders=[
                    dict(c=[x, y, z + 1], r=.15, h=2.), dict(c=[x, y, z + 3], r=.12, h=2.)])
            trees.append(tree)
            root = UsdGeom.Xform.Define(stage, f"/World/Forest/Trees/{tree['id']}")
            root.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            root.AddRotateZOp().Set(tree["yaw"])
            instance = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("trunk"))
            instance.GetPrim().GetReferences().AddInternalReference(f"/World/Prototypes/{name}")
            instance.GetPrim().SetInstanceable(True)
            instance.AddScaleOp().Set(Gf.Vec3f(*tree["trunk_scale"]))
            before_world.append(world_vertices(UsdGeom.Mesh(stage.GetPrimAtPath(instance.GetPath().AppendChild("trunk_0")))))
            before_proxy.append(trunk_mesh(tree["colliders"])[0])
        stand = dict(terrain=terrain, trees=trees, summary=dict(spawn=[0, 0, .07]))
        before = copy.deepcopy(stand)
        stats = seat_tree_roots(stage, stand)
        self.assertGreater(stats["max_extra_sink_m"], .3)
        for index, (tree, original) in enumerate(zip(trees, before["trees"])):
            sink = tree["extra_root_sink_m"]
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}/trunk/trunk_0"))
            after = world_vertices(mesh)
            np.testing.assert_allclose(after, before_world[index] - [0, 0, sink], atol=1e-7)
            proxy, _ = trunk_mesh(tree["colliders"])
            np.testing.assert_allclose(proxy, before_proxy[index] - [0, 0, sink], atol=1e-7)
            for segment, old in zip(tree["colliders"], original["colliders"]):
                np.testing.assert_allclose(segment["c"], np.asarray(old["c"]) - [0, 0, sink])
                self.assertEqual((segment["r"], segment["h"]), (old["r"], old["h"]))
            for field in ("z", "height", "crown_base", "collider_top"):
                self.assertAlmostEqual(tree[field], original[field] - sink)
            outer_roots = after[1:13][radius > .6]
            gap = outer_roots[:, 2] - TerrainZ(terrain).z_at(outer_roots[:, 0], outer_roots[:, 1])
            self.assertLessEqual(float(gap.max()), -.035 + 1e-7)
        self.assertEqual(trees[1]["extra_root_sink_m"], 0, "already buried roots must never be raised")
        self.assertEqual(stand["terrain"], before["terrain"])
        self.assertEqual(stand["summary"]["spawn"], before["summary"]["spawn"])
        np.testing.assert_array_equal(np.asarray(terrain["z"])[np.hypot(X, Y) <= 3.8], 0)
        np.testing.assert_array_equal(ground.GetPointsAttr().Get(), ground_v.astype(np.float32))
        np.testing.assert_array_equal(ground.GetFaceVertexIndicesAttr().Get(), ground_f.ravel())
        self.assertTrue(ground.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(ground).GetApproximationAttr().Get(), "none")
        np.testing.assert_array_equal(trunk.GetPointsAttr().Get(), vertices.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
