"""Offline material contracts: no Kit or GPU; preserve terrain/collision/instances."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "forest_fi"))
from forest_pine_hills.materials import author_materials, nonvisual, terrain_fields
from scene import define_material, write_mesh


def grid():
    x, y = np.meshgrid(np.linspace(-12, 12, 49), np.linspace(-12, 12, 49))
    z = .045 * x + 1.2 * np.sin(y / 7) + .7 * np.cos(x / 3)
    points = np.c_[x.ravel(), y.ravel(), z.ravel()].astype(np.float32)
    ids = np.arange(x.size).reshape(x.shape)
    a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
    return points, np.concatenate((np.c_[a, b, c], np.c_[b, d, c])).astype(np.int32)


class MaterialTest(unittest.TestCase):
    def test_ecological_weights_are_smooth_normalized_and_canopy_driven(self):
        points, _ = grid()
        a, wet_a = terrain_fields(points, [])
        b, wet_b = terrain_fields(points, [{"x": -8, "y": -8}, {"x": -7, "y": -8}])
        np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(b.sum(axis=1), 1, atol=1e-6)
        self.assertTrue(np.isfinite(b).all())
        self.assertTrue((b >= 0).all())
        near = np.hypot(points[:, 0] + 8, points[:, 1] + 8) < 2
        self.assertGreater(b[near, 1].mean(), a[near, 1].mean())
        self.assertGreater(wet_b[near].mean(), wet_a[near].mean())
        # Trail center must visibly expose soil and litter, not a moss carpet.
        trail = np.abs(points[:, 0] - .55 * points[:, 1] - .45 * np.sin(points[:, 1] / 4.5)) < .2
        self.assertLess(b[trail, 1].mean(), .15)
        self.assertGreater(b[trail, 2].mean(), .55)

    def test_usd_preserves_collision_mesh_instances_and_authors_sensor_materials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "textures"
            source.mkdir()
            output.mkdir()
            manifest = {}
            assets = {
                "forrest_ground_03": ["forrest_ground_03_diff", "forrest_ground_03_nor_gl", "forrest_ground_03_rough"],
                "forest_leaves_02": ["forest_leaves_02_diffuse", "forest_leaves_02_nor_gl", "forest_leaves_02_rough"],
                "rock_moss_set_01": ["rock_moss_set_01_diff", "rock_moss_set_01_nor_gl", "rock_moss_set_01_rough"],
                "pine_tree_01": ["pine_tree_01_bark_nor_gl", "pine_tree_01_twig_diff", "pine_tree_01_twig_nor_gl", "pine_tree_01_twig_arm"],
            }
            for asset, stems in assets.items():
                record = {"license": "CC0", "files": []}
                for stem in stems:
                    path = source / (stem + "_2k.jpg")
                    Image.new("RGB", (8, 8), (120, 160, 130)).save(path)
                    record["files"].append({"file": path.name, "md5": hashlib.md5(path.read_bytes()).hexdigest()})
                manifest[asset] = record
            (source / "polyhaven_manifest.json").write_text(json.dumps(manifest))
            stage = Usd.Stage.CreateInMemory()
            points, faces = grid()
            ground = write_mesh(stage, "/World/Terrain/ground", points, faces, points[:, :2])
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr("none")
            legacy_subsets = {}
            for index, name in enumerate(("floor_moss", "floor_needles", "floor_bilberry")):
                old_material = define_material(stage, Sdf.Path("/World/Looks/" + name),
                                               {"color": (index / 3, .2, .3)}, "textures/")
                old_indices = np.arange(index, len(faces), 3).tolist()
                old_subset = UsdGeom.Subset.CreateGeomSubset(ground, name, UsdGeom.Tokens.face,
                                                            old_indices, "materialBind", UsdGeom.Tokens.partition)
                UsdShade.MaterialBindingAPI.Apply(old_subset.GetPrim()).Bind(old_material)
                legacy_subsets[old_subset.GetPath()] = old_indices
            stage.CreateClassPrim("/World/Prototypes")
            proto = "/World/Prototypes/pine_test"
            mesh = write_mesh(stage, proto + "/foliage_0", points[:3], np.array([[0, 1, 2]]), points[:3, :2])
            material = define_material(stage, Sdf.Path(proto + "/Looks/pine_tree_01_twig"),
                                       {"diffuse": "pine_tree_01_twig_diff_2k.jpg", "normal": "pine_tree_01_twig_nor_gl_2k.jpg",
                                        "orm": "pine_tree_01_twig_arm_2k.jpg"}, "textures/")
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            tree = UsdGeom.Xform.Define(stage, "/World/Forest/Trees/pine_0000")
            instance = UsdGeom.Xform.Define(stage, tree.GetPath().AppendChild("crown")).GetPrim()
            instance.GetReferences().AddInternalReference(proto)
            instance.SetInstanceable(True)
            stand = {"trees": [{"id": "pine_0000", "x": 3, "y": 3}], "objects": []}
            before_points = np.asarray(ground.GetPointsAttr().Get()).copy()
            before_faces = np.asarray(ground.GetFaceVertexIndicesAttr().Get()).copy()
            report = author_materials(stage, stand, output, "textures/", source)
            np.testing.assert_array_equal(ground.GetPointsAttr().Get(), before_points)
            np.testing.assert_array_equal(ground.GetFaceVertexIndicesAttr().Get(), before_faces)
            self.assertTrue(ground.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
            self.assertEqual(UsdPhysics.MeshCollisionAPI(ground).GetApproximationAttr().Get(), "none")
            self.assertTrue(instance.IsInstanceable())
            proxy = stage.GetPrimAtPath(instance.GetPath().AppendChild("foliage_0"))
            self.assertTrue(proxy.IsInstanceProxy())
            self.assertFalse(mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI))
            strength = UsdGeom.PrimvarsAPI(proxy).FindPrimvarWithInheritance("pineHillsVariationStrength")
            self.assertTrue(strength)
            self.assertAlmostEqual(strength.Get(), .14)
            self.assertFalse(tree.GetPrim().GetAttribute("primvars:pineHillsTint"))
            self.assertFalse(tree.GetPrim().GetAttribute("primvars:pineHillsWetness"))
            surface = material.GetSurfaceOutput("mdl").GetConnectedSource()[0]
            self.assertEqual(surface.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier").Get(), "Foliage")
            self.assertEqual(material.GetPrim().GetAttribute("inputs:nonvisual:base").Get(), "leaf_grass")
            self.assertEqual(material.GetPrim().GetAttribute("omni:simready:nonvisual:base").Get(), "leaf_grass")
            self.assertEqual(report["thin_walled_foliage_materials"], 1)
            self.assertEqual(report["unique_instance_color_primvars"], 0)
            self.assertEqual(report["prototype_strength_primvars"], 1)
            self.assertLess(report["ground_source_texel_mm"], 1.0)
            ground_shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/pine_hills_ground_needles/PineHillsShader"))
            self.assertAlmostEqual(ground_shader.GetInput("diffuse_gain").Get(), .65)
            self.assertEqual(report["ground_diffuse_gain"], .65)
            # Existing FI material slots must be retired, not just zeroed: RTX
            # can retain active empty slots while USD resolves the new bindings.
            # Repeat authoring to verify the new nonempty subsets reactivate.
            for repeat in range(2):
                if repeat:
                    author_materials(stage, stand, output, "textures/", source)
                for path, old_indices in legacy_subsets.items():
                    old_prim = stage.GetPrimAtPath(path)
                    self.assertTrue(old_prim)
                    self.assertFalse(old_prim.IsActive())
                    self.assertEqual(list(old_prim.GetAttribute("indices").Get()), old_indices)
                subsets = UsdGeom.Subset.GetGeomSubsets(ground, UsdGeom.Tokens.face, "materialBind")
                self.assertTrue(subsets)
                for subset in subsets:
                    self.assertGreater(len(subset.GetIndicesAttr().Get()), 0)
                    bound, _ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
                    shader = UsdShade.Shader(bound.GetSurfaceOutput("mdl").GetConnectedSource()[0])
                    self.assertEqual(shader.GetSourceAssetSubIdentifier("mdl"), "Ground")
                indices = np.concatenate([np.asarray(s.GetIndicesAttr().Get(), dtype=int) for s in subsets])
                np.testing.assert_array_equal(np.sort(indices), np.arange(len(faces)))

    def test_rejects_unknown_nonvisual_preset(self):
        stage = Usd.Stage.CreateInMemory()
        material = UsdShade.Material.Define(stage, "/Material")
        with self.assertRaises(ValueError):
            nonvisual(material, "made_up_moss")


if __name__ == "__main__":
    unittest.main()
