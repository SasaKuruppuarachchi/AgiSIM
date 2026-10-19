"""Offline checks for non-flat contact, interpolation and matching physics proxies."""
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from scene import Terrain, TerrainZ, write_mesh, xform, world_pts
from forest_pine_hills.terrain import PineHillsTerrain, terrain_description, normal_at, trail_distance
from forest_pine_hills.scatter import (support_matrix, transform_points, canopy_gap,
    ecology_weights, _move_object, _crooked_stems, _move_tree_before_seating,
    open_trail_and_canopy, _free_footprint, frame_edge_ferns, vary_tree_age,
    seat_trail_root_edges, _open_scan_boundary)


def test_actual_triangle_slopes_relief_and_spawn_pad():
    for seed in (2027, 3, 47):
        terrain = PineHillsTerrain(np.random.default_rng(seed))
        report = terrain_description(terrain)
        assert 6 <= report["relief_m"] <= 15
        assert 25 <= report["max_triangle_slope_deg"] <= 28.50001
        x, y = np.meshgrid(np.linspace(-3, 3, 31), np.linspace(-3, 3, 31))
        inside = np.hypot(x, y) < 3.0
        assert np.max(np.abs(terrain.z_at(x[inside], y[inside]))) == 0
        assert terrain.z_at(0., 0.) == 0
        assert report["triangle_count"] == 320000


def test_height_queries_are_barycentric_on_saved_mesh_for_both_grid_sizes():
    for terrain in (PineHillsTerrain(np.random.default_rng(2027)), Terrain(np.random.default_rng(2027))):
        points, faces = terrain.mesh()
        rng = np.random.default_rng(3)
        triangles = points[faces[rng.choice(len(faces), 200)]]
        weights = rng.dirichlet([1, 1, 1], len(triangles))
        samples = np.einsum("ij,ijk->ik", weights, triangles)
        saved = TerrainZ(dict(area=100., z=terrain.Z.tolist()))
        np.testing.assert_allclose(saved.z_at(samples[:, 0], samples[:, 1]), samples[:, 2], atol=1e-10)


def test_support_orientation_and_transform_preserve_rigid_shape():
    normal = np.array([-.25, .15, 1.])
    normal /= np.linalg.norm(normal)
    matrix = support_matrix([2, 3, 5], 33., [1, 1, 1], normal)
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(matrix[2, :3], normal)
    assert np.linalg.det(matrix[:3, :3]) > .99999
    endpoints = transform_points([[-2, 0, 0], [2, 0, 0]], matrix)
    assert abs(np.linalg.norm(endpoints[1] - endpoints[0]) - 4.) < 1e-12
    assert abs(np.dot(endpoints[1] - endpoints[0], normal)) < 1e-12


def test_canopy_and_ecology_preferences_have_expected_direction():
    trees = [dict(x=0., y=0., crown_scale=[1., 1., 1.])]
    gap = canopy_gap(trees, [[0, 0], [10, 0]])
    assert gap.shape == (2,)
    assert gap[0] < gap[1]
    shade, sun = np.array([.1]), np.array([.9])
    flat, steep = np.array([0.]), np.array([.55])
    assert ecology_weights("grass", sun, flat, .5) > ecology_weights("grass", shade, flat, .5)
    assert ecology_weights("moss", shade, flat, .5) > ecology_weights("moss", sun, flat, .5)
    assert ecology_weights("fern", shade, flat, .5) > ecology_weights("fern", shade, steep, .5)
    y = np.linspace(-25, 25, 50)
    assert trail_distance(.55 * y + .45 * np.sin(y / 4.5), y).max() < 1e-14


def test_sloped_log_render_and_triangle_collider_use_same_affine_map():
    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/World/Prototypes")
    local = np.array([[-1, -.1, 0], [1, -.1, 0], [1, .1, .2], [-1, .1, .2]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_mesh(stage, "/World/Prototypes/log/deadwood_0", local, faces)
    obj = dict(id="log_0000", kind="log", x=1., y=2., z=.3, yaw=25., scale=[1., 1., 1.])
    root = UsdGeom.Xform.Define(stage, "/World/Forest/Understory/log/log_0000")
    root.GetPrim().GetReferences().AddInternalReference(Sdf.Path("/World/Prototypes/log"))
    root.GetPrim().SetInstanceable(True)
    xform(root, [obj["x"], obj["y"], obj["z"]], obj["yaw"], obj["scale"])
    old = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim()), float)
    proxy = write_mesh(stage, "/World/Forest/Colliders/Deadwood/log_0000_h0", transform_points(local, old), faces)
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
    new = support_matrix([7, -3, 2], 50., [1.2, .8, .8], normal=[-.3, .15, 1])
    _move_object(stage, obj, new)
    actual_render = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim()), float)
    np.testing.assert_allclose(transform_points(local, actual_render), proxy.GetPointsAttr().Get(), atol=1e-6)
    assert UsdPhysics.MeshCollisionAPI(proxy).GetApproximationAttr().Get() == "none"
    assert not stage.GetPrimAtPath("/World/Prototypes/log/deadwood_0").HasAPI(UsdPhysics.CollisionAPI)


def test_crooked_stem_variant_keeps_root_and_crown_join_and_moves_proxies():
    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/World/Prototypes")
    points = np.array([[0, 0, 0], [.1, 0, 0], [0, 0, 4.3], [.1, 0, 4.3], [0, 0, 9], [.1, 0, 9]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]])
    write_mesh(stage, "/World/Prototypes/pine_a_trunk_80/trunk_0", points, faces)
    tree = dict(id="pine_0", species="pine", trunk="pine_a_trunk_80", trunk_scale=[1., 1., 1.],
                yaw=0., z=0., colliders=[dict(c=[0., 0., 4.3], h=1., r=.1)])
    prim = UsdGeom.Xform.Define(stage, "/World/Forest/Trees/pine_0/trunk").GetPrim()
    prim.GetReferences().AddInternalReference(Sdf.Path("/World/Prototypes/pine_a_trunk_80"))
    class SelectAll:
        def random(self): return 0.
    result = _crooked_stems(stage, {"trees": [tree]}, SelectAll())
    changed = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Prototypes/pine_a_trunk_80_crooked/trunk_0"))
    actual = np.asarray(changed.GetPointsAttr().Get())
    np.testing.assert_allclose(actual[[0, 1, 4, 5]], points[[0, 1, 4, 5]], atol=1e-8)
    np.testing.assert_allclose(actual[2, :2] - points[2, :2], [.23, 0], atol=1e-8)
    np.testing.assert_allclose(tree["colliders"][0]["c"], [.23, 0, 4.3], atol=1e-8)
    assert result["extra_trunk_prototypes"] == 1
    assert not changed.GetPrim().HasAPI(UsdPhysics.CollisionAPI)


def test_tree_relocation_moves_visual_and_world_proxies_preserving_agl_dimensions():
    terrain = PineHillsTerrain(np.random.default_rng(2027))
    stage = Usd.Stage.CreateInMemory()
    old_xy = [-4.47, -8.33]
    base = float(terrain.z_at(*old_xy)) - .06
    tree = dict(id="pine_0037", x=old_xy[0], y=old_xy[1], z=base,
                height=21., crown_base=10., collider_top=base + 19.,
                colliders=[dict(c=[old_xy[0] + .1, old_xy[1] - .1, base + 3.], h=2., r=.2)])
    root = UsdGeom.Xform.Define(stage, "/World/Forest/Trees/pine_0037")
    xform(root, [tree["x"], tree["y"], tree["z"]], 23.)
    old_center = np.array(tree["colliders"][0]["c"])
    new_xy = [-2.48, -8.46]
    result = _move_tree_before_seating(stage, tree, new_xy, terrain)
    delta = np.array(result["to_xyz_m"]) - result["from_xyz_m"]
    np.testing.assert_allclose(np.array(tree["colliders"][0]["c"]) - old_center, delta)
    np.testing.assert_allclose(root.GetPrim().GetAttribute("xformOp:translate").Get(), result["to_xyz_m"])
    assert tree["height"] == 21. and tree["crown_base"] == 10.
    assert abs(tree["z"] - float(terrain.z_at(*new_xy)) + .06) < 1e-10
    assert abs(tree["collider_top"] - tree["z"] - 19.) < 1e-10
    assert root.GetPrim().GetAttribute("xformOp:rotateZ").Get() == 23.


def test_canopy_adjustment_and_trail_gap_preserve_stems_and_scale_z():
    terrain = PineHillsTerrain(np.random.default_rng(2027))
    stage = Usd.Stage.CreateInMemory()
    trees = []
    for index, xy in enumerate([[-4.47, -8.33], [20, 20], [-20, 20], [20, -20]]):
        base = float(terrain.z_at(*xy)) - .06
        species = "pine" if index % 2 == 0 else "spruce"
        tree = dict(id=f"tree_{index}", species=species, x=xy[0], y=xy[1], z=base,
                    crown_scale=[1.1, 1.1, 1.2], collider_top=base + 20,
                    colliders=[dict(c=[xy[0], xy[1], base + 3], h=2, r=.2)])
        root = UsdGeom.Xform.Define(stage, f"/World/Forest/Trees/tree_{index}")
        xform(root, [xy[0], xy[1], base], 17.)
        crown = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("crown"))
        xform(crown, [0, 0, 0], scale=tree["crown_scale"])
        trees.append(tree)
    report = open_trail_and_canopy(stage, {"trees": trees, "objects": []}, terrain, {})
    assert report["stems_preserved"] == 4
    assert len(report["relocated_trees"]) == 1
    assert len(trees) == 4
    assert trail_distance(trees[0]["x"], trees[0]["y"]) >= 1.8
    for tree in trees:
        expected = .78 if tree["species"] == "pine" else .70
        shoulder = tree["trail_shoulder_crown_multiplier"]
        assert .75 <= shoulder <= 1
        if trail_distance(tree["x"], tree["y"]) >= 4:
            assert shoulder == 1
        expected *= shoulder
        np.testing.assert_allclose(tree["crown_scale"], [1.1 * expected, 1.1 * expected, 1.2])
        prim = stage.GetPrimAtPath(f"/World/Forest/Trees/{tree['id']}/crown")
        np.testing.assert_allclose(prim.GetAttribute("xformOp:scale").Get(), tree["crown_scale"], atol=1e-7)


def test_foreground_footprint_rejects_pad_and_existing_hard_geometry():
    tree_xy = np.array([[6., 6.]])
    tree_radius = np.array([.3])
    hard = np.array([[8., 6., .8]])
    assert not _free_footprint([3.9, 0], .5, tree_xy, tree_radius, hard)
    assert not _free_footprint([6.5, 6], .5, tree_xy, tree_radius, hard)
    assert not _free_footprint([8.7, 6], .5, tree_xy, tree_radius, hard)
    assert _free_footprint([10, 6], .5, tree_xy, tree_radius, hard)


def test_existing_ferns_move_to_supported_trail_edges_without_count_or_scale_change():
    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/World/Prototypes")
    write_mesh(stage, "/World/Prototypes/fern/foliage_0",
               np.array([[-.1, -.1, 0], [.1, -.1, 0], [0, .1, .3]]), np.array([[0, 1, 2]]))
    objects = []
    for index in range(16):
        obj = dict(id=f"fern_{index:02d}", kind="fern", proto="fern", collider=None,
                   x=30. + index * .1, y=30., z=0., yaw=17., scale=[1., 1., 1.])
        prim = UsdGeom.Xform.Define(stage, f"/World/Forest/Understory/fern/{obj['id']}")
        prim.GetPrim().GetReferences().AddInternalReference(Sdf.Path("/World/Prototypes/fern"))
        prim.GetPrim().SetInstanceable(True)
        xform(prim, [obj["x"], obj["y"], obj["z"]], obj["yaw"], obj["scale"])
        objects.append(obj)
    terrain = PineHillsTerrain(np.random.default_rng(2027))
    stand = {"trees": [dict(x=30., y=30., dbh=.3, crown_scale=[1., 1., 1.])]}
    report = frame_edge_ferns(stage, stand, objects, terrain, {})
    assert report["total_fern_count_before"] == report["total_fern_count_after"] == 16
    assert report["new_instances"] == 0
    assert report["repositioned_count"] == 16
    for obj in objects:
        assert obj["scale"] == [1., 1., 1.]
        assert np.hypot(obj["x"], obj["y"]) > 3.8
        assert trail_distance(obj["x"], obj["y"]) >= 1.05
        assert abs(obj["z"] - terrain.z_at(obj["x"], obj["y"]) + .015) < 1e-10
        actual = np.asarray(obj["transform"])
        np.testing.assert_allclose(actual[2, :3], normal_at(terrain, obj["x"], obj["y"]), atol=1e-9)


def test_younger_cohort_scales_whole_tree_and_proxy_about_same_root():
    import copy
    stage = Usd.Stage.CreateInMemory()
    trees = []
    for index in range(10):
        tree = dict(id=f"pine_{index:04d}", species="pine", x=10. + index, y=4., z=2.,
                    trunk_scale=[.9, .9, 1.1], crown_scale=[1.05, 1.05, 1.1],
                    height=22., target_height=21., dbh=.28, target_dbh=.27, crown_base=9.,
                    prof_scale=[.9, 1.1], collider_top=23.,
                    colliders=[dict(c=[10.2 + index, 3.9, 5.], h=2., r=.14)])
        root = UsdGeom.Xform.Define(stage, f"/World/Forest/Trees/{tree['id']}")
        xform(root, [tree["x"], tree["y"], tree["z"]], 23.)
        for name in ("trunk", "crown", "skirt_low"):
            part = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild(name))
            xform(part, [0, 0, 0], scale=tree["trunk_scale"] if name == "trunk" else tree["crown_scale"])
        trees.append(tree)
    originals = copy.deepcopy(trees)
    report = vary_tree_age(stage, {"trees": trees, "summary": {"seed": 2026}}, young_fraction=.2)
    assert report["young_pine_stems"] == 2
    assert report["mature_pine_stems"] == 8
    assert report["new_prototype_meshes"] == 0
    assert report["stems_preserved"] == 10
    for old, new in zip(originals, trees):
        if "age_cohort" not in new:
            assert old == new
            continue
        factors = np.array(new["whole_tree_age_scale_xyz"])
        assert .60 <= factors[2] <= .80
        assert .75 <= factors[0] <= .90 and factors[0] == factors[1]
        for name, key in (("trunk", "trunk_scale"), ("crown", "crown_scale"), ("skirt_low", "crown_scale")):
            np.testing.assert_allclose(new[key], np.asarray(old[key]) * factors)
            prim = stage.GetPrimAtPath(f"/World/Forest/Trees/{new['id']}/{name}")
            np.testing.assert_allclose(prim.GetAttribute("xformOp:scale").Get(), new[key], atol=1e-7)
        base = np.array([old["x"], old["y"], old["z"]])
        np.testing.assert_allclose(new["colliders"][0]["c"], base + (np.array(old["colliders"][0]["c"]) - base) * factors)
        assert abs(new["colliders"][0]["r"] - old["colliders"][0]["r"] * factors[0]) < 1e-12
        assert abs(new["colliders"][0]["h"] - old["colliders"][0]["h"] * factors[2]) < 1e-12
        assert abs(new["dbh"] - old["dbh"] * factors[0]) < 1e-12
        assert abs(new["height"] - old["height"] * factors[2]) < 1e-12
        assert abs(new["crown_base"] - old["crown_base"] * factors[2]) < 1e-12
        assert abs(new["collider_top"] - (base[2] + (old["collider_top"] - base[2]) * factors[2])) < 1e-12
        root = stage.GetPrimAtPath(f"/World/Forest/Trees/{new['id']}")
        np.testing.assert_allclose(root.GetAttribute("xformOp:translate").Get(), base)


def test_younger_scaled_roots_seat_with_matching_collider_vertical_shift():
    from forest_mid_snow.scene import seat_tree_roots
    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/World/Prototypes")
    angle = np.arange(8) * 2 * np.pi / 8
    radius = np.array([.60, .52, .48, .43, .40, .32, .27, .21])
    base_ring = np.c_[np.cos(angle) * radius, np.sin(angle) * radius, [.1, .05, .12, .06, .13, .08, .1, .09]]
    upper_ring = np.c_[np.cos(angle) * .15, np.sin(angle) * .15, np.ones(8)]
    points = np.vstack((base_ring, upper_ring))
    faces = np.array([[i, (i + 1) % 8, i + 8] for i in range(8)] +
                     [[(i + 1) % 8, (i + 1) % 8 + 8, i + 8] for i in range(8)])
    write_mesh(stage, "/World/Prototypes/pine_a_trunk_80/trunk_0", points, faces)
    terrain = PineHillsTerrain(np.random.default_rng(2027))
    root_z = float(terrain.z_at(9., 4.)) - .06
    tree = dict(id="pine_root", species="pine", x=9., y=4., z=root_z, yaw=19., trunk="pine_a_trunk_80",
                crown_scale=[1., 1., 1.], trunk_scale=[1., 1., 1.], height=20., dbh=.28,
                crown_base=9., collider_top=root_z + 19., colliders=[dict(c=[9., 4., root_z + 2], h=2., r=.14)])
    root = UsdGeom.Xform.Define(stage, "/World/Forest/Trees/pine_root")
    xform(root, [9., 4., root_z], 19.)
    for name in ("crown", "trunk"):
        part = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild(name))
        xform(part, [0, 0, 0], scale=[1, 1, 1])
    stand = dict(trees=[tree], terrain=dict(area=100., z=terrain.Z.tolist()), summary={"seed": 2026})
    vary_tree_age(stage, stand, young_fraction=1.)
    old_segment = dict(tree["colliders"][0], c=list(tree["colliders"][0]["c"]))
    old_z = tree["z"]
    seat_tree_roots(stage, stand)
    sink = old_z - tree["z"]
    assert sink > 0
    np.testing.assert_allclose(tree["colliders"][0]["c"], np.array(old_segment["c"]) - [0, 0, sink])
    assert tree["colliders"][0]["r"] == old_segment["r"]
    assert tree["colliders"][0]["h"] == old_segment["h"]
    np.testing.assert_allclose(root.GetPrim().GetAttribute("xformOp:translate").Get(), [9, 4, tree["z"]])
    center = base_ring[:, :2].mean(0)
    radial = np.linalg.norm(base_ring[:, :2] - center, axis=1)
    exposed_edge = base_ring[radial > np.percentile(radial, 65)]
    world = world_pts(exposed_edge, [9, 4, tree["z"]], tree["yaw"], tree["trunk_scale"])
    assert (world[:, 2] - terrain.z_at(world[:, 0], world[:, 1])).max() <= -.035 + 1e-7


def test_root_skirt_seating_welds_uv_seams_and_preserves_raised_wood_and_instances():
    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/World/Prototypes")
    # The central raised wood vertex is duplicated per triangle, as at UV seams.
    # A non-welded boundary test would incorrectly classify its top as an edge.
    corners = np.array([[-.6, -.4, .04], [.6, -.4, .04], [.6, .4, .04], [-.6, .4, .04]])
    points = np.concatenate([np.array([corners[i], corners[(i + 1) % 4], [0., 0., .14]]) for i in range(4)])
    faces = np.arange(12).reshape(-1, 3)
    mesh = write_mesh(stage, "/World/Prototypes/roots_test/litter_0", points, faces)
    edges, _ = _open_scan_boundary(stage, "roots_test")
    assert len(edges) == 4
    assert edges[:, 2].max() < .041
    terrain = TerrainZ(dict(area=100., z=np.zeros((5, 5)).tolist()))
    prim = UsdGeom.Xform.Define(stage, "/World/Forest/Understory/trail_roots/trail_roots_test")
    prim.GetPrim().GetReferences().AddInternalReference(Sdf.Path("/World/Prototypes/roots_test"))
    prim.GetPrim().SetInstanceable(True)
    xform(prim, [6., 7., .02], 31., [1.2, .8, 1.])
    obj = dict(id="trail_roots_test", kind="trail_roots", proto="roots_test", collider=None,
               x=6., y=7., z=.02, scale=[1.2, .8, 1.], yaw=31.)
    metadata = {"placements": [dict(id=obj["id"], xyz_m=[6., 7., .02])]}
    old_points = np.array(mesh.GetPointsAttr().Get())
    before = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetPrim()))
    report = seat_trail_root_edges(stage, [obj], terrain, foreground_metadata=metadata)
    after = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetPrim()))
    sink = report["objects"][0]["extra_sink_m"]
    assert abs(sink - .062) < 1e-7
    np.testing.assert_allclose(after[:3], before[:3])
    np.testing.assert_allclose(transform_points(points, after) - transform_points(points, before),
                               np.tile([0., 0., -sink], (len(points), 1)), atol=1e-12)
    np.testing.assert_array_equal(mesh.GetPointsAttr().Get(), old_points)
    assert prim.GetPrim().IsInstance()
    assert report["instances"] == 1 and report["new_or_removed_instances"] == 0
    assert report["colliders_changed"] == 0
    assert report["objects"][0]["highest_root_above_ground_after_m"] > .09
    np.testing.assert_allclose(metadata["placements"][0]["xyz_m"], after[3, :3])
    np.testing.assert_allclose(obj["transform"], after, atol=1e-9)
    assert abs(obj["z"] - after[3, 2]) < 1e-12
    # Re-running the in-place repair must not gradually bury the root mesh.
    again = seat_trail_root_edges(stage, [obj], terrain, foreground_metadata=metadata)
    assert again["shifted_instances"] == 0
    np.testing.assert_allclose(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetPrim()), after)
