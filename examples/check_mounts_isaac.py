#!/usr/bin/env python
"""Isaac Sim 6 cross-check of camera-bracket occlusion for the setting-H fisheyes (no PX4, no flight).

Renders the four setting-H RTX opencvFisheye cameras (agipix_splat_camera.SplatCamera, without the placeholder
/Module cube) on the agipix vehicle on an empty stage, with each input STL added as a render-only mesh under
{vehicle}/body/CheckMounts/<stem>, plus prop proxy discs. One frame is rendered with the timeline stopped
(rep.orchestrator.step(delta_time=0)); rgb + semantic_segmentation per camera give obstruction % per label.
It is the sim-pipeline counterpart of assets/agipixv2_setting_H_handover/scripts/drone_cams/check_mounts.py.

    ~/isaacsim/python.sh examples/check_mounts_isaac.py --out OUT --poses poses.json --stl brackets.stl cameras.stl

Inputs: STL in mm, drone body frame (X fwd, Y left, Z up), already placed, one label per file (file stem).
--poses only cross-checks the poses; the sim always renders from agipix_splat_geometry.MOUNTS.
Labels: {vehicle}/body -> "drone", {vehicle}/rotorN (sim prop meshes) -> "sim_rotor", STL -> stem,
proxy discs (r --prop-r at Z --prop-z on the motor axes) -> "prop_disc". Obstruction = pixels whose label is not
BACKGROUND (UNLABELLED counts and is reported separately).
Fisheye check: each segmentation silhouette is compared with (a) the RGB difference of a second render with the
CheckMounts/discs hidden and (b) the checker's ideal equidistant projection (check_mounts.render_ids) of the same
meshes; IoU close to 1 for both means the mask follows the fisheye distortion.
Near clip: RTX clips the fisheye by DISTANCE from the viewpoint (a sphere of radius --near-mm), while check_mounts.py
clips a PLANE z_cam = near_mm. A camera STL face that sits < near_mm in front of the lens plane but extends sideways
(e.g. the OV9782 barrel face 0.6 mm ahead, rho <= 4.55 mm) is dropped by the checker but rendered here as a ring at
theta >= atan(rho/z), i.e. the image periphery. Use --near-mm just past that face's farthest point (4.7 mm for the v1
camera STL) after confirming no other geometry lies within that radius in front of the lens.
Frame: the sim body prim frame is compared vertex-by-vertex with the checker's agipixv2_mesh.npz (report frame_check).
Outputs in --out: cam_<CAM>_rgb.png, cam_<CAM>_mask.png (colour per label, legend), cam_<CAM>_labels.png (uint8 id),
cam_<CAM>_compare.png (when anything is in view), report.json. Exit code 1 if any camera sees non-background pixels.
"""

import argparse
import json
import pathlib
import sys
import time

T_START = time.time()

from isaacsim import SimulationApp

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--stl", nargs="+", required=True, help="STL files, body-frame mm, label = file stem")
ap.add_argument("--out", required=True, type=pathlib.Path, help="output directory")
ap.add_argument("--poses", help="poses.json to compare against MOUNTS (lens_centre_mm, yaw/pitch/roll)")
ap.add_argument("--near-mm", type=float, default=1.2, help="camera near clip (default 1.2 mm, checker near_mm)")
ap.add_argument("--render-poses", action="store_true",
                help="render from --poses (lens_centre_mm, yaw_deg, pitch_deg) instead of agipix_splat_geometry.MOUNTS")
ap.add_argument("--body-offset-mm", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                help="added to every STL/disc point: CAD body frame -> sim body prim frame (default 0 0 0)")
ap.add_argument("--prop-r", type=float, default=63.5)
ap.add_argument("--prop-z", type=float, default=29.3)
ap.add_argument("--rotor-xy", type=float, nargs=2, default=(98.84, 131.5))
ap.add_argument("--rt-subframes", type=int, default=4)
ap.add_argument("--npz", default=None, help="handover airframe npz for the body-frame check "
                "(default: the checker's agipixv2_mesh.npz; '' to skip)")
ap.add_argument("--gui", action="store_true", help="show the Isaac window (default headless)")
args = ap.parse_args()

simulation_app = SimulationApp({"headless": not args.gui, "anti_aliasing": 2})
T_APP = time.time()

import carb
import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
import isaacsim.storage.native as nucleus
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, Vt
from scipy.spatial import cKDTree

carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", False)

from isaacsim.core.experimental.utils.semantics import add_labels
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS

HERE = pathlib.Path(__file__).resolve().parent
sys.path.append(str(HERE))
from agipix_splat_camera import SplatCamera
from agipix_splat_geometry import CX, CY, FOCAL_PX, HEIGHT, MOUNTS, WIDTH, body_to_camera_cv, rendered_camera_to_opencv

CHECKER_DIR = HERE.parent / "assets/agipixv2_setting_H_handover/scripts/drone_cams"
sys.path.insert(0, str(CHECKER_DIR))
import check_mounts as CM  # noqa: E402  (numpy/cv2/scipy only)
import fisheye_mount as FM  # noqa: E402

PALETTE = [(255, 0, 200), (0, 200, 255), (255, 120, 0), (0, 220, 120), (160, 0, 255), (255, 60, 90),
           (0, 140, 140), (200, 200, 0), (100, 60, 0), (255, 170, 220)]
FIXED = {"BACKGROUND": (0, 0, 0), "UNLABELLED": (255, 255, 255), "drone": (150, 150, 160),
         "sim_rotor": (110, 40, 40), "prop_disc": (220, 60, 60)}


def matrix(m):
    """Gf row-vector matrix -> column-vector numpy 4x4."""
    return np.array(m, dtype=np.float64).T


def define_mesh(stage, path, tris_m, colour, label):
    mesh = UsdGeom.Mesh.Define(stage, path)
    n = len(tris_m)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tris_m.reshape(-1, 3).astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(n, 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(3 * n, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*(np.asarray(colour) / 255.0))])
    add_labels(mesh.GetPrim(), labels=[label])  # no CollisionAPI / RigidBodyAPI / MassAPI: render-only
    return mesh


def resolve_label(raw, stems):
    """Replicator joins ancestor labels ("drone,brackets") and lower-cases them; the most specific known label
    wins and is returned with the STL stem's original case."""
    parts = [p.strip().lower() for p in raw.split(",")]
    for want in list(stems) + ["prop_disc", "sim_rotor", "drone", "BACKGROUND", "UNLABELLED"]:
        if want.lower() in parts:
            return want
    return raw


def body_frame_check(stage, vehicle, npz_path):
    """Sim vehicle meshes (in the body prim frame) vs the handover npz the checker uses (same prim names).
    The checker's author verified npz == CAD body frame, so a zero difference here closes the chain sim == CAD."""
    body = stage.GetPrimAtPath(vehicle + "/body")
    inv = np.linalg.inv(matrix(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
    npz = np.load(npz_path) if npz_path else {}
    out = dict(npz=str(npz_path), meshes={})
    for prim in Usd.PrimRange(stage.GetPrimAtPath(vehicle)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        p = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), np.float64)
        if not len(p):
            continue
        m = inv @ matrix(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        P = (p @ m[:3, :3].T + m[:3, 3]) * 1000.0
        rel = str(prim.GetPath())[len(vehicle) + 1:]
        r = dict(n=len(P), bbox_mm=[P.min(0).round(2).tolist(), P.max(0).round(2).tolist()])
        key = "vehicle__" + rel.replace("/", "__") + "__P"
        if key in npz:
            Q = npz[key] * 1000.0
            r["npz_n"] = len(Q)
            if len(Q) == len(P):
                r["max_abs_diff_vs_npz_mm"] = round(float(np.abs(P - Q).max()), 4)
            else:
                d, _ = cKDTree(Q).query(P)
                r["nn_diff_vs_npz_mm"] = dict(median=round(float(np.median(d)), 4), max=round(float(d.max()), 4))
        else:
            r["npz"] = "absent"
        out["meshes"][rel] = r
    return out


def log(msg):
    print(f"[check_mounts_isaac {time.time() - T_START:7.1f} s] {msg}", flush=True)


def resolve_vehicle_materials(stage, root, assets_root):
    """As 8_agipix.AgipixApp._resolve_vehicle_material_assets: legacy material URLs -> this installation."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = next((spec.default for spec in attribute.GetPropertyStack() if spec.HasInfo("default")), None)
            if not isinstance(value, Sdf.AssetPath):
                continue
            path = value.path
            if "/Isaac/5.1/Isaac/Materials/" in path:
                relative = "Isaac/Materials/" + path.split("/Isaac/5.1/Isaac/Materials/", 1)[1]
            elif path.startswith("omniverse://localhost/NVIDIA/Materials/"):
                relative = "Isaac/Materials/" + path.split("/NVIDIA/Materials/", 1)[1]
            else:
                continue
            attribute.Set(Sdf.AssetPath(assets_root.rstrip("/") + "/" + relative))


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


def main():
    global MOUNTS
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    offset_m = np.asarray(args.body_offset_mm) / 1000.0
    stems = [pathlib.Path(f).stem for f in args.stl]
    if len({x.lower() for x in stems}) != len(stems):
        raise SystemExit("STL file stems must be unique ignoring case (they are the labels)")
    report = dict(tool="check_mounts_isaac.py", command=" ".join(sys.argv), near_mm=args.near_mm,
                  body_offset_mm=list(args.body_offset_mm), stl=args.stl, cams={})

    # ---- world + vehicle (as 8_agipix.py, no environment, no backends -> no PX4/ROS) -----------------------
    log("app up")
    pg = PegasusInterface()
    pg.set_world_settings(physics_dt=1.0 / 250.0, stage_units_in_meters=1.0, rendering_dt=1.0 / 30.0)
    pg.initialize_world()
    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/CheckDome").CreateIntensityAttr(1000.0)
    config = MultirotorConfig()
    config.backends = []
    if args.render_poses and args.poses:
        from agipix_splat_geometry import CameraMount
        _pj = json.load(open(args.poses))
        MOUNTS = tuple(CameraMount(m.name, tuple(v / 1000.0 for v in _pj[m.name]["lens_centre_mm"]),
                                   float(_pj[m.name].get("yaw_deg", m.yaw_degrees)),
                                   float(_pj[m.name].get("pitch_deg", m.pitch_degrees))) for m in MOUNTS)
        log("rendering from --poses: " + ", ".join(f"{m.name} {m.position} yaw {m.yaw_degrees}" for m in MOUNTS))
    cameras = [SplatCamera(m) for m in MOUNTS]
    config.graphical_sensors = cameras
    drone = Multirotor("/World/drone0", ROBOTS["Agipix v2"], 0, [0.0, 0.0, 0.07], [0.0, 0.0, 0.0, 1.0], config=config)
    vehicle = drone.prim_path
    body_path = vehicle + "/body"
    log("vehicle spawned")
    resolve_vehicle_materials(stage, vehicle, nucleus.get_assets_root_path())
    simulation_app.update()
    log("materials resolved")

    for cam in cameras:
        stage.RemovePrim(cam.prim_path + "/Module")
        UsdGeom.Camera(stage.GetPrimAtPath(cam.prim_path)).GetClippingRangeAttr().Set(
            Gf.Vec2f(args.near_mm / 1000.0, 1000.0))

    # ---- frame check before adding anything to body ------------------------------------------------------
    body_prim = stage.GetPrimAtPath(body_path)
    T_world_body = matrix(UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    report["body_prim"] = dict(path=body_path, world_matrix=T_world_body.round(6).tolist(),
                               scale=np.linalg.norm(T_world_body[:3, :3], axis=0).round(6).tolist())
    report["frame_check"] = body_frame_check(stage, vehicle, CM.DEFAULT_MESH if args.npz is None else args.npz)
    log("frame check done")
    report["vehicle_children"] = [str(c.GetPath()) for c in stage.GetPrimAtPath(vehicle).GetChildren()]

    # ---- labels + test geometry ---------------------------------------------------------------------------
    add_labels(body_prim, labels=["drone"])
    for child in stage.GetPrimAtPath(vehicle).GetChildren():
        if child.GetName().startswith("rotor"):
            add_labels(child, labels=["sim_rotor"])
    UsdGeom.Scope.Define(stage, body_path + "/CheckMounts")
    stl_tris = {}
    for k, (f, stem) in enumerate(zip(args.stl, stems)):
        T = CM.read_stl(f) / 1000.0 + offset_m
        stl_tris[stem] = T
        define_mesh(stage, f"{body_path}/CheckMounts/{stem}", T, PALETTE[k % len(PALETTE)], stem)
        print(f"STL {stem}: {len(T)} triangles", flush=True)
    rx, ry = np.asarray(args.rotor_xy) / 1000.0
    discs = [FM._disc(sx * rx, sy * ry, args.prop_z / 1000.0, args.prop_r / 1000.0, 96) + offset_m
             for sx, sy in ((1, -1), (-1, 1), (1, 1), (-1, -1))]
    UsdGeom.Scope.Define(stage, body_path + "/PropDiscs")
    for i, D in enumerate(discs):
        define_mesh(stage, f"{body_path}/PropDiscs/disc{i}", D, FIXED["prop_disc"], "prop_disc")
    disc_tris = np.concatenate(discs)

    # ---- annotators -----------------------------------------------------------------------------------------
    annot = {}
    for cam in cameras:
        a = dict(rgb=rep.annotators.get("rgb"),
                 seg=rep.annotators.get("semantic_segmentation", init_params={"colorize": False}),
                 params=rep.annotators.get("camera_params"))
        for x in a.values():
            x.attach([cam.render_product])
        annot[cam.mount.name] = a

    log("scene built")
    t0 = time.time()
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)
    report["render_s"] = round(time.time() - t0, 2)
    frames = {}
    for cam in cameras:
        a = annot[cam.mount.name]
        seg = a["seg"].get_data()
        frames[cam.mount.name] = dict(rgb=np.asarray(a["rgb"].get_data())[..., :3].copy(),
                                      ids=np.asarray(seg["data"]).copy(), id_to_labels=seg["info"]["idToLabels"],
                                      params=a["params"].get_data())

    # second render with the test geometry hidden: RGB difference = what RGB actually shows of it
    for p in ("CheckMounts", "PropDiscs"):
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{body_path}/{p}")).MakeInvisible()
    rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)
    for cam in cameras:
        frames[cam.mount.name]["rgb_clean"] = np.asarray(annot[cam.mount.name]["rgb"].get_data())[..., :3].copy()

    log("rendered")
    # ---- poses: MOUNTS vs poses.json vs rendered -----------------------------------------------------------
    ref_poses = CM.camera_poses("H", args.poses) if args.poses else None
    pj = json.load(open(args.poses)) if args.poses else {}
    pj = pj.get("cams", pj)

    colours = dict(FIXED)
    for k, s in enumerate(stems):
        colours[s] = PALETTE[k % len(PALETTE)]
    any_fail = False
    for cam in cameras:
        n = cam.mount.name
        fr = frames[n]
        ids, rgb = fr["ids"], fr["rgb"]
        if ids.shape != (HEIGHT, WIDTH):
            raise RuntimeError(f"{n}: unexpected segmentation shape {ids.shape}")
        labmap = {int(k): resolve_label(v.get("class", ""), stems) for k, v in fr["id_to_labels"].items()}
        names = sorted(set(labmap.values()) | {"BACKGROUND"}, key=lambda s: (s != "BACKGROUND", s))
        idx = {s: i for i, s in enumerate(names)}
        lab = np.zeros(ids.shape, np.uint8)
        for k, s in labmap.items():
            lab[ids == k] = idx[s]
        unknown = ~np.isin(ids, list(labmap)) if labmap else np.ones(ids.shape, bool)
        npx = ids.size
        by = {s: round(float((lab == i).sum() / npx * 100), 4) for s, i in idx.items() if s != "BACKGROUND"}
        by = {s: v for s, v in by.items() if v > 0}
        obst = round(float(((lab != idx["BACKGROUND"]) | unknown).sum() / npx * 100), 4)
        any_fail |= obst > 0

        # pose cross-check
        T_body_cv = body_to_camera_cv(cam.mount)
        T_world_cv = rendered_camera_to_opencv(fr["params"]["cameraViewTransform"])
        T_body_cv_rendered = np.linalg.inv(T_world_body) @ T_world_cv
        local = np.linalg.inv(T_world_body) @ matrix(UsdGeom.Xformable(stage.GetPrimAtPath(cam.prim_path))
                                                     .ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        pose = dict(mounts_lens_mm=(np.asarray(cam.mount.position) * 1000).round(4).tolist(),
                    mounts_yaw_deg=cam.mount.yaw_degrees, mounts_pitch_deg=cam.mount.pitch_degrees,
                    rendered_lens_in_body_mm=(T_body_cv_rendered[:3, 3] * 1000).round(4).tolist(),
                    usd_lens_in_body_mm=(local[:3, 3] * 1000).round(4).tolist(),
                    rendered_vs_mounts_rot_deg=round(float(np.degrees(np.arccos(np.clip(
                        (np.trace(T_body_cv_rendered[:3, :3].T @ T_body_cv[:3, :3]) - 1) / 2, -1, 1)))), 5),
                    camera_model=str(fr["params"].get("cameraModel")),
                    fisheye_params=np.asarray(fr["params"].get("cameraFisheyeParams", [])).round(4).tolist(),
                    clipping_range=np.asarray(fr["params"].get("cameraNearFar", [])).tolist())
        if ref_poses:
            c_ref, R_ref = ref_poses[n]["c"], ref_poses[n]["R"]
            pose["poses_json_lens_mm"] = pj.get(n, {}).get("lens_centre_mm")
            pose["lens_diff_mm"] = ((np.asarray(cam.mount.position) - c_ref) * 1000).round(4).tolist()
            pose["rot_diff_deg_vs_checker"] = round(float(np.degrees(np.arccos(np.clip(
                (np.trace(R_ref.T @ T_body_cv[:3, :3]) - 1) / 2, -1, 1)))), 5)

        # images
        cv2.imwrite(str(out / f"cam_{n}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"cam_{n}_labels.png"), lab)
        col = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        for s, i in idx.items():
            col[lab == i] = colours.get(s, (128, 128, 128))
        y = 28
        cv2.putText(col, f"{n} Isaac seg  obstruction {obst:.3f} %", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        for s, v in by.items():
            y += 26
            cv2.rectangle(col, (10, y - 14), (26, y + 2), colours.get(s, (128, 128, 128)), -1)
            cv2.putText(col, f"{s} {v:.3f} %", (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"cam_{n}_mask.png"), cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

        # fisheye check: seg silhouette of test geometry vs RGB difference vs checker's equidistant projection
        test_ids = [idx[s] for s in idx if s in stems or s == "prop_disc"]
        seg_test = np.isin(lab, test_ids)
        diff = np.abs(rgb.astype(np.int16) - fr["rgb_clean"].astype(np.int16)).max(-1)
        rgb_test = diff > 12
        # checker frame (CAD body): STLs/discs as given, lens = MOUNTS position minus the body offset
        parts = [(s, stl_tris[s] - offset_m) for s in stems] + [("prop_disc", disc_tris - offset_m)]
        geo = CM.render_ids(parts, np.asarray(cam.mount.position) - offset_m, T_body_cv[:3, :3], 1.0,
                            args.near_mm / 1000.0)
        geo_test = geo > 0
        fish = dict(seg_px=int(seg_test.sum()), rgb_diff_px=int(rgb_test.sum()), checker_px=int(geo_test.sum()),
                    checker_pct=round(float(geo_test.mean() * 100), 4),
                    iou_seg_vs_rgbdiff=iou(seg_test, rgb_test), iou_seg_vs_checker=iou(seg_test, geo_test))
        if seg_test.any() or geo_test.any():
            for key, m in (("seg", seg_test), ("checker", geo_test)):
                ys, xs = np.nonzero(m)
                fish[f"{key}_centroid_px"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)] if len(xs) else None
            cmp_img = (rgb * 0.5).astype(np.uint8)
            cmp_img[seg_test & geo_test] = (0, 200, 0)
            cmp_img[seg_test & ~geo_test] = (255, 0, 0)
            cmp_img[~seg_test & geo_test] = (0, 80, 255)
            cs, _ = cv2.findContours(rgb_test.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(cmp_img, cs, -1, (255, 255, 0), 1)
            cv2.putText(cmp_img, f"{n}: green seg&checker, red seg only, blue checker only, yellow = RGB diff edge",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imwrite(str(out / f"cam_{n}_compare.png"), cv2.cvtColor(cmp_img, cv2.COLOR_RGB2BGR))
            fish["compare_png"] = str(out / f"cam_{n}_compare.png")
        report["cams"][n] = dict(obstruction_pct=obst, by_label_pct=by, unknown_id_px=int(unknown.sum()),
                                 raw_id_to_labels=fr["id_to_labels"], pose=pose, fisheye_check=fish,
                                 rgb_png=str(out / f"cam_{n}_rgb.png"), mask_png=str(out / f"cam_{n}_mask.png"),
                                 labels_png=str(out / f"cam_{n}_labels.png"), label_index=names)

    report["startup_s"] = round(T_APP - T_START, 1)
    report["runtime_s"] = round(time.time() - T_START, 1)
    report["all_clear"] = not any_fail
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("\ncam  obstr%   by label                                   checker%  IoU(seg,rgbdiff) IoU(seg,checker)")
    for n, r in report["cams"].items():
        f = r["fisheye_check"]
        g = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"{n:3s} {r['obstruction_pct']:7.3f}   {', '.join(f'{k} {v:.3f}' for k, v in r['by_label_pct'].items()) or '-':42s}"
              f" {f['checker_pct']:7.3f}  {g(f['iou_seg_vs_rgbdiff']):>8s}  {g(f['iou_seg_vs_checker']):>8s}")
    for k, v in report["frame_check"]["meshes"].items():
        print(f"frame check {k}: " + json.dumps({x: y for x, y in v.items() if x != 'bbox_mm'}))
    print(f"wrote {out}/report.json  (startup {report['startup_s']} s, total {report['runtime_s']} s)", flush=True)
    return 1 if any_fail else 0


if __name__ == "__main__":
    code = 2
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=code) if code else simulation_app.close()
