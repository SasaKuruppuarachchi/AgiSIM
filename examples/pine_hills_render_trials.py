"""Low-resolution, reversible PINE_HILLS comparisons in one owned simulator.

Called by validate_forest_snow --pine-render-trials. All production sensors stay
enabled. An extra 580x326 comparison product makes these diagnostic timings,
not the acceptance benchmark; final RTF uses bench_agipix_splat_rtf instead.
"""
import json
import os
from pathlib import Path
import time

TRIALS = {
    "baseline": ("realtime", "full"),
    "no_detail": ("realtime", "no_detail"),
    "opaque_foliage": ("realtime", "opaque_foliage"),
    "no_sheen": ("realtime", "no_sheen"),
    "gi5": ("gi5", "full"),
    "uncached": ("uncached", "full"),
    "volumetric": ("volumetric", "full"),
    "volumetric_fast": ("volumetric_fast", "full"),
    "volumetric_light": ("volumetric_light", "full"),
    "uncached_haze": ("uncached_haze", "full"),
    "haze": ("haze", "full"),
    "pathtracing": ("pathtracing", "full"),
    "camera_neutral": ("realtime", "camera_neutral"),
    "post_neutral": ("realtime", "full"),
    "closed_canopy": ("realtime", "closed_canopy"),
    "exposure2": ("realtime", "exposure2"),
    "exposure4": ("realtime", "exposure4"),
    "exposure6": ("realtime", "exposure6"),
    "no_variation": ("realtime", "no_variation"),
    "ground_gain_one": ("uncached_haze", "ground_gain_one"),
    "ground_gain_low": ("uncached_haze", "ground_gain_low"),
    "ground_single": ("uncached_haze", "ground_single"),
    "final_no_detail": ("uncached_haze", "no_detail"),
    "final_opaque_foliage": ("uncached_haze", "opaque_foliage"),
    "final_no_sheen": ("uncached_haze", "no_sheen"),
    "final_no_variation": ("uncached_haze", "no_variation"),
}


def material_trial(stage, mode):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    layer = Sdf.Layer.CreateAnonymous("pine_material_trial.usda")
    stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
    with Usd.EditContext(stage, layer):
        if mode == "ground_single":
            ground = UsdGeom.Mesh.Get(stage, "/World/layout/Terrain/ground")
            for subset in UsdGeom.Subset.GetAllGeomSubsets(ground):
                subset.GetPrim().SetActive(False)
        if mode == "camera_neutral" or mode.startswith("exposure"):
            for p in stage.Traverse():
                if p.IsA(UsdGeom.Camera):
                    camera = UsdGeom.Camera(p)
                    if mode == "camera_neutral":
                        camera.CreateExposureTimeAttr(1.)
                        camera.CreateExposureFStopAttr(1.)
                    else:
                        # RTX 6's camera API consumes ISO/time/fStop, but not
                        # UsdGeom's generic exposure-compensation attribute.
                        camera.CreateExposureIsoAttr(100. * 2. ** float(mode[len("exposure"):]))
        if mode == "closed_canopy":
            from pxr import Gf
            trees = stage.GetPrimAtPath("/World/layout/Forest/Trees")
            for tree in trees.GetChildren():
                factor = .78 if tree.GetName().startswith("pine_") else .70
                for child in tree.GetChildren():
                    if child.GetName() == "crown" or child.GetName().startswith("skirt_"):
                        attr = child.GetAttribute("xformOp:scale")
                        scale = attr.Get()
                        attr.Set(Gf.Vec3d(scale[0]/factor, scale[1]/factor, scale[2]))
        if mode == "no_variation":
            for proto in stage.GetPrimAtPath("/World/layout/Prototypes").GetAllChildren():
                attr = proto.GetAttribute("primvars:pineHillsVariationStrength")
                if attr:
                    attr.Set(0.)
        for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.PrimAllPrimsPredicate):
            if not prim.IsA(UsdShade.Material) or prim.IsInstanceProxy():
                continue
            material = UsdShade.Material(prim)
            shader = UsdShade.Shader(stage.GetPrimAtPath(prim.GetPath().AppendChild("PineHillsShader")))
            if not shader:
                continue
            if mode in ("ground_gain_one", "ground_gain_low") and prim.GetName().startswith("pine_hills_ground_"):
                shader.CreateInput("diffuse_gain", Sdf.ValueTypeNames.Float).Set(1. if mode == "ground_gain_one" else .01)
            elif mode == "no_detail":
                if shader.GetInput("detail_enabled"):
                    shader.GetInput("detail_enabled").Set(False)
                if shader.GetInput("detail_weight"):
                    shader.GetInput("detail_weight").Set(0.)
            elif mode == "opaque_foliage" and shader.GetInput("translucency"):
                shader.GetInput("translucency").Set(0.)
            elif mode == "no_sheen" and shader.GetInput("sheen"):
                shader.GetInput("sheen").Set(0.)
    return layer


def run_trials(module, app, annotator, config, eye, target):
    import numpy as np
    from PIL import Image, ImageDraw
    from forest_pine_hills.runtime import ScopedRenderSettings

    output = Path(config["output"])
    reference = Image.open(output.parents[0] / "reference.png") if (output.parents[0] / "reference.png").is_file() else Image.open(Path(__file__).resolve().parents[1] / "artifacts/forest_pine_hills/reference.png")
    reference = reference.convert("RGB").resize((580, 326))
    names = os.environ.get("AGIPIX_PINE_TRIALS", ",".join(TRIALS)).split(",")
    if any(name not in TRIALS for name in names):
        raise ValueError(f"Unknown trial in {names}")
    state = dict(index=-1, scope=None, layer=None, phase_start=0., measured=None)
    reports = []
    original_update = module.simulation_app.update
    app._render_scope.restore()

    def restore():
        if state["scope"] is not None:
            state["scope"].restore()
        if state["layer"] is not None:
            session = app.stage.GetSessionLayer()
            session.subLayerPaths.remove(state["layer"].identifier)

    def next_phase(t):
        restore()
        state["index"] += 1
        if state["index"] >= len(names):
            state["scope"] = state["layer"] = None
            app.stop_sim = True
            return
        name = names[state["index"]]
        preset, material_mode = TRIALS[name]
        state["scope"] = ScopedRenderSettings(app.environment, preset=preset).apply()
        if name == "post_neutral":
            for key in ("/rtx/post/tonemap/exposureTime", "/rtx/post/tonemap/fNumber"):
                state["scope"].settings.set(key, 1.)
                state["scope"].applied[key] = 1.
        state["scope"].fix_camera_exposure(app.stage)
        state["layer"] = material_trial(app.stage, material_mode)
        # Interactive path tracing measured around0.02 RTF with all sensors.
        # A short bounded comparison is enough to reject it for production;
        # the final acceptance benchmark always uses its full60-second windows.
        warmup, duration = (.5, 1.) if name == "pathtracing" else (5., 10.)
        state.update(phase_start=t, measured=None, warmup=warmup, duration=duration)
        print("PINE_TRIAL_START=" + name, flush=True)

    def update():
        t = module.SimulationManager.get_simulation_time()
        if state["index"] < 0:
            next_phase(t)
        body = app.drone._body_rigid_prim
        if body is not None:
            body.set_world_poses(positions=[[0, 0, 2]], orientations=[[1, 0, 0, 0]])
            body.set_velocities(linear_velocities=[[0, 0, 0]], angular_velocities=[[0, 0, 0]])
        original_update()
        t = module.SimulationManager.get_simulation_time()
        elapsed = t - state["phase_start"]
        if elapsed >= state["warmup"] and state["measured"] is None:
            state["measured"] = (t, time.monotonic())
        if elapsed < state["warmup"] + state["duration"]:
            return
        end = time.monotonic()
        rgba = np.asarray(annotator.get_data())
        if not rgba.size:
            raise RuntimeError("No comparison render")
        rgb = rgba[..., :3].copy()
        name = names[state["index"]]
        rendered = Image.fromarray(rgb).resize((580, 326))
        pair = Image.new("RGB", (1160, 354), (24, 28, 28))
        pair.paste(reference, (0, 28))
        pair.paste(rendered, (580, 28))
        draw = ImageDraw.Draw(pair)
        draw.text((12, 8), "Poly Haven reference", fill="white")
        draw.text((592, 8), name + " | Isaac Sim 6.0 RTX", fill="white")
        pair.save(output / (name + "_comparison.jpg"), quality=90)
        start_sim, start_wall = state["measured"]
        from pxr import UsdGeom, UsdShade
        camera = UsdGeom.Camera.Get(app.stage, "/World/ForestSnowEvidenceCamera")
        ground_shader = UsdShade.Shader.Get(app.stage, "/World/layout/Looks/pine_hills_ground_needles/PineHillsShader")
        report = dict(name=name, sim_seconds=t-start_sim, wall_seconds=end-start_wall,
                      warmup_sim_seconds=state["warmup"], requested_sim_seconds=state["duration"],
                      diagnostic_rtf=(t-start_sim)/(end-start_wall), render=state["scope"].metadata(),
                      actual_camera_exposure=dict(iso=camera.GetExposureIsoAttr().Get(),
                          time=camera.GetExposureTimeAttr().Get(), f_stop=camera.GetExposureFStopAttr().Get()),
                      ground_shader_asset=str(ground_shader.GetSourceAsset("mdl")),
                      rgb_mean=rgb.mean(axis=(0, 1)).tolist(), black_fraction=float((rgb.max(axis=2)<5).mean()),
                      clipped_fraction=float((rgb.min(axis=2)>250).mean()))
        reports.append(report)
        (output / "report.json").write_text(json.dumps(dict(trials=reports, camera_eye=eye,
            camera_target=target, focal_mm=28, aperture_mm=36,
            note="Diagnostic sequential trials with extra 580x326 product; final acceptance benchmark is separate."), indent=2))
        print("PINE_TRIAL_RESULT=" + json.dumps(report), flush=True)
        next_phase(t)

    module.simulation_app.update = update
    try:
        app.run()
    finally:
        restore()
        module.simulation_app.update = original_update
    print("FOREST_PINE_HILLS_VALIDATION_PASSED=" + json.dumps(dict(trials=len(reports))), flush=True)
