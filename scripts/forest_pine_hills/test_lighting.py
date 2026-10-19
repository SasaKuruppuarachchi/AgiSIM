"""CPU checks for solar-energy accounting, light alignment and scoped exposure."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forest_pine_hills.lighting import build_lighting, latlong_directions, split_sun
from forest_pine_hills.runtime import BASE_SETTINGS, EXPOSURE, PRESETS, ScopedRenderSettings


class Settings:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def destroy_item(self, key):
        self.values.pop(key, None)


def solar_image():
    directions = latlong_directions(512, 1024)
    direction = directions[225, 1022]  # compact disk straddling the latlong seam
    angular = np.arccos(np.clip(directions @ direction, -1, 1))
    source = np.full((512, 1024, 3), .2, dtype=np.float32)
    source += (2000 * np.exp(-.5 * (angular / .006) ** 2))[..., None] * np.array([1, .8, .55])
    return source, direction


def test_solar_split_preserves_energy_and_handles_seam():
    source, direction = solar_image()
    sky, metadata = split_sun(source)
    extracted_direction = np.asarray(metadata["source_sun_direction_y_up"])
    assert np.degrees(np.arccos(np.clip(direction @ extracted_direction, -1, 1))) < .4
    directions = latlong_directions(*source.shape[:2])
    solid_angle = np.sqrt(1 - directions[..., 1] ** 2) * np.pi / 512 * 2 * np.pi / 1024
    projected = np.maximum(directions @ extracted_direction, 0) * solid_angle
    removed = ((source - sky) * projected[..., None]).sum(axis=(0, 1))
    np.testing.assert_allclose(removed, metadata["source_solar_normal_irradiance_rgb"], rtol=1e-5)
    assert np.all(sky <= source)
    assert np.all(sky >= 0)
    with pytest.raises(ValueError, match="no resolved solar"):
        split_sun(np.full_like(source, .2))


def test_built_sun_and_hdri_are_aligned(tmp_path):
    import cv2
    from pxr import Gf, Usd, UsdGeom, UsdLux
    source, _ = solar_image()
    hdr = tmp_path / "verified_source.hdr"
    assert cv2.imwrite(str(hdr), source[..., ::-1])
    md5 = hashlib.md5(hdr.read_bytes()).hexdigest()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    meta = build_lighting(stage, hdr, tmp_path / "textures", "textures/",
                          expected_md5=md5, sun_azimuth_deg=225)
    sky = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lights/Sky"))
    sun = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Lights/Sun"))
    local = Gf.Vec3d(*meta["source_sun_direction_y_up"])
    sky_solar_direction = sky.GetLocalTransformation().TransformDir(local)
    sun_ray = sun.GetLocalTransformation().TransformDir(Gf.Vec3d(0, 0, -1))
    np.testing.assert_allclose(sky_solar_direction, -np.asarray(sun_ray), atol=1e-7)
    assert meta["sun_diameter_deg"] == .53
    assert UsdLux.DistantLight(sun).GetNormalizeAttr().Get()
    assert len([p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)]) == 1
    assert hdr.read_bytes() and hashlib.md5(hdr.read_bytes()).hexdigest() == md5


def test_runtime_is_noop_for_other_world_and_restores_every_setting():
    settings = Settings({"/rtx/post/histogram/enabled": True, "/unrelated": 123})
    before = settings.values.copy()
    with ScopedRenderSettings("FOREST_FI", settings=settings, preset="invalid"):
        assert settings.values == before
    scope = ScopedRenderSettings("FOREST_PINE_HILLS", settings=settings, preset="volumetric").apply()
    assert settings.get("/rtx/post/histogram/enabled") is False
    assert settings.get("/rtx/raytracing/globalVolumetricEffects/enabled") is True
    scope.apply()  # idempotence must not replace the original restore snapshot
    scope.restore()
    scope.restore()
    assert settings.values == before


def test_runtime_rollback_on_partial_setting_failure():
    class FailingSettings(Settings):
        def set(self, key, value):
            if key == "/rtx/post/tonemap/op" and value == 6:
                raise RuntimeError("simulated failure")
            super().set(key, value)
    settings = FailingSettings({"/rtx/rendermode": "Other", "/rtx/post/tonemap/op": 1})
    before = settings.values.copy()
    with pytest.raises(RuntimeError, match="simulated failure"):
        ScopedRenderSettings("FOREST_PINE_HILLS", settings=settings).apply()
    assert settings.values == before


def test_fast_volume_changes_only_grid_sampling_and_restores():
    changes = {key for key, value in PRESETS["volumetric_fast"].items()
               if value != PRESETS["volumetric"][key]}
    assert changes == {"/rtx/raytracing/inscattering/depthSlices",
                       "/rtx/raytracing/inscattering/pixelRatio",
                       "/rtx/raytracing/inscattering/maxAccumulationFrames"}
    settings = Settings({"/rtx/raytracing/inscattering/depthSlices": 96})
    before = settings.values.copy()
    with ScopedRenderSettings("FOREST_PINE_HILLS", settings=settings, preset="volumetric_fast"):
        assert settings.get("/rtx/raytracing/inscattering/depthSlices") == 16
        assert settings.get("/rtx/raytracing/inscattering/pixelRatio") == 32
        assert settings.get("/rtx/raytracing/inscattering/maxAccumulationFrames") == 8
        assert settings.get("/rtx/raytracing/inscattering/maxDistance") == 12000.0
    assert settings.values == before


def test_camera_exposure_restores_authored_values_and_timesamples():
    from pxr import Sdf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/World/camera")
    camera.CreateExposureIsoAttr().Set(800.0)
    camera.GetExposureIsoAttr().Set(400.0, 1)
    camera.GetPrim().CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool,
                                    custom=False).Set(True)
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()
    settings = Settings()
    with ScopedRenderSettings("FOREST_PINE_HILLS", settings=settings) as scope:
        assert scope.fix_camera_exposure(stage) == ["/World/camera"]
        assert camera.GetExposureIsoAttr().Get() == EXPOSURE["iso"]
        assert camera.GetExposureIsoAttr().Get(1) == EXPOSURE["iso"]
        assert camera.GetExposureFStopAttr().Get() == EXPOSURE["f_stop"]
        assert camera.GetPrim().GetAttribute("omni:rtx:autoExposure:enabled").Get() is False
        scope.fix_camera_exposure(stage)
        assert len(stage.GetSessionLayer().subLayerPaths) == 1
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before
    assert camera.GetExposureIsoAttr().Get(1) == 400
    assert camera.GetPrim().GetAttribute("omni:rtx:autoExposure:enabled").Get() is True
    assert not settings.values
