"""Reversible RTX settings for FOREST_PINE_HILLS only (Isaac Sim 6.0).

The settings are deliberately not saved in a USD layer or Kit preferences.
Create after loading this environment, call ``apply()``, pin camera exposure
once after the cameras are created, and call ``restore()`` on shutdown or before
loading another environment. There is no frame/physics callback.

``AGIPIX_PINE_RENDER_PRESET`` selects controlled validation trials; production
defaults to ``uncached_haze``. It has no effect on any other environment.
"""
from copy import deepcopy
import os

ENVIRONMENT = "FOREST_PINE_HILLS"
DEFAULT_PRESET = "uncached_haze"
# The controlled ISO trial exposed the shaded forest without auto-exposure:
# ISO1600 (+4 stops from100) reduced near-black pixels from54.3% to0.37%.
# RTX consumes exposure:iso; generic USD exposure compensation was ignored in
# the measured Isaac 6.0 renderer path. All sensors use this same fixed gain.
EXPOSURE = dict(iso=1600.0, time=1 / 50, f_stop=5.0, responsivity=1.0, compensation=0.0)

# All paths verified in Isaac 6.0's omni.rtx.settings.core 0.7.1 widget sources
# and isaacsim.simulation_app/simulation_app.py. RT2 has physical occlusion/GI;
# the separate /rtx/ambientOcclusion controls belong to the legacy RT renderer.
BASE_SETTINGS = {
    "/rtx/rendermode": "RealTimePathTracing",
    "/rtx/rtpt/maxBounces": 3,
    "/rtx/rtpt/maxSpecularAndTransmissionBounces": 3,
    "/rtx/rtpt/cached/enabled": True,
    "/rtx/rtpt/lightcache/cached/enabled": True,
    "/rtx/post/histogram/enabled": False,
    "/rtx/post/tonemap/op": 6,  # ACES; identical fixed operator for all cameras
    "/rtx/post/tonemap/filmIso": EXPOSURE["iso"],
    "/rtx/post/tonemap/exposureTime": EXPOSURE["time"],
    "/rtx/post/tonemap/fNumber": EXPOSURE["f_stop"],
    "/rtx/post/tonemap/responsivity": EXPOSURE["responsivity"],
    "/rtx/raytracing/globalVolumetricEffects/enabled": False,
    "/rtx/fog/enabled": False,
}

PRESETS = {
    "realtime": {},
    "gi5": {"/rtx/rtpt/maxBounces": 5},
    "uncached": {"/rtx/rtpt/maxBounces": 5, "/rtx/rtpt/cached/enabled": False},
    "volumetric": {
        "/rtx/raytracing/globalVolumetricEffects/enabled": True,
        # These volume distances use renderer centimetres, as the local UI
        # explicitly documents: a 30 m atmosphere and 120 m frustum depth.
        "/rtx/raytracing/inscattering/atmosphereHeight": 3000.0,
        "/rtx/pathtracing/ptvol/fogHeightFallOff": 600.0,
        "/rtx/raytracing/inscattering/maxDistance": 12000.0,
        "/rtx/raytracing/inscattering/densityMult": 1.0,
        "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": 10000.0,
        "/rtx/raytracing/inscattering/transmittanceColor": [0.80, 0.83, 0.87],
        "/rtx/raytracing/inscattering/singleScatteringAlbedo": [0.92, 0.92, 0.92],
        "/rtx/raytracing/inscattering/anisotropyFactor": 0.55,
        "/rtx/raytracing/inscattering/useDetailNoise": False,
        "/rtx/raytracing/inscattering/depthSlices": 32,
        "/rtx/raytracing/inscattering/pixelRatio": 16,
        "/rtx/raytracing/inscattering/maxAccumulationFrames": 16,
        "/rtx/raytracing/inscattering/sliceDistributionExponent": 2.0,
    },
    "haze": {
        "/rtx/fog/enabled": True,
        "/rtx/fog/fogColor": [0.57, 0.66, 0.72],
        "/rtx/fog/fogColorIntensity": 1.0,
        "/rtx/fog/fogZup/enabled": True,
        "/rtx/fog/fogHeightDensity": 0.0,
        # Simple fog's local UI labels these distances in metres.
        "/rtx/fog/fogStartDist": 25.0,
        "/rtx/fog/fogEndDist": 150.0,
        "/rtx/fog/fogDistanceDensity": 0.12,
    },
    "pathtracing": {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 2,
        "/rtx/pathtracing/totalSpp": 64,
        "/rtx/pathtracing/maxBounces": 6,
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces": 6,
        "/rtx/pathtracing/optixDenoiser/enabled": True,
    },
}

# Installed rt_widgets.py accepts depthSlices16..1024, pixelRatio4..64 and
# accumulationFrames0..256. Change only sampling quality for this A/B preset;
# fog density, colour and 120 m distance remain identical to "volumetric".
PRESETS["volumetric_fast"] = {
    **PRESETS["volumetric"],
    "/rtx/raytracing/inscattering/depthSlices": 16,
    "/rtx/raytracing/inscattering/pixelRatio": 32,
    "/rtx/raytracing/inscattering/maxAccumulationFrames": 8,
}

# Controlled follow-up trials: iteration_07's full-density volume visibly
# veiled the near forest bed. Keep its physical parameters and cheap grid,
# changing only density; separately evaluate uncached GI with distance haze.
# These are candidates, not the production default or accepted RTF results.
PRESETS["volumetric_light"] = {
    **PRESETS["volumetric_fast"],
    "/rtx/raytracing/inscattering/densityMult": 0.25,
}
PRESETS["uncached_haze"] = {**PRESETS["uncached"], **PRESETS["haze"]}


class ScopedRenderSettings:
    """Snapshot, apply and restore only the keys this scene changes.

    The camera overrides live in their own anonymous session sublayer, so even
    pre-existing time samples or weaker authored exposure values survive restore.
    ``settings`` accepts an injected Carb-compatible object for offline testing.
    """

    def __init__(self, environment, *, settings=None, preset=None):
        self.environment = environment
        self.preset = preset or os.environ.get("AGIPIX_PINE_RENDER_PRESET", DEFAULT_PRESET)
        self.settings = settings
        self.previous = {}
        self.applied = {}
        self.active = False
        self.camera_paths = []
        self._camera_layers = []

    def apply(self):
        if self.active or self.environment != ENVIRONMENT:
            return self
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown pine render preset {self.preset!r}; choose {tuple(PRESETS)}")
        if self.settings is None:
            import carb.settings
            self.settings = carb.settings.get_settings()
        desired = {**BASE_SETTINGS, **PRESETS[self.preset]}
        self.active = True
        try:
            for key, value in desired.items():
                self.previous[key] = deepcopy(self.settings.get(key))
                self.settings.set(key, value)
                self.applied[key] = deepcopy(value)
        except Exception:
            self.restore()
            raise
        return self

    def fix_camera_exposure(self, stage, camera_paths=None):
        """Pin existing cameras after creation; safe to call again for new ones."""
        if not self.active:
            return []
        from pxr import Sdf, Usd, UsdGeom
        paths = (list(camera_paths) if camera_paths is not None else
                 [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)])
        cameras = []
        for path in paths:
            camera = UsdGeom.Camera.Get(stage, path)
            if not camera:
                raise ValueError(f"Not a USD camera: {path}")
            cameras.append(camera)
        if not cameras:
            return []
        session = stage.GetSessionLayer()
        layer = next((layer for owner, layer in self._camera_layers if owner == session), None)
        if layer is None:
            layer = Sdf.Layer.CreateAnonymous("forest_pine_hills_exposure.usda")
            session.subLayerPaths.insert(0, layer.identifier)
            self._camera_layers.append((session, layer))
        with Usd.EditContext(stage, layer):
            for camera in cameras:
                camera.CreateExposureIsoAttr(EXPOSURE["iso"])
                camera.CreateExposureTimeAttr(EXPOSURE["time"])
                camera.CreateExposureFStopAttr(EXPOSURE["f_stop"])
                camera.CreateExposureResponsivityAttr(EXPOSURE["responsivity"])
                camera.CreateExposureAttr(EXPOSURE["compensation"])
                # Global histogram settings can be overridden per camera. Use
                # the same schema/attribute as installed CameraExposurePrimHandler.
                prim = camera.GetPrim()
                prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
                prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool,
                                     custom=False).Set(False)
        self.camera_paths = sorted(set(self.camera_paths).union(paths))
        return paths

    def metadata(self):
        return dict(environment=self.environment, preset=self.preset,
                    active=self.active, settings=deepcopy(self.applied),
                    fixed_exposure=EXPOSURE.copy(), camera_paths=self.camera_paths[:],
                    exposure_selection="ISO1600, fixed +4 stops from the ISO100 trial; "
                    "scene exposure selection, not sensor radiometric calibration")

    def restore(self):
        for session, layer in self._camera_layers:
            if layer.identifier in session.subLayerPaths:
                session.subLayerPaths.remove(layer.identifier)
        self._camera_layers.clear()
        for key, value in reversed(tuple(self.previous.items())):
            if value is None:
                self.settings.destroy_item(key)
            else:
                self.settings.set(key, value)
        self.previous.clear()
        self.active = False

    def __enter__(self):
        return self.apply()

    def __exit__(self, *_):
        self.restore()
