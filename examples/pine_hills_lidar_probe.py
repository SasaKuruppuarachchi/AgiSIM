"""Diagnostic GMO intensity reader for an existing Agipix lidar render product.

Isaac 6.0 APIs verified in sensors.experimental.rtx/impl/utils.py, its GMO
type stubs/test_lidar_sensor.py, and Replicator's Annotator implementation.
No sensor, render product, publisher, channel setting or topic is created.
Use only during evidence capture, never in the production app or RTF benchmark.

Integration: probe = LidarScalarProbe(app); call probe.sample() after each
simulation_app.update(), probe.observe_cloud(message) in the existing ROS cloud
callback, probe.summary() for the report, and probe.close() in finally.
"""
import time

import numpy as np


class _ScalarStats:
    def __init__(self):
        self.count = self.finite = self.negative = self.zero = 0
        self.total = 0.0
        self.minimum = self.maximum = None

    def add(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        self.count += len(values)
        values = values[np.isfinite(values)]
        if not len(values):
            return
        self.finite += len(values)
        self.negative += int(np.count_nonzero(values < 0))
        self.zero += int(np.count_nonzero(values == 0))
        self.total += float(values.sum())
        lo, hi = float(values.min()), float(values.max())
        self.minimum = lo if self.minimum is None else min(self.minimum, lo)
        self.maximum = hi if self.maximum is None else max(self.maximum, hi)

    def summary(self):
        return dict(elements=self.count, finite=self.finite,
                    nonfinite=self.count - self.finite, negative=self.negative, zero=self.zero,
                    minimum=self.minimum, maximum=self.maximum,
                    mean=self.total / self.finite if self.finite else None,
                    all_finite=bool(self.count and self.count == self.finite),
                    all_nonnegative=bool(self.finite and not self.negative),
                    nonconstant=bool(self.finite and self.maximum - self.minimum >
                                     max(1e-8, abs(self.maximum) * 1e-6)))


class LidarScalarProbe:
    """Lazily attach after Pegasus creates its normal ROS lidar render product."""
    def __init__(self, app):
        sensors = [sensor for sensor in app.drone._graphical_sensors if sensor.sensor_type == "Lidar"]
        if len(sensors) != 1:
            raise ValueError("Intensity probe requires exactly one existing lidar")
        self.stage = app.stage
        self.sensor_path = sensors[0]._stage_prim_path
        self.product = None
        self.annotator = None
        self._parse = None
        self._retry_at = 0.0
        self._seen = set()
        self._all = _ScalarStats()
        self._returns = _ScalarStats()
        self._range_stats = {label: _ScalarStats() for label in ("0_5", "5_10", "10_20", "20_50", "50_plus")}
        self.ros_fields = []
        self.ros_clouds = self.empty_updates = 0
        self.errors = []
        self.coords_types = set()
        self.aux_before = self._aux_channels()

    def _aux_channels(self):
        attr = self.stage.GetPrimAtPath(self.sensor_path).GetAttribute(
            "_replicator:rendervar:GenericModelOutput:channels")
        value = attr.Get() if attr else None
        return list(value) if value is not None else None

    def observe_cloud(self, message):
        """Read the existing ROS message's schema without changing publication."""
        self.ros_clouds += 1
        if not self.ros_fields:
            self.ros_fields = [dict(name=field.name, offset=int(field.offset),
                                    datatype=int(field.datatype), count=int(field.count))
                               for field in message.fields]

    def _attach_if_ready(self):
        if self.annotator is not None:
            return True
        if time.monotonic() < self._retry_at:
            return False
        self._retry_at = time.monotonic() + .25
        from pxr import UsdRender
        products = [str(prim.GetPath()) for prim in self.stage.Traverse()
                    if prim.IsA(UsdRender.Product)
                    and self.sensor_path in map(str, UsdRender.Product(prim).GetCameraRel().GetTargets())]
        if not products:
            return False  # The production ROS backend attaches on its first update.
        if len(products) != 1:
            raise RuntimeError(f"Ambiguous existing lidar render products: {products}")
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import parse_generic_model_output_data
        self.product = products[0]
        self._parse = parse_generic_model_output_data
        self.annotator = rep.annotators.get("GenericModelOutput")
        # A path attaches a CPU reader to this product; never create/toggle a
        # HydraTexture or request FULL auxiliary sensor channels.
        self.annotator.attach(self.product)
        return False  # Its data is available after a subsequent renderer update.

    def sample(self):
        """Read basic scalar output once per new GMO frame; retain no point data."""
        try:
            if not self._attach_if_ready():
                return
            raw = self.annotator.get_data(device="cpu", do_array_copy=True)
            if isinstance(raw, dict):
                raw = raw.get("data")
            if raw is None or (isinstance(raw, np.ndarray) and not raw.size):
                self.empty_updates += 1
                return
            gmo = self._parse(raw)
            count = int(gmo.numElements)
            if not count:
                self.empty_updates += 1
                return
            key = (int(gmo.timestampNs), int(gmo.frameId))
            if key in self._seen:
                return
            values = np.asarray(gmo.scalar).reshape(-1)
            if len(values) != count:
                raise RuntimeError(f"GMO scalar length {len(values)} differs from numElements={count}")
            self._seen.add(key)
            self._all.add(values)
            coords = np.stack((gmo.x, gmo.y, gmo.z), axis=1)
            kind = int(gmo.elementsCoordsType)
            self.coords_types.add(kind)
            if kind not in (0, 1):
                raise RuntimeError(f"Unsupported GMO coordinate type {kind}")
            # Installed CoordsType: CARTESIAN=0; SPHERICAL=1 has range in z.
            ranges = np.linalg.norm(coords, axis=1) if kind == 0 else coords[:, 2]
            hits = np.isfinite(coords).all(axis=1) & np.isfinite(ranges) & (ranges > 0)
            self._returns.add(values[hits])
            for label, lo, hi in (("0_5", 0, 5), ("5_10", 5, 10), ("10_20", 10, 20),
                                  ("20_50", 20, 50), ("50_plus", 50, float("inf"))):
                self._range_stats[label].add(values[hits & (ranges >= lo) & (ranges < hi)])
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if message not in self.errors and len(self.errors) < 5:
                self.errors.append(message)

    def summary(self):
        import carb.settings
        settings = carb.settings.get_settings()
        names = [field["name"] for field in self.ros_fields]
        result = self._returns.summary()
        return dict(status="sampled" if result["elements"] else "unavailable",
                    sensor_path=self.sensor_path, existing_render_product=self.product,
                    annotator="GenericModelOutput", scalar_source="basic GMO scalar (lidar intensity)",
                    unique_frames=len(self._seen), empty_updates=self.empty_updates,
                    ros_clouds=self.ros_clouds, ros_field_names=names, ros_fields=self.ros_fields,
                    ros_has_intensity=any(name in names for name in ("intensity", "reflectivity")),
                    scalar_all_elements=self._all.summary(), scalar_finite_range_returns=result,
                    scalar_by_range_m={label: stats.summary() for label, stats in self._range_stats.items()},
                    coords_types=sorted(self.coords_types), errors=self.errors[:],
                    aux_channels_before=self.aux_before, aux_channels_after=self._aux_channels(),
                    nonvisual_prefix=settings.get("/rtx/materialDb/nonVisualMaterialSemantics/prefix"),
                    nonvisual_csv_enabled=settings.get("/rtx/materialDb/nonVisualMaterialCSV/enabled"),
                    limitation="Observed scalar variation is not proof of calibrated material reflectance or "
                               "material-specific causality. No material IDs requested and no sensor/publisher changes.")

    def close(self):
        if self.annotator is not None:
            self.annotator.detach(self.product)
            self.annotator = None
