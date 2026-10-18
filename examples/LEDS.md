# Agipix navigation lights (`--leds`)

```bash
python3 examples/8_agipix_launch.py --gui --leds --environment AKW_C
python3 examples/8_agipix_launch.py --gui --splat-mode --leds --environment FOREST_FI
```

`--leds` is opt-in in both the standalone script and its clean launcher. Without
it, no LED module is imported, no LED prim/material/light is created, and no
topic or render/physics-loop code is added. No sensor, collider, mass or rigid
body is added in either mode. Existing lidar mounts, publishers, physics dt,
camera calibration and dataset code are unchanged.

## Geometry

Reference: `assets/agipixv2_setting_H_handover/drone_base/agipixv2_cams/handover_H/setting_H_geometry.json`
and its handover report. The installed `agipixv2_1.usd` has metre units, an
identity body transform and rotor translations confirming motor axes at
X=±98.56 mm, Y=±131.50 mm. The guard annulus has radius 67–87.3 mm and spans
Z=28–32 mm.

The diagonal here is the radial line **from the body origin through each motor**
(front-left bearing 53.148°), not an arbitrary 45° line. Let `u` be its unit
direction. The 8 mm bulb centre is `motor + (87.3 + 4) mm * u`, at Z=30 mm:
its inward surface is tangent to the guard's outer edge.

| LED | Bulb centre in body FLU, metres | Colour | Outward unit direction |
| --- | --- | --- | --- |
| FL | (+0.15331689, +0.20455733, +0.030) | White | (+0.59974685, +0.80018980, 0) |
| FR | (+0.15331689, −0.20455733, +0.030) | White | (+0.59974685, −0.80018980, 0) |
| RL | (−0.15331689, +0.20455733, +0.030) | Red | (−0.59974685, +0.80018980, 0) |
| RR | (−0.15331689, −0.20455733, +0.030) | Red | (−0.59974685, −0.80018980, 0) |

Each lives at `<vehicle>/body/NavigationLights/{FL,FR,RL,RR}`. The parent
translation and orientation inherit the body's complete rigid motion. No
per-frame callback is needed. The analytic light centre is 8 mm farther outward
than the bulb centre: approximately (±0.15811486, ±0.21095885, 0.030) m. This
puts its emitting sphere outside the opaque bulb shell, avoiding self-shadowing.

## Light parameters and APIs

Constants live in `agipix_leds.py`:

- `UsdGeom.Sphere`: radius 0.004 m; `UsdShade.Material` / `UsdPreviewSurface`
  emission `(8,8,8)` in front or `(8,0,0)` behind, roughness 0.5.
- `UsdLux.SphereLight`: radius 0.004 m, intensity **40**, exposure 0,
  normalized power enabled, white `(1,1,1)` or red `(1,0,0)`.
- `UsdLux.ShapingAPI`: 75° cone half-angle, softness 0.3, pointed outward
  by rotating the source's local −Z to the diagonal direction.
- The analytic source has `visibleInPrimaryRay=false`; the emissive bulb is
  the visible package. Normal environment shadows remain enabled.

Range is deliberately **effective range from low power and inverse-square
falloff**, not a hard distance cutoff. Illumination is useful on nearby surfaces;
no unverified RTX distance-limit attribute or global renderer setting is used.
At 25 cm, the matte-target test changes front target RGB averages from roughly
5–7 to 34–36/255; rear targets change to roughly (35,5,4)/255. This confirms
white/red environmental illumination without saturation. These pixel values
depend on exposure and are not a photometric calibration.

APIs were checked against this installation's `pxr/UsdLux` bindings and
`omni.kit.property.light-1.0.14` implementation. The
[NVIDIA lighting reference](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/lighting.html)
documents normalized power, shaping and primary-ray visibility.

## Verification protocol

`validate_agipix_leds.py` uses the canonical launcher, checks existing
Isaac/PX4 processes, holds `/tmp/agipix-sim-benchmark.lock`, and moves only its
own unarmed drone. It records ROS lidar clouds and camera instance masks; it
saves all four fisheye samples, the panorama, an inspection view and (with
`--gui`) the actual chase viewport. `--night` dims scene lights only in that
test process; existing emissive scene fixtures remain. `--light-panels` adds
four matte targets 25 cm outside the sources. Neither affects production runs.

```bash
python3 examples/validate_agipix_leds.py --splat-mode --leds --night \
  --output /tmp/agipix-leds-final-night --validation-seconds 12
python3 examples/validate_agipix_leds.py --gui --splat-mode --leds --tilt \
  --output /tmp/agipix-leds-final-gui --validation-seconds 12
python3 examples/bench_agipix_splat_rtf.py --environments AKW_C \
  --modes off preview --repeats 2 --seconds 60 --output /tmp/agipix-leds-rtf-off
python3 examples/bench_agipix_splat_rtf.py --environments AKW_C \
  --modes off preview --repeats 2 --seconds 60 --leds --output /tmp/agipix-leds-rtf-on
```

The RTF measurements use ordinary production rendering, without diagnostic
targets, extra inspection cameras, segmentation annotators or night changes.
All modes retain 250 Hz physics, timestamped 10 Hz lidar, and (when enabled)
four full-resolution 10 Hz fisheyes. Each run has five simulated seconds of
warm-up. GPU telemetry covers the whole device.

During development, the first validator used an incorrect unprefixed lidar
frame ID; it was fixed to `drone0/lidar_link`. The initial non-normalized
intensity 1500 was too dim at this emitter size. A shadow-link experiment did
not improve it and was removed. A temporary high-power diagnostic established
that the lights rendered correctly; normalized intensity 40 with the outward
source offset is the retained setting. No production renderer settings changed.

## Final sensor and visual results

- Final 12 s night check and 12 s GUI roll/pitch/yaw check both passed. Each
  inspected 120 frames per fisheye: **zero LED pixels in every camera**. Dense
  sampled bulb-surface projections also clear all four calibrated images.
- Both final runs received 101 lidar scans after the initial two-second settling
  interval. No point was within 7 mm of a bulb centre, and **no point at all was
  within 0.4 m of the body origin** (also excluding noise-displaced LED returns).
- The matching lights-off night run had 2,674,583 points; final lights-on had
  2,673,570 across the same 101 timestamps: mean count change **−0.0379%**.
  Topic lists and timestamps match exactly. The installed Mid-360 has angular
  and range noise, so the clouds are not byte-identical. A pooled-cloud nearest
  neighbour comparison gives median 10.9 mm / p95 40.9 mm; no LED return cluster
  appears. No lidar exclusion, mount change or range filter was needed.
- All four night images were inspected. No bulb or LED glare appears; the bright
  ceiling fixtures/reflections are already present in the lights-off scene.
  Mean absolute on/off RGB differences are 0.34–0.70/255; p99 is 2–3/255.
  Rear frames have zero saturated pixels in both runs. Front saturation changes
  by only 2/8 pixels around existing bright fixtures. Nearby target illumination
  was checked separately, as described above.
- Default, GUI and splat launch plans with LEDs off compare **exactly equal** to
  the pre-change launcher (command, environment and plan). The lights-off live
  validator asserts that `NavigationLights` does not exist.

Screenshots and raw evidence (full paths):

- Chase viewport with the drone tilted: `/tmp/agipix-leds-final-gui/chase.png`.
- GUI-run close-up: `/tmp/agipix-leds-final-gui/inspection.png`.
- Night close-up: `/tmp/agipix-leds-final-night/inspection.png`.
- Four night fisheyes: `/tmp/agipix-leds-final-night/{FL,FR,RL,RR}.png`.
- Four daytime fisheyes: `/tmp/agipix-leds-final-gui/{FL,FR,RL,RR}.png`.
- Illumination targets: `/tmp/agipix-leds-panels-final/` versus
  `/tmp/agipix-leds-panels-off-01/`.
- Numerical on/off comparison: `/tmp/agipix-leds-comparison.json`.
- Each validator directory retains `report.json`, `lidar.npz` and `simulator.log`.

Tests: **42 passed, 18 subtests passed**, including CLI, LED geometry/USD,
existing splat/chase math and lifecycle tests. Host CLI tests separately pass
12 tests / 14 subtests. Preflight, default command printing, Python compilation
and `git diff --check` pass. The full suite used Isaac's Python plus its installed
USD libraries and `/tmp/agipix-test-deps` for pytest:

```bash
AGIPIX_TEST_USD=/home/sasa/isaacsim/extscache/omni.usd.libs-1.0.3+f9bf0dda.lx64.r.cp312
PYTHONPATH=/tmp/agipix-test-deps:$AGIPIX_TEST_USD LD_LIBRARY_PATH=$AGIPIX_TEST_USD/bin \
  /home/sasa/isaacsim/python.sh -m pytest examples/test_8_agipix_launch_cli.py \
  examples/test_agipix_splat.py examples/test_agipix_chase_camera.py \
  examples/test_agipix_splat_lifecycle.py examples/test_agipix_leds.py -q
```

## RTF comparison

AKW_C, both configurations headless; two 60 s repeats each after 5 s warm-up.
Means average the two run means; p5 pools one-sim-second windows. No dataset saving.

| Mode | LEDs off mean / p5 | LEDs on mean / p5 | RTF decrease | Off range | On range | GPU % off → on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal (RealSense) | 1.101 / 1.007 | 1.093 / 1.018 | 0.72% | 1.097–1.105 | 1.092–1.095 | 29.9 → 29.7 |
| Splat preview | 0.966 / 0.896 | 0.936 / 0.894 | 3.08% | 0.960–0.972 | 0.934–0.938 | 33.8 → 33.3 |

Every splat run rendered/captured 600 frames per camera and delivered 600 raw
previews, 600 JPEG previews and 600 lidar messages. Normal mode observes 599
lidar messages because its final delayed render crosses the measurement boundary;
timestamp spacing remains 100 ms, equally with LEDs on/off. Topics match within
each mode. All eight production-source hashes match the final working tree.

| Mode | App update ms off → on | GPU frame ms off → on |
| --- | ---: | ---: |
| off | 28.55 → 28.77 | 6.42 → 6.42 |
| preview | 32.45 → 33.52 | 9.04 → 9.12 |

Evidence: `/tmp/agipix-leds-rtf-off/reports.json`,
`/tmp/agipix-leds-rtf-on/reports.json`, `/tmp/agipix-leds-rtf-summary.json`.
Hardware: Threadripper 9960X, RTX PRO 6000 Blackwell Max-Q; eight CPU threads,
zero configured PhysX worker threads. GUI visuals were checked separately;
GUI RTF, other environments, and autonomous PX4 flight were not benchmarked.

## Changed files and handoff

- `examples/8_agipix.py`: opt-in body attachment and standalone CLI flag.
- `examples/8_agipix_launch.py`: flag parsing, canonical argv and enabled launch-plan entry.
- `examples/agipix_leds.py`: geometry, materials and four static USD light definitions.
- `examples/test_8_agipix_launch_cli.py`: default-off and forwarding checks.
- `examples/test_agipix_leds.py`: geometric clearance, colours, rigid transforms and no-physics checks.
- `examples/validate_agipix_leds.py`: bounded sensor and screenshot checks.
- `examples/bench_agipix_splat_rtf.py`: optional LED benchmarking and source hash.
- `examples/validate_agipix_splat.py`: forward the LED flag to the existing dataset validator.
- `examples/LEDS.md`: this geometry/API/test/result record.

No commits, deletions or extension edits. The pre-existing dirty `isaac_envs`
submodule is untouched. Final host process checks found no PX4, Kit or
`8_agipix` process left running.
