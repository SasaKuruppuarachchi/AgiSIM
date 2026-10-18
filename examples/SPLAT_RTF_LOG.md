# Splat-mode RTF optimisation log

Baseline source: `e84e15c` (2026-10-18). The existing `isaac_envs` submodule change is preserved.

## Measurement protocol

- One Isaac/PX4 instance at a time, parent-held `/tmp/agipix-sim-benchmark.lock`, process checks before every run.
- FOREST_FI and AKW_C; mode off, preview, save RGB, save RGB+depth; two repetitions each.
- Five simulated seconds warm-up, then sixty measured seconds; fixed bounded body motion, no arming.
- Physics 250 Hz, lidar 10 Hz with timestamps, cameras/preview 10 Hz simulated time. All measured frame counts checked.
- Raw and JPEG preview consumers active during measurements; separate ROS domain 193.
- Native PhysX profile stats, app-update wall time, Python writer/stitch/publish/submit timings, one-Hz GPU and per-thread CPU telemetry.
- GPU profiler time overlaps CPU time. App-update residual includes rendering, native Kit/ROS work and synchronization; it is not falsely labelled pure render time.
- cProfile/native profiler captures are separate diagnostic runs because profiling adds overhead.

## Investigation before changes

- RtxCamera already applies OmniSensorAPI with 10 Hz tick rate; Isaac enables multi-tick-rate rendering. Need measured render counts before assuming 30 Hz waste.
- Headless SimulationApp currently leaves its hidden active viewport updating. A splat-only `disable_viewport_updates` experiment is proposed.
- CPU microbench: noncontiguous RGB slices from RGBA cost ~13.9 ms to stitch versus ~3.7 ms contiguous; exact RGB conversion can remove redundant packing without changing pixels.
- PNG writer already uses Pillow compression level 1, four workers and raw NPY depth. Disk-agent measurements compare lossless encoders; no production change made yet.

## Experiments

### Diagnostic, before any production change

Evidence: `/tmp/agipix-rtf-commission-01/FOREST_FI-preview-1/` (2 s warm-up,
8 s measured; **cProfile + native tracing enabled, not an RTF baseline**).
All four cameras captured 80 frames; raw/JPEG preview and timestamped lidar each
delivered 80 messages. The native trace establishes non-overlapping CPU scopes:

| App-update scope | Total seconds | Share |
| --- | ---: | ---: |
| Replicator writer | 18.845 | 68.4% |
| Physics including callbacks | 4.827 | 17.5% |
| Hydra renderer CPU submission/work/waits | 2.466 | 9.0% |
| Other Kit work | 1.397 | 5.1% |
| Total app update | 27.536 | 100% |

Writer includes 17.707 s ROS publication and 1.131 s panorama stitching. Physics
includes 4.390 s of post-step callbacks, not just the solver. GPU frame time is
12.18 ms mean, GPU utilization 12.7%; these overlap CPU work. Profiling magnifies
Python per-byte message validation: unprofiled microbenchmarks of the actual
bundled ROS setters measured 26.60 ms raw + 2.05 ms JPEG with `bytes`, versus
0.017 + 0.004 ms using `array.array('B', payload)`. Complete messages/payloads
compare equal. This becomes the first candidate, ahead of GPU changes.

Native graph traces show 80 GPU interop/host-copy executions per camera in 8 sim
seconds. The 242 viewport setup calls are not 242 sensor captures. Manual 30→10 Hz
render gating is therefore not justified by this trace.

Commissioning caught and fixed benchmark-only issues: CPU telemetry initially
followed the `python.sh` parent; it now samples the actual Isaac PID. Lidar and
camera timestamps now require unique, approximately 100 ms intervals. A 3-second
mode-off run can observe 29 messages because the final lidar render is delayed;
the count tolerance is one boundary message, using integer counts rather than
floating-point rate subtraction. Capture and shutdown/drain RTF are reported
separately. Native drawable event counts are diagnostic only; they may include
setup without a new sensor image.

Baseline matrix and per-change results follow below.

### Baseline matrix — complete

Command: `python3 examples/bench_agipix_splat_rtf.py --output /tmp/agipix-rtf-baseline`
(defaults: 5 + 60 simulated seconds, two repeats, both scenes, all four modes).
All 16 runs passed and source hashes match across runs. Short commissioning in
`/tmp/agipix-rtf-commission-03` is excluded.

| Scene | Mode | RTF runs | p5 RTF runs | Mean GPU % |
| --- | --- | --- | --- | ---: |
| FOREST_FI | off | 1.0819 / 1.0673 | 1.0104 / 1.0051 | 36.9 |
| FOREST_FI | preview | 0.4540 / 0.4606 | 0.4333 / 0.4527 | 21.6 |
| FOREST_FI | save | 0.4101 / 0.4059 | 0.3979 / 0.3899 | 18.0 |
| FOREST_FI | depth | 0.4016 / 0.3993 | 0.3898 / 0.3902 | 19.6 |
| AKW_C | off | 1.1692 / 1.1496 | 1.0831 / 1.0561 | 32.0 |
| AKW_C | preview | 0.4704 / 0.4684 | 0.4527 / 0.4544 | 19.0 |
| AKW_C | save | 0.4275 / 0.4172 | 0.4108 / 0.4051 | 16.9 |
| AKW_C | depth | 0.4101 / 0.4116 | 0.3998 / 0.3958 | 18.8 |

Every measured splat run captured 600 frames per camera and delivered 600 raw
and 600 JPEG previews. Saved manifests contain all 2,400 measured images (plus
200 warm-up images); depth runs also wrote every corresponding NPY. Lidar stamp
spacing is 100 ms: 600 messages in splat modes; mode off sees 599 because its
last lidar render crosses the stop boundary. Peak dataset queue depth was one
batch, including warm-up. Both mode-off topic lists agree in each scene.

### Round 1 — typed ROS byte buffers (kept)

Only `utils/drone_location_pub.py` changes: assign `array('B', payload)` to raw
and compressed `data`, retaining exactly the same serialized bytes. The bundled
ROS setter recognizes the uint8 array and avoids per-byte Python assertions.
Candidate benchmarks: FOREST_FI preview/save, two 60-second repeats each, then
independent invariant review. No image, camera, dataset or cadence changes.

Results (`/tmp/agipix-rtf-ros`):

- preview: RTF 0.8646 / 0.8734; mean 0.8690, gain 90.0%.
- save: RTF 0.7210 / 0.7094; mean 0.7152, gain 75.3%.

Keep: gains greatly exceed baseline spread; every measured camera/preview
frame and saved image is accounted for. ROS publication fell from ~105 ms to
~0.9 ms/capture; RGB disk-batch wall time fell from ~163 to ~67 ms as Python
contention eased. Queue peak remains one batch. 38 tests + 14 subtests pass;
host CLI separately passes 11 tests + 10 subtests. Live 8 s FOREST_FI validator
passes: 320 images, 10 Hz preview, max RGB reprojection 3.235 px, instance-mask
reprojection 1.835 px, no own-module pixels. Preflight/print-command pass.
Independent reviewer confirms pose/calibration/cadence/headers/mode-off paths
are unchanged.

### Round 2 — contiguous RGB packing (kept)

The remaining saved capture spends ~10.7 ms copying strided RGB and ~19.1 ms
stitching. Replace the RGBA `[..., :3]` view with `cv2.cvtColor(..., COLOR_RGBA2RGB)`
once per camera. Pixels are identical, but both downstream consumers receive
packed RGB; the dataset retains its own independent copy. No resolution, lens,
pose, image-quality or rate changes.

Results (`/tmp/agipix-rtf-packed`):

- preview: RTF 0.9474 / 0.9722; mean 0.9598, incremental gain 10.4%.
- save: RTF 0.8668 / 0.8447; mean 0.8557, incremental gain 19.7%.

Keep: improvements exceed the ~2.6% spread. Both stretch targets are met by
the repeated FOREST_FI means, so production optimization stops here. No lowered
resolution, JPEG-quality change, rendering-setting change or dropped previews.
Full suite again passes 38 tests + 14 subtests; host CLI passes 11 + 10. The
randomized RGBA test checks every output RGB byte and arbitrary alpha. Live
8-second FOREST_FI RGB+depth validator passes: 320 images, max RGB reprojection
3.505 px, instance-mask reprojection 1.835 px, no own-module pixels, 10 Hz preview.

The final matrix reuses these four confirmed, unchanged-source FOREST_FI
preview/save runs. Two fresh repeats of FOREST_FI off/depth and all four AKW_C
modes complete the requested eight configurations without redundant repeats.

### Live lifecycle verification

`python3 /tmp/agipix_splat_runtime_lifecycle.py --output /tmp/agipix-splat-lifecycle-final-01`
passed on the final source in headless AKW_C. Eight update calls after pause/drain
kept physics time, capture stamp and frame count unchanged. Disconnecting the
JPEG subscriber produced ten camera batches with zero JPEG encodes; reconnecting
resumed correctly stamped/framed JPEGs. All 80 raw previews and 320 RGB images
were retained; 70 JPEGs were expected and received. Shutdown drained in 0.147 s,
with no live disk worker, no queued batch, and no error. DDS subscriber lifecycle
was exercised directly; Foxglove's WebSocket/UI layer was not part of this test.
Stop/restart refusal is covered by CPU tests; live normal stop and pause/resume
are covered here.

### Investigated alternatives, not applied

No production candidate was reverted: both measured changes improved RTF beyond
noise and met the targets. These other proposals were investigated, not claimed
as end-to-end RTF experiments:

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Manual camera render gating | Native interop/readback and drawable counts already show 10 Hz, not 30 Hz | No extra gating; preserve native sensor scheduling |
| Hidden headless viewport | Isaac 6 `SimulationApp(disable_viewport_updates=True)` verified in local source; explicit sensor render products are independent | Not needed after targets met; GUI behavior remains unchanged |
| 640×400 preview-only cameras | Local `RtxCamera` fisheye schema/RP APIs support scaled intrinsics | Retain original rendering resolution and preview quality |
| ROI/fixed-point panorama remaps | Contiguous-source CPU microbench ~3.71 → 2.22 ms, pixel-identical | Smaller benefit; packing already removes the dominant stitch cost |
| Lossless PNG encoder | Four-worker CPU tests: Pillow level1 ~53 ms forest /42 ms indoor; OpenCV level1 HUFFMAN_ONLY ~34/27 ms, decoded RGB identical, smaller files | Keep existing encoder; queue stays at one batch and save stretch target is met |
| More disk workers/queue capacity | Already four concurrent encoders; no growing queue | Keep bounds and loud overload behavior |
| Async panorama / skipped previews | Current path now meets targets and delivers every preview | No asynchronous lifecycle or frame-drop behavior added |
| Annotator copy suppression / CPU thread tuning / RTX quality changes | APIs/settings inspected; no evidence they are required to meet targets | Keep ownership, thread limits and rendered content unchanged |

Microbench evidence: `/tmp/agipix-ros-data-microbench.json`,
`/tmp/agipix-panorama-microbench.json`, `/tmp/agipix-disk-png-strategies-20261018.json`.

### Matching diagnostic profile after optimization

Evidence: `/tmp/agipix-rtf-final-profile/FOREST_FI-preview-1/`, same 2 + 8 s
protocol and profiler settings as the initial diagnostic. Both have 242 updates,
80 captured/rendered frames per camera, 80 raw/JPEG previews and 80 lidar messages.
These are profiled CPU wall times, **not production RTF measurements**:

| Non-overlapping main-thread scope | Before (s) | Final (s) | Final share |
| --- | ---: | ---: | ---: |
| Replicator writer | 18.845 | 0.482 | 4.9% |
| Physics including callbacks | 4.827 | 5.247 | 53.3% |
| Hydra renderer CPU/submission/waits | 2.466 | 2.631 | 26.7% |
| Other Kit/update work | 1.397 | 1.486 | 15.1% |
| Total app update | 27.536 | 9.846 | 100% |

Component interval unions equal sums: there is no double-counting. Python call
count fell from 212.4 million to 7.55 million. GPU frame duration stayed close
(12.18 → 12.39 ms), while utilization rose (12.7% → 34.1%) as CPU stalls shrank.
The final main thread used about 99% of one CPU core.

The remaining physics scope is mostly existing callback work: 4.780 s post-step
subscriptions versus 0.230 s native PhysX simulate. Hotspots include Pegasus
sensor updates, repeated SciPy rotation/array conversions, ROS state publication
and PX4 callbacks. Further substantial CPU gains would need reducing redundant
work there while preserving 250 Hz physics and existing sensor/publication
cadence, likely involving extensions outside this task's intended scope. Native
rendering is the next substantial component. No rates or extension code changed.

### Notes for the next profiling pass

- Keep cProfile/native tracing separate from reported RTF. Profiling amplifies
  per-byte Python loops dramatically; confirm with unprofiled complete runs.
- For ROS uint8 sequences, `array('B', payload)` takes the generated setter's
  supported fast path and retains identical serialized data. Check publishers
  before assuming GPU rendering is the limiting cost.
- A NumPy `RGBA[..., :3]` view is not packed RGB. Native `cv2.cvtColor` makes
  downstream resize/copy much cheaper; retain explicit dataset buffer ownership.
- Native RtxCamera tick rate already gates actual sensor work. Count completed
  drawable frames and GPU interop/readback executions; viewport setup events do
  not demonstrate extra rendered images.
- Isaac 6 APIs used: `omni.physics.core.get_physics_benchmarks_interface()` with
  `subscribe_profile_stats_events` (milliseconds); `omni.hydra.engine.stats.HydraEngineStats`
  with `get_gpu_profiler_result`; `carb.profiler.acquire_profiler_interface`
  with `carb.profiler-mux.plugin`, capture mask and CPU profiler saveProfile/filePath;
  `omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED` via
  `carb.eventdispatcher.get_eventdispatcher().observe_event`, then
  `hydra_texture.get_frame_info(...)["frame_number"]`.
- PhysX Update includes Python callbacks, not only the solver. GPU timings can
  overlap CPU wall time. The ordinary app-update residual also includes Kit,
  ROS, rendering and waits; only native diagnostic scopes isolate render CPU.
- `isaacsim/python.sh` is a parent shell. Sample the actual Isaac child PID and
  retain per-thread telemetry. GPU utilization/memory are whole-device numbers,
  including any desktop work, so compare repeated runs with no other simulator.
- Preserve all measured image/stamp counts, source hashes, queue pressure and
  shutdown drain time. One queued batch means more queue capacity cannot make
  the current producer faster. Keep no-silent-drop failure handling intact.

## Final unprofiled results

Two 60-second measured runs per cell after 5 s warm-up. RTF is the mean of
the two run means; p5 pools their one-sim-second windows. GPU % is the mean
of whole-device samples (including desktop activity). All runs are headless.
Final source hashes match across all 16 runs and the current working tree.

| Scene | Mode | Baseline RTF / p5 | Final RTF / p5 | GPU % before → final | Frames/camera/run |
| --- | --- | ---: | ---: | ---: | ---: |
| FOREST_FI | off | 1.075 / 1.007 | 1.077 / 1.005 | 36.9 → 37.0 | — |
| FOREST_FI | preview | 0.457 / 0.437 | 0.960 / 0.902 | 21.6 → 42.0 | 600 |
| FOREST_FI | save | 0.408 / 0.394 | 0.856 / 0.812 | 18.0 → 37.5 | 600 |
| FOREST_FI | depth | 0.400 / 0.390 | 0.812 / 0.773 | 19.6 → 36.8 | 600 |
| AKW_C | off | 1.159 / 1.062 | 1.181 / 1.085 | 32.0 → 31.9 | — |
| AKW_C | preview | 0.469 / 0.453 | 0.978 / 0.915 | 19.0 → 34.5 | 600 |
| AKW_C | save | 0.422 / 0.406 | 0.855 / 0.796 | 16.9 → 30.8 | 600 |
| AKW_C | depth | 0.411 / 0.398 | 0.818 / 0.767 | 18.8 → 30.6 | 600 |

Each saving run wrote all 2,400 measured RGB images, plus 200 warm-up images.
Each depth run wrote matching NPY counts. All 41,600 baseline/final PNGs and
20,800 corresponding depth files exist; all manifests retain 1280×800, f577.5,
zero KB4 distortion coefficients and no recording errors. Splat runs receive
600 raw previews, 600 JPEG previews and 600 lidar messages. Mode-off sees
599 lidar messages due to the delayed final boundary render; stamp spacing
is still 100 ms. Mode-off topics match exactly before/after in both scenes.

| Scene / mode | Baseline RTF range | Final RTF range | Final RTF including close (range) |
| --- | ---: | ---: | ---: |
| FOREST_FI / off | 1.067–1.082 | 1.072–1.082 | 1.072–1.082 |
| FOREST_FI / preview | 0.454–0.461 | 0.947–0.972 | 0.947–0.972 |
| FOREST_FI / save | 0.406–0.410 | 0.845–0.867 | 0.841–0.863 |
| FOREST_FI / depth | 0.399–0.402 | 0.805–0.820 | 0.801–0.815 |
| AKW_C / off | 1.150–1.169 | 1.177–1.184 | 1.177–1.184 |
| AKW_C / preview | 0.468–0.470 | 0.971–0.985 | 0.970–0.985 |
| AKW_C / save | 0.417–0.427 | 0.852–0.858 | 0.849–0.854 |
| AKW_C / depth | 0.410–0.412 | 0.814–0.822 | 0.810–0.818 |

### Unprofiled timing breakdown

Entries are baseline → final milliseconds. Physics includes callbacks;
GPU timing overlaps CPU wall time. Other update time includes render CPU,
native Kit/ROS work and waits; it is not a pure render measurement.
The matching native diagnostics above isolate renderer CPU scopes.

| Scene / mode | App update | Physics/update | Writer/capture | Other update incl. render | GPU frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| FOREST_FI / off | 29.33 → 29.25 | 15.02 → 14.98 | 0.00 → 0.00 | 14.31 → 14.27 | 8.75 → 8.74 |
| FOREST_FI / preview | 70.79 → 32.67 | 15.36 → 15.46 | 120.20 → 5.61 | 15.34 → 15.34 | 12.11 → 12.15 |
| FOREST_FI / save | 78.97 → 36.33 | 16.89 → 16.29 | 135.49 → 7.45 | 16.89 → 17.56 | 12.08 → 12.18 |
| FOREST_FI / depth | 80.55 → 38.15 | 17.02 → 16.44 | 136.95 → 9.78 | 17.86 → 18.44 | 12.13 → 12.11 |
| AKW_C / off | 27.10 → 26.61 | 14.56 → 14.35 | 0.00 → 0.00 | 12.54 → 12.26 | 6.47 → 6.46 |
| AKW_C / preview | 68.89 → 32.05 | 15.30 → 15.28 | 115.81 → 5.54 | 14.97 → 14.92 | 9.07 → 9.13 |
| AKW_C / save | 76.50 → 36.81 | 16.50 → 17.87 | 130.07 → 7.42 | 16.62 → 16.47 | 9.07 → 9.16 |
| AKW_C / depth | 78.71 → 38.56 | 16.58 → 17.84 | 133.47 → 9.55 | 17.62 → 17.54 | 9.08 → 9.04 |

Writer parts overlap its inclusive time above. Disk time runs on workers;
it must not be added to main-thread app-update totals.

| Scene / mode | Stitch/capture | ROS/capture | Dataset submit/capture | Disk/batch |
| --- | ---: | ---: | ---: | ---: |
| FOREST_FI / preview | 14.71 → 4.36 | 105.45 → 0.86 | 0.00 → 0.00 | 0.00 → 0.00 |
| FOREST_FI / save | 18.27 → 5.11 | 106.70 → 0.92 | 10.22 → 0.66 | 163.18 → 67.41 |
| FOREST_FI / depth | 18.60 → 5.28 | 106.31 → 0.91 | 10.76 → 1.62 | 164.81 → 70.57 |
| AKW_C / preview | 14.38 → 4.37 | 101.39 → 0.74 | 0.00 → 0.00 | 0.00 → 0.00 |
| AKW_C / save | 17.14 → 5.21 | 102.58 → 0.81 | 10.05 → 0.63 | 141.76 → 55.19 |
| AKW_C / depth | 18.42 → 5.20 | 102.93 → 0.83 | 10.83 → 1.61 | 144.29 → 57.13 |

### Resource and shutdown checks

| Scene / mode | GPU peak GiB before → final | Queue peak before → final | Final close drain range (s) | Main CPU % before → final |
| --- | ---: | ---: | ---: | ---: |
| FOREST_FI / off | 7.90 → 7.90 | 0 → 0 | 0.004–0.004 | 98.8 → 98.8 |
| FOREST_FI / preview | 8.41 → 8.41 | 0 → 0 | 0.024–0.026 | 99.5 → 98.8 |
| FOREST_FI / save | 8.40 → 8.40 | 1 → 1 | 0.305–0.317 | 97.1 → 95.3 |
| FOREST_FI / depth | 8.47 → 8.46 | 1 → 1 | 0.374–0.384 | 96.9 → 94.8 |
| AKW_C / off | 6.94 → 6.94 | 0 → 0 | 0.004–0.004 | 99.0 → 99.0 |
| AKW_C / preview | 7.43 → 7.43 | 0 → 0 | 0.025–0.026 | 99.5 → 99.1 |
| AKW_C / save | 7.43 → 7.43 | 1 → 1 | 0.305–0.319 | 97.2 → 94.2 |
| AKW_C / depth | 7.49 → 7.49 | 1 → 1 | 0.361–0.376 | 97.0 → 93.7 |

All per-thread CPU records, raw one-Hz GPU samples, timings, source hashes and
assertion results are retained in each case report/system.jsonl. Queue
peaks include warm-up and are measured at enqueue, before publication.
Combined final reports: `/tmp/agipix-rtf-final-reports.json`.

## Final validation and handoff

The final 30-second AKW_C RGB validator passed on the unchanged production source:

```bash
python3 examples/validate_agipix_splat.py --headless --splat-mode --save-splat \
  --environment AKW_C --splat-out /tmp/agipix-rtf-final-akw-validation \
  --validation-seconds 30
```

It saved all 1,200 images (300 per camera), received 300 previews at 10 Hz in
simulation time, and reported no recording error. Maximum rendered-target
reprojection error was 2.3421 px; scene-geometry reprojection error was 1.9407 px.
All four cameras had zero own-module pixels across 300 segmentation samples each.
Queue peak was one batch. The validator enables additional diagnostic annotators;
its RTF is not substituted for the ordinary benchmark results above. Evidence:
`/tmp/agipix-rtf-final-akw-validation/validation.json`.

The following suite passed after each production change (38 tests, 14 subtests):

```bash
PYTHONPATH=/tmp/agipix-test-deps /home/sasa/isaacsim/python.sh -m pytest \
  examples/test_8_agipix_launch_cli.py examples/test_agipix_splat.py \
  examples/test_agipix_chase_camera.py examples/test_agipix_splat_lifecycle.py -q
```

Host CLI tests also passed (11 tests, 10 subtests) using
`PYTHONPATH=/tmp/agipix-test-deps python3 -m pytest examples/test_8_agipix_launch_cli.py`.
The dependency path supplies pytest, which is not installed in the host Python.
Launcher `--preflight` and `--print-command` passed with unchanged headless,
physics-dt and lidar timestamp defaults. The independent reviewer verified the
final tables, source hashes, topic lists, frame counts and lifecycle evidence.

Hardware: AMD Ryzen Threadripper 9960X (24 cores), NVIDIA RTX PRO 6000 Blackwell
Max-Q Workstation Edition, Isaac Sim 6.0. All benchmark runs used eight CPU
threads and zero configured PhysX worker threads. The benchmark now records these
settings and its own hash explicitly in future case configs; this final
metadata-only addition does not change the measured runtime or production code.

Limits: measurements use headless scripted unarmed motion, not a GUI flight.
GPU telemetry covers the whole device, including desktop work. Subscribe/
unsubscribe testing exercises DDS directly, not the Foxglove UI/WebSocket layer.
Saved-recording restart refusal and slow-disk fault injection are CPU tests;
pause/resume and normal shutdown were also exercised live. Stretch targets are
met by repeat means; one FOREST_FI preview run was 0.9474 RTF.

No commits or file deletions were made. Only the two production files, benchmark,
lifecycle tests and this log belong to this change; the existing dirty
`isaac_envs` submodule is untouched. Main benchmark datasets retain approximately
139 GiB under `/tmp/agipix-rtf-*`, plus smaller diagnostic/validator datasets.
The final disk check showed 46 GiB free. Final host process checks found no PX4,
Kit or `8_agipix` process left running.
