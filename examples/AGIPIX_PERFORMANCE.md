# Agipix runtime performance

`isaac_run 8_agipix.py` uses 8 Kit/TBB threads and 0 PhysX worker threads for the single-vehicle CPU physics workload. Zero workers keeps physics on the calling thread; it does not disable physics. Physics defaults to 250 Hz, rendering remains 30 Hz, and sensors and PX4 remain enabled.

Set the physics timestep in seconds with `--physics-dt` (default `0.004`):

```bash
isaac_run 8_agipix.py --physics-dt 0.005  # 200 Hz physics
```

Override the defaults when measuring another machine or a larger scene:

```bash
AGIPIX_CPU_THREADS=32 AGIPIX_PHYSICS_THREADS=8 isaac_run 8_agipix.py
```

The example restores the previous persistent PhysX worker setting on normal shutdown and exceptions handled by `main`. A forcibly terminated process cannot guarantee cleanup. Code that imports the example and owns its application loop should call `restore_thread_settings()` before closing `simulation_app`.

Multirotors use `MultirotorTensors` to own public PhysX simulation, rigid-body and articulation views for each play session. Force, torque, position and index buffers are reused. Body positions and velocities are read directly from tensors; attitude retains the existing USD read. Local force frames, rotor drive targets, timestep and control logic are unchanged. Views are discarded on stop and recreated on play. The adapter validates rigid-body ordering before enabling the fast path.

Unsupported physics engines, GPU tensor pipelines or failed adapter initialization retain the prim-wrapper path and issue a warning. Set `MultirotorConfig.use_runtime_tensors = False` on the configuration instance before creating a vehicle to explicitly use wrappers for comparison. The optimization currently targets CPU PhysX and does not implement a GPU-resident control loop.

Validation scripts:

```bash
isaac_run tests/isaacsim/test_multirotor_batching.py
isaac_run tests/isaacsim/performance/probe.py \
  --output /tmp/agipix-production-tensors \
  --log /tmp/agipix-production-tensors.log \
  --seconds 30 --windows 3 --profile-seconds 0 \
  > /tmp/agipix-production-tensors.log 2>&1
```

Run from the repository root for these validation commands. The regression compares the tensor path against scalar wrapper calls for tilted initial attitude and unequal rotor forces, including accelerations, joint velocities and two play/stop cycles. The benchmark waits for PX4 readiness and warmup, records effective thread settings and tensor activation, and measures the full stationary workload. Flight and active ROS consumers must be validated separately. Historical `--tensor-fast` results used a private-handle prototype; that flag is unnecessary for the production implementation.

## Match ROS versions across the host and container

The Agipix development container uses ROS 2 Humble. Source `scripts/isaac_ros_env.sh` before launching Isaac Sim's Python to use its bundled Humble libraries, even when the host runs Ubuntu 24.04. The host OS alone does not determine the ROS distribution used by the other DDS participants. The helper removes stale native ROS and bundled Jazzy/Humble library paths before adding the selected `isaacsim.ros2.core/<distro>/lib` directory. It preserves `ROS_DOMAIN_ID`.

The host `isaac_run` function sources this helper and defaults to Humble/CycloneDDS, matching the Agipix container. Reload `~/.bashrc` in existing terminals. Explicit overrides are supported:

```bash
ISAACSIM_ROS_DISTRO=humble isaac_run 8_agipix.py
ISAACSIM_ROS_DISTRO=jazzy isaac_run 8_agipix.py
# Optional Fast DDS comparison, using the selected distro's bundled libraries:
ISAACSIM_ROS_RMW=rmw_fastrtps_cpp isaac_run 8_agipix.py
```

For a shell without the `isaac_run` function, from the repository root:

```bash
source scripts/isaac_ros_env.sh
"${ISAACSIM_PATH:-$HOME/isaacsim}/python.sh" examples/8_agipix.py
```

Matching distributions addresses the observed DDS discovery/deserialization errors. It does not establish a fix for the separate intermittent native RTX lidar crash. See the crash investigation artifacts under `tests/isaacsim/crash_2026-09-26/` for the limits of validation.

## Mid-360 firing cadence

The bundled rotary approximation uses `patternFiringRateHz=7760` and
`scanRateBaseHz=10`, giving 776 firing ticks per scan. NVIDIA recommends a firing
rate divisible by the scan rate because the rotary scheduler uses integer ticks
([sensor configuration reference](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.common/latest/spec_sheet_configuration.html)).
The previous 7761 Hz setting reproduced a complete loss of native sensor output
at 216.8 simulation seconds in three full-stack runs, including a diagnostic run
that retained invalid returns. The 7760 Hz profile continued publishing through 956 simulation seconds
(22.3 wall minutes), including dynamic exploration and ground operation. This changes the firing cadence by approximately 0.013%; physics,
rendering and the 10 Hz lidar publication rate remain unchanged.

The setup script copies the profile into the configured Isaac Sim asset root.
Existing installations need that updated asset as well as the repository change;
changing only a ROS subscriber or middleware cannot repair empty native output.
The full-stack measurements and remaining autonomy/performance issues are in
`tests/isaacsim/full_stack_2026-10-02/` in the test workspace.
