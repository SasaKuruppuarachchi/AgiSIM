# Agipix runtime performance

`isaac_run 8_agipix.py` uses 8 Kit/TBB threads and 0 PhysX worker threads for the single-vehicle CPU physics workload. Zero workers keeps physics on the calling thread; it does not disable physics. Physics remains 250 Hz, rendering remains 30 Hz, and sensors and PX4 remain enabled.

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
