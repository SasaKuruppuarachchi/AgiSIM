"""Reusable CPU PhysX tensors owned by one multirotor for one play session."""
import numpy as np
import omni.physics.tensors as tensors
import warp as wp
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager


class MultirotorTensors:
    """Use public tensor views; keep NumPy owners alive for zero-copy Warp inputs.

    Forces and application positions are in each link's local frame. State reads
    remain in world coordinates. GPU pipelines use the existing prim wrappers.
    """

    def __init__(self, path, forces, torques, positions):
        if SimulationManager.get_active_physics_engine() != "physx":
            raise ValueError("Reusable multirotor tensors currently support PhysX only")
        stage_id = stage_utils.get_stage_id(stage_utils.get_current_stage(backend="usd"))
        self.simulation = tensors.create_simulation_view("warp", stage_id=stage_id, backend="physx")
        if self.simulation.device_ordinal != -1:
            raise ValueError("Reusable multirotor buffers require a CPU tensor pipeline")
        self.simulation.set_subspace_roots("/")
        paths = [path + "/body"] + [path + f"/rotor{i}" for i in range(len(forces) - 1)]
        self.links = self.simulation.create_rigid_body_view(paths)
        self.body = self.simulation.create_rigid_body_view(path + "/body")
        self.articulation = self.simulation.create_articulation_view(path)
        # Never assume that tensor rows follow the requested pattern order.
        if list(self.links.prim_paths) != paths:
            raise ValueError("Unexpected multirotor rigid-body ordering")
        if self.body.count != 1 or self.articulation.count != 1:
            raise ValueError("Expected exactly one body and one articulation")
        if not self.links.check() or not self.body.check() or not self.articulation.check():
            raise ValueError("Invalid multirotor physics views")
        self._owners = (forces, torques, positions)
        self.forces, self.torques, self.positions = [
            wp.array(data, dtype=wp.float32, device="cpu", copy=False) for data in self._owners
        ]
        self.link_indices = wp.array(np.arange(len(paths), dtype=np.uint32), dtype=wp.uint32, device="cpu")
        self.articulation_indices = wp.array([0], dtype=wp.uint32, device="cpu")

    @property
    def valid(self):
        return self.simulation.is_valid

    def apply_forces(self):
        self.links.apply_forces_and_torques_at_position(
            self.forces, self.torques, self.positions, self.link_indices, False
        )

    def set_rotor_velocities(self, values, dof_indices):
        # Preserve other DOFs and the wrapper's velocity + drive-target semantics.
        data = self.articulation.get_dof_velocities()
        data.numpy()[:, dof_indices] = values
        self.articulation.set_dof_velocities(data, self.articulation_indices)
        self.articulation.set_dof_velocity_targets(data, self.articulation_indices)

    def read_body_state(self):
        transforms = self.body.get_transforms().numpy()
        velocities = self.body.get_velocities().numpy()
        return transforms[0, :3], velocities[0, :3], velocities[0, 3:]
