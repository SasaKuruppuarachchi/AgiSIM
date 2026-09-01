#!/usr/bin/env python
"""
| File: 9_people.py
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: This file serves as an example on how to build an app that makes use of the Pegasus API to run a simulation
| where people move around in the world. (Updated for Isaac Sim 6.0)
"""

# Imports to start Isaac Sim from this script
import carb

from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from isaacsim.core.experimental.utils.app import enable_extension

# Enable ROS bridge extensions
enable_extension("isaacsim.ros2.bridge")

# Update the simulation app with the new extensions
simulation_app.update()

# -------------------------------------------------------------------------------------------------
# These lines are needed to restart the USD stage and make sure that the people extension is loaded
# -------------------------------------------------------------------------------------------------
import omni.usd
omni.usd.get_context().new_stage()

import numpy as np

# Drone start exclusion zone — people must not enter or spawn here
_DRONE_EXCLUSION_CENTER = np.array([0.0, 0.0, 0.0])
_DRONE_EXCLUSION_RADIUS = 2.0


def _clamp_away_from_zone(target, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS):
    """Push target to the nearest point on the exclusion zone boundary if it falls inside."""
    target = np.array(target, dtype=float)
    delta = target[:2] - zone_center[:2]
    dist = np.linalg.norm(delta)
    if dist < zone_radius:
        if dist < 1e-6:
            delta = np.array([zone_radius, 0.0])
        else:
            delta = delta / dist * zone_radius
        target = target.copy()
        target[0] = zone_center[0] + delta[0]
        target[1] = zone_center[1] + delta[1]
    return target


def _apply_exclusion_zone(person, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS):
    """Check the actor's current position and force an escape target if inside the zone.
    Returns True if the person was inside the zone; callers should skip normal update logic."""
    pos = person.state.position
    delta = pos[:2] - zone_center[:2]
    dist = np.linalg.norm(delta)
    if dist < zone_radius:
        if dist < 1e-6:
            escape_dir = np.array([1.0, 0.0])
        else:
            escape_dir = delta / dist
        escape_point = zone_center[:2] + escape_dir * (zone_radius + 1.0)
        person.update_target_position([escape_point[0], escape_point[1], pos[2]], 2.0)
        return True
    return False


def _detour_around_zone(pos, target, zone_center=_DRONE_EXCLUSION_CENTER, zone_radius=_DRONE_EXCLUSION_RADIUS, margin=0.5):
    """If the straight-line path from pos to target passes through the exclusion zone,
    return a tangent boundary waypoint that routes around it. Otherwise returns target unchanged."""
    p = np.array(pos[:2], dtype=float)
    t = np.array(target[:2], dtype=float)
    c = np.array(zone_center[:2], dtype=float)
    r = zone_radius + margin

    direction = t - p
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return np.array(target, dtype=float)
    dir_unit = direction / length

    proj = np.clip(np.dot(c - p, dir_unit), 0.0, length)
    closest = p + proj * dir_unit
    dist_closest = np.linalg.norm(closest - c)

    if dist_closest >= r:
        return np.array(target, dtype=float)

    # Path intersects: steer to one of the two tangent points on the zone boundary
    perp = np.array([-dir_unit[1], dir_unit[0]])
    detour_left = c + perp * r
    detour_right = c - perp * r

    def total_dist(mid):
        return np.linalg.norm(mid - p) + np.linalg.norm(t - mid)

    detour_2d = detour_left if total_dist(detour_left) <= total_dist(detour_right) else detour_right
    z = float(np.asarray(target, dtype=float).flat[2]) if np.asarray(target).size > 2 else 0.0
    return np.array([detour_2d[0], detour_2d[1], z], dtype=float)


# Import the Pegasus API for simulating drones
from pegasus.simulator.params import FLAT_ENVIRONMENTS, ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.people.person import Person
from pegasus.simulator.logic.people.person_controller import PersonController
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig


class WaypointPatrolController(PersonController):
    """Patrols through a list of 3D waypoints in sequence, looping back to start."""

    def __init__(self, waypoints, speed=1.2, arrival_radius=0.8):
        super().__init__()
        self._waypoints = [np.array(wp) for wp in waypoints]
        self._speed = speed
        self._arrival_radius = arrival_radius
        self._current_wp_idx = 0

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        target = self._waypoints[self._current_wp_idx]
        pos = self._person.state.position
        dist = np.linalg.norm(target[:2] - pos[:2])
        if dist < self._arrival_radius:
            self._current_wp_idx = (self._current_wp_idx + 1) % len(self._waypoints)
            target = self._waypoints[self._current_wp_idx]
        target = _clamp_away_from_zone(target)
        immediate = _detour_around_zone(pos, target)
        speed = self._speed * min(1.0, dist / (self._arrival_radius * 2.0))
        speed = max(speed, 0.3)
        self._person.update_target_position(immediate.tolist(), speed)


class LemniscateController(PersonController):
    """Person traces a figure-8 (lemniscate of Bernoulli) path around a center point."""

    def __init__(self, center, scale=4.0, speed=0.35):
        super().__init__()
        self._center = np.array(center)
        self._scale = scale
        self._gamma = 0.0
        self._gamma_dot = speed

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        self._gamma += self._gamma_dot * dt
        denom = 1 + np.sin(self._gamma) ** 2
        x = self._center[0] + self._scale * np.cos(self._gamma) / denom
        y = self._center[1] + self._scale * np.sin(self._gamma) * np.cos(self._gamma) / denom
        z = self._center[2]
        target = _clamp_away_from_zone([x, y, z])
        immediate = _detour_around_zone(self._person.state.position, target)
        self._person.update_target_position(immediate.tolist(), 1.0)


class ReactivePersonController(PersonController):
    """Person behavior reacts to drone proximity:
    - If drone within flee_radius: flee at high speed.
    - If drone beyond approach_radius: slowly approach.
    - Otherwise: idle.
    """

    IDLE = "idle"
    FLEE = "flee"
    FOLLOW = "follow"

    def __init__(self, shared_state, initial_pos, flee_radius=4.0, approach_radius=10.0):
        super().__init__()
        self._shared_state = shared_state
        self._initial_pos = np.array(initial_pos)
        self._flee_radius = flee_radius
        self._approach_radius = approach_radius
        self._mode = self.IDLE

    def update(self, dt: float):
        if _apply_exclusion_zone(self._person):
            return
        drone_pos = self._shared_state.get("drone_position", None)
        my_pos = self._person.state.position

        if drone_pos is None:
            safe_idle = _clamp_away_from_zone(self._initial_pos)
            self._person.update_target_position(safe_idle.tolist(), 0.5)
            return

        drone_pos = np.array(drone_pos)
        dist = np.linalg.norm(drone_pos[:2] - my_pos[:2])

        if dist < self._flee_radius:
            self._mode = self.FLEE
            flee_dir = my_pos[:2] - drone_pos[:2]
            norm = np.linalg.norm(flee_dir)
            if norm > 0.01:
                flee_dir /= norm
            flee_target = my_pos[:2] + flee_dir * 6.0
            flee = _clamp_away_from_zone([flee_target[0], flee_target[1], my_pos[2]])
            flee = _detour_around_zone(my_pos, flee)
            self._person.update_target_position(flee.tolist(), 1.8)
        elif dist > self._approach_radius:
            self._mode = self.FOLLOW
            approach_target = drone_pos.copy()
            approach_target[2] = my_pos[2]
            approach_target = _clamp_away_from_zone(approach_target)
            approach_target = _detour_around_zone(my_pos, approach_target)
            self._person.update_target_position(approach_target.tolist(), 0.8)
        else:
            self._mode = self.IDLE
            safe_idle = _clamp_away_from_zone(self._initial_pos)
            self._person.update_target_position(safe_idle.tolist(), 0.4)


from scipy.spatial.transform import Rotation


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg.initialize_world()

        # Load environment
        if "Full Warehouse" in FLAT_ENVIRONMENTS:
            self.pg.load_asset(FLAT_ENVIRONMENTS["Full Warehouse"], "/World/layout")
        else:
            self.pg.load_asset(SIMULATION_ENVIRONMENTS["Box Room"], "/World/layout")

        self._shared_drone_state = {"drone_position": None}

        patrol_waypoints = [
            [5.0, 5.0, 0.0],
            [5.0, -5.0, 0.0],
            [-5.0, -5.0, 0.0],
            [-5.0, 5.0, 0.0],
            [0.0, 4.0, 0.0],
        ]
        p1 = Person(
            "person1",
            "original_male_adult_construction_05",
            init_pos=[5.0, 5.0, 0.0],
            init_yaw=0.0,
            controller=WaypointPatrolController(patrol_waypoints, speed=1.2, arrival_radius=0.8),
        )

        p2 = Person(
            "person2",
            "original_female_adult_business_02",
            init_pos=[4.0, 8.0, 0.0],
            init_yaw=0.0,
            controller=LemniscateController(center=[0.0, 8.0, 0.0], scale=4.0, speed=0.4),
        )

        p3 = Person(
            "person3",
            "original_male_adult_construction_05",
            init_pos=[-3.0, 3.0, 0.0],
            init_yaw=1.0,
            controller=ReactivePersonController(
                shared_state=self._shared_drone_state,
                initial_pos=[-3.0, 3.0, 0.0],
                flee_radius=4.0,
                approach_radius=10.0,
            ),
        )

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
        })

        config_multirotor.backends = [
            PX4MavlinkBackend(mavlink_config),
            ROS2Backend(
                vehicle_id=1,
                config={
                    "namespace": 'drone',
                    "pub_sensors": False,
                    "pub_graphical_sensors": True,
                    "pub_state": True,
                    "pub_tf": False,
                    "sub_control": False,
                },
            ),
        ]

        config_multirotor.graphical_sensors = [MonocularCamera("camera", config={"update_rate": 60.0})]

        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.pg.set_viewport_camera([5.0, 9.0, 6.5], [0.0, 0.0, 0.0])
        self.stop_sim = False

    def run(self):
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:
            simulation_app.update()

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
