#!/usr/bin/env python

import math
from array import array
import json
import time
import uuid
from pegasus.simulator.logic.backends.tools.simulation_timestamp_policy import ros_time_parts
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import StaticTransformBroadcaster
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32, Float32MultiArray, String
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from pegasus.simulator.logic.state import State
from rclpy.parameter import Parameter

GRAVITY = 9.81


class DroneLocationPublisher(Node):
    def __init__(
        self,
        namespace="drone",
        vehicle_id=0,
        lidar_trans=[0.0795, 0.0, 0.0323],
        lidar_ori=[0.9238795, 0.0, 0.3826834, 0.0],
    ):
        self.namespace = namespace
        self.vehicle_id = vehicle_id
        self.vehicle_name = f"{self.namespace}{self.vehicle_id}"
        super().__init__(f"{self.vehicle_name}_location_publisher")
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.gt_publisher_ = self.create_publisher(
            PoseStamped, f"{self.vehicle_name}/gt_pose", qos_profile
        )
        # Direct physics observation for the native GT authority. This is
        # simulator State, never PX4 estimation or differentiated poses.
        self.physics_gt_publisher_ = self.create_publisher(
            Odometry, f"{self.vehicle_name}/physics_gt/odom", qos_profile_sensor_data
        )
        self._physics_gt_last_stamp_ns = -1
        self.rtf_publisher_ = self.create_publisher(Float32, "real_Time_factor", qos_profile)
        self.self_imu_publisher_ = self.create_publisher(
            Imu, f"{self.vehicle_name}/self_imu", qos_profile
        )
        self.imu_publisher_ = self.create_publisher(
            Imu, f"{self.vehicle_name}/gt_imu", qos_profile
        )
        self.forces_publisher = self.create_publisher(
            Float32MultiArray, f"{self.vehicle_name}/gt_forces", qos_profile
        )
        self.time_publisher = self.create_publisher(Clock, "clock", qos_profile)
        self.timeline_publisher = self.create_publisher(String, "/agisim/timeline_state", 1)
        self.timeline_session = uuid.uuid4().hex
        self.timeline_sequence = 0
        self.timeline_last_wall = 0.0

        self.lidar_trans = lidar_trans
        self.lidar_ori = lidar_ori
        lidar_frame_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_transform(
            f"{self.vehicle_name}/lidar_link",
            f"{self.vehicle_name}/base_link",
            self.lidar_trans,
            self.lidar_ori,
            lidar_frame_broadcaster,
        )

    def enable_livox_imu(self):
        """Opt-in emulated Livox IMU, /<ns>/livox/imu: reliable like the Livox driver (LIOM subscribes reliable)."""
        self._livox_imu_publisher = self.create_publisher(Imu, f"{self.vehicle_name}/livox/imu", 200)
        self._livox_imu_frame = f"{self.vehicle_name}/lidar_link"  # the sim lidar cloud's frame

    def publish_livox_imu(self, stamp_s, accel_g, gyro):
        msg = Imu()
        msg.header.stamp.sec, msg.header.stamp.nanosec = ros_time_parts(stamp_s)
        msg.header.frame_id = self._livox_imu_frame
        msg.orientation.w = 1.0  # the Livox driver publishes no orientation
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(v) for v in accel_g)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(v) for v in gyro)
        self._livox_imu_publisher.publish(msg)

    def enable_splat_preview(self):
        """Opt-in preview only; existing publishers and mode-off topics are unchanged."""
        from sensor_msgs.msg import CompressedImage, Image
        from agipix_splat_preview import encode_preview_jpeg

        self._splat_image_type = Image
        self._splat_compressed_image_type = CompressedImage
        self._encode_splat_jpeg = encode_preview_jpeg
        self._splat_preview_publisher = self.create_publisher(
            Image, f"{self.vehicle_name}/splat/preview", qos_profile_sensor_data
        )
        self._splat_preview_compressed_publisher = self.create_publisher(
            CompressedImage, f"{self.vehicle_name}/splat/preview/compressed", qos_profile_sensor_data
        )

    def publish_splat_preview(self, rgb, timestamp_ns):
        msg = self._splat_image_type()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(timestamp_ns), 10**9)
        msg.header.frame_id = f"{self.vehicle_name}/splat_panorama"
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = msg.width * 3
        # Typed byte arrays use the ROS setter's fast path instead of checking
        # every pixel channel in Python. The serialized payload is identical.
        msg.data = array("B", rgb.tobytes())
        self._splat_preview_publisher.publish(msg)
        # The bridge subscribes when a Foxglove panel selects this topic. Avoid
        # JPEG work when no compressed-image consumer is connected.
        if self._splat_preview_compressed_publisher.get_subscription_count() > 0:
            compressed = self._splat_compressed_image_type()
            compressed.header = msg.header
            compressed.format = "rgb8; jpeg compressed bgr8"
            compressed.data = array("B", self._encode_splat_jpeg(rgb))
            self._splat_preview_compressed_publisher.publish(compressed)

    def publish_static_transform(
        self, ch_frame, pr_frame, translation_xyz, orient_wxyz, broadcaster
    ):
        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = pr_frame
        static_transform_stamped.child_frame_id = ch_frame

        static_transform_stamped.transform.translation.x = float(translation_xyz[0])
        static_transform_stamped.transform.translation.y = float(translation_xyz[1])
        static_transform_stamped.transform.translation.z = float(translation_xyz[2])

        static_transform_stamped.transform.rotation.w = float(orient_wxyz[0])
        static_transform_stamped.transform.rotation.x = float(orient_wxyz[1])
        static_transform_stamped.transform.rotation.y = float(orient_wxyz[2])
        static_transform_stamped.transform.rotation.z = float(orient_wxyz[3])

        broadcaster.sendTransform(static_transform_stamped)

    def quattovec(self, quatf):
        vec = []
        real_part = quatf.GetReal()
        imaginary_part = quatf.GetImaginary()
        vec.append(imaginary_part[0])
        vec.append(imaginary_part[1])
        vec.append(imaginary_part[2])
        vec.append(real_part)
        return vec

    def rotate_vector_by_quaternion(self, vector, quat):
        vector = np.array(vector)
        r = Rotation.from_quat(quat)
        return r.apply(vector)

    def publish_clock(self, sim_time):
        time_msg = Clock()
        time_msg.clock.sec, time_msg.clock.nanosec = ros_time_parts(sim_time)
        self.time_publisher.publish(time_msg)

    def publish_timeline_state(self, timeline, sim_time):
        # App updates continue while physics is paused. This is trusted timeline
        # evidence, independent of /clock loss, and never publishes commands.
        wall = time.monotonic()
        state = "playing" if timeline.is_playing() else "stopped" if timeline.is_stopped() else "paused"
        if state == getattr(self, "timeline_last_state", None) and wall - self.timeline_last_wall < 0.05:
            return
        self.timeline_last_wall = wall
        self.timeline_last_state = state
        self.timeline_sequence += 1
        sec, ns = ros_time_parts(sim_time)
        msg = String()
        msg.data = json.dumps({"schema": "agisim.timeline.v1", "session": self.timeline_session,
                               "sequence": self.timeline_sequence, "sim_ns": sec * 10**9 + ns,
                               "state": state}, separators=(",", ":"))
        self.timeline_publisher.publish(msg)

    def publish_self_imu(self, self_imu):
        if self_imu is None:
            return
        msg = Imu()
        sim_time = self_imu.get("time", 0.0)
        sim_time_ = Time()
        sim_time_.sec, sim_time_.nanosec = ros_time_parts(sim_time)
        msg.header.stamp = sim_time_
        msg.header.frame_id = f"{self.vehicle_name}/base_link"

        lin_acc = self_imu.get("linear_acceleration", self_imu.get("lin_acc", [0.0, 0.0, 0.0]))
        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])

        ori = self_imu.get("orientation", [1.0, 0.0, 0.0, 0.0])
        msg.orientation.w = float(ori[0])
        msg.orientation.x = float(ori[1])
        msg.orientation.y = float(ori[2])
        msg.orientation.z = float(ori[3])

        ang_vel = self_imu.get("angular_velocity", self_imu.get("ang_vel", [0.0, 0.0, 0.0]))
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])

        self.self_imu_publisher_.publish(msg)

    def publish_gt(self, state: State, sim_time):
        msg = PoseStamped()
        sim_time_ = Time()
        sim_time_.sec, sim_time_.nanosec = ros_time_parts(sim_time)
        msg.header.stamp = sim_time_
        msg.header.frame_id = "vicon_map"

        position = state.position
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        orientation = state.attitude
        msg.pose.orientation.x = float(orientation[0])
        msg.pose.orientation.y = float(orientation[1])
        msg.pose.orientation.z = float(orientation[2])
        msg.pose.orientation.w = float(orientation[3])

        self.gt_publisher_.publish(msg)
        self.publish_physics_gt(state, sim_time)

    def publish_physics_gt(self, state: State, sim_time):
        """Publish original ENU pose and FLU body twist at the physics stamp."""
        def finite_vector(values, size):
            result = tuple(float(value) for value in values)
            if len(result) != size or not all(math.isfinite(value) for value in result):
                raise ValueError("Invalid original physics State")
            return result

        try:
            position = finite_vector(state.position, 3)
            orientation = finite_vector(state.attitude, 4)  # xyzw, body -> world
            linear = finite_vector(state.linear_body_velocity, 3)
            angular = finite_vector(state.angular_velocity, 3)
            if abs(sum(value * value for value in orientation) - 1.0) > 1e-5:
                return
            sec, nanosec = ros_time_parts(sim_time)
        except (TypeError, ValueError, OverflowError):
            return  # Withhold invalid observations; never renew their lease.
        stamp_ns = sec * 1_000_000_000 + nanosec
        if stamp_ns <= self._physics_gt_last_stamp_ns:
            return  # Clock rewind is visible on /clock; do not restamp State.

        msg = Odometry()
        msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
        msg.header.frame_id = f"{self.vehicle_name}/world"
        msg.child_frame_id = f"{self.vehicle_name}/base_link"
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        p.x, p.y, p.z = position
        q.x, q.y, q.z, q.w = orientation
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        v.x, v.y, v.z = linear
        w.x, w.y, w.z = angular
        # ROS zero covariance convention; no calibrated uncertainty is claimed.
        self.physics_gt_publisher_.publish(msg)
        self._physics_gt_last_stamp_ns = stamp_ns

    def publish_gt_imu(self, sim_time, state: State):
        msg = Imu()
        sim_time_ = Time()
        sim_time_.sec, sim_time_.nanosec = ros_time_parts(sim_time)
        msg.header.stamp = sim_time_
        msg.header.frame_id = f"{self.vehicle_name}/base_link"

        if hasattr(state, "get_linear_body_acceleration_ned_frd"):
            linear_acceleration = state.get_linear_body_acceleration_ned_frd()
        else:
            linear_acceleration = state.get_linear_body_velocity_ned_frd()

        msg.linear_acceleration.x = float(linear_acceleration[0])
        msg.linear_acceleration.y = float(linear_acceleration[1])
        msg.linear_acceleration.z = float(linear_acceleration[2] - GRAVITY)

        angular_velocity = state.get_angular_velocity_frd()
        msg.angular_velocity.x = float(angular_velocity[0])
        msg.angular_velocity.y = float(angular_velocity[1])
        msg.angular_velocity.z = float(angular_velocity[2])

        attitude = state.get_attitude_ned_frd()
        msg.orientation.x = float(attitude[0])
        msg.orientation.y = float(attitude[1])
        msg.orientation.z = float(attitude[2])
        msg.orientation.w = float(attitude[3])

        self.imu_publisher_.publish(msg)

    def publish_gt_forces(self, prop_forces, rolling_torque):
        forces = Float32MultiArray()
        prop_forces = np.array(prop_forces, dtype=np.float32)
        forces.data = [
            float(prop_forces[0]),
            float(prop_forces[1]),
            float(prop_forces[2]),
            float(prop_forces[3]),
            float(np.sum(prop_forces)),
            float(rolling_torque),
        ]
        self.forces_publisher.publish(forces)

    def publish_rtf(self, real_dt, sim_dt):
        msg = Float32()
        if real_dt and real_dt > 0.0:
            msg.data = float(sim_dt / real_dt)
            self.rtf_publisher_.publish(msg)

    def check_clock_topic(self):
        topics_and_types = self.get_topic_names_and_types()
        clock_topic_exists = any(topic == "/clock" for topic, _ in topics_and_types)
        if clock_topic_exists:
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
            self.timer.cancel()
        self.get_logger().info("/clock topic not available. use_sim_time remains False.")
