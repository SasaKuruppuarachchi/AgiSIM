#!/usr/bin/env python

import math
import numpy as np
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_ros import StaticTransformBroadcaster
from rclpy.node import Node
from sensor_msgs.msg import Imu
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32, Float32MultiArray
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
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
        time_msg.clock.sec = math.floor(sim_time)
        time_msg.clock.nanosec = int((sim_time - time_msg.clock.sec) * 1e9)
        self.time_publisher.publish(time_msg)

    def publish_self_imu(self, self_imu):
        if self_imu is None:
            return
        msg = Imu()
        sim_time = self_imu.get("time", 0.0)
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
        msg.header.stamp = sim_time_
        msg.header.frame_id = f"{self.vehicle_name}/base_link"

        lin_acc = self_imu.get("lin_acc", [0.0, 0.0, 0.0])
        msg.linear_acceleration.x = float(lin_acc[0])
        msg.linear_acceleration.y = float(lin_acc[1])
        msg.linear_acceleration.z = float(lin_acc[2])

        ori = self_imu.get("orientation", [1.0, 0.0, 0.0, 0.0])
        msg.orientation.w = float(ori[0])
        msg.orientation.x = float(ori[1])
        msg.orientation.y = float(ori[2])
        msg.orientation.z = float(ori[3])

        ang_vel = self_imu.get("ang_vel", [0.0, 0.0, 0.0])
        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])

        self.self_imu_publisher_.publish(msg)

    def publish_gt(self, state: State, sim_time):
        msg = PoseStamped()
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
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

    def publish_gt_imu(self, sim_time, state: State):
        msg = Imu()
        sim_time_ = Time()
        sim_time_.sec = math.floor(sim_time)
        sim_time_.nanosec = int((sim_time - sim_time_.sec) * 1e9)
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
