import numpy as np
import threading
from discoverse.examples.hardware_in_loop.airbot_arm import AirbotArm, cfg

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class AirbotPlayShortROS2(AirbotArm, Node):
    def __init__(self, config):
        super().__init__(config)
        Node.__init__(self, "Airbot_play_short_node")

        self.joint_state_puber = self.create_publisher(
            JointState, f"/airbot_{config.arm_type}/joint_states", 5
        )
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(self.nj)]
        if config.eef_type != "none":
            self.joint_state.name.append(config.eef_type)
        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            self.joint_state.header.stamp = self.get_clock().now().to_msg()
            self.joint_state.position = self.sensor_joint_qpos.tolist()
            self.joint_state.velocity = self.sensor_joint_qvel.tolist()
            self.joint_state.effort = self.sensor_joint_force.tolist()
            self.joint_state_puber.publish(self.joint_state)
            rate.sleep()


if __name__ == "__main__":
    rclpy.init()

    import argparse

    parser = argparse.ArgumentParser(
        description="Run arm with specified parameters. \ne.g. python3 airbot_play_short.py --arm_type play_short --eef_type none"
    )
    parser.add_argument(
        "--arm_type",
        type=str,
        choices=["play_long", "play_short", "lite", "pro", "replay"],
        help="Name of the arm",
        default="play_short",
    )
    parser.add_argument(
        "--eef_type",
        type=str,
        choices=["G2", "E2B", "PE2", "none"],
        help="Name of the eef",
        default="none",
    )
    parser.add_argument(
        "--discoverse_viewer", action="store_true", help="Use discoverse viewer"
    )
    args = parser.parse_args()

    cfg.arm_type = args.arm_type
    cfg.eef_type = args.eef_type
    exec_node = AirbotPlayShortROS2(cfg)

    spin_thread = threading.Thread(target=lambda: rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    action = np.zeros(exec_node.nj * 3)
    obs = exec_node.reset()
    action[: exec_node.nj] = 0.2

    while rclpy.ok() and exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step()

    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()
