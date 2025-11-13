import os
import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from sensor_msgs.msg import Joy

from discoverse.robots import AirbotPlayIK
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils.joy_stick_ros2 import JoyTeleopRos2
from discoverse.examples.ros2.airbot_play_ros2 import AirbotPlayROS2, cfg

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"

class AirbotPlayROS2JoyCtl(AirbotPlayROS2):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)

        self.arm_ik = AirbotPlayIK()
    
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)

        self.teleop = JoyTeleopRos2()
        self.sub = self.create_subscription(Joy, '/joy', self.teleop.joy_callback, 10)

    def resetState(self):
        super().resetState()
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.teleop.reset()

    def teleopProcess(self):
        calc_ik = False
        tmp_arm_target_pose = self.tar_end_pose.copy()
        if self.teleop.joy_cmd.axes[0] or self.teleop.joy_cmd.axes[1] or self.teleop.joy_cmd.axes[4]:
            calc_ik = True
            tmp_arm_target_pose[0] += 0.15 * self.teleop.joy_cmd.axes[1] * self.delta_t
            tmp_arm_target_pose[1] += 0.15 * self.teleop.joy_cmd.axes[0] * self.delta_t
            tmp_arm_target_pose[2] += 0.1 * self.teleop.joy_cmd.axes[4] * self.delta_t

        el = self.tar_end_euler.copy()
        if self.teleop.joy_cmd.axes[3] or self.teleop.joy_cmd.axes[6] or self.teleop.joy_cmd.axes[7]:
            calc_ik = True
            el[0] += 0.01 * self.teleop.joy_cmd.axes[3]
            el[1] += 0.01 * self.teleop.joy_cmd.axes[7]
            el[2] += 0.01 * self.teleop.joy_cmd.axes[6]

        if calc_ik:
            rot = Rotation.from_euler('xyz', el).as_matrix()
            try:
                tarjq = self.arm_ik.properIK(self.tar_end_pose, rot, self.sensor_joint_qpos[:6])
                self.tar_end_pose[:] = tmp_arm_target_pose[:]
                self.tar_end_euler[:] = el[:]
            except ValueError:
                tarjq = None

            if not tarjq is None:
                self.tar_jq[:6] = tarjq
            else:
                self.get_logger().warn("Fail to solve inverse kinematics trans={} euler={}".format(self.tar_end_pose, self.tar_end_euler))

        if self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]:
            self.tar_jq[6] += 1. * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * self.delta_t
            self.tar_jq[6] = np.clip(self.tar_jq[6], 0, 1.)

    def printMessage(self):
        super().printMessage()
        print("target end posi  = {}".format(np.array2string(self.tar_end_pose, separator=', ')))
        print("target end euler = {}".format(np.array2string(self.tar_end_euler, separator=', ')))

if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = AirbotPlayROS2JoyCtl(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.teleopProcess()
        exec_node.step(exec_node.tar_jq)

    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()