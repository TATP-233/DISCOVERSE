import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from discoverse.robots_env.tok2_base import TOK2Base, TOK2Cfg
from discoverse.utils.joy_stick_ros2 import JoyTeleopRos2

class TOK2JOY(TOK2Base, Node):

    target_control = np.zeros(16)
    def __init__(self, config: TOK2Cfg):
        self.tctr_base = self.target_control[:2]
        self.tctr_left_arm = self.target_control[2:8]
        self.tctr_lft_gripper = self.target_control[8:9]
        self.tctr_right_arm = self.target_control[9:15]
        self.tctr_rgt_gripper = self.target_control[15:16]

        super().__init__(config)
        Node.__init__(self, 'TOK2_node')

        self.teleop = JoyTeleopRos2()
        self.sub = self.create_subscription(Joy, '/joy_throttle', self.teleop.joy_callback, 10)

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.teleop.reset()

    def teleopProcess(self):
        linear_vel  = 1.0 * self.teleop.joy_cmd.axes[1]**2 * np.sign(self.teleop.joy_cmd.axes[1])
        angular_vel = 2.0 * self.teleop.joy_cmd.axes[0]**2 * np.sign(self.teleop.joy_cmd.axes[0])
        self.base_move(linear_vel, angular_vel)

    def base_move(self, linear_vel, angular_vel):
        self.tctr_base[0] = linear_vel
        self.tctr_base[1] = angular_vel

    def printMessage(self):
        super().printMessage()

        print("    lta local = {}".format(self.lft_arm_target_pose))
        print("    rta local = {}".format(self.rgt_arm_target_pose))
        print("       euler  = {}".format(self.lft_end_euler))
        print("       euler  = {}".format(self.rgt_end_euler))

if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = TOK2Cfg()
    
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = None
    cfg.obs_depth_cam_id = None

    cfg.render_set     = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080
    }
    cfg.mjcf_file_path = "mjcf/tok2_floor.xml"

    exec_node = TOK2JOY(cfg)
    exec_node.reset()

    while rclpy.ok() and exec_node.running:
        exec_node.teleopProcess()
        obs, _, _, _, _ = exec_node.step(exec_node.target_control)
        rclpy.spin_once(exec_node)

    exec_node.destroy_node()
    rclpy.shutdown()