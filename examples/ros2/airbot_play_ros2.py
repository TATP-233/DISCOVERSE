import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, JointState

from discoverse.robots_env.airbot_play_base import AirbotPlayCfg, AirbotPlayBase

cfg = AirbotPlayCfg()
cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"
cfg.decimation = 4
cfg.timestep = 0.0025
cfg.obs_rgb_cam_id = [0,1]
cfg.obs_depth_cam_id = [0,1]
cfg.use_gaussian_renderer = False

class AirbotPlayROS2(AirbotPlayBase, Node):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        Node.__init__(self, 'Airbot_play_node')

        self.tar_jq = np.zeros(self.nj)
        self.joint_state_puber = self.create_publisher(JointState, '/airbot_play/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(6)] + ["gripper"]

        self.bridge = CvBridge()
        self.side_color_puber = self.create_publisher(Image, '/airbot_play/side_camera/color/image_raw', 2)
        self.side_depth_puber = self.create_publisher(Image, '/airbot_play/side_camera/aligned_depth_to_color/image_raw', 2)
        self.arm_color_puber = self.create_publisher(Image, '/airbot_play/arm_camera/color/image_raw', 2)
        self.arm_depth_puber = self.create_publisher(Image, '/airbot_play/arm_camera/aligned_depth_to_color/image_raw', 2)

        self.arm_cmd_suber = self.create_subscription(Float64MultiArray, '/airbot_play/forward_position_controller/commands', self.cmd_arm_callback, 5)

    def cmd_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == self.nj:
            self.tar_jq[:] = np.array(msg.data)
        else:
            print("Invalid arm command length: {}".format(len(msg.data)))

    def resetState(self):
        super().resetState()
        self.tar_jq = np.zeros(self.nj)

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            time_stamp = self.get_clock().now().to_msg()

            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_joint_qpos.tolist()
            self.joint_state.velocity = self.sensor_joint_qvel.tolist()
            self.joint_state.effort = self.sensor_joint_force.tolist()
            self.joint_state_puber.publish(self.joint_state)

            side_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][0], encoding="rgb8")
            side_color_img_msg.header.stamp = time_stamp
            side_color_img_msg.header.frame_id = "side_camera"
            self.side_color_puber.publish(side_color_img_msg)

            side_depth_img = np.array(np.clip(self.obs["depth"][0]*1e3, 0, 65535), dtype=np.uint16)
            side_depth_img_msg = self.bridge.cv2_to_imgmsg(side_depth_img, encoding="mono16")
            side_depth_img_msg.header.stamp = time_stamp
            side_depth_img_msg.header.frame_id = "side_camera"
            self.side_depth_puber.publish(side_depth_img_msg)

            arm_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][1], encoding="rgb8")
            arm_color_img_msg.header.stamp = time_stamp
            arm_color_img_msg.header.frame_id = "arm_camera"
            self.arm_color_puber.publish(arm_color_img_msg)

            arm_depth_img = np.array(np.clip(self.obs["depth"][1]*1e3, 0, 65535), dtype=np.uint16)
            arm_depth_img_msg = self.bridge.cv2_to_imgmsg(arm_depth_img, encoding="mono16")
            arm_depth_img_msg.header.stamp = time_stamp
            arm_depth_img_msg.header.frame_id = "arm_camera"
            self.arm_depth_puber.publish(arm_depth_img_msg)

            rate.sleep()

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("joint tar_q = {}".format(np.array2string(self.tar_jq, separator=', ')))
        print("joint q     = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("joint v     = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))

        print("sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    exec_node = AirbotPlayROS2(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.step(exec_node.tar_jq)

    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()