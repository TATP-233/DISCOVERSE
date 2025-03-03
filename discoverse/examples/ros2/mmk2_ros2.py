import threading
import numpy as np

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image, CameraInfo, JointState

from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.utils import camera2k

cfg = MMK2Cfg()
cfg.mjcf_file_path = "mjcf/tasks_mmk2/plate_coffeecup.xml"
cfg.use_gaussian_renderer = False
cfg.obs_rgb_cam_id = [0, 1, 2]
cfg.obs_depth_cam_id = [0]
cfg.render_set = {"fps": 30, "width": 640, "height": 480}


class MMK2ROS2(MMK2Base, Node):
    target_control = np.zeros(19)

    def __init__(self, config: MMK2Cfg):
        self.arm_action = config.init_key
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)
        Node.__init__(self, "MMK2_mujoco_node")

        # joint state
        self.joint_state_puber = self.create_publisher(
            JointState, "/mmk2/joint_states", 5
        )
        self.joint_state = JointState()
        self.joint_state.name = [
            "slide_joint",
            "head_yaw_joint",
            "head_pitch_joint",
            "left_arm_joint1",
            "left_arm_joint2",
            "left_arm_joint3",
            "left_arm_joint4",
            "left_arm_joint5",
            "left_arm_joint6",
            "left_arm_eef_gripper_joint",
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
            "right_arm_eef_gripper_joint",
        ]

        self.joint_state.position = self.sensor_qpos[2:].tolist()
        self.joint_state.velocity = self.sensor_qvel[2:].tolist()
        self.joint_state.effort = self.sensor_force[2:].tolist()

        # geometry
        self.odom_puber = self.create_publisher(Odometry, "/mmk2/odom", 5)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "mmk2_footprint"

        # image
        self.bridge = CvBridge()
        self.head_color_puber = self.create_publisher(
            Image, "/mmk2/head_camera/color/image_raw", 2
        )
        self.head_depth_puber = self.create_publisher(
            Image, "/mmk2/head_camera/aligned_depth_to_color/image_raw", 2
        )
        self.left_color_puber = self.create_publisher(
            Image, "/mmk2/left_camera/color/image_raw", 2
        )
        self.right_color_puber = self.create_publisher(
            Image, "/mmk2/right_camera/color/image_raw", 2
        )

        # camera info publishers
        self.head_color_info_puber = self.create_publisher(
            CameraInfo, "/mmk2/head_camera/color/camera_info", 2
        )
        self.head_depth_info_puber = self.create_publisher(
            CameraInfo, "/mmk2/head_camera/aligned_depth_to_color/camera_info", 2
        )
        self.left_color_info_puber = self.create_publisher(
            CameraInfo, "/mmk2/left_camera/color/camera_info", 2
        )
        self.right_color_info_puber = self.create_publisher(
            CameraInfo, "/mmk2/right_camera/color/camera_info", 2
        )

        # Initialize camera info messages
        self.head_color_info = CameraInfo()
        self.head_depth_info = CameraInfo()
        self.left_color_info = CameraInfo()
        self.right_color_info = CameraInfo()

        # Set camera info parameters
        self.head_color_info.width = self.config.render_set["width"]
        self.head_color_info.height = self.config.render_set["height"]
        self.head_color_info.k = (
            camera2k(
                self.mj_model.cam_fovy[0] * np.pi / 180.0,
                self.config.render_set["width"],
                self.config.render_set["height"],
            )
            .flatten()
            .tolist()
        )

        self.head_depth_info.width = self.config.render_set["width"]
        self.head_depth_info.height = self.config.render_set["height"]
        self.head_depth_info.k = (
            camera2k(
                self.mj_model.cam_fovy[0] * np.pi / 180.0,
                self.config.render_set["width"],
                self.config.render_set["height"],
            )
            .flatten()
            .tolist()
        )

        self.left_color_info.width = self.config.render_set["width"]
        self.left_color_info.height = self.config.render_set["height"]
        self.left_color_info.k = (
            camera2k(
                self.mj_model.cam_fovy[1] * np.pi / 180.0,
                self.config.render_set["width"],
                self.config.render_set["height"],
            )
            .flatten()
            .tolist()
        )

        self.right_color_info.width = self.config.render_set["width"]
        self.right_color_info.height = self.config.render_set["height"]
        self.right_color_info.k = (
            camera2k(
                self.mj_model.cam_fovy[2] * np.pi / 180.0,
                self.config.render_set["width"],
                self.config.render_set["height"],
            )
            .flatten()
            .tolist()
        )

        # Publish camera info periodically
        self.create_timer(1.0, self.publish_camera_info)

        # command subscriber
        self.cmd_vel_suber = self.create_subscription(
            Twist, "/mmk2/cmd_vel", self.cmd_vel_callback, 5
        )
        self.spine_cmd_suber = self.create_subscription(
            Float64MultiArray,
            "/mmk2/spine_forward_position_controller/commands",
            self.cmd_spine_callback,
            5,
        )
        self.head_cmd_suber = self.create_subscription(
            Float64MultiArray,
            "/mmk2/head_forward_position_controller/commands",
            self.cmd_head_callback,
            5,
        )
        self.left_arm_cmd_suber = self.create_subscription(
            Float64MultiArray,
            "/mmk2/left_arm_forward_position_controller/commands",
            self.cmd_left_arm_callback,
            5,
        )
        self.right_arm_cmd_suber = self.create_subscription(
            Float64MultiArray,
            "/mmk2/right_arm_forward_position_controller/commands",
            self.cmd_right_arm_callback,
            5,
        )

    def publish_camera_info(self):
        self.head_color_info_puber.publish(self.head_color_info)
        self.head_depth_info_puber.publish(self.head_depth_info)
        self.left_color_info_puber.publish(self.left_color_info)
        self.right_color_info_puber.publish(self.right_color_info)

    def cmd_vel_callback(self, msg: Twist):
        self.tctr_base[0] = msg.linear.x
        self.tctr_base[1] = msg.angular.z

    def cmd_spine_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 1:
            self.tctr_slide[:] = msg.data[:]
        else:
            print("spine command length error")

    def cmd_head_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.tctr_head[:] = msg.data[:]
        else:
            print("head command length error")

    def cmd_left_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_left_arm[:] = msg.data[:6]
            self.tctr_lft_gripper[:] = msg.data[6:]
        else:
            print("left arm command length error")

    def cmd_right_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_right_arm[:] = msg.data[:6]
            self.tctr_rgt_gripper[:] = msg.data[6:]
        else:
            print("right arm command length error")

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            time_stamp = self.get_clock().now().to_msg()
            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_qpos[2:].tolist()
            self.joint_state.velocity = self.sensor_qvel[2:].tolist()
            self.joint_state.effort = self.sensor_force[2:].tolist()
            self.joint_state_puber.publish(self.joint_state)

            self.odom_msg.pose.pose.position.x = self.sensor_base_position[0]
            self.odom_msg.pose.pose.position.y = self.sensor_base_position[1]
            self.odom_msg.pose.pose.position.z = self.sensor_base_position[2]
            self.odom_msg.pose.pose.orientation.w = self.sensor_base_orientation[0]
            self.odom_msg.pose.pose.orientation.x = self.sensor_base_orientation[1]
            self.odom_msg.pose.pose.orientation.y = self.sensor_base_orientation[2]
            self.odom_msg.pose.pose.orientation.z = self.sensor_base_orientation[3]
            self.odom_puber.publish(self.odom_msg)

            head_color_img_msg = self.bridge.cv2_to_imgmsg(
                self.obs["img"][0], encoding="rgb8"
            )
            head_color_img_msg.header.stamp = time_stamp
            head_color_img_msg.header.frame_id = "head_camera"
            self.head_color_puber.publish(head_color_img_msg)

            head_depth_img = np.array(
                np.clip(self.obs["depth"][0] * 1e3, 0, 65535), dtype=np.uint16
            )
            head_depth_img_msg = self.bridge.cv2_to_imgmsg(
                head_depth_img, encoding="mono16"
            )
            head_depth_img_msg.header.stamp = time_stamp
            head_depth_img_msg.header.frame_id = "head_camera"
            self.head_depth_puber.publish(head_depth_img_msg)

            left_color_img_msg = self.bridge.cv2_to_imgmsg(
                self.obs["img"][1], encoding="rgb8"
            )
            left_color_img_msg.header.stamp = time_stamp
            left_color_img_msg.header.frame_id = "left_camera"
            self.left_color_puber.publish(left_color_img_msg)

            right_color_img_msg = self.bridge.cv2_to_imgmsg(
                self.obs["img"][2], encoding="rgb8"
            )
            right_color_img_msg.header.stamp = time_stamp
            right_color_img_msg.header.frame_id = "right_camera"
            self.right_color_puber.publish(right_color_img_msg)

            rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg.init_key = "pick"
    exec_node = MMK2ROS2(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda: rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.step(exec_node.target_control)

    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()
