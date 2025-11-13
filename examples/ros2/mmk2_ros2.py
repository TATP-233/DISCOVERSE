import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo, JointState, Imu, LaserScan

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.utils import PIDarray, camera2k, get_site_tmat

class MMK2ROS2(MMK2Base, Node):
    target_control = np.zeros(19)
    def __init__(self, config: MMK2Cfg):
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)
        Node.__init__(self, 'MMK2_mujoco_node')

        self.pid_base_vel = PIDarray(
            kps=np.array([ 7.5 ,  7.5 ]),
            kis=np.array([  .0 ,   .0 ]),
            kds=np.array([  .0 ,   .0 ]),
            integrator_maxs=np.array([5.0, 5.0]),
        )

        self.init_topic_publisher()
        self.init_topic_subscriber()

        if self.config.lidar_s2_sim:
            from mujoco_lidar.lidar_wrapper import MjLidarWrapper
            from mujoco_lidar.scan_gen import create_lidar_single_line

            self.lidar_frame_id = "laser"
            
            self.rays_theta, self.rays_phi = create_lidar_single_line(360, np.pi*2.)

            self.lidar_s2 = MjLidarWrapper(self.mj_model, self.mj_data, site_name=self.lidar_frame_id)
            self.lidar_s2.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data)
    
            self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self.publish_static_transform(header_frame_id='base_link', child_frame_id=self.lidar_frame_id)

    def init_topic_publisher(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # rostopic joint state
        self.joint_state_puber = self.create_publisher(JointState, '/joint_states', 5)
        self.joint_state = JointState()
        self.joint_state.name = [
            "slide_joint", "head_yaw_joint", "head_pitch_joint",
            "left_arm_joint1" , "left_arm_joint2" , "left_arm_joint3" , "left_arm_joint4" , "left_arm_joint5" , "left_arm_joint6" , "left_arm_eef_gripper_joint" ,
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
        ]
        self.joint_state.position = self.sensor_qpos[2:].tolist()
        self.joint_state.velocity = self.sensor_qvel[2:].tolist()
        self.joint_state.effort = self.sensor_force[2:].tolist()

        # rostopic imu
        # # # self.imu_puber = self.create_publisher(Imu, '/imu', 5)
        # # self.imu_msg = Imu()
        # # self.imu_msg.header.frame_id = "mmk2_imu_link"

        # rostopic odometry
        self.odom_puber = self.create_publisher(Odometry, '/slamware_ros_sdk_server_node/odom', 5)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "/odom"
        # self.odom_msg.child_frame_id = "base_link"

        # image
        self.bridge = CvBridge()

        # lidar
        if self.config.lidar_s2_sim:
            self.lidar_s2_puber = self.create_publisher(LaserScan, '/slamware_ros_sdk_server_node/scan', 1)

        # image publisher, camera info publisher,  Initialize camera info messages
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_puber  = self.create_publisher(Image, '/head_camera/color/image_raw', 2)
            self.head_color_info_puber  = self.create_publisher(CameraInfo, '/head_camera/color/camera_info', 2)
            self.head_color_info = CameraInfo()
            self.head_color_info.width = self.config.render_set["width"]
            self.head_color_info.height = self.config.render_set["height"]
            self.head_color_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_puber  = self.create_publisher(Image, '/left_camera/color/image_raw', 2)
            self.left_color_info_puber  = self.create_publisher(CameraInfo, '/left_camera/color/camera_info', 2)
            self.left_color_info = CameraInfo()
            self.left_color_info.width = self.config.render_set["width"]
            self.left_color_info.height = self.config.render_set["height"]
            self.left_color_info.k = camera2k(self.mj_model.cam_fovy[1] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_puber = self.create_publisher(Image, '/right_camera/color/image_raw', 2)
            self.right_color_info_puber = self.create_publisher(CameraInfo, '/right_camera/color/camera_info', 2)
            self.right_color_info = CameraInfo()
            self.right_color_info.width = self.config.render_set["width"]
            self.right_color_info.height = self.config.render_set["height"]
            self.right_color_info.k = camera2k(self.mj_model.cam_fovy[2] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()

        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_puber  = self.create_publisher(Image, '/head_camera/aligned_depth_to_color/image_raw', 2)
            self.head_depth_info_puber  = self.create_publisher(CameraInfo, '/head_camera/aligned_depth_to_color/camera_info', 2)
            self.head_depth_info = CameraInfo()
            self.head_depth_info.width = self.config.render_set["width"]
            self.head_depth_info.height = self.config.render_set["height"]
            self.head_depth_info.k = camera2k(self.mj_model.cam_fovy[0] * np.pi / 180., self.config.render_set["width"], self.config.render_set["height"]).flatten().tolist()
 
        if (self.config.obs_rgb_cam_id is not None) or (self.config.obs_depth_cam_id is not None):            
            # Publish camera info periodically
            self.create_timer(1.0, self.publish_camera_info)

    def init_topic_subscriber(self):
        self.cmd_vel_suber = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 5)
        self.spine_cmd_suber = self.create_subscription(Float64MultiArray, '/spine_forward_position_controller/commands', self.cmd_spine_callback, 5)
        self.head_cmd_suber = self.create_subscription(Float64MultiArray, '/head_forward_position_controller/commands', self.cmd_head_callback, 5)
        self.left_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/left_arm_forward_position_controller/commands', self.cmd_left_arm_callback, 5)
        self.right_arm_cmd_suber = self.create_subscription(Float64MultiArray, '/right_arm_forward_position_controller/commands', self.cmd_right_arm_callback, 5)

    def publish_camera_info(self):
        if 0 in self.config.obs_rgb_cam_id:
            self.head_color_info_puber.publish(self.head_color_info)
        if 1 in self.config.obs_rgb_cam_id:
            self.left_color_info_puber.publish(self.left_color_info)
        if 2 in self.config.obs_rgb_cam_id:
            self.right_color_info_puber.publish(self.right_color_info)
        if 0 in self.config.obs_depth_cam_id:
            self.head_depth_info_puber.publish(self.head_depth_info)

    def cmd_vel_callback(self, msg: Twist):
        self.tctr_base[0] = (msg.linear.x - msg.angular.z * self.wheel_distance) / self.wheel_radius
        self.tctr_base[1] = (msg.linear.x + msg.angular.z * self.wheel_distance) / self.wheel_radius

    def cmd_spine_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 1:
            self.tctr_slide[:] = msg.data[:]
        else:
            self.get_logger().error('Spine command length error')

    def cmd_head_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.tctr_head[:] = msg.data[:]
        else:
            self.get_logger().error("head command length error")

    def cmd_left_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_left_arm[:] = msg.data[:6]
            self.tctr_lft_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("left arm command length error")

    def cmd_right_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_right_arm[:] = msg.data[:6]
            self.tctr_rgt_gripper[:] = msg.data[6:]
        else:
            self.get_logger().error("right arm command length error")

    def resetState(self):
        super().resetState()
        self.pid_base_vel.reset()
        self.target_control[:] = self.init_joint_ctrl[:]

    def updateControl(self, action):
        wheel_force = self.pid_base_vel.output(np.clip(self.tctr_base - self.sensor_wheel_qvel, -2.5, 2.5), self.mj_model.opt.timestep)
        self.mj_data.ctrl[:2] = np.clip(wheel_force, self.mj_model.actuator_ctrlrange[:2,0], self.mj_model.actuator_ctrlrange[:2,1])
        self.mj_data.ctrl[2:self.njctrl] = np.clip(action[2:self.njctrl], self.mj_model.actuator_ctrlrange[2:self.njctrl,0], self.mj_model.actuator_ctrlrange[2:self.njctrl,1])

    def publish_static_transform(self, header_frame_id, child_frame_id):
        stfs_msg = TransformStamped()
        stfs_msg.header.stamp = self.get_clock().now().to_msg()
        stfs_msg.header.frame_id = header_frame_id
        stfs_msg.child_frame_id = child_frame_id

        tmat_base = get_site_tmat(self.mj_data, header_frame_id)
        tmat_child = get_site_tmat(self.mj_data, child_frame_id)
        tmat_trans = np.linalg.inv(tmat_base) @ tmat_child
        
        stfs_msg.transform.translation.x = tmat_trans[0, 3]
        stfs_msg.transform.translation.y = tmat_trans[1, 3]
        stfs_msg.transform.translation.z = tmat_trans[2, 3]

        quat = Rotation.from_matrix(tmat_trans[:3, :3]).as_quat()
        stfs_msg.transform.rotation.x = quat[0]
        stfs_msg.transform.rotation.y = quat[1]
        stfs_msg.transform.rotation.z = quat[2]
        stfs_msg.transform.rotation.w = quat[3]

        self.static_broadcaster.sendTransform(stfs_msg)

    def thread_publidartopic(self, freq=12):
        if not self.config.lidar_s2_sim:
            return
                      
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            points = self.lidar_s2.get_lidar_points(self.rays_phi, self.rays_theta, self.mj_data)
            dists = np.linalg.norm(points[:, :2], axis=1)
            if np.all(dists < 1e-6):
                range_min = 0.0
            else:
                range_min = np.min(dists[dists >= 1e-6])
            range_max = np.max(dists)

            scan_msg = LaserScan()
            scan_msg.header.frame_id = self.lidar_frame_id
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.angle_min = float(np.pi)
            scan_msg.angle_max = float(-np.pi * 179. / 180.)
            scan_msg.angle_increment = float(-2. * np.pi / 360.)
            scan_msg.time_increment = 0.0
            scan_msg.range_min = float(range_min)
            scan_msg.range_max = float(range_max)
            scan_msg.ranges = dists[::-1].astype(np.float32).tolist()
            scan_msg.intensities = []

            self.lidar_s2_puber.publish(scan_msg)
            rate.sleep()

    def thread_pubros2topic(self, freq=30):
        rate = self.create_rate(freq)
        while rclpy.ok() and self.running:
            time_stamp = self.get_clock().now().to_msg()

            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_qpos[2:].tolist()
            self.joint_state.velocity = self.sensor_qvel[2:].tolist()
            self.joint_state.effort = self.sensor_force[2:].tolist()
            self.joint_state_puber.publish(self.joint_state)

            self.odom_msg.header.stamp = time_stamp
            self.odom_msg.pose.pose.position.x = self.sensor_base_position[0]
            self.odom_msg.pose.pose.position.y = self.sensor_base_position[1]
            self.odom_msg.pose.pose.position.z = self.sensor_base_position[2]
            self.odom_msg.pose.pose.orientation.w = self.sensor_base_orientation[0]
            self.odom_msg.pose.pose.orientation.x = self.sensor_base_orientation[1]
            self.odom_msg.pose.pose.orientation.y = self.sensor_base_orientation[2]
            self.odom_msg.pose.pose.orientation.z = self.sensor_base_orientation[3]
            self.odom_msg.twist.twist.linear.x = self.sensor_base_linear_vel[0]
            self.odom_msg.twist.twist.linear.y = self.sensor_base_linear_vel[1]
            self.odom_msg.twist.twist.linear.z = self.sensor_base_linear_vel[2]
            self.odom_msg.twist.twist.angular.x = self.sensor_base_gyro[0]
            self.odom_msg.twist.twist.angular.y = self.sensor_base_gyro[1]
            self.odom_msg.twist.twist.angular.z = self.sensor_base_gyro[2]
            self.odom_puber.publish(self.odom_msg)
            
            trans_msg = TransformStamped()
            trans_msg.header.stamp = time_stamp
            trans_msg.header.frame_id = "odom"
            trans_msg.child_frame_id = "base_link"
            trans_msg.transform.translation.x = self.sensor_base_position[0]
            trans_msg.transform.translation.y = self.sensor_base_position[1]
            trans_msg.transform.translation.z = self.sensor_base_position[2]
            trans_msg.transform.rotation.w = self.sensor_base_orientation[0]
            trans_msg.transform.rotation.x = self.sensor_base_orientation[1]
            trans_msg.transform.rotation.y = self.sensor_base_orientation[2]
            trans_msg.transform.rotation.z = self.sensor_base_orientation[3]            
            self.tf_broadcaster.sendTransform(trans_msg)

            # self.imu_msg.header.stamp = time_stamp
            # self.imu_msg.orientation.w = self.sensor_base_orientation[0]
            # self.imu_msg.orientation.x = self.sensor_base_orientation[1]
            # self.imu_msg.orientation.y = self.sensor_base_orientation[2]
            # self.imu_msg.orientation.z = self.sensor_base_orientation[3]
            # self.imu_msg.angular_velocity.x = self.sensor_base_gyro[0]
            # self.imu_msg.angular_velocity.y = self.sensor_base_gyro[1]
            # self.imu_msg.angular_velocity.z = self.sensor_base_gyro[2]
            # self.imu_msg.linear_acceleration.x = self.sensor_base_acc[0]
            # self.imu_msg.linear_acceleration.y = self.sensor_base_acc[1]
            # self.imu_msg.linear_acceleration.z = self.sensor_base_acc[2]
            # # self.imu_puber.publish(self.imu_msg)

            if 0 in self.config.obs_rgb_cam_id:
                head_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][0], encoding="rgb8")
                head_color_img_msg.header.stamp = time_stamp
                head_color_img_msg.header.frame_id = "head_camera"
                self.head_color_puber.publish(head_color_img_msg)

            if 1 in self.config.obs_rgb_cam_id:
                left_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][1], encoding="rgb8")
                left_color_img_msg.header.stamp = time_stamp
                left_color_img_msg.header.frame_id = "left_camera"
                self.left_color_puber.publish(left_color_img_msg)

            if 2 in self.config.obs_rgb_cam_id:
                right_color_img_msg = self.bridge.cv2_to_imgmsg(self.obs["img"][2], encoding="rgb8")
                right_color_img_msg.header.stamp = time_stamp
                right_color_img_msg.header.frame_id = "right_camera"
                self.right_color_puber.publish(right_color_img_msg)

            if 0 in self.config.obs_depth_cam_id:
                head_depth_img = np.array(np.clip(self.obs["depth"][0]*1e3, 0, 65535), dtype=np.uint16)
                head_depth_img_msg = self.bridge.cv2_to_imgmsg(head_depth_img, encoding="mono16")
                head_depth_img_msg.header.stamp = time_stamp
                head_depth_img_msg.header.frame_id = "head_camera"
                self.head_depth_puber.publish(head_depth_img_msg)

            rate.sleep()


if __name__ == "__main__":
    rclpy.init()
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = [0,1,2]
    cfg.obs_depth_cam_id = [0]
    cfg.lidar_s2_sim = True
    cfg.render_set     = {
        "fps"    : 24,
        "width"  : 640,
        "height" : 480
    }

    exec_node = MMK2ROS2(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    publidar_thread = threading.Thread(target=exec_node.thread_publidartopic, args=(12,))
    publidar_thread.start()
    
    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while rclpy.ok() and exec_node.running:
        exec_node.step(exec_node.target_control)

    exec_node.destroy_node()
    rclpy.shutdown()
    publidar_thread.join()
    pubtopic_thread.join()
    spin_thread.join()