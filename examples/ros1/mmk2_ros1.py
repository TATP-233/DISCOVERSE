import threading
import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu

from discoverse.robots_env.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.utils import PIDarray

class MMK2ROS1(MMK2Base):
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

        self.pid_base_vel = PIDarray(
            kps=np.array([ 7.5 ,  7.5 ]),
            kis=np.array([  .0 ,   .0 ]),
            kds=np.array([  .0 ,   .0 ]),
            integrator_maxs=np.array([5.0, 5.0]),
        )

        self.init_topic_publisher()
        self.init_topic_subscriber()

    def init_topic_publisher(self):
        # rostopic joint state
        self.joint_state_puber = rospy.Publisher('/mmk2/joint_states', JointState, queue_size=1)
        self.joint_state = JointState()
        self.joint_state.name = [
            "left_wheel", "right_wheel", "slide_joint", "head_yaw_joint", "head_pitch_joint",
            "left_arm_joint1" , "left_arm_joint2" , "left_arm_joint3" , "left_arm_joint4" , "left_arm_joint5" , "left_arm_joint6" , "left_arm_eef_gripper_joint" ,
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
        ]
        self.joint_state.position = self.sensor_qpos.tolist()
        self.joint_state.velocity = self.sensor_qvel.tolist()
        self.joint_state.effort = self.sensor_force.tolist()

        # rostopic imu
        self.imu_puber = rospy.Publisher('/mmk2/imu', Imu, queue_size=5)
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = "mmk2_imu_link"

        # rostopic odometry
        self.odom_puber = rospy.Publisher('/mmk2/odom', Odometry, queue_size=1)
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "mmk2_footprint"

    def init_topic_subscriber(self):
        rospy.Subscriber('/mmk2/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/mmk2/spine_forward_position_controller/commands', Float64MultiArray, self.cmd_spine_callback)
        rospy.Subscriber('/mmk2/head_forward_position_controller/commands', Float64MultiArray, self.cmd_head_callback)
        rospy.Subscriber('/mmk2/left_arm_forward_position_controller/commands', Float64MultiArray, self.cmd_left_arm_callback)
        rospy.Subscriber('/mmk2/right_arm_forward_position_controller/commands', Float64MultiArray, self.cmd_right_arm_callback)

    def cmd_vel_callback(self, msg: Twist):
        self.tctr_base[0] = (msg.linear.x - msg.angular.z * self.wheel_distance) / self.wheel_radius
        self.tctr_base[1] = (msg.linear.x + msg.angular.z * self.wheel_distance) / self.wheel_radius

    def cmd_spine_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 1:
            self.tctr_slide[:] = msg.data[:]
        else:
            rospy.logerr("spine command length error")

    def cmd_head_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 2:
            self.tctr_head[:] = msg.data[:]
        else:
            rospy.logerr("head command length error")

    def cmd_left_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_left_arm[:] = msg.data[:6]
            self.tctr_lft_gripper[:] = msg.data[6:]
        else:
            rospy.logerr("left arm command length error")

    def cmd_right_arm_callback(self, msg: Float64MultiArray):
        if len(msg.data) == 7:
            self.tctr_right_arm[:] = msg.data[:6]
            self.tctr_rgt_gripper[:] = msg.data[6:]
        else:
            rospy.logerr("right arm command length error")

    def resetState(self):
        super().resetState()
        self.pid_base_vel.reset()
        self.target_control[:] = self.init_joint_ctrl[:]

    def updateControl(self, action):
        wheel_force = self.pid_base_vel.output(np.clip(self.tctr_base - self.sensor_wheel_qvel, -2.5, 2.5), self.mj_model.opt.timestep)
        self.mj_data.ctrl[:2] = np.clip(wheel_force, self.mj_model.actuator_ctrlrange[:2,0], self.mj_model.actuator_ctrlrange[:2,1])
        self.mj_data.ctrl[2:self.njctrl] = np.clip(action[2:self.njctrl], self.mj_model.actuator_ctrlrange[2:self.njctrl,0], self.mj_model.actuator_ctrlrange[2:self.njctrl,1])

    def thread_pubrostopic(self, freq=30):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown() and self.running:
            time_stamp = rospy.Time.now()
            
            self.joint_state.header.stamp = time_stamp
            self.joint_state.position = self.sensor_qpos.tolist()
            self.joint_state.velocity = self.sensor_qvel.tolist()
            self.joint_state.effort = self.sensor_force.tolist()
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

            self.imu_msg.header.stamp = time_stamp
            self.imu_msg.orientation.w = self.sensor_base_orientation[0]
            self.imu_msg.orientation.x = self.sensor_base_orientation[1]
            self.imu_msg.orientation.y = self.sensor_base_orientation[2]
            self.imu_msg.orientation.z = self.sensor_base_orientation[3]
            self.imu_msg.angular_velocity.x = self.sensor_base_gyro[0]
            self.imu_msg.angular_velocity.y = self.sensor_base_gyro[1]
            self.imu_msg.angular_velocity.z = self.sensor_base_gyro[2]
            self.imu_msg.linear_acceleration.x = self.sensor_base_acc[0]
            self.imu_msg.linear_acceleration.y = self.sensor_base_acc[1]
            self.imu_msg.linear_acceleration.z = self.sensor_base_acc[2]
            self.imu_puber.publish(self.imu_msg)

            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('mmk2_mujoco_node', anonymous=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = None
    cfg.obs_depth_cam_id = None
    cfg.render_set     = {
        "fps"    : 30,
        "width"  : 640,
        "height" : 480
    }

    exec_node = MMK2ROS1(cfg)
    exec_node.reset()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubrostopic, args=(30,))
    pubtopic_thread.start()

    while exec_node.running and not rospy.is_shutdown():
        exec_node.step(exec_node.target_control)

    pubtopic_thread.join()