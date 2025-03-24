import numpy as np
import cv2
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation

from discoverse.mmk2 import MMK2FIK

class MMK2TaskBase(Node):
    target_control = np.zeros(19)
    action_done_dict = {
        "slide"         : False,
        "head"          : False,
        "left_arm"      : False,
        "left_gripper"  : False,
        "right_arm"     : False,
        "right_gripper" : False,
        "delay"         : False,
    }
    delay_cnt = 0
    set_left_arm_new_target = False
    set_right_arm_new_target = False

    def __init__(self):
        super().__init__('mmk2_node')
        self.get_odom = False
        self.get_joint = False
        self.arm_action = "pick"
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:12]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:19]
        self.tctr_rgt_gripper = self.target_control[18:19]

        self.imshow = False
        self.data_renew = False
        self.task_renew = False
        self.task_info = None
        self.logger = False
        self.obs = {
            "time": None,
            "jq": [0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0.,
                   0., 0., 0., 0.],
            "base_position": [0., 0., 0.],
            "base_orientation": [1., 0., 0., 0.],
            "img": {},
            "depth": None
        }

        self.init_subscription()
        self.init_publisher()
        self.initial_pose = [0,0,0]
        self.arm_action_init_position = np.array([
            [0.223,  0.21, 1.07055],
            [0.223, -0.21, 1.07055],
        ])

        self.lft_arm_target_pose = self.arm_action_init_position[0].copy()
        self.rgt_arm_target_pose = self.arm_action_init_position[1].copy()
        self.running = True # robot running state
        self.tmat_cam_head = [[ 0.003, -0.962,  0.273,  0.406],
                            [ -1,    -0.003,  0.001,  0.037],
                            [ 0,    -0.273, -0.962,  1.454 ],
                            [ 0,     0,     0,       1.    ]]


    def get_tmat_wrt_mmk2base(self, pose):
        current_pos  = self.obs["base_position"]   # [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # [qw, qx, qy, qz]
        # transfer to 4x4 matrix
        tmat_mmk2 = np.eye(4)
        tmat_mmk2[:3,:3] = Rotation.from_quat([current_quat[1], current_quat[2], 
                                               current_quat[3], current_quat[0]]).as_matrix()
        tmat_mmk2[:3,3] = current_pos
        return (np.linalg.inv(tmat_mmk2) @ np.append(pose, 1))[:3]

    def setArmEndTarget(self, target_pose, arm_action, arm, q_ref, a_rot):
        print(f"Setting {arm.upper()} arm target local : {np.array2string(target_pose, separator=', ')}")
    
        try:
            rq = MMK2FIK().get_armjoint_pose_wrt_footprint(target_pose, 
                                                           arm_action, arm, 
                                                           self.tctr_slide[0], 
                                                           q_ref, a_rot)
            if arm == "l":
                self.tctr_left_arm[:] = rq
                self.set_left_arm_new_target = True
            elif arm == "r":
                self.tctr_right_arm[:] = rq
                self.set_right_arm_new_target = True
            return True
        
        except ValueError as e:
            print(f"IK failed: {e} parameters: arm={arm}, target={target_pose}, slide={self.tctr_slide[0]:.2f}")
            return False

    def updateControl(self):
        twist_msg = Twist()
        twist_msg.linear.x = self.tctr_base[0]
        twist_msg.angular.z = self.tctr_base[1]
        self.publisher_cmd_vel.publish(twist_msg)

        float_array_msg_head = Float64MultiArray()
        float_array_msg_head.data = self.tctr_head.tolist()
        self.publisher_head.publish(float_array_msg_head)

        float_array_msg_left_arm = Float64MultiArray()
        float_array_msg_left_arm.data = self.tctr_left_arm.tolist()  # Adjust with desired values
        if len(float_array_msg_left_arm.data)!=7:
                print("left arm error:",len(float_array_msg_left_arm.data))
                print(self.tctr_left_arm)
        self.publisher_left_arm.publish(float_array_msg_left_arm)

        float_array_msg_right_arm = Float64MultiArray()
        float_array_msg_right_arm.data = self.tctr_right_arm.tolist()  # Adjust with desired values
        if len(float_array_msg_right_arm.data)!=7:
                print("right arm error:",len(float_array_msg_right_arm.data))
        self.publisher_right_arm.publish(float_array_msg_right_arm)

        float_array_msg_spine = Float64MultiArray()
        float_array_msg_spine.data = self.tctr_slide.tolist()  # Adjust with desired values
        self.publisher_spine.publish(float_array_msg_spine)

    def step(self, action=None):
        self.tctr_base = action[:2]
        self.tctr_slide = action[2:3]
        self.tctr_head = action[3:5]
        self.tctr_left_arm = action[5:12]
        self.tctr_right_arm = action[12:19]
        # self.updateControl()

    def get_base_pose(self):
        current_pos  = self.obs["base_position"]   # [X, Y, Z]
        current_quat = self.obs["base_orientation"]  # [qw, qx, qy, qz]
        yaw = Rotation.from_quat([current_quat[1], current_quat[2], 
                                  current_quat[3], current_quat[0]]).as_euler('zyx')[0]
        return np.array([current_pos[0], current_pos[1], yaw])
    
    def checkReceiveMessage(self):
        return self.get_odom and self.get_joint

    def checkBaseDone(self, translation:float=0.0, rotation:float=0.0):
        ori_x, ori_y, ori_yaw = self.initial_pose
        current_pos = self.get_base_pose()

        if abs(translation) > 0.01:
            move_dist = np.hypot(current_pos[0] - ori_x, current_pos[1] - ori_y)
            position_done = move_dist >= max(translation - 0.01, 0)
            if position_done:
                print(f"<MMK2>  \033[1;36mmove complete | distance: {move_dist:.3f}m/{translation:.3f}m | diff: {abs(move_dist-abs(translation)):.3f}m\033[0m")
            return position_done

        if abs(rotation) > 0.01:
            rotation = np.clip(rotation, -np.pi, np.pi)            
            delta_angle = abs(current_pos[2] - ori_yaw)
            rotation_done = (rotation == 0) or (delta_angle >= max(rotation - 0.001, 0))
            if rotation_done and rotation > 0:
                print(f"<MMK2>  \033[1;36mrotation complete | angle: {np.degrees(delta_angle):.1f}°/{np.degrees(rotation):.1f}°")
            return rotation_done

    def checkActionDone(self, debug=False):
        slide_done = np.allclose(self.tctr_slide, self.sensor_slide_qpos, atol=3e-2)
        head_done  = np.allclose(self.tctr_head,  self.sensor_head_qpos,  atol=3e-2)

        if self.set_left_arm_new_target:
            left_arm_done = (
                np.allclose(self.tctr_left_arm, self.sensor_lft_arm_qpos, atol=3e-2)
                )
            if left_arm_done:
                self.set_left_arm_new_target = False
        else:
            left_arm_done = True

        if self.set_right_arm_new_target:
            right_arm_done = (
                np.allclose(self.tctr_right_arm, self.sensor_rgt_arm_qpos, atol=3e-2) 
                )
            if right_arm_done:
                self.set_right_arm_new_target = False
        else:
            right_arm_done = True

        left_gripper_done  = np.allclose(self.tctr_lft_gripper, self.sensor_lft_gripper_qpos, 
                                         atol=0.1)
        right_gripper_done = np.allclose(self.tctr_rgt_gripper, self.sensor_rgt_gripper_qpos, 
                                         atol=0.2)

        self.delay_cnt -= 1
        delay_done = (self.delay_cnt<=0)

        self.action_done_dict = {
            "slide"         : slide_done,
            "head"          : head_done,
            "left_arm"      : left_arm_done,
            "left_gripper"  : left_gripper_done,
            "right_arm"     : right_arm_done,
            "right_gripper" : right_gripper_done,
            "delay"         : delay_done,
        }

        if debug:
            error_msgs = []
            if not slide_done:
                pos_err = np.abs(self.tctr_slide - self.sensor_slide_qpos)
                error_msgs.append(f"  slide: slide diff={pos_err[0]:.4f}m")

            if not head_done:
                pos_err = np.abs(self.tctr_head - self.sensor_head_qpos)
                error_msgs.append(f"  head: yaw diff={pos_err[0]:.4f}°, pitch diff={pos_err[1]:.4f}°")

            if not left_arm_done and self.set_left_arm_new_target:
                pos_err = np.linalg.norm(self.lft_arm_target_pose - self.sensor_lftarm_ep)
                error_msgs.append(f"  left_arm: left joints diff={pos_err:.4f}m")

            if not right_arm_done and self.set_right_arm_new_target:
                pos_err = np.linalg.norm(self.rgt_arm_target_pose - self.sensor_rgtarm_ep)
                error_msgs.append(f"  right_arm: right joints diff={pos_err:.4f}m")

            if not left_gripper_done:
                pos_err = np.abs(self.tctr_lft_gripper - self.sensor_lft_gripper_qpos)[0]
                error_msgs.append(f"  left_gripper: diff={pos_err:.3f}m")

            if not right_gripper_done:
                pos_err = np.abs(self.tctr_rgt_gripper - self.sensor_rgt_gripper_qpos)[0]
                error_msgs.append(f"  right_gripper: diff={pos_err:.3f}m")
                
            if not delay_done:
                error_msgs.append(f"  delay: {self.delay_cnt:.3f} > 0")

            if error_msgs:
                print("\n".join(error_msgs))

        return slide_done and head_done and left_arm_done and left_gripper_done and right_arm_done and right_gripper_done and delay_done
      
    def init_subscription(self):
        self.sub_odom = self.create_subscription(Odometry, '/mmk2/odom', self.odom_callback, 10)
        self.sub_joint_states = self.create_subscription(
            JointState, '/mmk2/joint_states', self.joint_states_callback, 10)
        
        # 1Hz
        self.sub_taskinfo = self.create_subscription(
            String, '/s2r2025/taskinfo', self.taskinfo_callback, 10)

    def init_publisher(self):
        self.publisher_cmd_vel = self.create_publisher(Twist, '/mmk2/cmd_vel', 10)
        self.publisher_head = self.create_publisher(Float64MultiArray, '/mmk2/head_forward_position_controller/commands', 10)
        self.publisher_left_arm = self.create_publisher(Float64MultiArray, '/mmk2/left_arm_forward_position_controller/commands', 10)
        self.publisher_right_arm = self.create_publisher(Float64MultiArray, '/mmk2/right_arm_forward_position_controller/commands', 10)
        self.publisher_spine = self.create_publisher(Float64MultiArray, '/mmk2/spine_forward_position_controller/commands', 10)

    def pub_thread(self):
        rate = self.create_rate(30)
        while rclpy.ok():
            if self.checkReceiveMessage():
                self.updateControl()
            try:
                rate.sleep()
            except rclpy.exceptions.ROSInterruptException:
                print("done navigation")
                break

    def clock_callback(self, msg):
        if self.logger:
            self.get_logger().info('Received clock: %s' % msg)

    def odom_callback(self, msg):
        timestamp = msg.header.stamp
        seconds = timestamp.sec
        nanoseconds = timestamp.nanosec

        position_x = msg.pose.pose.position.x
        position_y = msg.pose.pose.position.y
        position_z = msg.pose.pose.position.z
        self.obs["base_position"] = [position_x, position_y, position_z]

        orientation_x = msg.pose.pose.orientation.x
        orientation_y = msg.pose.pose.orientation.y
        orientation_z = msg.pose.pose.orientation.z
        orientation_w = msg.pose.pose.orientation.w
        self.obs["base_orientation"] = [orientation_w, orientation_x, orientation_y, orientation_z]
        self.get_odom = True

        if self.logger:
            self.get_logger().info('Received odometry:')
            self.get_logger().info('Timestamp: %d seconds, %d nanoseconds' % (seconds, nanoseconds))
            self.get_logger().info('Position: x=%f, y=%f, z=%f' % (position_x, position_y, position_z))
            self.get_logger().info(
                'Orientation: x=%f, y=%f, z=%f, w=%f' % (orientation_x, orientation_y, orientation_z, orientation_w))

    def joint_states_callback(self, msg):
        self.obs["jq"][2:] = msg.position
        self.sensor_slide_qpos = self.obs["jq"][2:3]
        self.sensor_head_qpos  = self.obs["jq"][3:5]
        self.sensor_lft_arm_qpos  = self.obs["jq"][5:11]
        self.sensor_lft_gripper_qpos  = self.obs["jq"][11:12]
        self.sensor_rgt_arm_qpos  = self.obs["jq"][12:18]
        self.sensor_rgt_gripper_qpos  = self.obs["jq"][18:19]
        self.get_joint = True

    def taskinfo_callback(self, msg):
        self.task_renew = True
        self.task_info = msg.data
        if self.logger:
            self.get_logger().info('Received task info: %s' % msg)


if __name__ == '__main__':
    rclpy.init()
    mmk2_node = MMK2TaskBase()
    spin_thead = threading.Thread(target=lambda: rclpy.spin(mmk2_node))
    spin_thead.start()
    pub_thread = threading.Thread(target=mmk2_node.pub_thread)
    pub_thread.start()

    spin_thead.join()
    pub_thread.join()
    mmk2_node.destroy_node()
    rclpy.shutdown()