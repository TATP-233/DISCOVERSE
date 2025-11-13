import os
import threading
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from sensor_msgs.msg import JointState, Imu

from discoverse.utils import step_func
from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.robots import AirbotPlayIK
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg, AirbotPlayBase
from discoverse.utils.joy_stick_ros1 import JoyTeleopRos1


class AirbotPlayJoyCtl(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)

        self.arm_ik = AirbotPlayIK()
    
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.tar_jq = np.zeros(self.nj)

        self.joint_state_puber = rospy.Publisher('/airbot_play/joint_states', JointState, queue_size=5)
        self.joint_state = JointState()
        self.joint_state.name = [f"joint{i+1}" for i in range(6)] + ["gripper"]

        self.joint_state.position = self.sensor_joint_qpos.tolist()
        self.joint_state.velocity = self.sensor_joint_qvel.tolist()
        self.joint_state.effort = self.sensor_joint_force.tolist()

        self.imu_puber = rospy.Publisher('/airbot_play/imu', Imu, queue_size=5)
        self.imu = Imu()

        self.teleop = JoyTeleopRos1()

    def resetState(self):
        super().resetState()
        self.tar_jq = np.zeros(self.nj)
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.teleop.reset()

    def updateControl(self, action):
        super().updateControl(self.tar_jq)

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
                rospy.logwarn("Fail to solve inverse kinematics trans={} euler={}".format(self.tar_end_pose, self.tar_end_euler))

        if self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]:
            self.tar_jq[6] += 1. * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * self.delta_t
            self.tar_jq[6] = np.clip(self.tar_jq[6], 0, 1.)

    def thread_pubros2topic(self, freq=30):
        rate = rospy.Rate(freq)
        while not rospy.is_shutdown() and self.running:
            self.joint_state.header.stamp = rospy.Time.now()
            self.joint_state.position = self.sensor_joint_qpos.tolist()
            self.joint_state.velocity = self.sensor_joint_qvel.tolist()
            self.joint_state.effort = self.sensor_joint_force.tolist()

            self.imu.header.stamp = rospy.Time.now()
            self.imu.orientation.w = self.sensor_endpoint_quat_local[0]
            self.imu.orientation.x = self.sensor_endpoint_quat_local[1]
            self.imu.orientation.y = self.sensor_endpoint_quat_local[2]
            self.imu.orientation.z = self.sensor_endpoint_quat_local[3]
            self.imu.angular_velocity.x = self.sensor_endpoint_gyro[0]
            self.imu.angular_velocity.y = self.sensor_endpoint_gyro[1]
            self.imu.angular_velocity.z = self.sensor_endpoint_gyro[2]
            self.imu.linear_acceleration.x = self.sensor_endpoint_acc[0]
            self.imu.linear_acceleration.y = self.sensor_endpoint_acc[1]
            self.imu.linear_acceleration.z = self.sensor_endpoint_acc[2]

            self.joint_state_puber.publish(self.joint_state)
            self.imu_puber.publish(self.imu)
            rate.sleep()

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("joint tar_q = {}".format(np.array2string(self.tar_jq, separator=', ')))
        print("joint q     = {}".format(np.array2string(self.sensor_joint_qpos, separator=', ')))
        print("joint v     = {}".format(np.array2string(self.sensor_joint_qvel, separator=', ')))

        print("target end posi  = {}".format(np.array2string(self.tar_end_pose, separator=', ')))
        print("target end euler = {}".format(np.array2string(self.tar_end_euler, separator=', ')))

        print("sensor end posi  = {}".format(np.array2string(self.sensor_endpoint_posi_local, separator=', ')))
        print("sensor end euler = {}".format(np.array2string(Rotation.from_quat(self.sensor_endpoint_quat_local[[1,2,3,0]]).as_euler("xyz"), separator=', ')))

        if self.cam_id == -1:
            print(self.free_camera)
        else:
            print(self.mj_data.camera(self.camera_names[self.cam_id]))

if __name__ == "__main__":
    rospy.init_node('Airbot_play_mujoco_node', anonymous=True)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/tasks_airbot_play/laptop_close.xml"
    cfg.decimation = 4
    cfg.timestep = 0.002
    cfg.use_gaussian_renderer = False

    exec_node = AirbotPlayJoyCtl(cfg)

    # from discoverse.examples.robots.airbot_replay import AirbotReplay
    # replay = AirbotReplay("can0", with_eef=True, auto_control=True, control_period=0.03)

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    move_speed = 5.
    while not rospy.is_shutdown() and exec_node.running:
        exec_node.teleopProcess()
        # jt = np.array([encoder.pos for encoder in replay.encoders])

        # for i in range(exec_node.nj-1):
        #     exec_node.tar_jq[i] = step_func(exec_node.tar_jq[i], jt[i], move_speed * exec_node.delta_t)
        # exec_node.tar_jq[exec_node.nj-1] = jt[exec_node.nj-1]

        exec_node.step()

    pubtopic_thread.join()