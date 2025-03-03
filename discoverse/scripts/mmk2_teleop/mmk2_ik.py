import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
import tf2_ros
from std_msgs.msg import Float32

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg


class LowpassFilter:
    def __init__(self, newWeight, shape) -> None:
        self.v = None
        self.w = newWeight
        self.lw = 1.0 - self.w

    def update(self, v):
        if self.v is None:
            self.v = v.copy()
        else:
            self.v = self.v * self.lw + v * self.w
        return self.v


class SimNode(MMK2Base):
    def getObservation(self):
        self.reset_flag = False
        self.obs = {
            "time": self.mj_data.time,
            "jq": self.mj_data.qpos[: self.njq].tolist(),
            "left_endposi": (
                self.mj_data.site("lft_endpoint").xpos
                - self.mj_data.body("lft_arm_base_link").xpos
            )
            @ self.mj_data.body("lft_arm_base_link").xmat.reshape((3, 3)),
            "left_endrmat": self.mj_data.site("lft_endpoint").xmat.reshape((3, 3))
            @ self.mj_data.body("lft_arm_base_link").xmat.reshape((3, 3)).T,
            "right_endposi": (
                self.mj_data.site("rgt_endpoint").xpos
                - self.mj_data.body("rgt_arm_base_link").xpos
            )
            @ self.mj_data.body("rgt_arm_base_link").xmat.reshape((3, 3)),
            "right_endrmat": self.mj_data.site("rgt_endpoint").xmat.reshape((3, 3))
            @ self.mj_data.body("rgt_arm_base_link").xmat.reshape((3, 3)).T,
            "reset": self.reset_flag,
        }
        if self.reset_flag:
            self.reset_flag = False
        return self.obs

    def updateControl(self, action):
        super().updateControl(action)

    def printMessage(self):
        super().printMessage()
        print(self.obs)


def left_finger_callback(data: Float32):
    global left_finger_cmd
    rospy.loginfo_once("Recv left finger control message.")
    left_finger_cmd[0] = 0.04 * (np.clip(data.data, 20, 50) - 20) / 30.0


def right_finger_callback(data: Float32):
    global right_finger_cmd
    rospy.loginfo_once("Recv right finger control message.")
    right_finger_cmd[0] = 0.04 * (np.clip(data.data, 20, 50) - 20) / 30.0


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    rospy.init_node("mmk2_blocks_television")

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    urdf_path = os.path.join(
        DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
    )
    arm_fik = AirbotPlayFIK(urdf_path)

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/mmk2_blocks_teleop.xml"
    cfg.render_set = {"fps": 30, "width": 1920, "height": 1080}
    sim_node = SimNode(cfg)
    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    sim_node.options.frame = mujoco.mjtFrame.mjFRAME_SITE.value

    action = sim_node.init_ctrl.copy()

    left_arm_cmd = action[5:11]
    left_finger_cmd = action[11:12]

    right_arm_cmd = action[12:18]
    right_finger_cmd = action[18:19]

    rospy.Subscriber("Left_finger", Float32, left_finger_callback)
    rospy.Subscriber("Right_finger", Float32, right_finger_callback)

    obs = sim_node.reset()

    lft_hand_pos_local = obs["left_endposi"].copy()
    lft_hand_rot_local = obs["left_endrmat"].copy()
    lh_p_filter = LowpassFilter(0.1, 3)
    lh_o_filter = LowpassFilter(0.1, 3)

    rgt_hand_pos_local = obs["right_endposi"].copy()
    rgt_hand_rot_local = obs["right_endrmat"].copy()
    rh_p_filter = LowpassFilter(0.1, 3)
    rh_o_filter = LowpassFilter(0.1, 3)

    arm_rot_mat = np.array([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

    while sim_node.running:
        if obs["reset"]:
            action = sim_node.init_ctrl.copy()
            left_arm_cmd = action[5:11]
            left_finger_cmd = action[11:12]

            right_arm_cmd = action[12:18]
            right_finger_cmd = action[18:19]

            lft_hand_pos_local = obs["left_endposi"].copy()
            lft_hand_rot_local = obs["left_endrmat"].copy()
            lh_p_filter = LowpassFilter(0.1, 3)
            lh_o_filter = LowpassFilter(0.1, 3)

            rgt_hand_pos_local = obs["right_endposi"].copy()
            rgt_hand_rot_local = obs["right_endrmat"].copy()
            rh_p_filter = LowpassFilter(0.1, 3)
            rh_o_filter = LowpassFilter(0.1, 3)

        try:
            tf_l = tf_buffer.lookup_transform(
                "world", "Left_wrist", rospy.Time()
            ).transform
            Tmat_world_Ltarget = np.eye(4)
            target_world_posi = np.array(
                [tf_l.translation.x, tf_l.translation.y, max(tf_l.translation.z, 0.345)]
            )
            target_world_quat_xyzw = np.array(
                [tf_l.rotation.x, tf_l.rotation.y, tf_l.rotation.z, tf_l.rotation.w]
            )
            Tmat_world_Ltarget[:3, 3] = target_world_posi
            # Tmat_world_Ltarget[:3,:3] = Rotation.from_quat(target_world_quat_xyzw).as_matrix() @ Rotation.from_euler("xyz", [0, np.pi/2., 0]).as_matrix() @ Rotation.from_euler("zyx", [0, np.pi/6., -np.pi/2.]).as_matrix()
            Tmat_world_Ltarget[:3, :3] = (
                Rotation.from_euler("xyz", [0, np.pi / 2.0, 0]).as_matrix()
                @ Rotation.from_euler("zyx", [0, np.pi / 6.0, -np.pi / 2.0]).as_matrix()
            )
            quat = Rotation.from_matrix(Tmat_world_Ltarget[:3, :3]).as_quat()

            sim_node.mj_model.body("left_target").pos = target_world_posi
            sim_node.mj_model.body("left_target").quat = quat[[3, 0, 1, 2]]

            Tmat_world_Larmbase = np.eye(4)
            Tmat_world_Larmbase[:3, :3] = sim_node.mj_data.body(
                "lft_arm_base_link"
            ).xmat.reshape((3, 3))
            Tmat_world_Larmbase[:3, 3] = sim_node.mj_data.body("lft_arm_base_link").xpos
            Tmat_Ltarget_local = np.linalg.inv(Tmat_world_Larmbase) @ Tmat_world_Ltarget

            lft_hand_pos_local = lh_p_filter.update(Tmat_Ltarget_local[:3, 3])
            lft_hand_euler_local = lh_o_filter.update(
                Rotation.from_matrix(Tmat_Ltarget_local[:3, :3] @ arm_rot_mat).as_euler(
                    "xyz"
                )
            )
            lft_hand_rot_local = Rotation.from_euler(
                "xyz", lft_hand_euler_local
            ).as_matrix()

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

        try:
            tf_r = tf_buffer.lookup_transform(
                "world", "Right_wrist", rospy.Time()
            ).transform
            Tmat_world_Rtarget = np.eye(4)
            target_world_posi = np.array(
                [tf_r.translation.x, tf_r.translation.y, max(tf_r.translation.z, 0.345)]
            )
            target_world_quat_xyzw = np.array(
                [tf_r.rotation.x, tf_r.rotation.y, tf_r.rotation.z, tf_r.rotation.w]
            )
            Tmat_world_Rtarget[:3, 3] = target_world_posi
            # Tmat_world_Rtarget[:3,:3] = Rotation.from_quat(target_world_quat_xyzw).as_matrix() @ Rotation.from_euler("xyz", [0, np.pi/2., 0]).as_matrix() @ Rotation.from_euler("zyx", [0, -np.pi/6., -np.pi/2.]).as_matrix() @ Rotation.from_euler("xyz", [np.pi, 0.0, 0.0]).as_matrix()
            Tmat_world_Rtarget[:3, :3] = (
                Rotation.from_euler("xyz", [0, np.pi / 2.0, 0]).as_matrix()
                @ Rotation.from_euler(
                    "zyx", [0, -np.pi / 6.0, -np.pi / 2.0]
                ).as_matrix()
                @ Rotation.from_euler("xyz", [np.pi, 0.0, 0.0]).as_matrix()
            )
            quat = Rotation.from_matrix(Tmat_world_Rtarget[:3, :3]).as_quat()

            sim_node.mj_model.body("right_target").pos = target_world_posi
            sim_node.mj_model.body("right_target").quat = quat[[3, 0, 1, 2]]

            Tmat_world_Rarmbase = np.eye(4)
            Tmat_world_Rarmbase[:3, :3] = sim_node.mj_data.body(
                "rgt_arm_base_link"
            ).xmat.reshape((3, 3))
            Tmat_world_Rarmbase[:3, 3] = sim_node.mj_data.body("rgt_arm_base_link").xpos
            Tmat_Rtarget_local = np.linalg.inv(Tmat_world_Rarmbase) @ Tmat_world_Rtarget

            rgt_hand_pos_local = rh_p_filter.update(Tmat_Rtarget_local[:3, 3])
            rgt_hand_euler_local = rh_o_filter.update(
                Rotation.from_matrix(Tmat_Rtarget_local[:3, :3] @ arm_rot_mat).as_euler(
                    "xyz"
                )
            )
            rgt_hand_rot_local = Rotation.from_euler(
                "xyz", rgt_hand_euler_local
            ).as_matrix()

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            pass

        try:
            left_arm_jres = arm_fik.inverseKin(
                lft_hand_pos_local, lft_hand_rot_local, np.array(obs["jq"][12:18])
            )
        except ValueError:
            left_arm_jres = None
        try:
            right_arm_jres = arm_fik.inverseKin(
                rgt_hand_pos_local, rgt_hand_rot_local, np.array(obs["jq"][20:26])
            )
        except ValueError:
            right_arm_jres = None

        if left_arm_jres != None:
            for i in range(6):
                left_arm_cmd[i] = sim_node.step_func(
                    left_arm_cmd[i],
                    left_arm_jres[i],
                    2.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep,
                )

        if not right_arm_jres is None:
            for i in range(6):
                right_arm_cmd[i] = sim_node.step_func(
                    right_arm_cmd[i],
                    right_arm_jres[i],
                    2.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep,
                )

        obs, pri_obs, rew, ter, info = sim_node.step(action)
