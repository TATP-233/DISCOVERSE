import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from os.path import abspath, dirname, join
import os
import sys

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import json
import copy
import mediapy
import logging
import multiprocessing as mp

from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayBase, AirbotPlayCfg
from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg

ROBOT_TYPE = ["MMK2", "AIRBOTPLAY"]

TASK_INSTANCES = ["plate_coffeecup", "stack_1block1bowl"]

AIRBOT_MOTIONS = ["approach", "move_to", "close_gripper", "open_gripper", "departs"]


class AirbotPlaySimNode(AirbotPlayBase):
    def __init__(self, task_instances):
        assert task_instances in TASK_INSTANCES, "Invalid task instances!"
        self.task_instances = task_instances

        cfg = AirbotPlayCfg()
        cfg.use_gaussian_renderer = True
        cfg.timestep = 1 / 240
        cfg.decimation = 4
        cfg.sync = False
        cfg.headless = False
        cfg.decimation = 4
        cfg.render_set = {"fps": 60, "width": 640, "height": 480}
        cfg.obs_rgb_cam_id = [0, 1]
        cfg.init_joint_pose = {
            "joint1": 0,
            "joint2": -0.71966516,
            "joint3": 1.2772779,
            "joint4": -1.57079633,
            "joint5": 1.72517278,
            "joint6": 1.57079633,
            "gripper": 1,
        }

        self.robot_state = ("holding", None)

        if self.task_instances == "plate_coffeecup":
            cfg.gs_model_dict["coffeecup_white"] = "object/teacup.ply"
            cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"
            cfg.gs_model_dict["cup_lid"] = "object/teacup_lid.ply"
            cfg.gs_model_dict["wood"] = "object/wood.ply"
            cfg.gs_model_dict["background"] = "scene/flower_table/point_cloud.ply"

            cfg.mjcf_file_path = "mjcf/plate_coffeecup.xml"
            cfg.obj_list = ["plate_white", "coffeecup_white", "wood", "cup_lid"]

        elif self.task_instances == "stack_1block1bowl":
            cfg.gs_model_dict["block_green"] = "object/block_green.ply"
            cfg.gs_model_dict["bowl_pink"] = "object/bowl_pink.ply"
            cfg.gs_model_dict[
                "background"
            ] = "scene/discover_operation_studio/point_cloud.ply"

            cfg.mjcf_file_path = "mjcf/1block1bowl_stack.xml"
            cfg.obj_list = ["block_green", "bowl_pink"]

        super().__init__(cfg)
        self.njq = self.nj

        # Forward/Inverse Kinematic
        base_posi, base_quat = self.getObjPose("arm_base")
        Tmat_base = np.eye(4)
        Tmat_base[:3, :3] = Rotation.from_quat(base_quat[[1, 2, 3, 0]]).as_matrix()
        Tmat_base[:3, 3] = base_posi
        self.Tmat_base_inv = np.linalg.inv(Tmat_base)

        arm_rot_mat = np.array([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        tar_end_rot = np.array([[0.0, -0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

        self.rot = tar_end_rot @ arm_rot_mat

        urdf_path = os.path.join(
            DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
        )
        self.arm_fik = AirbotPlayFIK(urdf_path)

    def getRobotState(self):
        return self.robot_state

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)
        self.mj_data.qpos[: self.nj] = self.init_joint_pose.copy()
        self.mj_data.qpos[self.nj] = -self.mj_data.qpos[self.nj - 1]
        self.mj_data.ctrl[: self.nj] = self.init_joint_pose.copy()

        if self.task_instances == "plate_coffeecup":
            # 随机初始化coffeecup的位置和姿态
            initial_quaternion = [
                self.mj_data.qpos[self.nj + 4],
                self.mj_data.qpos[self.nj + 5],
                self.mj_data.qpos[self.nj + 6],
                self.mj_data.qpos[self.nj + 7],
            ]
            initial_rotation = Rotation.from_quat(initial_quaternion)

            self.angle_change_deg = 5
            # max_angle_deg = 15
            # self.angle_change_deg = np.random.uniform(-max_angle_deg, max_angle_deg)
            axis = np.array([1, 0, 0])
            small_rotation = Rotation.from_rotvec(
                np.radians(self.angle_change_deg) * axis
            )
            new_rotation = initial_rotation * small_rotation
            [
                self.mj_data.qpos[self.nj + 4],
                self.mj_data.qpos[self.nj + 5],
                self.mj_data.qpos[self.nj + 6],
                self.mj_data.qpos[self.nj + 7],
            ] = new_rotation.as_quat()

            self.mj_data.qpos[self.nj + 1] += np.random.random() * 0.01
            self.mj_data.qpos[self.nj + 2] += np.random.random() * 0.01

        elif self.task_instances == "stack_1block1bowl":
            self.mj_data.qpos[self.nj + 8] += np.random.random() * 0.1 - 0.05
            self.mj_data.qpos[self.nj + 9] += np.random.random() * 0.1 - 0.05

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def getEnvDescription(self):
        description = ""
        for obj_name in self.config.obj_list:
            posi, quat = self.getObjPose(obj_name)
            description += (
                "The "
                + str(obj_name)
                + " is at position "
                + str(posi)
                + " with rotation described by a quaternion "
                + str(quat)
                + ". "
            )
            """
            ToDo:
            height, width = self.getObjSize(obj_name)
            """
        return description

    def getObservation(self):
        obj_pose = {}
        for name in self.config.obj_list + self.config.rb_link_list:
            obj_pose[name] = self.getObjPose(name)
        obj_pose["camera0"] = self.getCameraPose(0)
        obj_pose["camera1"] = self.getCameraPose(1)

        self.obs = {
            "time": self.mj_data.time,
            "jq": self.jq.tolist(),
            "img": self.img_rgb_obs_s,
            "obj_pose": copy.deepcopy(obj_pose),
        }
        self.obs["jq"][6] *= 25.0  # gripper normalization
        return self.obs

    def getArmTargetPose(self, start_pose, motion_to_do, obj_name=None, param=None):
        if not motion_to_do in AIRBOT_MOTIONS:
            print("AirbotPlay has no such motion primitives!")
            return None

        target_pose = start_pose.copy()

        if obj_name:
            posi, quat = self.getObjPose(obj_name)
            if posi.any() == None or quat.any() == None:
                print(
                    "{} does not exist in the current task instance.".format(obj_name)
                )
                return None

        if motion_to_do == "close_gripper" or motion_to_do == "open_gripper":
            if motion_to_do == "close_gripper":
                target_pose[6] = 0
            else:
                target_pose[6] = 1
            return target_pose

        Tmat_block_g_global = np.eye(4)
        Tmat_block_g_global[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        Tmat_block_g_global[:3, 3] = posi

        Tmat_move_bias = np.eye(4)
        if self.task_instances == "plate_coffeecup":
            if obj_name != "cup_lid":
                theta_rad = -np.radians(self.angle_change_deg)
                pick_pose_bias = [-0.055, 0, 0.059]
                translated_point = np.array(pick_pose_bias)
                rotation_matrix = np.array(
                    [
                        [np.cos(theta_rad), -np.sin(theta_rad), 0],
                        [np.sin(theta_rad), np.cos(theta_rad), 0],
                        [0, 0, 1],
                    ]
                )
                rotated_point = np.dot(rotation_matrix, translated_point)
                pick_pose_bias_r = rotated_point
                Tmat_move_bias[:3, 3] = pick_pose_bias_r

            if (
                self.robot_state == ("holding", "cup_lid") and obj_name == "plate_white"
            ):  # if move cup_lid to plate_white
                Tmat_move_bias[:3, 3] = [0.004, 0.000, 0.147]

        if motion_to_do == "approach" and param:
            Tmat_move_bias[2, 3] = param

        print(Tmat_move_bias)
        Tmat_block_g_local = Tmat_move_bias @ self.Tmat_base_inv @ Tmat_block_g_global
        tar_end_pose = Tmat_block_g_local[:3, 3]
        target_pose[:6] = self.arm_fik.inverseKin(
            tar_end_pose, self.rot, start_pose[:6]
        )  # start_pose = np.array(obs["jq"])[:6]

        return target_pose
