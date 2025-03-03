import os
import json
import shutil
import mujoco
import mediapy
import numpy as np

from discoverse import DISCOVERSE_ASSERT_DIR
from discoverse.utils import get_body_tmat
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.envs.tok2_base import TOK2Base, TOK2Cfg


def recoder_tok2(save_path, act_lst, obs_lst, cfg):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
                "base_position": [o["base_position"] for o in obs_lst],
                "base_orientation_wxyz": [o["base_orientation"] for o in obs_lst],
            },
            "act": act_lst,
        }
        json.dump(obj, fp)

    for id in cfg.obs_rgb_cam_id:
        mediapy.write_video(
            os.path.join(save_path, f"cam_{id}.mp4"),
            [o["img"][id] for o in obs_lst],
            fps=cfg.render_set["fps"],
        )


class TOK2TaskBase(TOK2Base):
    target_control = np.zeros(16)
    joint_move_ratio = np.zeros(16)
    action_done_dict = {
        "left_arm": False,
        "left_gripper": False,
        "right_arm": False,
        "right_gripper": False,
        "delay": False,
    }
    delay_cnt = 0
    reset_sig = False
    cam_id = 0

    gripper_done_limit = 0.35
    set_left_arm_new_target = False
    set_right_arm_new_target = False

    def __init__(self, config: TOK2Cfg):
        self.arm_action = config.init_key
        self.tctr_base = self.target_control[:2]
        self.tctr_left_arm = self.target_control[2:8]
        self.tctr_lft_gripper = self.target_control[8:9]
        self.tctr_right_arm = self.target_control[9:15]
        self.tctr_rgt_gripper = self.target_control[15:16]

        super().__init__(config)

        self.arm_action_init_position = np.array(
            [
                [0.223, 0.21, 1.07055],
                [0.223, -0.21, 1.07055],
            ]
        )

        self.lft_arm_target_pose = self.arm_action_init_position[0].copy()
        self.rgt_arm_target_pose = self.arm_action_init_position[1].copy()

        self.arm_fik = AirbotPlayFIK(
            urdf=os.path.join(
                DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"
            )
        )

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl.copy()
        self.lft_arm_target_pose = self.arm_action_init_position[0].copy()
        self.rgt_arm_target_pose = self.arm_action_init_position[1].copy()
        self.set_left_arm_new_target = False
        self.set_right_arm_new_target = False
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def domain_randomization(self):
        pass

    def get_tmat_wrt_mmk2base(self, pose):
        tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
        if pose.shape == (4, 4):
            return np.linalg.inv(tmat_mmk2) @ pose
        else:
            return (np.linalg.inv(tmat_mmk2) @ np.append(pose, 1))[:3]

    def setArmEndTarget(self, target_pose, target_rot, arm, q_ref):
        rq = self.arm_fik.properIK(target_pose, target_rot, q_ref)
        if arm == "l":
            self.tctr_left_arm[:] = rq
            self.set_left_arm_new_target = True
        elif arm == "r":
            self.tctr_right_arm[:] = rq
            self.set_right_arm_new_target = True

    def checkActionDone(self):
        if self.set_left_arm_new_target:
            left_arm_done = (
                np.allclose(self.lft_arm_target_pose, self.sensor_lftarm_ep, atol=3e-2)
                and np.abs(self.sensor_lft_arm_qvel).sum() < 0.1
            )
            if left_arm_done:
                self.set_left_arm_new_target = False
        else:
            left_arm_done = True
        left_gripper_done = (
            np.allclose(
                self.tctr_lft_gripper,
                self.sensor_lft_gripper_qpos,
                atol=self.gripper_done_limit,
            )
            and np.abs(self.sensor_lft_gripper_qvel).sum() < 0.125
        )

        if self.set_right_arm_new_target:
            right_arm_done = (
                np.allclose(self.rgt_arm_target_pose, self.sensor_rgtarm_ep, atol=3e-2)
                and np.abs(self.sensor_rgt_arm_qvel).sum() < 0.1
            )
            if right_arm_done:
                self.set_right_arm_new_target = False
        else:
            right_arm_done = True
        right_gripper_done = (
            np.allclose(
                self.tctr_rgt_gripper,
                self.sensor_rgt_gripper_qpos,
                atol=self.gripper_done_limit,
            )
            and np.abs(self.sensor_rgt_gripper_qvel).sum() < 0.125
        )

        self.delay_cnt -= 1
        delay_done = self.delay_cnt <= 0

        self.action_done_dict = {
            "left_arm": left_arm_done,
            "left_gripper": left_gripper_done,
            "right_arm": right_arm_done,
            "right_gripper": right_gripper_done,
            "delay": delay_done,
        }
        return (
            left_arm_done
            and left_gripper_done
            and right_arm_done
            and right_gripper_done
            and delay_done
        )

    def printMessage(self):
        super().printMessage()
        print("target control : ")
        print(
            "    tctr_base        = {}".format(
                np.array2string(self.tctr_base, separator=", ")
            )
        )
        print(
            "    tctr_left_arm    = {}".format(
                np.array2string(self.tctr_left_arm, separator=", ")
            )
        )
        print(
            "    tctr_lft_gripper = {}".format(
                np.array2string(self.tctr_lft_gripper, separator=", ")
            )
        )
        print(
            "    tctr_right_arm   = {}".format(
                np.array2string(self.tctr_right_arm, separator=", ")
            )
        )
        print(
            "    tctr_rgt_gripper = {}".format(
                np.array2string(self.tctr_rgt_gripper, separator=", ")
            )
        )
        print(
            "    lft  endpoint    = {}".format(
                np.array2string(self.lft_arm_target_pose, separator=", ")
            )
        )
        print(
            "    rgt  endpoint    = {}".format(
                np.array2string(self.rgt_arm_target_pose, separator=", ")
            )
        )

        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

    def check_success(self):
        raise NotImplementedError
