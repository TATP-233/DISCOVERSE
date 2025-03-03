import os
import json
import glfw
import shutil
import mujoco
import mediapy
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.envs.airbot_play_base import AirbotPlayBase


def recoder_airbot_play(save_path, act_lst, obs_lst, cfg):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
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


class AirbotPlayTaskBase(AirbotPlayBase):
    target_control = np.zeros(7)
    joint_move_ratio = np.zeros(7)
    action_done_dict = {
        "joint": False,
        "gripper": False,
        "delay": False,
    }
    delay_cnt = 0
    reset_sig = False
    cam_id = 0

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def domain_randomization(self):
        pass

    def checkActionDone(self):
        joint_done = (
            np.allclose(self.sensor_joint_qpos[:6], self.target_control[:6], atol=3e-2)
            and np.abs(self.sensor_joint_qvel[:6]).sum() < 0.1
        )
        gripper_done = (
            np.allclose(self.sensor_joint_qpos[6], self.target_control[6], atol=0.4)
            and np.abs(self.sensor_joint_qvel[6]).sum() < 0.125
        )
        self.delay_cnt -= 1
        delay_done = self.delay_cnt <= 0
        self.action_done_dict = {
            "joint": joint_done,
            "gripper": gripper_done,
            "delay": delay_done,
        }
        return joint_done and gripper_done and delay_done

    def printMessage(self):
        super().printMessage()
        print("    target control = ", self.target_control)
        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

        print("camera foyv = ", self.mj_model.vis.global_.fovy)
        cam_xyz, cam_wxyz = self.getCameraPose(self.cam_id)
        print(
            f"    camera_{self.cam_id} =\n({cam_xyz}\n{Rotation.from_quat(cam_wxyz[[1,2,3,0]]).as_matrix()})"
        )

    def check_success(self):
        raise NotImplementedError

    def on_key(self, window, key, scancode, action, mods):
        ret = super().on_key(window, key, scancode, action, mods)
        if action == glfw.PRESS:
            if key == glfw.KEY_MINUS:
                self.mj_model.vis.global_.fovy = np.clip(
                    self.mj_model.vis.global_.fovy * 0.95, 5, 175
                )
            elif key == glfw.KEY_EQUAL:
                self.mj_model.vis.global_.fovy = np.clip(
                    self.mj_model.vis.global_.fovy * 1.05, 5, 175
                )
        return ret
