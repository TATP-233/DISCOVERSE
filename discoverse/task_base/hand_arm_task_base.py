import os
import json
import shutil
import mujoco
import mediapy
import numpy as np
from scipy.spatial.transform import Rotation
from discoverse.envs.hand_with_arm_base import HandWithArmBase


def recoder_hand_with_arm(save_path, act_lst, obs_lst, cfg):
    # 采集仿真数据并保存

    # TODO:在补充好HandWithArmBase中的传感器等数据后，修改这里的数据保存方式

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time": [o["time"] for o in obs_lst],
            "obs": {
                "jq": [o["jq"] for o in obs_lst],
                "fq": [o["fq"] for o in obs_lst],
            },
            "act": act_lst,
        }
        json.dump(obj, fp)

    print("data saved")
    for id in cfg.obs_rgb_cam_id:
        # 保存相机数据
        mediapy.write_video(
            os.path.join(save_path, f"cam_{id}.mp4"),
            [o["img"][id] for o in obs_lst],
            fps=cfg.render_set["fps"],
        )
    print("mp4 saved")


class HandArmTaskBase(HandWithArmBase):
    # 机器人执行器数量 机械臂 6 + 手 6 = 12
    target_control = np.zeros(12)
    joint_move_ratio = np.zeros(12)

    action_done_dict = {
        "joint": False,
        "gripper": False,
        "delay": False,
    }
    delay_cnt = 0
    reset_sig = False
    cam_id = -1

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.domain_randomization()
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.reset_sig = True

    def domain_randomization(self):
        pass

    def checkActionDone(self):
        # 根据反馈和设定值error判断动作完成
        # TODO:融入触觉数据进行任务完成的判断

        joint_done = (
            np.allclose(self.sensor_arm_qpos[:6], self.target_control[:6], atol=3e-2)
            and np.abs(self.sensor_arm_qvel[:6]).sum() < 0.1
            and np.allclose(
                self.sensor_finger_qpos[:6], self.target_control[6:12], atol=1e-2
            )
        )
        # gripper_done = np.allclose(self.sensor_joint_qpos[6], self.target_control[6], atol=0.4) and np.abs(self.sensor_joint_qvel[6]).sum() < 0.125
        self.delay_cnt -= 1
        delay_done = self.delay_cnt <= 0
        # self.action_done_dict = {
        #     "joint"   : joint_done,
        #     "gripper" : gripper_done,
        #     "delay"   : delay_done,
        # }
        if joint_done and delay_done:
            print("check_done")
        return joint_done and delay_done

        # return True

    def printMessage(self):
        super().printMessage()
        print("    target control = ", self.target_control)
        print("    action done: ")
        for k, v in self.action_done_dict.items():
            print(f"        {k}: {v}")

        print("camera foyv = ", self.mj_model.vis.global_.fovy)

        # cam_xyz, cam_wxyz = self.getCameraPose(self.cam_id)
        # print(f"    camera_{self.cam_id} =\n({cam_xyz}\n{Rotation.from_quat(cam_wxyz[[1,2,3,0]]).as_matrix()})")

    def check_success(self):
        # return False
        raise NotImplementedError

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord("-"):
            self.mj_model.vis.global_.fovy = np.clip(
                self.mj_model.vis.global_.fovy * 0.95, 5, 175
            )
        elif key == ord("="):
            self.mj_model.vis.global_.fovy = np.clip(
                self.mj_model.vis.global_.fovy * 1.05, 5, 175
            )
        return ret
