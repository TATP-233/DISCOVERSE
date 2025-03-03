import mujoco
import numpy as np
import sys
import os

from discoverse.utils.base_config import BaseConfig
from discoverse.envs import SimulatorBase


class LeapHandCfg(BaseConfig):
    mjcf_file_path = "mjcf/leaphand_sensor_env.xml"
    decimation = 4
    timestep = 0.001
    sync = True
    headless = False
    init_key = "0"
    render_set = {
        "fps": 24,
        "width": 1920,
        "height": 1080,
    }
    obs_rgb_cam_id = None
    # rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list = []
    use_gaussian_renderer = False


class LeapHandBase(SimulatorBase):
    def __init__(self, config: LeapHandCfg):
        self.nj = 16  # number of joints
        super().__init__(config)

        self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[: self.nj]
        self.init_joint_ctrl = self.mj_model.key(self.config.init_key).ctrl[: self.nj]

        self.resetState()

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # if self.teleop:
        #    self.teleop.reset()

        self.mj_data.qpos[: self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[: self.nj] = self.init_joint_ctrl.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateControl(self, action):
        for i in range(self.nj):
            self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(
                self.mj_data.ctrl[i],
                self.mj_model.actuator_ctrlrange[i][0],
                self.mj_model.actuator_ctrlrange[i][1],
            )

    def step_func(self, current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

    def checkTerminated(self):
        return False

    def getObservation(self):
        self.obs = {
            "jq": self.mj_data.qpos[: self.nj].tolist(),
            "jv": self.mj_data.qvel[: self.nj].tolist(),
            "img": self.img_rgb_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None


if __name__ == "__main__":
    cfg = LeapHandCfg()
    exec_node = LeapHandBase(cfg)

    obs = exec_node.reset()

    action = exec_node.init_joint_pose[: exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
