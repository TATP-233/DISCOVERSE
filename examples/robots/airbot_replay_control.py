import os

import cv2
import numpy as np

from airbot_replay import AirbotReplay

from scipy.spatial.transform import Rotation

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.robots import AirbotPlayIK
from discoverse.robots_env.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

def step_func(current, target, step):
    if current < target - step:
        return current + step
    elif current > target + step:
        return current - step
    else:
        return target

if __name__ == "__main__":
    replay = AirbotReplay("can0", with_eef=True, auto_control=True, control_period=0.05)
    cfg = AirbotPlayCfg()

    cfg.timestep = 1e-3
    cfg.obs_rgb_cam_id = [0]
    cfg.decimation = 4
    cfg.render_set = {
        "fps"    : 60,
        "width"  : 1280,
        "height" : 720,
    }

    cfg.mjcf_file_path = "mjcf/airbot_play_floor.xml"
    cfg.use_gaussian_renderer = False
    exec_node = AirbotPlayBase(cfg)

    arm_ik = AirbotPlayIK()
    target_position = np.array([0.295, -0., 0.219])
    target_euler = np.zeros(3)
    target_action = np.zeros(7)

    obs = exec_node.reset()

    action = exec_node.init_joint_pose[:exec_node.nj]
    cv2.namedWindow("img0", cv2.WINDOW_GUI_NORMAL)

    while exec_node.running:
        replay_position = np.array([encoder.pos for encoder in replay.encoders])
        target_action = replay_position
        for i in range(exec_node.nj-1):
            action[i] = step_func(action[i], target_action[i], 20. * exec_node.delta_t)
        action[exec_node.nj-1] = target_action[exec_node.nj-1]
        obs, _, _, _, _ = exec_node.step(action)
        
        cv2.imshow("img0", cv2.cvtColor(obs["img"][0], cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    cv2.destroyAllWindows()
