import mujoco
import numpy as np

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
from scipy.spatial.transform import Rotation

from discoverse.robots_env.skyrover_base import SkyRoverBase, SkyRoverCfg

if __name__ == "__main__":

    cfg = SkyRoverCfg()
    cfg.use_gaussian_renderer = True

    cfg.gs_model_dict["background"] = "scene/tsimf_library_0/point_cloud_for_skyrover.ply"
    # cfg.gs_model_dict["background"] = "scene/student_center/point_cloud_trans.ply"
    # cfg.gs_model_dict["background"] = "scene/solid_background/white.ply"

    exec_node = SkyRoverBase(cfg)
    obs = exec_node.reset()

    cnt = 0
    action = np.zeros(15)
    while exec_node.running:
        if cnt * exec_node.delta_t < 1.:
            action[1] += 80 * exec_node.delta_t / 1.
            action[6] += 80 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 1.5:
            action[0] -= 545 * exec_node.delta_t / 0.5
        elif cnt * exec_node.delta_t < 2.:
            action[2] = action[3] = 0.25
            action[7] = action[8] = 0.25
        elif cnt * exec_node.delta_t < 2.5:
            action[2] = action[3] = -0.3
            action[7] = action[8] = -0.3
        elif cnt * exec_node.delta_t < 3.5:
            action[2] = action[3] = 0.
            action[7] = action[8] = 0.
            action[0] += 545 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 4.5:
            action[1] -= 80 * exec_node.delta_t / 1.
            action[6] -= 80 * exec_node.delta_t / 1.
        elif cnt * exec_node.delta_t < 5.:
            action[0] -= 545 * exec_node.delta_t / 0.5
        elif cnt * exec_node.delta_t < 6.:
            action[[4,5,9,10]] = 10
            exec_node.mj_data.qpos[0] = last_x
            exec_node.mj_data.qpos[2] = last_z + 0.5 * exec_node.delta_t / 1.
            exec_node.mj_data.qpos[[3,4,5,6]] = [1,0,0,0]
        elif cnt * exec_node.delta_t < 7.:
            exec_node.mj_data.qpos[0] = last_x
            exec_node.mj_data.qpos[2] = last_z - 0.5 * exec_node.delta_t / 1.
            exec_node.mj_data.qpos[[3,4,5,6]] = [1,0,0,0]
        else:
            action[[4,5,9,10]] = 0
            cnt = 0
        cnt += 1

        last_x = exec_node.mj_data.qpos[0]
        last_z = exec_node.mj_data.qpos[2]

        obs, pri_obs, rew, ter, info = exec_node.step(action)
