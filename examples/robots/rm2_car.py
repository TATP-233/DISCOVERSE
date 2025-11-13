import os
import cv2
import mujoco
import mediapy
import numpy as np

from discoverse.robots_env.rm2_car_base import CARCfg, CARBase

if __name__ == "__main__":
    # import rospy
    # rospy.init_node('mujoco_node', anonymous=True)
    # np.set_printoptions(precision=4, suppress=True, linewidth=500)

    cfg = CARCfg()
    cfg.use_gaussian_renderer = True
    # cfg.gs_model_dict["background"] = "scene/riverside1/point_cloud.ply"
    # cfg.gs_model_dict["background_env"] = "scene/riverside1/environment.ply"
    cfg.gs_model_dict["background"] = "scene/solid_background/white.ply"

    cfg.timestep = 1/240
    cfg.decimation = 4
    cfg.render_set["fps"] = 60

    # cfg.obs_rgb_cam_id = [0]
    # cfg.obs_depth_cam_id = [0]

    exec_node = CARBase(cfg)
    obs = exec_node.reset()

    # exec_node.cam_id = 0
    # exec_node.mj_data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]

    action = np.zeros(10)
    # action[0] : lineal_velocity_x  local    朝前为正方向
    # action[1] : lineal_velocity_y  local    朝左为正方向
    # action[2] : angular_velocity_z          为从上向下看逆时针旋转为正方向

    depth_lst = []
    rgb_lst = []
    cnt = 0
    while exec_node.running:
        # if cnt * exec_node.delta_t < 1:
        #     action[:3] = [2, 2, 0]
        # elif cnt * exec_node.delta_t < 2:
        #     action[:3] = [0, -1.4142, 0]
        # elif cnt * exec_node.delta_t < 3:
        #     action[:3] = [-2, 2, 0]
        # elif cnt * exec_node.delta_t < 4:
        #     action[:3] = [0, -1.4142, 0]
        # elif cnt * exec_node.delta_t < 5:
        #     action[:3] = [0, 0, 5]
        # else:
        #     cnt = 0
        # cnt += 1

        action[:3] = [0, 0, 5]

        # action[0] = min(cnt * 7 * exec_node.delta_t, 35)
        obs, pri_obs, rew, ter, info = exec_node.step(action)

    #     depth_lst.append(exec_node.obs["depth"][0].copy())
    #     rgb_lst.append(exec_node.obs["img"][0].copy())

