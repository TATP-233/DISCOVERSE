import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.laptop_ori_pos = self.mj_model.body("lp1").pos.copy()
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

    def domain_randomization(self):
        # 随机 笔记本位置
        self.mj_model.body("lp1").pos[:2] = self.laptop_ori_pos[:2] + 2.*(np.random.random(2) - 0.5) * 0.1

        # 随机 笔记本角度
        self.set_joint_position("laptop_joint", np.random.random() * np.pi/4)

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        return (abs(self.mj_data.qpos[8]) < 0.1)

cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]   = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]   = "hinge/drawer_2.ply"
cfg.gs_model_dict["laptop_1"]   = "hinge/laptop_1.ply"
cfg.gs_model_dict["laptop_2"]   = "hinge/laptop_2.ply"
cfg.init_qpos[:] = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599,  0.0]

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/close_laptop.xml"
cfg.obj_list     = ["drawer_1", "drawer_2", "laptop_1", "laptop_2"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs
    
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", os.path.splitext(os.path.basename(__file__))[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    arm_ik = AirbotPlayIK()

    trmat_01 = Rotation.from_euler("xyz", [0., 0.8, 0.], degrees=False).as_matrix()
    trmat_2 = Rotation.from_euler("xyz", [0., np.pi/2., 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 4
    max_time = 15.0 #s

    action = np.zeros(7)
    move_speed = 0.75
    sim_node.reset()

    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(20, cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到电脑屏幕上方
                    tmat_laptop = get_site_tmat(sim_node.mj_data, "laptop_cam_site")
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_laptop
                    tmat_tgt_local[:3, 3] = tmat_tgt_local[:3, 3] + np.array([0.05, 0.0, 0.1])
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat_01, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 0.75
                elif stm.state_idx == 1: # 靠近电脑屏幕
                    tmat_tgt_local[:3, 3] = tmat_tgt_local[:3, 3] + np.array([0.05, 0.0, 0.0])
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat_01, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 合起屏幕
                    tmat_laptop = get_site_tmat(sim_node.mj_data, "laptop_bottom_site")
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_laptop
                    tmat_tgt_local[:3, 3] = tmat_tgt_local[:3, 3] + np.array([0.0, 0.0, 0.03])
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat_2, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 3: # 抬起夹爪
                    tmat_tgt_local[:3, 3] = tmat_tgt_local[:3, 3] + np.array([0.0, 0.0, 0.05])
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat_2, sim_node.mj_data.qpos[:6])

                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()
            act_lst, obs_lst = [], []

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop("img")
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                for encoder in encoders.values():
                    encoder.close()
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
