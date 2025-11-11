import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

import discoverse
from discoverse.envs import make_env
from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder

class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (self.mj_model.camera("eye_side").pos.copy(), self.mj_model.camera("eye_side").quat.copy())

    def domain_randomization(self):
        # 随机 block_green位置
        flag_position = False
        while not flag_position:

            self.object_pose("block_red")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])
            self.object_pose("block_green")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])
            self.object_pose("block_blue")[:2] += 2.*(np.random.random() - 0.5) * np.array([0.08, 0.06])

            position_list = np.array([
                self.object_pose("block_red")[:2], 
                self.object_pose("block_green")[:2], 
                self.object_pose("block_blue")[:2]])
            
            flag_position = self.check_position(position_list, 0.03)

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        tmat_block_green = get_body_tmat(self.mj_data, "block_green")
        tmat_block_blue = get_body_tmat(self.mj_data, "block_blue")
        tmat_block_red = get_body_tmat(self.mj_data, "block_red")
        return (abs(tmat_block_blue[2, 2]) > 0.99) and \
            np.hypot(tmat_block_green[0, 3] - tmat_block_blue[0, 3], tmat_block_green[1, 3] - tmat_block_blue[1, 3]) < 0.02 and \
            np.hypot(tmat_block_red[0, 3] - tmat_block_blue[0, 3], tmat_block_red[1, 3] - tmat_block_blue[1, 3]) < 0.02

    def check_position(self, position_list, tolerance):
        for i in range(len(position_list)-1):
            if i < len(position_list) - 1:
                res = np.linalg.norm(position_list[i] - position_list[i+1:], axis=1)
            else:
                res = np.linalg.norm(position_list[i] - position_list[i+1])
            if np.any(res < tolerance):
                return False
        return True
    
cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"]   = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"
cfg.init_qpos[:] = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599,  0.0]

robot_name = "airbot_play"
task_name = "stack_block"
cfg.mjcf_file_path = f"mjcf/tmp/{robot_name}_{task_name}.xml"
env = make_env(robot_name, task_name)
env.export_xml(os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path))

cfg.obj_list     = ["drawer_1", "drawer_2", "bowl_pink", "block_green"]
cfg.timestep     = 1/240
cfg.decimation   = 4
cfg.sync         = True
cfg.headless     = False
cfg.render_set   = {
    "fps"    : 20,
    "width"  : 448,
    "height" : 448
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=100, help="data set size")
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

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 18
    max_time = 12.0 # seconds
    
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
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}

        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到方块上方
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 0.04
                elif stm.state_idx == 1: # 伸到方块
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.028 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 抓住方块
                    sim_node.target_control[6] = 0.0
                elif stm.state_idx == 3: # 抓稳方块
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # 提起来方块
                    tmat_tgt_local[2,3] += 0.07
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 把方块放到碗上空
                    tmat_plate = get_body_tmat(sim_node.mj_data, "block_blue")
                    tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.0, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 降低高度 把方块放到碗上
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 松开方块
                    sim_node.target_control[6] = 0.04
                elif stm.state_idx == 8: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 9: # 伸到red block上方
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_red")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 0.04
                elif stm.state_idx == 10: # 伸到red block
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_red")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.028 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 11: # 抓住red block
                    sim_node.target_control[6] = 0.0
                elif stm.state_idx == 12: # 抓稳red block
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 13: # 提起来red block
                    tmat_tgt_local[2,3] += 0.07
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 14: # 把red block放到block_blue
                    tmat_plate = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_plate[:3,3] = tmat_plate[:3, 3] + np.array([0.0, 0.0, 0.10])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 15: # 降低高度
                    tmat_tgt_local[2,3] -= 0.04
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 16: # 松开block_red
                    sim_node.target_control[6] = 0.04
                elif stm.state_idx == 17: # 抬升高度
                    tmat_tgt_local[2,3] += 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])

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
            print("Error: ", ve)
            print("Current Errot idx: ", stm.state_idx)
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            imgs = obs.pop('img')
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            for encoder in encoders.values():
                encoder.close()
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")
                for encoder in encoders.values():
                    encoder.remove_av_file()

            sim_node.reset()
