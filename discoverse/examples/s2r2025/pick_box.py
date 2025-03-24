import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):

    def domain_randomization(self):
        self.start_mmk2_base = self.mj_data.qpos[0:2].copy()
        random_x = (np.random.random() - 0.5) * 0.1
        # random position of box
        self.mj_data.qpos[self.njq+7*1+0] += random_x
        # random position of disk
        self.mj_data.qpos[self.njq+7*4+0] += random_x
        # random rotation of disk
        # angle = (np.random.random() - 0.5) * np.radians(90) 
        # random_rotation = Rotation.from_euler('x', angle).as_quat()
        # current_quat = self.mj_data.qpos[self.njq+7*4+3 : self.njq+7*4+7]
        # new_quat = (Rotation.from_quat(random_rotation) * Rotation.from_quat(current_quat)).as_quat()
        # self.mj_data.qpos[self.njq+7*4+3 : self.njq+7*4+7] = new_quat

        # random height of box
        # if np.random.random() < 0.33:
        #     self.mj_data.qpos[self.njq+7*1+2] += -0.3
        # elif np.random.random() < 0.667:
        #     self.mj_data.qpos[self.njq+7*1+2] += 0.0
        # else:
        #     self.mj_data.qpos[self.njq+7*1+2] += 0.3

    def check_success(self):
        pick_success = self.mj_data.qpos[self.njq+7*1+1] - 1.21 < 0.01
        move = abs(self.mj_data.qpos[0:2] - self.start_mmk2_base)
        if pick_success and move[0] < 0.01 and move[1] < 0.01:
            return True

        return False

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "pick"
cfg.gs_model_dict["box_disk"]       = "s2r2025/box.ply"
cfg.gs_model_dict["disk_01"]        = "s2r2025/disk_01.ply"
cfg.gs_model_dict["background"]     = "scene/s2r2025/point_cloud.ply"

cfg.mjcf_file_path = "mjcf/s2r2025_env_pick_box.xml"
cfg.obj_list    = ["box_disk", "disk_01"]
cfg.sync     = False
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    parser.add_argument("--dim17", action="store_true", help="genegrate 17 joint num mmk2 data")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    if args.dim17:
        cfg.io_dim = 17
    else:
        cfg.io_dim = 19

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_pick_box")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    sim_node.teleop = None
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 7
    max_time = 20.0 #s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

    pick_lip_arm = "l"
    move_speed = 1.
    obs = sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                if stm.state_idx == 0: # slide down
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    sim_node.tctr_head[1] = 0.4
                    sim_node.tctr_slide[0] = 1.22 - 1.08 * tmat_box[2, 3]
                elif stm.state_idx == 1: # prepare action
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    target_posi1 = tmat_box[:3, 3] + np.array([-0.13, -0.15, 0.1]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.13, -0.15, 0.1]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                elif stm.state_idx == 2: # before box
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    target_posi1 = tmat_box[:3, 3] + np.array([-0.13, 0.0, 0.1]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.13, 0.0, 0.1]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                elif stm.state_idx == 3: # hug box
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    target_posi1 = tmat_box[:3, 3] + np.array([-0.08, 0.0, 0.1]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.08, 0.0, 0.1]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                elif stm.state_idx == 4: # slide up
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    sim_node.tctr_slide[0] = 1.22 - 1.08 * tmat_box[2, 3] - 0.05
                elif stm.state_idx == 5: # take out box
                    tmat_box = get_body_tmat(sim_node.mj_data, "box_disk")
                    target_posi1 = tmat_box[:3, 3] + np.array([-0.08, -0.25, 0.14]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.08, -0.25, 0.14]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 7]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 7]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                    print(sim_node.mj_data.qpos[sim_node.njq+0])
                elif stm.state_idx == 6:
                    sim_node.delay_cnt = int(3.0/sim_node.delta_t)
                
                dif = np.abs(action - sim_node.target_control)
                sim_node.joint_move_ratio = dif / (np.max(dif) + 1e-6)
                sim_node.joint_move_ratio[2] *= 0.25

            elif sim_node.mj_data.time > max_time:
                raise ValueError("Time out")

            else:
                stm.update()

            if sim_node.checkActionDone():
                stm.next()

        except ValueError as ve:
            # traceback.print_exc()
            sim_node.reset()

        for i in range(2, sim_node.njctrl):
                action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)

        if cfg.io_dim == 19:
            obs, _, _, _, _ = sim_node.step(action)
        elif cfg.io_dim == 17:
            obs, _, _, _, _ = sim_node.step(action[2:])
        else:
            raise ValueError(f"Wrong io dim: {sim_node.io_dim}")
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            if cfg.io_dim == 19:
                act_lst.append(action.tolist().copy())
            elif cfg.io_dim == 17:
                act_lst.append(action[2:].tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(target=recoder_mmk2, args=(save_path, act_lst, obs_lst, cfg))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            obs = sim_node.reset()

    for p in process_list:
        p.join()
