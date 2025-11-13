import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase, recoder_mmk2, copypy2
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine

class SimNode(MMK2TaskBase):

    def domain_randomization(self):
        # 随机 盒子位置
        
        self.mj_data.qpos[self.njq+0] += (np.random.random() - 0.5) * 0.05
        if np.random.random() < 0.33:
            self.mj_data.qpos[self.njq+2] += -0.3
        elif np.random.random() < 0.667:
            self.mj_data.qpos[self.njq+2] += 0.0
        else:
            self.mj_data.qpos[self.njq+2] += 0.3

    def check_success(self):
        if self.mj_data.qpos[self.njq+1] - 1.21 < 0.01:
            return True

        return False

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = False
cfg.gs_model_dict["box"]             = "object/box.ply"
cfg.gs_model_dict["background"]      = "scene/tsimf_library_0/point_cloud_for_mmk2.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_box.xml"
cfg.obj_list    = ["box"]
cfg.sync     = True
cfg.headless = False
cfg.render_set  = {
    "fps"    : 20,
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
    parser.add_argument('--use_gs', action='store_true', help='Use gaussian splatting renderer')
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False
    cfg.use_gaussian_renderer = args.use_gs

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_pick_box")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 5
    max_time = 20.0 #s

    action = np.zeros_like(sim_node.target_control)
    process_list = []

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
                if stm.state_idx == 0: # 降高度
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    sim_node.tctr_head[1] = -0.4
                    sim_node.tctr_slide[0] = 1.22 - 1.08 * tmat_box[2, 3]
                elif stm.state_idx == 1: # 伸到盒子前
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi1 = tmat_box[:3, 3] + np.array([0.0, 0.15, 0.13]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.0,-0.15, 0.13]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                elif stm.state_idx == 2: # 抱住盒子
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi1 = tmat_box[:3, 3] + np.array([0.0, 0.08, 0.13]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([0.0,-0.08, 0.13]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 8]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 8]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                elif stm.state_idx == 3: # 抬升一定高度
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    sim_node.tctr_head[1] = -0.8
                    sim_node.tctr_slide[0] = 1.22 - 1.08 * tmat_box[2, 3] - 0.08
                elif stm.state_idx == 4: # 向后取出盒子
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi1 = tmat_box[:3, 3] + np.array([-0.35, 0.08, 0.14]) # y,x,z
                    target_posi2 = tmat_box[:3, 3] + np.array([-0.35,-0.08, 0.14]) # y,x,z
                    sim_node.lft_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi1)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi2)
                    sim_node.setArmEndTarget(sim_node.lft_arm_target_pose, sim_node.arm_action, "l", sim_node.sensor_lft_arm_qpos, Rotation.from_euler('zyx', [ np.pi / 2, -0.0551 + np.pi, np.pi / 7]).as_matrix())
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [-np.pi / 2, -0.0551 + np.pi, -np.pi / 7]).as_matrix())
                    sim_node.tctr_lft_gripper[:] = 0
                    print(sim_node.mj_data.qpos[sim_node.njq+0])
                
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
        yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2]
        action[1] = -10 * yaw

        obs, _, _, _, _ = sim_node.step(action)
        
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
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
