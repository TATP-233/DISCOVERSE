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
        # 随机 盒子和盘子位置
        rand_choice = 0
        rand_angle = 0
        if rand_choice < 0.5:
            rand_angle = 0
            rand_x_pos = (np.random.random() - 0.6) * 0.07
            rand_y_pos = (np.random.random() - 0.5) * 0.009
            self.mj_data.qpos[self.njq + 7 * 0 + 0] += rand_x_pos
            self.mj_data.qpos[self.njq + 7 * 1 + 0] += rand_x_pos
            self.mj_data.qpos[self.njq + 7 * 0 + 1] += rand_y_pos
            self.mj_data.qpos[self.njq + 7 * 1 + 1] += rand_y_pos
        else:
            rand_angle = 1

        # 旋转角度
        Yaw_Angle = np.pi / 2 + rand_angle * (np.random.random() - 0.5)*0.219
        Quat2 = Rotation.from_euler('zyx', [Yaw_Angle, 0, 0]).as_quat()

        Quat1 = Rotation.from_euler('zyx', [0, Yaw_Angle, np.pi / 2]).as_quat()
        
        self.mj_data.qpos[self.njq + 7 * 0 + 3] = Quat1[3]
        self.mj_data.qpos[self.njq + 7 * 0 + 4] = Quat1[0]
        self.mj_data.qpos[self.njq + 7 * 0 + 5] = Quat1[1]
        self.mj_data.qpos[self.njq + 7 * 0 + 6] = Quat1[2]

        self.mj_data.qpos[self.njq + 7 * 1 + 3] = Quat2[3]
        self.mj_data.qpos[self.njq + 7 * 1 + 4] = Quat2[0]
        self.mj_data.qpos[self.njq + 7 * 1 + 5] = Quat2[1]
        self.mj_data.qpos[self.njq + 7 * 1 + 6] = Quat2[2]
        # 随机 盘子位置
        rand_data = np.random.random()
        if rand_data < 0.33:
            self.mj_data.qpos[self.njq + 7 * 0 + 0] += -0.1
        elif rand_data < 0.67:
            self.mj_data.qpos[self.njq + 7 * 0 + 0] += -0.05
        else:
            pass

    def check_success(self):
        distance = self.mj_data.qpos[self.njq+7*0+2] - 0.75
        if distance < 0.03:
            return True
        return False

cfg = MMK2Cfg()
cfg.use_gaussian_renderer = False
cfg.gs_model_dict["pan"]             = "object/pan.ply"
cfg.gs_model_dict["box"]             = "object/box.ply"
cfg.gs_model_dict["background"]      = "scene/tsimf_library_0/point_cloud_for_mmk2.ply"

cfg.mjcf_file_path = "mjcf/tasks_mmk2/pan_pick.xml"
cfg.obj_list    = ["pan", "box"]
cfg.sync     = True
cfg.headless = False
cfg.render_set  = {
    "fps"    : 25,
    "width"  : 640,
    "height" : 480
}
cfg.obs_rgb_cam_id = [0,1,2]
cfg.save_mjb_and_task_config = True

cfg.init_state["base_position"] = [0.4, -0.4, 0.0]
cfg.init_state["base_orientation"] = [0.707, 0.0, 0.0, -0.707]
cfg.init_state["lft_arm_qpos"] = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
cfg.init_state["rgt_arm_qpos"] = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/mmk2_pick_pan")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 13
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
                    sim_node.tctr_head[1] = -0.4
                    sim_node.tctr_slide[0] = 0.12
                elif stm.state_idx == 1: # 伸到盘子上1
                    tmat_pan = get_body_tmat(sim_node.mj_data, "pan")
                    target_posi = tmat_pan[:3, 3] + np.array([0.0, 0.0, 0.10]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_pan[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1
                elif stm.state_idx == 2: # 伸到盘子上2
                    tmat_pan = get_body_tmat(sim_node.mj_data, "pan")
                    target_posi = tmat_pan[:3, 3] + np.array([0.01, 0.0, 0.06]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_pan[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1
                elif stm.state_idx == 3: # 抓住盘子
                    tmat_pan = get_body_tmat(sim_node.mj_data, "pan")
                    target_posi = tmat_pan[:3, 3] + np.array([0.01, 0.0, 0.05]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_pan[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 1
                elif stm.state_idx == 4: # 夹住盘子
                    tmat_pan = get_body_tmat(sim_node.mj_data, "pan")
                    target_posi = tmat_pan[:3, 3] + np.array([0.01, 0.0, 0.05]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_pan[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 5: # 拿起盘子
                    tmat_pan = get_body_tmat(sim_node.mj_data, "pan")
                    target_posi = tmat_pan[:3, 3] + np.array([0.01, 0.0, 0.13]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_pan[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 6: # 移出盘子
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.35, 0.0, 0.25]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [0, -0.0551, euler_angles[1]]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 7: # 降高度
                    sim_node.tctr_head[1] = -0.4
                    sim_node.tctr_slide[0] = 0.12
                elif stm.state_idx == 8: # 放下盘子1
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.3, 0.12, 0.25]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [np.pi / 4, 0, np.pi / 2 + np.pi / 3]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 9: # 放下盘子1
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.3, 0.12, 0.13]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [np.pi / 3, 0, np.pi / 2 + np.pi / 3]).as_matrix())
                    
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 10: # 放下盘子1
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.3, 0.12, 0.04]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [np.pi / 2.1, 0, np.pi / 2 + np.pi / 3]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 11: # 放下盘子1
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.3, 0.12, 0.035]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [np.pi / 2.1, 0, np.pi / 2 + np.pi / 3]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0
                elif stm.state_idx == 12: # 放下盘子1
                    tmat_box = get_body_tmat(sim_node.mj_data, "box")
                    target_posi = tmat_box[:3, 3] + np.array([-0.3, 0.12, 0.03]) # y,x,z
                    # 将旋转矩阵转换为欧拉角 (指定旋转轴顺序为 'zyx')
                    euler_angles = Rotation.from_matrix(tmat_box[:3, :3]).as_euler('zyx', degrees=False)
                    sim_node.rgt_arm_target_pose[:] = sim_node.get_tmat_wrt_mmk2base(target_posi)
                    sim_node.setArmEndTarget(sim_node.rgt_arm_target_pose, sim_node.arm_action, "r", sim_node.sensor_rgt_arm_qpos, Rotation.from_euler('zyx', [np.pi / 2.1, 0, np.pi / 2 + np.pi / 3]).as_matrix())
                    sim_node.tctr_rgt_gripper[:] = 0

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
        yaw = Rotation.from_quat(np.array(obs["base_orientation"])[[1,2,3,0]]).as_euler("xyz")[2] + np.pi / 2
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
