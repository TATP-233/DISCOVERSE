import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import shutil
import argparse
import multiprocessing as mp

from discoverse.airbot_play import AirbotPlayFIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSERT_DIR
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, copypy2


class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_0_pose = (
            self.mj_model.camera("eye_side").pos.copy(),
            self.mj_model.camera("eye_side").quat.copy(),
        )

    def domain_randomization(self):
        # 随机 杯子位置
        self.mj_data.qpos[self.nj + 1 + 0] += 2.0 * (np.random.random() - 0.5) * 0.2
        self.mj_data.qpos[self.nj + 1 + 1] += 2.0 * (np.random.random() - 0.5) * 0.06

        # wood_base_x_bias = 2.*(np.random.random() - 0.5) * 0.035
        # 随机 盘子位置
        self.mj_data.qpos[self.nj + 7 * 1 + 1 + 0] += (
            2.0 * (np.random.random() - 0.5) * 0.0625
        )
        # self.mj_data.qpos[self.nj+7*1+1+1] += 2.*(np.random.random() - 0.5) * 0.02

        # # 随机 木板位置
        # self.mj_data.qpos[self.nj+7*2+1+0] += wood_base_x_bias
        # self.mj_data.qpos[self.nj+7*2+1+1] += 2.*(np.random.random() - 0.5) * 0.02

        # # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

        # 随机 free camera 视角
        # self.free_camera.lookat[:] = np.array([-0.041, 0.878, 0.879]) + 2.*(np.random.random(3) - 0.5) * 0.1
        # self.free_camera.distance  =   0.55 + 2.*(np.random.random()- 0.5) * 0.25
        # self.free_camera.azimuth   =  48    + 2.*(np.random.random()- 0.5) * 40
        # self.free_camera.elevation = -40    + 2.*(np.random.random()- 0.5) * 20

    def check_success(self):
        tmat_cup = get_body_tmat(self.mj_data, "coffeecup_white")
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")
        return (abs(tmat_cup[2, 2]) > 0.99) and np.hypot(
            tmat_plate[0, 3] - tmat_cup[0, 3], tmat_plate[1, 3] - tmat_cup[1, 3]
        ) < 0.02


cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = True
cfg.init_key = "ready"
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"] = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"] = "hinge/drawer_2.ply"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"
cfg.gs_model_dict["coffeecup_white"] = "object/teacup.ply"
cfg.gs_model_dict["wood"] = "object/wood.ply"
# cfg.gs_model_dict["tablecloth"]      = "object/table_cloth.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/coffeecup_place.xml"
cfg.obj_list = ["drawer_1", "drawer_2", "plate_white", "coffeecup_white", "wood"]
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = False  # True
cfg.headless = False
cfg.render_set = {
    "fps": 20,
    "width": 448,  # 1920, #640,
    "height": 448,  # 1080  #480
}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_idx", type=int, default=0, help="data index")
    parser.add_argument("--data_set_size", type=int, default=1, help="data set size")
    parser.add_argument("--auto", action="store_true", help="auto run")
    args = parser.parse_args()

    data_idx, data_set_size = args.data_idx, args.data_idx + args.data_set_size
    if args.auto:
        cfg.headless = True
        cfg.sync = False

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/coffeecup_place_448_448")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    sim_node = SimNode(cfg)
    if (
        hasattr(cfg, "save_mjb_and_task_config")
        and cfg.save_mjb_and_task_config
        and data_idx == 0
    ):
        mujoco.mj_saveModel(
            sim_node.mj_model,
            os.path.join(
                save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")
            ),
        )
        copypy2(
            os.path.abspath(__file__),
            os.path.join(save_dir, os.path.basename(__file__)),
        )

    sim_node.free_camera.distance = 0.5
    sim_node.mj_model.vis.global_.fovy = 72.02033809218685
    sim_node.cam_id = 0
    # sim_node.gs_renderer.set_obj_pose("table_cloth", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0])
    arm_fik = AirbotPlayFIK(
        os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    )

    trmat = Rotation.from_euler("xyz", [0, np.pi * 0.4, 0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 9
    max_time = 15.0  # s

    action = np.zeros(7)
    process_list = []

    move_speed = 0.75
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []

        try:
            if stm.trigger():
                if stm.state_idx == 0:  # 伸到杯子前
                    tmat_coffee = get_body_tmat(sim_node.mj_data, "coffeecup_white")
                    tmat_coffee[:3, 3] = (
                        tmat_coffee[:3, 3]
                        + 0.1 * tmat_coffee[:3, 1]
                        + 0.1 * tmat_coffee[:3, 2]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_coffee
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1:  # 伸到杯把
                    tmat_coffee = get_body_tmat(sim_node.mj_data, "coffeecup_white")
                    tmat_coffee[:3, 3] = (
                        tmat_coffee[:3, 3]
                        + 0.06 * tmat_coffee[:3, 1]
                        + 0.05 * tmat_coffee[:3, 2]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_coffee
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 2:  # 抓住杯把
                    sim_node.target_control[6] = 0
                elif stm.state_idx == 3:  # 抓住杯把
                    sim_node.delay_cnt = int(0.5 / sim_node.delta_t)
                elif stm.state_idx == 4:  # 提起来杯子
                    tmat_tgt_local[2, 3] += 0.15
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 5:  # 把杯子放到盘子上空
                    tmat_plate = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_plate[:3, 3] = tmat_plate[:3, 3] + np.array([0.06, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 6:  # 把杯子放到盘子上空
                    tmat_plate = get_body_tmat(sim_node.mj_data, "plate_white")
                    tmat_plate[:3, 3] = tmat_plate[:3, 3] + np.array([0.06, 0.0, 0.08])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_plate
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 7:  # 降低高度 把杯子放到盘子上
                    tmat_tgt_local[2, 3] -= 0.02
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 8:  # 松开杯把 放下杯子
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 9:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.05
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

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

        for i in range(sim_node.nj - 1):
            action[i] = step_func(
                action[i],
                sim_node.target_control[i],
                move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t,
            )
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)

        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
                process = mp.Process(
                    target=recoder_airbot_play, args=(save_path, act_lst, obs_lst, cfg)
                )
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()

    for p in process_list:
        p.join()
