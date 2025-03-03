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
    def domain_randomization(self):
        pass

    def check_success(self):
        return self.mj_data.qpos[9] > 0.15


cfg = AirbotPlayCfg()
cfg.use_gaussian_renderer = False
cfg.init_key = "ready"
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"] = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"] = "hinge/drawer_2.ply"

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/drawer_open.xml"
cfg.obj_list = ["drawer_1", "drawer_2"]
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 640, "height": 480}
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

    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/drawer_open")
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

    arm_fik = AirbotPlayFIK(
        os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    )

    trmat = Rotation.from_euler(
        "xyz", [-np.pi / 2.0, 0.0, np.pi], degrees=False
    ).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 7
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
                if stm.state_idx == 0:  # 伸到柜子前
                    tmat_handle = get_site_tmat(sim_node.mj_data, "drawer_2_handle")
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + 0.1 * tmat_handle[:3, 0]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                    move_speed = 1.5
                elif stm.state_idx == 1:  # 伸到把手位置
                    tmat_handle = get_site_tmat(sim_node.mj_data, "drawer_2_handle")
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    move_speed = 0.5
                elif stm.state_idx == 2:  # 抓住把手
                    sim_node.target_control[6] = 0
                elif stm.state_idx == 3:  # 抓稳把手 sleep 0.5s
                    sim_node.delay_cnt = int(0.5 / sim_node.delta_t)
                elif stm.state_idx == 4:  # 拉开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, "drawer_2_handle")
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + 0.2 * tmat_handle[:3, 0]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
                    sim_node.target_control[:6] = arm_fik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 5:  # 松开把手
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 6:  # 离开抽屉
                    tmat_handle = get_site_tmat(sim_node.mj_data, "drawer_2_handle")
                    tmat_handle[:3, 3] = tmat_handle[:3, 3] + 0.025 * tmat_handle[:3, 0]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_handle
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
