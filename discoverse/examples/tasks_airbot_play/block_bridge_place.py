import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

import traceback
import discoverse
from discoverse.envs import make_env
from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder


class SimNode(AirbotPlayTaskBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.camera_1_pose = (
            self.mj_model.camera("eye_side").pos.copy(),
            self.mj_model.camera("eye_side").quat.copy(),
        )

    def domain_randomization(self):
        # 随机 2个绿色长方体位置

        for z in range(2):
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 2 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机 10个紫色方块位置

        for z in range(6):
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 0] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )
            self.mj_data.qpos[self.nj + 1 + 7 * 4 + z * 7 + 1] += (
                2.0 * (np.random.random() - 0.5) * 0.001
            )

        # 随机 eye side 视角
        # camera = self.mj_model.camera("eye_side")
        # camera.pos[:] = self.camera_0_pose[0] + 2.*(np.random.random(3) - 0.5) * 0.05
        # euler = Rotation.from_quat(self.camera_0_pose[1][[1,2,3,0]]).as_euler("xyz", degrees=False) + 2.*(np.random.random(3) - 0.5) * 0.05
        # camera.quat[:] = Rotation.from_euler("xyz", euler, degrees=False).as_quat()[[3,0,1,2]]

    def check_success(self):
        tmat_bridge1 = get_body_tmat(self.mj_data, "bridge1")
        tmat_bridge2 = get_body_tmat(self.mj_data, "bridge2")
        tmat_block1 = get_body_tmat(self.mj_data, "block1_green")
        tmat_block2 = get_body_tmat(self.mj_data, "block2_green")
        tmat_block01 = get_body_tmat(self.mj_data, "block_purple3")
        tmat_block02 = get_body_tmat(self.mj_data, "block_purple6")
        return (
            (abs(tmat_block1[2, 2]) < 0.001)
            and (abs(abs(tmat_bridge1[1, 3] - tmat_bridge2[1, 3]) - 0.03) <= 0.002)
            and (abs(tmat_block2[2, 2]) < 0.001)
            and np.hypot(
                tmat_block1[0, 3] - tmat_block01[0, 3],
                tmat_block2[1, 3] - tmat_block02[1, 3],
            )
            < 0.11
        )


cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"] = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"] = "hinge/drawer_2.ply"
cfg.gs_model_dict["bowl_pink"] = "object/bowl_pink.ply"
cfg.gs_model_dict["block_green"] = "object/block_green.ply"

robot_name = "airbot_play"
task_name = "block_bridge_place"
cfg.mjcf_file_path = f"mjcf/tmp/{robot_name}_{task_name}.xml"
env = make_env(robot_name, task_name)
env.export_xml(os.path.join(DISCOVERSE_ASSETS_DIR, cfg.mjcf_file_path))

cfg.obj_list = [
    "bridge1",
    "bridge2",
    "block1_green",
    "block2_green",
    "block_purple1",
    "block_purple2",
    "block_purple3",
    "block_purple4",
    "block_purple5",
    "block_purple6",
]
cfg.timestep = 1 / 240
cfg.decimation = 4
cfg.sync = True
cfg.headless = False
cfg.render_set = {"fps": 20, "width": 448, "height": 448}
cfg.obs_rgb_cam_id = [0, 1]
cfg.save_mjb_and_task_config = True

if __name__ == "__main__":
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)
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

    arm_ik = AirbotPlayIK()

    trmat = Rotation.from_euler("xyz", [0.0, np.pi / 2, 0.0], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 79
    max_time = 70.0  # seconds

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

                if stm.state_idx == 0:  # 伸到拱桥上方
                    trmat = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, np.pi / 2], degrees=False
                    ).as_matrix()
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
                        [0.03, -0.015, 0.12]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1:  # 伸到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 3:  # 伸到长方体
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 4:  # 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 5:  # 抓稳长方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 6:  # 提起长方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 7:  # 把长方体放到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [0.075 + 0.00005, -0.015, 0.1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 8:  # 保持夹爪角度 降低高度 把长方体放到桥旁边
                    tmat_tgt_local[2, 3] -= 0.03
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 9:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 10:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.06
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                elif stm.state_idx == 11:  # 伸到拱桥上方
                    tmat_bridge1 = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge1[:3, 3] = tmat_bridge1[:3, 3] + np.array(
                        [0.03, -0.015, 0.12]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 12:  # 伸到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 14:  # 伸到长方体
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array([0, 0, 0.04])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 15:  # 抓住长方体
                    sim_node.target_control[6] = 0.29
                elif stm.state_idx == 16:  # 抓稳长方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 17:  # 提起长方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 18:  # 把长方体放到桥旁边上方
                    tmat_bridge = get_body_tmat(sim_node.mj_data, "bridge1")
                    tmat_bridge[:3, 3] = tmat_bridge[:3, 3] + np.array(
                        [-0.015 - 0.0005, -0.015, 0.1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bridge
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 19:  # 保持夹爪角度 降低高度 把长方体放到桥旁边
                    tmat_tgt_local[2, 3] -= 0.03
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 20:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 21:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.06
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 1
                elif stm.state_idx == 22:  # 伸到立方体上方
                    trmat = Rotation.from_euler(
                        "xyz", [0.0, np.pi / 2, 0.0], degrees=False
                    ).as_matrix()
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 23:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple1")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 24:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 25:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 26:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 27:  # 把立方体放到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 28:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 29:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 30:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 2
                elif stm.state_idx == 31:  # 伸到立方体上方
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 32:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple2")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 33:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 34:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 35:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 36:  # 把立方体放到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 2]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 37:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 38:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 39:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 3
                elif stm.state_idx == 40:  # 伸到立方体上方
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 41:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple3")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 42:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 46:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 47:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 48:  # 把立方体放到长方体上方
                    tmat_block2 = get_body_tmat(sim_node.mj_data, "block2_green")
                    tmat_block2[:3, 3] = tmat_block2[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 3]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block2
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 49:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 50:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 51:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 4
                elif stm.state_idx == 52:  # 伸到立方体上方
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 53:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple4")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 54:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 55:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 56:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 57:  # 把立方体放到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 1]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 58:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 59:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 60:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 5
                elif stm.state_idx == 61:  # 伸到立方体上方
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 62:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple5")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 63:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 64:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 65:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 66:  # 把立方体放到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 2]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 67:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 68:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 69:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )

                # 6
                elif stm.state_idx == 70:  # 伸到立方体上方
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.12])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 71:  # 伸到立方体
                    tmat_block = get_body_tmat(sim_node.mj_data, "block_purple6")
                    tmat_block[:3, 3] = tmat_block[:3, 3] + np.array([0, 0, 0.03])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 72:  # 抓住立方体
                    sim_node.target_control[6] = 0.24
                elif stm.state_idx == 73:  # 抓稳立方体
                    sim_node.delay_cnt = int(0.35 / sim_node.delta_t)
                elif stm.state_idx == 74:  # 提起立方体
                    tmat_tgt_local[2, 3] += 0.09
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 75:  # 把立方体放到长方体上方
                    tmat_block1 = get_body_tmat(sim_node.mj_data, "block1_green")
                    tmat_block1[:3, 3] = tmat_block1[:3, 3] + np.array(
                        [0, 0, 0.04 + 0.031 * 3]
                    )
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_block1
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 76:  # 把立方体放到长方体上侧
                    tmat_tgt_local[2, 3] -= 0.01
                    sim_node.target_control[:6] = arm_ik.properIK(
                        tmat_tgt_local[:3, 3], trmat, sim_node.mj_data.qpos[:6]
                    )
                elif stm.state_idx == 77:  # 松开方块
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 78:  # 抬升高度
                    tmat_tgt_local[2, 3] += 0.02
                    sim_node.target_control[:6] = arm_ik.properIK(
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
            traceback.print_exc()
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
            imgs = obs.pop('img')
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)


        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
                for encoder in encoders.values():
                    encoder.close()
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
