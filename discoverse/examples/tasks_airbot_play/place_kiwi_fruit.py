import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import argparse
import multiprocessing as mp

from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, batch_encode_videos, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder

import traceback
class SimNode(AirbotPlayTaskBase):
    def domain_randomization(self):
        random_bias_xy = np.zeros(2)
        random_bias_xy[0] = 2.*(np.random.random() - 0.5) * 0.08
        random_bias_xy[1] = 2.*(np.random.random() - 0.5) * 0.03

        # 随机 猕猴桃位置
        self.object_pose("kiwi")[:2] += random_bias_xy[:2]

        # 随机 碟子位置
        self.object_pose("plate_white")[:2] += random_bias_xy[:2]

        # 随机 碗位置
        self.object_pose("flower_bowl")[0] += 2.*(np.random.random() - 0.5) * 0.05
        self.object_pose("flower_bowl")[1] += 2.*(np.random.random() - 0.5) * 0.03

    def check_success(self):
        tmat_jujube = get_body_tmat(self.mj_data, "kiwi")
        tmat_bowl = get_body_tmat(self.mj_data, "flower_bowl")
        return (abs(tmat_bowl[2, 2]) > 0.99) and np.hypot(tmat_jujube[0, 3] - tmat_bowl[0, 3], tmat_jujube[1, 3] - tmat_bowl[1, 3]) < 0.02

cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"]  = "scene/lab3/point_cloud.ply"
cfg.gs_model_dict["drawer_1"]    = "hinge/drawer_1.ply"
cfg.gs_model_dict["drawer_2"]    = "hinge/drawer_2.ply"
cfg.gs_model_dict["kiwi"]        = "object/kiwi.ply"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"
cfg.gs_model_dict["flower_bowl"] = "object/flower_bowl.ply"
cfg.init_qpos[:] = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599,  0.0]

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/place_kiwi_fruit.xml"
cfg.obj_list     = ["drawer_1", "drawer_2", "kiwi", "plate_white", "flower_bowl"]
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
    parser.add_argument("--save_segment", action="store_true", help="save segment videos")
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

    if args.save_segment:
        cfg.obs_depth_cam_id = list(set(cfg.obs_rgb_cam_id + ([] if cfg.obs_depth_cam_id is None else cfg.obs_depth_cam_id)))
        from discoverse.randomain.utils import SampleforDR
        samples = SampleforDR(objs=cfg.obj_list[2:], robot_parts=cfg.rb_link_list, cam_ids=cfg.obs_rgb_cam_id, save_dir=os.path.join(save_dir, "segment"), fps=cfg.render_set["fps"])

    sim_node = SimNode(cfg)
    if hasattr(cfg, "save_mjb_and_task_config") and cfg.save_mjb_and_task_config and data_idx == 0:
        mujoco.mj_saveModel(sim_node.mj_model, os.path.join(save_dir, os.path.basename(cfg.mjcf_file_path).replace(".xml", ".mjb")))
        copypy2(os.path.abspath(__file__), os.path.join(save_dir, os.path.basename(__file__)))

    arm_ik = AirbotPlayIK()

    trmat = Rotation.from_euler("xyz", [0., 1.4, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 9
    max_time = 15.0 #s

    action = np.zeros(7)
    act_lst, obs_lst = [], []
    move_speed = 0.75
    sim_node.reset()
    while sim_node.running:
        if sim_node.reset_sig:
            sim_node.reset_sig = False
            stm.reset()
            action[:] = sim_node.target_control[:]
            act_lst, obs_lst = [], []
            if args.save_segment:
                samples.reset()
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到猕猴桃上方
                    tmat_kiwi = get_body_tmat(sim_node.mj_data, "kiwi")
                    tmat_kiwi[:3, 3] = tmat_kiwi[:3, 3] + 0.1 * tmat_kiwi[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_kiwi
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 1: # 伸到猕猴桃
                    tmat_kiwi = get_body_tmat(sim_node.mj_data, "kiwi")
                    tmat_kiwi[:3, 3] = tmat_kiwi[:3, 3] + 0.027 * tmat_kiwi[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_kiwi
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 抓住猕猴桃
                    sim_node.target_control[6] = 0.5
                elif stm.state_idx == 3: # 抓稳猕猴桃
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # 提起来猕猴桃
                    tmat_tgt_local[2,3] += 0.15
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 5: # 把猕猴桃放到碗上空
                    tmat_bowl = get_body_tmat(sim_node.mj_data, "flower_bowl")
                    tmat_bowl[:3,3] = tmat_bowl[:3, 3] + np.array([0.0, 0.0, 0.13])
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_bowl
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 6: # 降低高度 把猕猴桃放到碗上
                    tmat_tgt_local[2,3] -= 0.05
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 7: # 松开猕猴桃
                    sim_node.target_control[6] = 1
                elif stm.state_idx == 8: # 抬升高度
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
            traceback.print_exc()
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        imgs = obs.pop("img")
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            for cam_id, img in imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)
            if args.save_segment:
                samples.sampling(sim_node)

        if stm.state_idx >= stm.max_state_cnt:
            if sim_node.check_success():
                recoder_airbot_play(save_path, act_lst, obs_lst, cfg)
                for encoder in encoders.values():
                    encoder.close()
                if args.save_segment:
                    seg_process = mp.Process(target=samples.save)
                    seg_process.start()
    
                data_idx += 1
                print("\r{:4}/{:4} ".format(data_idx, data_set_size), end="")
                if data_idx >= data_set_size:
                    break
            else:
                print(f"{data_idx} Failed")

            sim_node.reset()
