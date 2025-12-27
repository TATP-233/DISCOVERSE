import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import os
import argparse
from discoverse.robots import AirbotPlayIK
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_base import AirbotPlayCfg
from discoverse.utils import get_body_tmat, get_site_tmat, step_func, SimpleStateMachine
from discoverse.task_base import AirbotPlayTaskBase, recoder_airbot_play, copypy2
from discoverse.task_base.airbot_task_base import PyavImageEncoder
from pathlib import Path


class SimNode(AirbotPlayTaskBase):
    tmat_cam_trans: np.ndarray = np.array([
        [ 0, 0,-1, 0],
        [-1, 0, 0, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1],
    ])
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)
        self.dist_v2c = np.linalg.norm(self.mj_model.camera("eye_side").pos - self.mj_data.body("viewpoint").xpos)
        self.cam_id = 1

        self.camera_pose_random = True
        self.yaw_range = np.pi / 6. * np.array([-1, 1])
        self.pitch_range = np.pi / 36. * np.array([-1, 1])
        # self.dis_range = np.random.uniform(0.95, 1.05)
        self.dis_range = 1.0

    def domain_randomization(self):
        # 随机 枣位置
        self.object_pose("block_green")[:2] += 2.*(np.random.random(2) - 0.5) * np.array([0.1, 0.05])
        # 随机 eye side 视角
        if self.camera_pose_random:
            yaw = -np.pi / 3 + np.random.uniform(self.yaw_range[0], self.yaw_range[1])
            pitch = 0 + np.random.uniform(self.pitch_range[0], self.pitch_range[1])
            tmat_tmp = np.eye(4)
            tmat_tmp[:3, :3] = Rotation.from_euler("xyz", [0, pitch, yaw], degrees=False).as_matrix()
            tmat_trans = np.eye(4)
            tmat_trans[0, 3] = -self.dist_v2c * self.dis_range
            tpos = get_body_tmat(self.mj_data, "viewpoint")
            tmat_camera_base = tpos @ tmat_tmp @ tmat_trans
            tmat_camera = tmat_camera_base @ self.tmat_cam_trans
            camera = self.mj_model.camera("eye_side")
            camera.pos[:] = tmat_camera[:3, 3]
            camera.quat[:] = Rotation.from_matrix(tmat_camera[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def check_success(self):
        tmat_jujube = get_body_tmat(self.mj_data, "block_green")
        tmat_gripper = get_site_tmat(self.mj_data, "endpoint")
        return (np.linalg.norm(tmat_jujube[:3, 3] - tmat_gripper[:3, 3]) < 0.03)

cfg = AirbotPlayCfg()
cfg.gs_model_dict["background"] = "scene/lab3/point_cloud.ply"
# cfg.gs_model_dict["drawer_1"]   = "hinge/drawer_1.ply"
# cfg.gs_model_dict["drawer_2"]   = "hinge/drawer_2.ply"
# cfg.gs_model_dict["block_green"]     = "object/block_green.ply"
cfg.init_qpos[:] = [-0.055, -0.547, 0.905, 1.599, -1.398, -1.599,  0.0]

cfg.mjcf_file_path = "mjcf/tasks_airbot_play/lift_block.xml"
# cfg.obj_list     = ["drawer_1", "drawer_2", "block_green"]
cfg.obj_list     = ["block_green"]
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
cfg.obs_depth_cam_id = [0, 1]
cfg.obs_point_cloud_id = [0, 1]
cfg.save_mjb_and_task_config = True
save_depth = False

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

    trmat = Rotation.from_euler("xyz", [0., np.pi/2, 0.], degrees=False).as_matrix()
    tmat_armbase_2_world = np.linalg.inv(get_body_tmat(sim_node.mj_data, "arm_base"))

    stm = SimpleStateMachine()
    stm.max_state_cnt = 5
    max_time = 10.0 #s

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
            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            os.makedirs(save_path, exist_ok=True)
            encoders = {cam_id: PyavImageEncoder(cfg.render_set["width"], cfg.render_set["height"], save_path, cam_id) for cam_id in cfg.obs_rgb_cam_id}
        try:
            if stm.trigger():
                if stm.state_idx == 0: # 伸到枣上方
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.1 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                    sim_node.target_control[6] = 1.
                elif stm.state_idx == 1: # 伸到枣
                    tmat_jujube = get_body_tmat(sim_node.mj_data, "block_green")
                    tmat_jujube[:3, 3] = tmat_jujube[:3, 3] + 0.027 * tmat_jujube[:3, 2]
                    tmat_tgt_local = tmat_armbase_2_world @ tmat_jujube
                    sim_node.target_control[:6] = arm_ik.properIK(tmat_tgt_local[:3,3], trmat, sim_node.mj_data.qpos[:6])
                elif stm.state_idx == 2: # 抓住枣
                    sim_node.target_control[6] = 0.4
                elif stm.state_idx == 3: # 抓稳枣
                    sim_node.delay_cnt = int(0.35/sim_node.delta_t)
                elif stm.state_idx == 4: # 提起来枣
                    tmat_tgt_local[2,3] += 0.07
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
            sim_node.reset()

        for i in range(sim_node.nj-1):
            action[i] = step_func(action[i], sim_node.target_control[i], move_speed * sim_node.joint_move_ratio[i] * sim_node.delta_t)
        action[6] = sim_node.target_control[6]

        obs, _, _, _, _ = sim_node.step(action)
        if len(obs_lst) < sim_node.mj_data.time * cfg.render_set["fps"]:
            rgb_imgs = obs.pop('img')
            depth_imgs = obs.pop('depth')
            point_cloud = obs.pop('point_cloud')
            act_lst.append(action.tolist().copy())
            obs_lst.append(obs)
            for cam_id, img in rgb_imgs.items():
                encoders[cam_id].encode(img, obs["time"])
            if save_depth:
                for cam_id, depth in depth_imgs.items():
                    depth_path = Path(save_path) / f"depth/{cam_id}/{len(obs_lst) - 1}"
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(depth_path, depth)
            for cam_id, pc in point_cloud.items():
                pc_path = Path(save_path) / f"point_cloud/{cam_id}/{len(obs_lst) - 1}"
                pc_path.parent.mkdir(parents=True, exist_ok=True)
                # print(pc[0][0], len(pc[0]))
                np.save(pc_path, pc[0])

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
