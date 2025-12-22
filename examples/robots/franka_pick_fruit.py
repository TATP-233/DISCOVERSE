from typing import Any, Dict, Optional, Union

import time
import mink
import mujoco
import mujoco.viewer
import torch
import numpy as np
from etils import epath
import argparse

try:
    from .mink_arm_ik import MinkIK
except ImportError:
    from mink_arm_ik import MinkIK

from discoverse import DISCOVERSE_ASSETS_DIR
from discoverse.utils import update_assets
from discoverse.gaussian_renderer.gs_renderer_mujoco import GSRendererMuJoCo
from discoverse.gaussian_web_renderer.client import GSRendererRemote

H = 300; W = 400
_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FINGER_JOINTS = ["right_driver_joint", "left_driver_joint"]

_DISCOVERSE_ASSETS_DIR = epath.Path(DISCOVERSE_ASSETS_DIR)
_MJCF_FRANKA_DIR = _DISCOVERSE_ASSETS_DIR / "mjcf" / "manipulator" / "franka_robotiq"
_ASSETS_FRANKA_DIR = _DISCOVERSE_ASSETS_DIR / "meshes" / "franka_emika_panda"
_ASSETS_ROBOTIQ_DIR = _DISCOVERSE_ASSETS_DIR / "meshes" / "gripper" / "robotiq"
_ASSETS_BANANA_DIR = _DISCOVERSE_ASSETS_DIR / "meshes"/ "object" / "banana"
_3DGS_FRANKA_DIR = _DISCOVERSE_ASSETS_DIR / "3dgs" / "manipulator" / "franka_robotiq"

def get_assets() -> Dict[str, bytes]:
    assets = {}
    update_assets(assets, _MJCF_FRANKA_DIR, "*.xml")
    update_assets(assets, _ASSETS_FRANKA_DIR)
    update_assets(assets, _ASSETS_ROBOTIQ_DIR)
    update_assets(assets, _ASSETS_BANANA_DIR)
    return assets

class FrankaCfg:
    mjcf_file_path = "pick_fruit.xml"
    decimation     = 8
    timestep       = 0.005

    gaussians = {
        "background" : (_3DGS_FRANKA_DIR / "background.ply").as_posix(),

        "link0" : (_3DGS_FRANKA_DIR / "franka" / "link0.ply").as_posix(),
        "link1" : (_3DGS_FRANKA_DIR / "franka" / "link1.ply").as_posix(),
        "link2" : (_3DGS_FRANKA_DIR / "franka" / "link2.ply").as_posix(),
        "link3" : (_3DGS_FRANKA_DIR / "franka" / "link3.ply").as_posix(),
        "link4" : (_3DGS_FRANKA_DIR / "franka" / "link4.ply").as_posix(),
        "link5" : (_3DGS_FRANKA_DIR / "franka" / "link5.ply").as_posix(),
        "link6" : (_3DGS_FRANKA_DIR / "franka" / "link6.ply").as_posix(),
        "link7" : (_3DGS_FRANKA_DIR / "franka" / "link7.ply").as_posix(),

        "robotiq_base"      : (_3DGS_FRANKA_DIR / "robotiq" / "robotiq_base.ply").as_posix(),
        "left_driver"       : (_3DGS_FRANKA_DIR / "robotiq" / "left_driver.ply").as_posix(),
        "left_coupler"      : (_3DGS_FRANKA_DIR / "robotiq" / "left_coupler.ply").as_posix(),
        "left_spring_link"  : (_3DGS_FRANKA_DIR / "robotiq" / "left_spring_link.ply").as_posix(),
        "left_follower"     : (_3DGS_FRANKA_DIR / "robotiq" / "left_follower.ply").as_posix(),

        "right_driver"      : (_3DGS_FRANKA_DIR / "robotiq" / "right_driver.ply").as_posix(),
        "right_coupler"     : (_3DGS_FRANKA_DIR / "robotiq" / "right_coupler.ply").as_posix(),
        "right_spring_link" : (_3DGS_FRANKA_DIR / "robotiq" / "right_spring_link.ply").as_posix(),
        "right_follower"    : (_3DGS_FRANKA_DIR / "robotiq" / "right_follower.ply").as_posix(),
    }

class FrankaBase:
    def __init__(self, config: FrankaCfg, use_remote=False, remote_ip="127.0.0.1"):
        self.config = config
        self.free_camera = None

        xml_path = _MJCF_FRANKA_DIR / self.config.mjcf_file_path
        self.mjcf_xml = xml_path.read_text()
        self._model_assets = get_assets()
        self.mj_model = mujoco.MjModel.from_xml_string(self.mjcf_xml, assets=self._model_assets)
        self.mj_model.opt.timestep = self.config.timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        self._robot_arm_qposadr = np.array([
            self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in _ARM_JOINTS
        ])
        self._robot_qposadr = np.array([
            self.mj_model.jnt_qposadr[self.mj_model.joint(j).id] for j in _ARM_JOINTS + _FINGER_JOINTS
        ])
        
        if use_remote:
            print(f"Using Remote Renderer at {remote_ip}")
            self.renderer = GSRendererRemote(self.config.gaussians, server_ip=remote_ip)
        else:
            print("Using Local Renderer")
            self.renderer = GSRendererMuJoCo(self.config.gaussians)
            
        self.renderer.init_renderer(self.mj_model)
        self.timing_stats = []

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return self.getObservation()

    def step(self, action: np.ndarray = None):
        if action is not None:
            self.mj_data.ctrl[:] = action

        for _ in range(self.config.decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def checkSuccess(self):
        return False

    def getObservation(self):
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_update = torch.cuda.Event(enable_timing=True)
        evt_render = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)

        evt_start.record()
        self.renderer.update_gaussians(self.mj_data)
        evt_update.record()
        
        results_tensor = self.renderer.render(
            self.mj_model,
            self.mj_data,
            [-1, 0] if self.free_camera is not None else [0],  # camera id list
            W,
            H,
            self.free_camera
        )
        evt_render.record()
        
        # Handle both float (local GPU) and uint8 (remote CPU) tensors
        img_tensor = results_tensor[0][0]
        if img_tensor.dtype == torch.uint8:
            rgb_np = img_tensor.cpu().numpy()
        else:
            rgb_np = (255. * torch.clamp(img_tensor, 0.0, 1.0)).to(torch.uint8).cpu().numpy()
            
        evt_end.record()

        observation_dict = {
            "state" : self.mj_data.qpos[self._robot_qposadr].copy(),
            "rgb"   : rgb_np,
        }

        if self.free_camera is not None and -1 in results_tensor:
            # Handle free camera similarly
            free_img_tensor = results_tensor[-1][0]
            if free_img_tensor.dtype == torch.uint8:
                rgb_free = free_img_tensor.cpu().numpy()
            else:
                rgb_free = (255. * torch.clamp(free_img_tensor, 0.0, 1.0)).to(torch.uint8).cpu().numpy()
            observation_dict["free_camera"] = rgb_free

        return observation_dict

    def set_mocap_target(self, target_name, target_pos, target_quat, box_color=(0,1,0,0.1)):
        """设置Mocap目标位置和姿态"""
        mocap_id = self.mj_model.body(target_name).mocapid
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = target_pos
            self.mj_data.mocap_quat[mocap_id] = target_quat
            self.mj_model.geom(f'{target_name}_box').rgba = box_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action="store_true", help="Use remote rendering server")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Remote server IP")
    parser.add_argument("--latency", action="store_true", help="Monitor communication latency")
    args = parser.parse_args()

    cfg = FrankaCfg()
    cfg.gaussians["banana"] = (_DISCOVERSE_ASSETS_DIR / "3dgs" / "object" / "banana.ply").as_posix()

    exec_node = FrankaBase(cfg, use_remote=args.remote, remote_ip=args.ip)
    if args.remote and args.latency:
        exec_node.renderer.monitor_latency = True
    obs = exec_node.reset()
    exec_node.timing_stats = [] # Clear warmup stats

    use_mocap_ik = True

    if use_mocap_ik:
        ik_model = mujoco.MjModel.from_xml_string(exec_node.mjcf_xml, assets=exec_node._model_assets)
        ik_solver = MinkIK(ik_model, len(_ARM_JOINTS), frame_name="gripper")

        mocap_name = "mocap_target"
        mocap_box_name = mocap_name + "_box"
        mocap_id = exec_node.mj_model.body(mocap_name).mocapid[0]

        mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "gripper", "site")
        ik_solver.configuration.update(exec_node.mj_data.qpos)
        ik_solver.posture_task.set_target_from_configuration(ik_solver.configuration)

    sync = True
    _last_time = -1.
    with mujoco.viewer.launch_passive(exec_node.mj_model, exec_node.mj_data) as viewer:
        exec_node.free_camera = viewer.cam
        while viewer.is_running():
            if exec_node.mj_data.time < _last_time:
                _last_time = -1.
                exec_node.reset()
                if use_mocap_ik:
                    mink.move_mocap_to_frame(exec_node.mj_model, exec_node.mj_data, mocap_name, "gripper", "site")
                    ik_solver.configuration.update(exec_node.mj_data.qpos)
                    ik_solver.posture_task.set_target_from_configuration(ik_solver.configuration)
            _last_time = exec_node.mj_data.time

            step_time = time.time()

            if use_mocap_ik:
                mink_target_se3 = mink.SE3.from_mocap_name(exec_node.mj_model, exec_node.mj_data, mocap_name)
                ik_solver.end_effector_task.set_target(mink_target_se3)
                res = ik_solver.converge_ik()
                if res:
                    # 设置目标框为绿色（表示IK计算成功）
                    exec_node.mj_model.geom(mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
                else:
                    # 设置目标框为红色（表示IK计算失败）
                    exec_node.mj_model.geom(mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)
                solution = exec_node.mj_data.ctrl.copy()
                solution[:ik_solver.ndof_arm] = ik_solver.configuration.data.qpos[:ik_solver.ndof_arm]
            else:
                solution = None

            exec_node.step(solution)
            
            obs = exec_node.getObservation()

            if "free_camera" in obs:
                viewport = mujoco.MjrRect(viewer.viewport.left + viewer.viewport.width - W, 0, W, H * 2)
                viewer.set_images([(viewport, np.vstack([obs["free_camera"], obs["rgb"]]))])
            else:
                viewport = mujoco.MjrRect(viewer.viewport.left + viewer.viewport.width - W, 0, W, H)
                viewer.set_images([(viewport, obs["rgb"])])

            viewer.sync()
            if sync:
                time.sleep(max(0, exec_node.mj_model.opt.timestep * cfg.decimation - (time.time() - step_time)))
