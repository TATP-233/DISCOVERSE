"""
Discoverse 7DOF (Play G3) -> Zenoh Publisher (RGB JPEG + Depth ZSTD-Float32 + PCL)

与 publish.py 功能相同，使用 airplay_7dof_pick_blocks.xml 及 7DOF 机械臂 (8 actuators: 7 arm + 1 gripper)。

Publishes:
  - env/{CAMERA_ID}/rgb   : ImageData(jpeg)
  - env/{CAMERA_ID}/depth : ImageData(zstd-depth32 float32 meters)
  - env/{CAMERA_ID}/pcl   : raw bytes float32 (N,4) [x,y,z,packed_rgb_bits_as_float]
"""
import argparse
import os
import time

import numpy as np
import mujoco

from subscribe_joint_state_7dof import ArmGripperSubscriber7dof, GRIPPER_WIDTH_MAX

import discoverse
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.airbot_play_7dof_base import AirbotPlay7dofBase, AirbotPlay7dofCfg

from zenoh_bridge import (
    ZenohPub,
    ZenohPubConfig,
    GpuPclProjector,
    publish_rgb_depth,
    publish_pcl,
)

# -------------------------
# Config
# -------------------------
CAMERA_ID = "SN_57524755"
MJ_CAM_NAME = "stream_cam"
FPS_PUB = 20.0

PCL_HZ = 10.0
PCL_STRIDE = 1
PCL_MAX_DEPTH = 4.0

MJCF_PATH = os.path.join(
    DISCOVERSE_ROOT_DIR,
    "models/mjcf/manipulator/roombia/airplay_7dof_pick_blocks.xml",
)

# 3DGS: xml body name -> grasp_gen_obj ply (参照 open_drawer.py)
GS_GRASP_GEN = "grasp_gen_obj"
cfg_gs_model_dict = {
    "background": f"{GS_GRASP_GEN}/scene.ply",  # env
    "plate": f"{GS_GRASP_GEN}/plate.ply",  # xml body "plate"
    "block": f"{GS_GRASP_GEN}/bottle.ply",  # xml body "block"
}

# 保存 RGB/Depth 的默认目录，供 export_yaml 的 RGB_PATH / DEPTH_PATH 读取
DEBUG_RGBD_DIR = "debug_rgbd"


class SimNode(AirbotPlay7dofBase):
    def __init__(self, config):
        super().__init__(config)
        self._did_init = False

    def resetState(self):
        if not self._did_init:
            # 首次必须用 keyframe 初始化，否则仿真从不稳定状态开始会爆炸
            super().resetState()
            self._did_init = True
        # 后续 step() 触发 terminated 时不再 reset，保持 "不 reset" 语义


def _set_geom_contact_params(
    model: mujoco.MjModel,
    geom_name: str,
    friction=(3.0, 0.0, 0.0),
    condim=3,
    solref=(0.01, 1.0),
    solimp=(0.95, 0.95, 0.002),
):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        print(f"[WARN] geom '{geom_name}' not found, skip")
        return False

    model.geom_friction[gid, 0] = float(friction[0])
    model.geom_friction[gid, 1] = float(friction[1])
    model.geom_friction[gid, 2] = float(friction[2])
    model.geom_condim[gid] = int(condim)
    model.geom_solref[gid, 0] = float(solref[0])
    model.geom_solref[gid, 1] = float(solref[1])
    model.geom_solimp[gid, 0] = float(solimp[0])
    model.geom_solimp[gid, 1] = float(solimp[1])
    model.geom_solimp[gid, 2] = float(solimp[2])

    print(
        f"[OK] patched geom '{geom_name}': "
        f"friction={model.geom_friction[gid]} condim={model.geom_condim[gid]} "
        f"solref={model.geom_solref[gid]} solimp={model.geom_solimp[gid]}"
    )
    return True


def patch_finger_pads(sim):
    """7DOF Play G3 夹爪 pad 名称与 6DOF 不同，按需 patch"""
    pass


def patch_actuator_gains(sim, scale: float = 0.4):
    """降低 actuator kp 以提升仿真稳定性（Python 侧修改，无需改 XML）"""
    m = sim.mj_model
    for i in range(min(sim.nj, m.nu)):
        # position actuator: gainprm[0]=kp, gainprm[1]=kv
        if m.actuator_gainprm[i, 0] > 0:
            m.actuator_gainprm[i, 0] *= scale
    print(f"[OK] 降低 actuator kp 至 {scale*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Discoverse 7DOF -> Zenoh publisher (RGB + Depth + PCL)")
    parser.add_argument(
        "--debug-save-pcl",
        action="store_true",
        help="Save every published PCL to debug_pcl/pcl_000000.ply, ...",
    )
    parser.add_argument(
        "--debug-save-rgbd",
        action="store_true",
        help="Save RGB/depth to debug_rgbd/rgb.npy, depth.npy (for export_yaml RGB_PATH/DEPTH_PATH)",
    )
    parser.add_argument(
        "--invert-gripper",
        action="store_true",
        help="Invert gripper direction if open/close is reversed",
    )
    parser.add_argument(
        "--use-gs",
        action="store_true",
        help="Use gaussian splatting renderer (3DGS)",
    )
    args = parser.parse_args()

    print(discoverse.__logo__)
    np.set_printoptions(precision=3, suppress=True)

    # resolve camera numeric id
    _m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    cam_id = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_CAMERA, MJ_CAM_NAME)
    del _m
    if cam_id < 0:
        raise RuntimeError(f"Camera '{MJ_CAM_NAME}' not found in MJCF: {MJCF_PATH}")

    # discoverse cfg（与 publish.py 对齐）
    cfg = AirbotPlay7dofCfg()
    cfg.mjcf_file_path = MJCF_PATH
    cfg.obs_rgb_cam_id = [cam_id]
    cfg.obs_depth_cam_id = [cam_id]
    cfg.timestep = 1 / 240
    cfg.decimation = 10  # 与 airbot_play_7dof_base / test_7dof 一致
    cfg.sync = True
    cfg.headless = False
    cfg.render_set = {"fps": int(FPS_PUB), "width": 600, "height": 600}
    cfg.init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    cfg.enable_render = True  # 必须 True 才会有 img/depth

    # 3DGS: env(background) + plate + block，与 xml body 一一对应
    for k, v in cfg_gs_model_dict.items():
        cfg.gs_model_dict[k] = v
    cfg.obj_list = ["plate", "block"]
    cfg.use_gaussian_renderer = args.use_gs

    sim = SimNode(cfg)
    # patch_actuator_gains(sim, scale=0.4)
    sim.reset()

    # 首帧 obs 校验：确认 img/depth 已正确填充
    _first_obs = sim.getObservation()
    _img_keys = list(_first_obs.get("img", {}).keys())
    _depth_keys = list(_first_obs.get("depth", {}).keys())
    _rgb0 = _first_obs.get("img", {}).get(cam_id)
    _depth0 = _first_obs.get("depth", {}).get(cam_id)
    print(f"[obs] img keys={_img_keys} depth keys={_depth_keys} cam_id={cam_id}")
    if _rgb0 is not None and _depth0 is not None:
        print(f"[OK] 首帧 RGB shape={_rgb0.shape} depth shape={_depth0.shape}")
    else:
        print(f"[WARN] 首帧 RGB/depth 为 None! enable_render={cfg.enable_render} obs_rgb_cam_id={cfg.obs_rgb_cam_id} obs_depth_cam_id={cfg.obs_depth_cam_id}")

    print("sim.nj =", sim.nj)
    print("actuator ctrlrange =", sim.mj_model.actuator_ctrlrange[: sim.nj])

    # zenoh publishers + gpu projector
    pubs = ZenohPub(ZenohPubConfig(camera_id=CAMERA_ID, shm_mb=256, jpeg_quality=85, zstd_level=1))
    projector = GpuPclProjector(stride=PCL_STRIDE, max_depth=PCL_MAX_DEPTH, device="cuda")

    # --- subscribe arm + gripper -> action (7DOF 专用) ---
    # gripper position: 0=闭合, GRIPPER_WIDTH_MAX=0.0722=张开
    # fallback_arm_q: 仅夹爪数据时用其作为臂姿，支持夹爪单独验证开合
    sub = ArmGripperSubscriber7dof(
        arm_topic="robot/state/joint",
        gripper_topic="gripper/sim_two_finger/state",
        gripper_ctrl_min=0.0,
        gripper_ctrl_max=GRIPPER_WIDTH_MAX,
        invert_gripper=bool(args.invert_gripper),
        fallback_arm_q=sim.init_joint_ctrl[:7].copy(),
    )

    # action init (fallback)：无订阅数据时保持初始姿态
    action = sim.init_joint_ctrl.copy()

    # 预热期：前 N 秒只用 init，不跟订阅，让仿真稳定
    warmup_sec = 3.0
    t_start = time.time()

    # publish schedule
    period = 1.0 / FPS_PUB
    next_pub_wall = time.time()

    logging_every = 2.0
    next_log = time.time() + logging_every
    n_pub_ok, n_pub_none = 0, 0  # 图像发布统计
    last_rgbd_warn = 0.0  # 避免 rgb/depth 为 None 时刷屏

    try:
        while sim.running:
            # 1) 控制仿真：预热期内只用 init，之后订阅优先
            now_ = time.time()
            in_warmup = (now_ - t_start) < warmup_sec
            latest = None if in_warmup else sub.get_latest_action()
            if latest is not None and latest.shape[0] == sim.nj:
                ctrl_low = sim.mj_model.actuator_ctrlrange[: sim.nj, 0]
                ctrl_high = sim.mj_model.actuator_ctrlrange[: sim.nj, 1]
                # action[7] 已由 subscribe 映射为 ctrl [0, 1]，直接使用
                action = np.clip(latest.astype(np.float64), ctrl_low, ctrl_high)
                print(f"action = {action}")

            obs, *_ = sim.step(action)

            # 2) 发布图像
            now = time.time()
            if now < next_pub_wall:
                continue
            next_pub_wall += period

            rgb = obs.get("img", {}).get(cam_id, None)
            depth = obs.get("depth", {}).get(cam_id, None)
            if rgb is not None and depth is not None:
                n_pub_ok += 1
            else:
                n_pub_none += 1
                if now - last_rgbd_warn > 1.0:  # 最多 1 秒提醒一次
                    last_rgbd_warn = now
                    print(f"[WARN] rgb={rgb is not None} depth={depth is not None} obs.img.keys={list(obs.get('img',{}).keys())} obs.depth.keys={list(obs.get('depth',{}).keys())}")
            publish_rgb_depth(pubs, rgb, depth)
            if args.debug_save_rgbd and rgb is not None and depth is not None:
                os.makedirs(DEBUG_RGBD_DIR, exist_ok=True)
                np.save(os.path.join(DEBUG_RGBD_DIR, "rgb.npy"), rgb)
                np.save(os.path.join(DEBUG_RGBD_DIR, "depth.npy"), depth.astype(np.float32))

            # if rgb is not None and depth is not None:
            #     publish_pcl(pubs, projector, sim, cam_id, rgb, depth, save_to_disk=args.debug_save_pcl)

            if now >= next_log:
                next_log = now + logging_every
                grip = float(action[7]) if sim.nj >= 8 else float("nan")
                print(f"[ctrl] t={sim.mj_data.time:.2f}s grip={grip:.4f} action={action.round(3)} | rgbd pub ok={n_pub_ok} none={n_pub_none}")

    finally:
        sub.close()
        print("Exited.")


if __name__ == "__main__":
    # export MUJOCO_GL=glfw   # 如果 GLX BadAccess
    main()
