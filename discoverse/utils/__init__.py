from .controllor import PIDController, PIDarray
from .base_config import BaseConfig
from .statemachine import SimpleStateMachine
from .camera_spline_interpolation import interpolate_camera_poses

import os
import random
import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from discoverse import DISCOVERSE_ASSETS_DIR

def get_mocap_tmat(mj_data, mocap_id):
    tmat = np.eye(4)
    tmat[:3,:3] = Rotation.from_quat(mj_data.mocap_quat[mocap_id][[1,2,3,0]]).as_matrix()
    tmat[:3,3] = mj_data.mocap_pos[mocap_id]
    return tmat

def get_site_tmat(mj_data, site_name):
    tmat = np.eye(4)
    tmat[:3,:3] = mj_data.site(site_name).xmat.reshape((3,3))
    tmat[:3,3] = mj_data.site(site_name).xpos
    return tmat

def get_body_tmat(mj_data, body_name):
    tmat = np.eye(4)
    tmat[:3,:3] = Rotation.from_quat(mj_data.body(body_name).xquat[[1,2,3,0]]).as_matrix()
    tmat[:3,3] = mj_data.body(body_name).xpos
    return tmat

def get_control_idx(mj_model, control_names, check=False):
    control_idx = {
        ctr_name : mujoco.mj_name2id(
            mj_model, 
            mujoco.mjtObj.mjOBJ_ACTUATOR, 
            ctr_name
        ) 
        for ctr_name in control_names
    }
    if check:
        for k in control_idx:
            assert control_idx[k] >= 0, f"Control name not found in model: {k}"
    return control_idx

def get_sensor_idx(mj_model, sensor_names, check=False):
    sensor_id_cumsum = np.cumsum(mj_model.sensor_dim) - mj_model.sensor_dim
    sensor_idx = {
        sn : mujoco.mj_name2id(
            mj_model, 
            mujoco.mjtObj.mjOBJ_SENSOR, 
            sn
        ) 
        for sn in sensor_names
    }
    if check:
        for k in sensor_idx:
            assert sensor_idx[k] >= 0, f"Sensor name not found in model: {k}"
    sensor_data_id = {}
    for k, v in sensor_idx.items():
        sensor_data_id[k] = sensor_id_cumsum[v].item()
    return sensor_data_id

def step_func(current, target, step):
    if current < target - step:
        return current + step
    elif current > target + step:
        return current - step
    else:
        return target

def camera2k(fovy, width, height):
    cx = width / 2
    cy = height / 2
    fovx = 2 * np.arctan(np.tan(fovy / 2.) * width / height)
    fx = cx / np.tan(fovx / 2)
    fy = cy / np.tan(fovy / 2)
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]])

def get_random_texture():
    TEXTURE_1K_PATH = os.getenv("TEXTURE_1K_PATH", os.path.join(DISCOVERSE_ASSETS_DIR, "textures_1k"))
    if not TEXTURE_1K_PATH is None and os.path.exists(TEXTURE_1K_PATH):
        for _ in range(5):
            img_path = os.path.join(TEXTURE_1K_PATH, random.choice(os.listdir(TEXTURE_1K_PATH)))
            if img_path.endswith('.png') or img_path.endswith('.jpg'):
                return Image.open(img_path)
            continue
    else:
        # raise ValueError("TEXTURE_1K_PATH not found")
        print("Warning: TEXTURE_1K_PATH not found! Please set the TEXTURE_1K_PATH environment variable to the path of the textures_1k directory.")
        return Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))

__all__ = [
    "PIDController",
    "PIDarray",
    "BaseConfig",
    "SimpleStateMachine",
    "interpolate_camera_poses",
    "get_mocap_tmat",
    "get_site_tmat",
    "get_body_tmat",
    "get_control_idx",
    "get_sensor_idx",
    "step_func",
    "camera2k",
    "get_random_texture"
]