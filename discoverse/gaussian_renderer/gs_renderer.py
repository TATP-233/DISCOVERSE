# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2025 Yufei Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation

try:
    from gsplat.rendering import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available")

from .util_gau import load_ply
from .gaussiandata import GaussianData
from .batch_rasterization import batch_render, quaternion_multiply, transform_points

class GSRenderer:
    def __init__(self, models_dict:dict):
        """
        初始化高斯飞溅渲染器
        
        Args:
            models_dict: 模型字典,键为模型名称,值为模型路径
        """
        if not GSPLAT_AVAILABLE:
            raise RuntimeError("gsplat backend requested but not available. Please install gsplat.")

        self.gaussians = None
        self.need_rerender = True
        
        # Buffers for updates
        self.gau_ori_xyz_all_cu = None
        self.gau_ori_rot_all_cu = None
        self.gau_xyz_all_cu = None
        self.gau_rot_all_cu = None

        self.gaussians_all:dict[GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        bg_key = "background"
        if bg_key in models_dict:
            gs = load_ply(models_dict[bg_key])
            if "background_env" in models_dict.keys():
                bgenv_key = "background_env"
                bgenv_gs = load_ply(models_dict[bgenv_key])
                gs.xyz = np.concatenate([gs.xyz, bgenv_gs.xyz], axis=0)
                gs.rot = np.concatenate([gs.rot, bgenv_gs.rot], axis=0)
                gs.scale = np.concatenate([gs.scale, bgenv_gs.scale], axis=0)
                gs.opacity = np.concatenate([gs.opacity, bgenv_gs.opacity], axis=0)
                gs.sh = np.concatenate([gs.sh, bgenv_gs.sh], axis=0)

            self.gaussians_all[bg_key] = gs
            self.gaussians_idx[bg_key] = idx_sum
            self.gaussians_size[bg_key] = gs.xyz.shape[0]
            idx_sum = self.gaussians_size[bg_key]

        for (k, v) in models_dict.items():
            if k != "background" and k != "background_env":
                gs = load_ply(v)
                self.gaussians_all[k] = gs
                self.gaussians_idx[k] = idx_sum
                self.gaussians_size[k] = gs.xyz.shape[0]
                idx_sum += self.gaussians_size[k]

        self.update_gaussian_data(self.gaussians_all)
        
        self.gaussian_start_indices = self.gaussians_idx
        self.gaussian_end_indices = {k: v + self.gaussians_size[k] for k, v in self.gaussians_idx.items()}
        self.gaussian_model_names = list(self.gaussians_all.keys())

    @torch.no_grad()
    def update_gaussian_data(self, gaus: GaussianData):
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []

            max_sh_dim = 0
            for gaus_item in gaus.values():
                if gaus_item.sh.shape[1] > max_sh_dim:
                    max_sh_dim = gaus_item.sh.shape[1]

            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                
                current_sh = gaus_item.sh
                if current_sh.shape[1] < max_sh_dim:
                    padding = np.zeros((current_sh.shape[0], max_sh_dim - current_sh.shape[1]), dtype=current_sh.dtype)
                    current_sh = np.hstack([current_sh, padding])
                gau_c.append(current_sh)

            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_all.to_cuda()
        else:
            self.gaussians = gaus.to_cuda()

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)



    def init_renderer(self, mj_model):
        self.gs_idx_start = []
        self.gs_idx_end = []
        self.gs_body_ids = []
        
        for i in range(mj_model.nbody):
            body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name in self.gaussian_model_names:
                start_idx = self.gaussian_start_indices[body_name]
                end_idx = self.gaussian_end_indices[body_name]
                self.gs_idx_start.append(start_idx)
                self.gs_idx_end.append(end_idx)
                self.gs_body_ids.append(i)

        self.gs_idx_start = np.array(self.gs_idx_start)
        self.gs_idx_end = np.array(self.gs_idx_end)
        self.gs_body_ids = np.array(self.gs_body_ids)

        device = self.gaussians.device
        self.gs_idx_start_tensor = torch.tensor(self.gs_idx_start, device=device, dtype=torch.long)
        self.gs_idx_end_tensor = torch.tensor(self.gs_idx_end, device=device, dtype=torch.long)

        # Precompute point-to-body mapping for vectorized update
        num_points = self.gaussians.xyz.shape[0]
        self.dynamic_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        self.point_to_body_idx = torch.zeros(num_points, dtype=torch.long, device=device)
        
        for i in range(len(self.gs_idx_start)):
            start = self.gs_idx_start[i]
            end = self.gs_idx_end[i]
            self.dynamic_mask[start:end] = True
            self.point_to_body_idx[start:end] = i

    def update_gaussians(self, mj_data):
        if not hasattr(self, 'gs_idx_start') or len(self.gs_idx_start) == 0:
            return

        # Batch extract position (N, 3)
        pos_values = mj_data.xpos[self.gs_body_ids]
        
        # Batch extract quaternion (N, 4) - wxyz
        quat_values = mj_data.xquat[self.gs_body_ids]
        
        # Call batch update interface
        self.update_gaussian_properties(
            pos_values,
            quat_values
        )

    def update_gaussian_properties(self, pos, quat):
        """
        Batch update gaussian properties for multiple objects using vectorized operations.
        
        Args:
            pos: (N_objects, 3) array of positions
            quat: (N_objects, 4) array of quaternions (wxyz)
        """
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float().cuda()
        if not isinstance(quat, torch.Tensor):
            quat = torch.from_numpy(quat).float().cuda()
            
        if not self.dynamic_mask.any():
            return

        # Gather poses for all dynamic points
        # pos is (N_bodies, 3), point_to_body_idx maps each point to a body index
        # We only care about points where dynamic_mask is True
        
        mask = self.dynamic_mask
        body_indices = self.point_to_body_idx[mask]
        
        pos_expanded = pos[body_indices]   # (N_dynamic_points, 3)
        quat_expanded = quat[body_indices] # (N_dynamic_points, 4)
        
        xyz_ori = self.gau_ori_xyz_all_cu[mask]
        rot_ori = self.gau_ori_rot_all_cu[mask]
        
        # Vectorized transform
        xyz_new = transform_points(xyz_ori, pos_expanded, quat_expanded)
        rot_new = quaternion_multiply(quat_expanded, rot_ori)
        
        self.gaussians.xyz[mask] = xyz_new
        self.gaussians.rot[mask] = rot_new
        
        self.gau_xyz_all_cu = self.gaussians.xyz
        self.gau_rot_all_cu = self.gaussians.rot

    def render(self, mj_model, mj_data, cam_ids, width, height, free_camera=None):
        if len(cam_ids) == 0:
            return {}, {}, {}

        # 1. Get fixed camera poses
        fixed_cam_ids = [cid for cid in cam_ids if cid != -1]
        
        if len(fixed_cam_ids) > 0:
            fixed_cam_indices = np.array(fixed_cam_ids)
            cam_pos_fixed = mj_data.cam_xpos[fixed_cam_indices]
            cam_xmat_fixed = mj_data.cam_xmat[fixed_cam_indices]
            fovy_fixed = mj_model.cam_fovy[fixed_cam_indices]
        else:
            cam_pos_fixed = np.empty((0, 3))
            cam_xmat_fixed = np.empty((0, 9))
            fovy_fixed = np.empty((0,))

        # 2. Handle free camera (cam_id == -1)
        if -1 in cam_ids:
            if free_camera is None:
                raise ValueError("free_camera must be provided if cam_id -1 is requested")
            
            # Calculate free camera pose
            camera_rmat = np.array([
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0,  1,  0],
            ])
            rotation_matrix = camera_rmat @ Rotation.from_euler('xyz', [free_camera.elevation * np.pi / 180.0, free_camera.azimuth * np.pi / 180.0, 0.0]).as_matrix()
            camera_position = free_camera.lookat + free_camera.distance * rotation_matrix[:3,2]
            
            trans = camera_position
            rmat = rotation_matrix.flatten() # (9,)
            fovy = mj_model.vis.global_.fovy
            
            cam_pos = np.vstack([cam_pos_fixed, trans])
            cam_xmat = np.vstack([cam_xmat_fixed, rmat])
            fovy_arr = np.concatenate([fovy_fixed, [fovy]])
            
            batch_indices = {cid: i for i, cid in enumerate(fixed_cam_ids)}
            batch_indices[-1] = len(fixed_cam_ids)
        else:
            cam_pos = cam_pos_fixed
            cam_xmat = cam_xmat_fixed
            fovy_arr = fovy_fixed
            batch_indices = {cid: i for i, cid in enumerate(fixed_cam_ids)}

        # Call batch_render directly
        # batch_render expects numpy arrays for camera params, and handles tensor conversion internally
        rgb_tensor, depth_tensor = batch_render(
            self.gaussians,
            cam_pos,
            cam_xmat,
            height,
            width,
            fovy_arr
        )
        
        results = {}
        for cid, idx in batch_indices.items():
            results[cid] = (rgb_tensor[idx], depth_tensor[idx])
            
        return results