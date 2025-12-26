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
from typing import Tuple, List, Union, Dict, Optional

import numpy as np
import torch

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
    def __init__(self, models_dict:Dict[str, str]):
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

        # Mapping for dynamic updates
        self.dynamic_mask = None
        self.point_to_body_idx = None

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

    def set_objects_mapping(self, objects_info: List[Tuple[str, int, int]]):
        """
        Set mapping from points to objects for dynamic updates.
        
        Args:
            objects_info: List of (name, start_idx, end_idx)
        """
        device = self.gaussians.device
        num_points = self.gaussians.xyz.shape[0]
        self.dynamic_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        self.point_to_body_idx = torch.zeros(num_points, dtype=torch.long, device=device)
        
        for i, (name, start, end) in enumerate(objects_info):
            self.dynamic_mask[start:end] = True
            self.point_to_body_idx[start:end] = i

    def update_gaussian_properties(self, pos: Union[np.ndarray, torch.Tensor], quat: Union[np.ndarray, torch.Tensor]):
        """
        Batch update gaussian properties for multiple objects using vectorized operations.
        
        Args:
            pos: (N_objects, 3) array or tensor of positions
            quat: (N_objects, 4) array or tensor of quaternions (wxyz)
        """
        if self.dynamic_mask is None or not self.dynamic_mask.any():
            return

        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float().cuda()
        if not isinstance(quat, torch.Tensor):
            quat = torch.from_numpy(quat).float().cuda()

        # Gather poses for all dynamic points
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

    def render_batch(self, cam_pos: np.ndarray, cam_xmat: np.ndarray, height: int, width: int, fovy_arr: np.ndarray):
        """
        Pure rendering call using batch_render.
        
        Args:
            cam_pos: (N_cams, 3) array of camera positions
            cam_xmat: (N_cams, 9) array of camera rotation matrices
            height: int
            width: int
            fovy_arr: (N_cams,) array of fov values
            
        Returns:
            rgb_tensor, depth_tensor
        """
        return batch_render(
            self.gaussians,
            cam_pos,
            cam_xmat,
            height,
            width,
            fovy_arr
        )
