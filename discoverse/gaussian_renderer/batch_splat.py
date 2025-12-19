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

"""
Minimal batch Gaussian splat renderer that works with mjx state tensors.

Usage pattern (3 steps):
1) init: cfg = BatchSplatConfig(...); renderer = BatchSplatRenderer(cfg)
2) loop: gsb = renderer.batch_update_gaussians(body_pos, body_quat)
3) loop: rgb, depth = renderer.batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy)

Where body_pos/body_quat/cam_pos/cam_xmat can come from mjx (MuJoCo JAX) state via
`torch.utils.dlpack.from_dlpack(state.data.xxx)`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .gaussiandata import GaussianData
from .util_gau import load_ply
from .batch_rasterization import (
    batch_env_render as _batch_env_render,
    batch_update_gaussians as _batch_update_gaussians,
)


@dataclass
class BatchSplatConfig:
    # Mapping from MuJoCo body name -> PLY path (local coordinate frame).
    body_gaussians: Dict[str, str]
    # Optional static/background PLY that is not attached to any body.
    background_ply: Optional[str] = None
    # Device to place tensors on; if None, will pick CUDA if available else CPU.
    device: Optional[torch.device] = None
    # Minibatch size for rendering
    minibatch: Optional[int] = None


class BatchSplatRenderer:
    def __init__(self, cfg: BatchSplatConfig, mj_model=None):
        device = cfg.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.device = device
        self.minibatch = cfg.minibatch

        xyz_list: List[torch.Tensor] = []
        rot_list: List[torch.Tensor] = []
        scale_list: List[torch.Tensor] = []
        opacity_list: List[torch.Tensor] = []
        sh_list: List[torch.Tensor] = []

        self.gs_idx_start: List[int] = []
        self.gs_idx_end: List[int] = []
        self.gs_body_ids: List[int] = []

        body_name_to_id = {}
        if mj_model is not None:
            for i in range(mj_model.nbody):
                body_name = mj_model.body(i).name
                if body_name:
                    body_name_to_id[body_name] = i

        # Load per-body gaussians
        for body_name, ply_path in cfg.body_gaussians.items():
            g = load_ply(ply_path)
            start = len(torch.cat(xyz_list)) if xyz_list else 0
            end = start + len(g.xyz)
            self.gs_idx_start.append(start)
            self.gs_idx_end.append(end)
            if body_name not in body_name_to_id:
                raise ValueError(f"Body '{body_name}' not found in mj_model; available: {list(body_name_to_id.keys())}")
            self.gs_body_ids.append(body_name_to_id[body_name])
            xyz_list.append(torch.tensor(g.xyz, device=device, dtype=torch.float32))
            rot_list.append(torch.tensor(g.rot, device=device, dtype=torch.float32))
            scale_list.append(torch.tensor(g.scale, device=device, dtype=torch.float32))
            opacity_list.append(torch.tensor(g.opacity, device=device, dtype=torch.float32))
            sh_list.append(torch.tensor(g.sh, device=device, dtype=torch.float32))

        # Optional background/static gaussian
        if cfg.background_ply:
            g = load_ply(cfg.background_ply)
            xyz_list.append(torch.tensor(g.xyz, device=device, dtype=torch.float32))
            rot_list.append(torch.tensor(g.rot, device=device, dtype=torch.float32))
            scale_list.append(torch.tensor(g.scale, device=device, dtype=torch.float32))
            opacity_list.append(torch.tensor(g.opacity, device=device, dtype=torch.float32))
            sh_list.append(torch.tensor(g.sh, device=device, dtype=torch.float32))

        # Concatenate template
        xyz_all = torch.cat(xyz_list, dim=0)
        rot_all = torch.cat(rot_list, dim=0)
        scale_all = torch.cat(scale_list, dim=0)
        opacity_all = torch.cat(opacity_list, dim=0)
        sh_all = torch.cat(sh_list, dim=0).reshape(xyz_all.shape[0], -1, 3).contiguous()

        self.template = GaussianData(
            xyz=xyz_all,
            rot=rot_all,
            scale=scale_all,
            opacity=opacity_all,
            sh=sh_all,
        )

        self.gs_idx_start = torch.tensor(self.gs_idx_start, dtype=torch.long, device=device)
        self.gs_idx_end = torch.tensor(self.gs_idx_end, dtype=torch.long, device=device)
        self.gs_body_ids = torch.tensor(self.gs_body_ids, dtype=torch.long, device=device)

        # Precompute point-to-body mapping for vectorized update
        num_points = len(self.template)
        self.dynamic_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        self.point_to_body_idx = torch.zeros(num_points, dtype=torch.long, device=device)
        
        for i in range(len(self.gs_idx_start)):
            start = self.gs_idx_start[i]
            end = self.gs_idx_end[i]
            body_id = self.gs_body_ids[i]
            self.dynamic_mask[start:end] = True
            self.point_to_body_idx[start:end] = body_id

    def batch_update_gaussians(self, body_pos: torch.Tensor, body_quat: torch.Tensor):
        """Update gaussians using body poses.

        Args:
            body_pos: (Nenv, Nbody, 3) torch tensor
            body_quat: (Nenv, Nbody, 4) torch tensor (wxyz)
        Returns:
            GaussianBatchData with per-env gaussians.
        """
        # Ensure device
        body_pos = body_pos.to(self.device)
        body_quat = body_quat.to(self.device)
        return _batch_update_gaussians(
            self.template,
            body_pos,
            body_quat,
            point_to_body_idx=self.point_to_body_idx,
            dynamic_mask=self.dynamic_mask,
        )

    def batch_env_render(
        self,
        gsb,
        cam_pos: torch.Tensor,
        cam_xmat: torch.Tensor,
        height: int,
        width: int,
        fovy: np.ndarray,
        bg_imgs: Optional[torch.Tensor] = None,
    ):
        """Render RGBD for batch envs and cameras."""
        cam_pos = cam_pos.to(self.device)
        cam_xmat = cam_xmat.to(self.device)
        return _batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy, bg_imgs=bg_imgs, minibatch=self.minibatch)

__all__ = ["BatchSplatConfig", "BatchSplatRenderer"]
