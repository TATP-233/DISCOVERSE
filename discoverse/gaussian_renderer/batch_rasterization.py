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
from torch import Tensor

from gsplat.rendering import rasterization
from .gaussiandata import GaussianData, GaussianBatchData

@torch.compile
def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)

@torch.compile
def transform_points(points: Tensor, pos: Tensor, quat: Tensor) -> Tensor:
    # points: (N, 3)
    # pos: (N, 3)
    # quat: (N, 4) wxyz
    
    # Rotate points
    # v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    q_w = quat[..., 0]
    q_xyz = quat[..., 1:]
    
    t = 2.0 * torch.cross(q_xyz, points)
    points_rotated = points + q_w.unsqueeze(-1) * t + torch.cross(q_xyz, t)
    
    return points_rotated + pos

@torch.no_grad()
def batch_render(
    gaussians: GaussianData,
    cam_pos: np.ndarray, # (Ncam, 3)
    cam_xmat: np.ndarray, # (Ncam, 9)
    height: int,
    width: int,
    fovy: np.ndarray, # (Ncam,) degree
    bg_imgs: Optional[torch.Tensor] = None, # (Ncam, H, W, 3)
) -> Tuple[Tensor, Tensor]:
    
    device = gaussians.device
    
    # 1. Prepare Gaussians    
    if gaussians.sh.dim() == 2:
        gaussians.sh = gaussians.sh.reshape(gaussians.sh.shape[0], -1, 3).contiguous()
    
    sh_degree = int(np.round(np.sqrt(gaussians.sh.shape[1]))) - 1

    # 2. Prepare Cameras
    Ncam = cam_pos.shape[0]
    
    # Convert camera data to torch
    cam_pos_t = torch.tensor(cam_pos, dtype=torch.float32, device=device) # (N, 3)
    cam_xmat_t = torch.tensor(cam_xmat, dtype=torch.float32, device=device).reshape(Ncam, 3, 3) # (N, 3, 3)
    fovy_t = torch.tensor(np.radians(fovy), dtype=torch.float32, device=device) # (N,)
    
    # Compute Intrinsics (K)
    # tan(fovy/2) = H / (2*fy) => fy = H / (2 * tan(fovy/2))
    # Assume square pixels: fx = fy
    # cx = W/2, cy = H/2
    
    tan_half_fovy = torch.tan(fovy_t / 2.0)
    focal_y = height / (2.0 * tan_half_fovy)
    focal_x = focal_y # Square pixels assumption
    
    cx = width / 2.0
    cy = height / 2.0
    
    Ks = torch.zeros((Ncam, 3, 3), dtype=torch.float32, device=device)
    Ks[:, 0, 0] = focal_x
    Ks[:, 1, 1] = focal_y
    Ks[:, 0, 2] = cx
    Ks[:, 1, 2] = cy
    Ks[:, 2, 2] = 1.0
    
    # Compute Extrinsics (View Matrix)
    # Tmat construction similar to renderer_cuda.py
    # Tmat = [R | t]
    #        [0 | 1]
    
    Tmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(Ncam, 1, 1)
    Tmats[:, :3, :3] = cam_xmat_t
    Tmats[:, :3, 3] = cam_pos_t
    
    # Flip Y and Z columns of rotation (MuJoCo to OpenGL convention)
    Tmats[:, 0:3, 1] *= -1
    Tmats[:, 0:3, 2] *= -1
    
    # View Matrix = Inverse of World Matrix
    viewmats = torch.inverse(Tmats)
    
    # 3. Rasterization
    renders, alphas, meta = rasterization(
        means=gaussians.xyz,         # [G, 3]
        quats=gaussians.rot,         # [G, 4]
        scales=gaussians.scale,      # [G, 3]
        opacities=gaussians.opacity, # [G, 1]
        colors=gaussians.sh,         # [G, SH_coeffs]
        viewmats=viewmats,           # [Ncam, 4, 4]
        Ks=Ks,                       # [Ncam, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+D",
        packed=False,
    )
    
    # renders: (Ncam, H, W, 4) -> RGBD
    
    color_img = renders[..., :3]
    depth_img = renders[..., 3:4]

    if bg_imgs is not None:
        if bg_imgs.shape != (Ncam, height, width, 3):
            raise ValueError(f"bg_imgs shape mismatch. Expected {(Ncam, height, width, 3)}, got {bg_imgs.shape}")
        
        if bg_imgs.device != device:
            bg_imgs = bg_imgs.to(device)
            
        color_img.addcmul_(bg_imgs, 1.0 - alphas)
    
    return color_img, depth_img

@torch.no_grad()
def batch_env_render(
    gaussians: GaussianBatchData,
    cam_pos: torch.Tensor, # (Nenv, Ncam, 3)
    cam_xmat: torch.Tensor, # (Nenv, Ncam, 9)
    height: int,
    width: int,
    fovy: np.ndarray, # (Nenv, Ncam) degree
    bg_imgs: Optional[torch.Tensor] = None, # (Nenv, Ncam, H, W, 3)
    minibatch: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    
    device = gaussians.device
    Nenv = cam_pos.shape[0]
    Ncam = cam_pos.shape[1]

    if minibatch is not None and minibatch > 0 and minibatch < Nenv:
        out_color = torch.empty((Nenv, Ncam, height, width, 3), dtype=torch.float32, device=device)
        out_depth = torch.empty((Nenv, Ncam, height, width, 1), dtype=torch.float32, device=device)
        
        for i in range(0, Nenv, minibatch):
            end = min(i + minibatch, Nenv)
            
            g_slice = GaussianBatchData(
                xyz=gaussians.xyz[i:end],
                rot=gaussians.rot[i:end],
                scale=gaussians.scale[i:end],
                opacity=gaussians.opacity[i:end],
                sh=gaussians.sh[i:end]
            )
            
            bg_slice = bg_imgs[i:end] if bg_imgs is not None else None
            
            c, d = batch_env_render(
                g_slice, 
                cam_pos[i:end], 
                cam_xmat[i:end], 
                height, 
                width, 
                fovy[i:end] if len(fovy) == Nenv else fovy, 
                bg_imgs=bg_slice, 
                minibatch=None
            )
            out_color[i:end] = c
            out_depth[i:end] = d
        return out_color, out_depth
    
    # 1. Prepare Gaussians
    # gaussians.xyz is (Nenv, N, 3)
    
    if gaussians.sh.dim() == 3: # (Nenv, N, D)
        gaussians.sh = gaussians.sh.reshape(gaussians.sh.shape[0], gaussians.sh.shape[1], -1, 3).contiguous()
    
    sh_degree = int(np.round(np.sqrt(gaussians.sh.shape[2]))) - 1

    # 2. Prepare Cameras
    Nenv = cam_pos.shape[0]
    Ncam = cam_pos.shape[1]
    
    # Convert camera data to torch
    fovy_t = torch.tensor(np.radians(fovy), dtype=torch.float32, device=device) # (Nenv, Ncam)
    
    # Compute Intrinsics (K)
    tan_half_fovy = torch.tan(fovy_t / 2.0)
    focal_y = height / (2.0 * tan_half_fovy)
    focal_x = focal_y # Square pixels assumption
    
    cx = width / 2.0
    cy = height / 2.0
    
    Ks = torch.zeros((Nenv, Ncam, 3, 3), dtype=torch.float32, device=device)
    Ks[..., 0, 0] = focal_x
    Ks[..., 1, 1] = focal_y
    Ks[..., 0, 2] = cx
    Ks[..., 1, 2] = cy
    Ks[..., 2, 2] = 1.0
    
    # Compute Extrinsics (View Matrix)
    Tmats = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(Nenv, Ncam, 1, 1)
    Tmats[..., :3, :3] = cam_xmat.reshape(Nenv, Ncam, 3, 3)
    Tmats[..., :3, 3] = cam_pos
    
    # Flip Y and Z columns of rotation (MuJoCo to OpenGL convention)
    Tmats[..., 0:3, 1] *= -1
    Tmats[..., 0:3, 2] *= -1
    
    # View Matrix = Inverse of World Matrix
    viewmats = torch.inverse(Tmats)
    
    # 3. Rasterization
    renders, alphas, meta = rasterization(
        means=gaussians.xyz,         # [Nenv, G, 3]
        quats=gaussians.rot,         # [Nenv, G, 4]
        scales=gaussians.scale,      # [Nenv, G, 3]
        opacities=gaussians.opacity, # [Nenv, G, 1]
        colors=gaussians.sh,         # [Nenv, G, SH_coeffs]
        viewmats=viewmats,           # [Nenv, Ncam, 4, 4]
        Ks=Ks,                       # [Nenv, Ncam, 3, 3]
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode="RGB+D",
        packed=False,
    )
    
    # renders: (Nenv, Ncam, H, W, 4) -> RGBD
    
    color_img = renders[..., :3]
    depth_img = renders[..., 3:4]

    if bg_imgs is not None:
        if bg_imgs.shape != (Nenv, Ncam, height, width, 3):
            raise ValueError(f"bg_imgs shape mismatch. Expected {(Nenv, Ncam, height, width, 3)}, got {bg_imgs.shape}")
        
        if bg_imgs.device != device:
            bg_imgs = bg_imgs.to(device)
            
        color_img.addcmul_(bg_imgs, 1.0 - alphas)
    
    return color_img, depth_img

@torch.no_grad()
def batch_update_gaussians(
    gaussian_template: GaussianData,
    body_pos: torch.Tensor, # (Nenv, Nbody, 3)
    body_quat: torch.Tensor, # (Nenv, Nbody, 4)
    point_to_body_idx: Optional[torch.Tensor], # (N_points,)
    dynamic_mask: Optional[torch.Tensor], # (N_points,)
) -> GaussianBatchData:
    """
    Batch update gaussian positions and rotations based on body poses.
    """
    device = body_pos.device
    Nenv = body_pos.shape[0]
    
    # 1. Convert template to tensor if needed (cache this in practice!)
    if isinstance(gaussian_template.xyz, np.ndarray):
        tmpl_xyz = torch.tensor(gaussian_template.xyz, dtype=torch.float32, device=device)
        tmpl_rot = torch.tensor(gaussian_template.rot, dtype=torch.float32, device=device)
        tmpl_scale = torch.tensor(gaussian_template.scale, dtype=torch.float32, device=device)
        tmpl_opacity = torch.tensor(gaussian_template.opacity, dtype=torch.float32, device=device)
        tmpl_sh = torch.tensor(gaussian_template.sh, dtype=torch.float32, device=device)
    else:
        tmpl_xyz = gaussian_template.xyz
        tmpl_rot = gaussian_template.rot
        tmpl_scale = gaussian_template.scale
        tmpl_opacity = gaussian_template.opacity
        tmpl_sh = gaussian_template.sh

    # 2. Vectorized Update
    # Prepare output
    xyz_out = tmpl_xyz.unsqueeze(0).expand(Nenv, -1, -1).clone()
    rot_out = tmpl_rot.unsqueeze(0).expand(Nenv, -1, -1).clone()
    
    mask = dynamic_mask
    if mask.any():
        # (N_dynamic,)
        body_indices = point_to_body_idx[mask]
        
        # Gather body poses: (Nenv, N_dynamic, 3/4)
        # body_pos is (Nenv, Nbody, 3)
        # We want to select body_indices for each env.
        # body_pos[:, body_indices] works? 
        # Yes, standard indexing: (Nenv, N_dynamic, 3)
        pos_expanded = body_pos[:, body_indices]
        quat_expanded = body_quat[:, body_indices]
        
        # Template properties: (N_dynamic, 3/4) -> (1, N_dynamic, 3/4)
        xyz_ori = tmpl_xyz[mask].unsqueeze(0)
        rot_ori = tmpl_rot[mask].unsqueeze(0)
        
        # Apply transform
        # transform_points broadcasts: (1, N_dyn, 3) + (Nenv, N_dyn, 3) -> (Nenv, N_dyn, 3)
        xyz_new = transform_points(xyz_ori, pos_expanded, quat_expanded)
        rot_new = quaternion_multiply(quat_expanded, rot_ori)
        
        # Scatter back
        # xyz_out is (Nenv, N_total, 3)
        # We need to assign to xyz_out[:, mask, :]
        # This works in PyTorch
        xyz_out[:, mask, :] = xyz_new
        rot_out[:, mask, :] = rot_new

    # 3. Expand static properties
    scale_out = tmpl_scale.unsqueeze(0).expand(Nenv, -1, -1)
    opacity_out = tmpl_opacity.unsqueeze(0).expand(Nenv, -1)
    sh_out = tmpl_sh.unsqueeze(0).expand(Nenv, -1, -1, -1)
    
    return GaussianBatchData(
        xyz=xyz_out,
        rot=rot_out,
        scale=scale_out,
        opacity=opacity_out,
        sh=sh_out
    )