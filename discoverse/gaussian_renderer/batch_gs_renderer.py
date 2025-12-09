from typing import Tuple, List, Union, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from gsplat.rendering import rasterization
from discoverse.gaussian_renderer.gaussiandata import GaussianData, GaussianBatchData
from discoverse.gaussian_renderer.util_gau import multiple_quaternion_vector3d, multiple_quaternions

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
                fovy[i:end], 
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
    gs_idx_start: Union[List, np.ndarray, torch.Tensor], # (N_gs_body,)
    gs_idx_end: Union[List, np.ndarray, torch.Tensor], # (N_gs_body,)
    gs_body_ids: Union[List, np.ndarray, torch.Tensor], # (N_gs_body,)
) -> GaussianBatchData:
    """
    Batch update gaussian positions and rotations based on body poses.
    
    Args:
        gaussian_template: The template gaussian data (usually in local coordinates).
        body_pos: Body positions (Nenv, Nbody, 3).
        body_quat: Body quaternions (Nenv, Nbody, 4) in wxyz format.
        gs_idx_start: Start index of gaussians for each body in the template.
        gs_idx_end: End index of gaussians for each body in the template.
        gs_body_ids: Body indices corresponding to the gaussian groups.
        
    Returns:
        GaussianBatchData: The updated gaussian data for all environments.
    """
    device = body_pos.device
    Nenv = body_pos.shape[0]
    total_gaussians = len(gaussian_template)

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

    # 2. Prepare output tensors
    xyz_out = torch.zeros((Nenv, total_gaussians, 3), dtype=torch.float32, device=device)
    rot_out = torch.zeros((Nenv, total_gaussians, 4), dtype=torch.float32, device=device)
    xyz_out[...] = tmpl_xyz.unsqueeze(0).expand(Nenv, -1, -1)
    rot_out[...] = tmpl_rot.unsqueeze(0).expand(Nenv, -1, -1)

    # 3. Update each body's gaussians
    # We iterate over the bodies (objects), which is usually a small number.
    for i in range(len(gs_body_ids)):
        body_idx = gs_body_ids[i]
        start = gs_idx_start[i]
        end = gs_idx_end[i]
        
        # (Nenv, 1, 3)
        b_pos = body_pos[:, body_idx, :].unsqueeze(1)
        # (Nenv, 1, 4)
        b_quat = body_quat[:, body_idx, :].unsqueeze(1)
        
        # (1, N_g, 3)
        g_xyz = tmpl_xyz[start:end].unsqueeze(0)
        # (1, N_g, 4)
        g_rot = tmpl_rot[start:end].unsqueeze(0)
        
        # Apply transform: R * p + t
        # multiple_quaternion_vector3d broadcasts (Nenv, 1, 4) and (1, N_g, 3) -> (Nenv, N_g, 3)
        xyz_new = multiple_quaternion_vector3d(b_quat, g_xyz) + b_pos
        
        # Apply rotation: q_body * q_gaussian
        rot_new = multiple_quaternions(b_quat, g_rot)
        
        xyz_out[:, start:end, :] = xyz_new
        rot_out[:, start:end, :] = rot_new

    # 4. Expand static properties
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