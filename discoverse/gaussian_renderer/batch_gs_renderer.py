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

@torch.jit.script
def _update_gaussians_kernel(
    tmpl_xyz: Tensor,
    tmpl_rot: Tensor,
    body_pos: Tensor,
    body_quat: Tensor,
    gs_idx_start: Tensor,
    gs_idx_end: Tensor,
    gs_body_ids: Tensor,
    Nenv: int,
) -> Tuple[Tensor, Tensor]:
    
    # Prepare output tensors
    # We use clone() to ensure we have new memory that we can modify in-place
    xyz_out = tmpl_xyz.unsqueeze(0).expand(Nenv, -1, -1).clone()
    rot_out = tmpl_rot.unsqueeze(0).expand(Nenv, -1, -1).clone()

    # Iterate over bodies
    # JIT will optimize this loop
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
        # Inlined multiple_quaternion_vector3d logic for JIT compatibility if needed, 
        # but calling the function is also fine if it's JIT-compatible.
        # Here we inline to be safe and self-contained within the JIT kernel.
        
        # multiple_quaternion_vector3d(b_quat, g_xyz)
        qw, qx, qy, qz = b_quat[..., 0], b_quat[..., 1], b_quat[..., 2], b_quat[..., 3]
        vx, vy, vz = g_xyz[..., 0], g_xyz[..., 1], g_xyz[..., 2]
        
        qvw = -vx*qx - vy*qy - vz*qz
        qvx =  vx*qw - vy*qz + vz*qy
        qvy =  vx*qz + vy*qw - vz*qx
        qvz = -vx*qy + vy*qx + vz*qw
        
        vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
        vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
        vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
        
        xyz_new = torch.stack([vx_, vy_, vz_], dim=-1) + b_pos
        
        # multiple_quaternions(b_quat, g_rot)
        q1w, q1x, q1y, q1z = b_quat[..., 0], b_quat[..., 1], b_quat[..., 2], b_quat[..., 3]
        q2w, q2x, q2y, q2z = g_rot[..., 0], g_rot[..., 1], g_rot[..., 2], g_rot[..., 3]

        qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
        qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
        qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
        qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w
        
        rot_new = torch.stack([qw_, qx_, qy_, qz_], dim=-1)
        
        xyz_out[:, start:end, :] = xyz_new
        rot_out[:, start:end, :] = rot_new
        
    return xyz_out, rot_out

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

    # Ensure indices are tensors
    if not isinstance(gs_idx_start, torch.Tensor):
        gs_idx_start = torch.tensor(gs_idx_start, device=device, dtype=torch.long)
    else:
        gs_idx_start = gs_idx_start.to(device=device, dtype=torch.long)
        
    if not isinstance(gs_idx_end, torch.Tensor):
        gs_idx_end = torch.tensor(gs_idx_end, device=device, dtype=torch.long)
    else:
        gs_idx_end = gs_idx_end.to(device=device, dtype=torch.long)
        
    if not isinstance(gs_body_ids, torch.Tensor):
        gs_body_ids = torch.tensor(gs_body_ids, device=device, dtype=torch.long)
    else:
        gs_body_ids = gs_body_ids.to(device=device, dtype=torch.long)

    # 2. Run JIT kernel
    xyz_out, rot_out = _update_gaussians_kernel(
        tmpl_xyz, tmpl_rot, body_pos, body_quat, 
        gs_idx_start, gs_idx_end, gs_body_ids, 
        Nenv
    )

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