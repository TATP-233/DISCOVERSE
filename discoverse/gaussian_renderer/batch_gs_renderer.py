from typing import Tuple, List, Union, Dict

import numpy as np
import torch
from torch import Tensor

from gsplat.rendering import rasterization
from discoverse.gaussian_renderer.util_gau import GaussianData

@torch.no_grad()
def batch_render(
    gaussians: GaussianData,
    cam_pos: np.ndarray, # (Ncam, 3)
    cam_xmat: np.ndarray, # (Ncam, 9)
    height: int,
    width: int,
    fovy: np.ndarray, # (Ncam,) degree
    # minibatch: int = 15,
) -> Tuple[Tensor, Tensor]:
    
    device = "cuda"
    
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
    
    return color_img, depth_img