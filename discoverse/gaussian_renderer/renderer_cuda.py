'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from discoverse.gaussian_renderer import util_gau, GaussianData

import numpy as np
import torch
from torch import Tensor
from typing import Tuple

try:
    from gsplat.rendering import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available")

def gaus_cuda_from_cpu(gau: util_gau) -> GaussianData:
    gaus =  GaussianData(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

@torch.jit.script
def _update_gaussian_properties_kernel(
    gau_ori_xyz: Tensor,
    gau_ori_rot: Tensor,
    pos: Tensor,
    quat: Tensor,
    start_indices: Tensor,
    end_indices: Tensor,
    gau_xyz_out: Tensor,
    gau_rot_out: Tensor
) -> Tuple[Tensor, Tensor]:
    
    for i in range(len(start_indices)):
        start = int(start_indices[i].item())
        end = int(end_indices[i].item())
        
        # Slicing tensors
        xyz_ori = gau_ori_xyz[start:end]
        rot_ori = gau_ori_rot[start:end]
        
        cur_pos = pos[i]
        cur_quat = quat[i]
        
        # Inline quaternion math
        qw, qx, qy, qz = cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3]
        vx, vy, vz = xyz_ori[..., 0], xyz_ori[..., 1], xyz_ori[..., 2]
        
        qvw = -vx*qx - vy*qy - vz*qz
        qvx =  vx*qw - vy*qz + vz*qy
        qvy =  vx*qz + vy*qw - vz*qx
        qvz = -vx*qy + vy*qx + vz*qw
        
        vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
        vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
        vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
        
        xyz_new = torch.stack([vx_, vy_, vz_], dim=-1) + cur_pos
        
        q1w, q1x, q1y, q1z = cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3]
        q2w, q2x, q2y, q2z = rot_ori[..., 0], rot_ori[..., 1], rot_ori[..., 2], rot_ori[..., 3]

        qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
        qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
        qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
        qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w
        
        rot_new = torch.stack([qw_, qx_, qy_, qz_], dim=-1)
        
        gau_xyz_out[start:end] = xyz_new
        gau_rot_out[start:end] = rot_new
        
    return gau_xyz_out, gau_rot_out

class CUDARenderer:
    def __init__(self):
        """
        Initialize CUDA Renderer
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

    @torch.no_grad()
    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        if type(gaus) is dict:
            gau_xyz = []
            gau_rot = []
            gau_s = []
            gau_a = []
            gau_c = []
            for gaus_item in gaus.values():
                gau_xyz.append(gaus_item.xyz)
                gau_rot.append(gaus_item.rot)
                gau_s.append(gaus_item.scale)
                gau_a.append(gaus_item.opacity)
                gau_c.append(gaus_item.sh)
            gau_xyz = np.concatenate(gau_xyz, axis=0)
            gau_rot = np.concatenate(gau_rot, axis=0)
            gau_s = np.concatenate(gau_s, axis=0)
            gau_a = np.concatenate(gau_a, axis=0)
            gau_c = np.concatenate(gau_c, axis=0)
            gaus_all = util_gau.GaussianData(gau_xyz, gau_rot, gau_s, gau_a, gau_c)
            self.gaussians = gaus_cuda_from_cpu(gaus_all)
        else:
            self.gaussians = gaus_cuda_from_cpu(gaus)

        num_points = self.gaussians.xyz.shape[0]

        self.gau_ori_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_ori_xyz_all_cu[..., :] = torch.from_numpy(gau_xyz).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)
        self.gau_ori_rot_all_cu[..., :] = torch.from_numpy(gau_rot).cuda().requires_grad_(False)

        self.gau_xyz_all_cu = torch.zeros(num_points, 3).cuda().requires_grad_(False)
        self.gau_rot_all_cu = torch.zeros(num_points, 4).cuda().requires_grad_(False)

    def update_gaussian_properties(self, start_indices, end_indices, pos, quat):
        """
        Batch update gaussian properties for multiple objects.
        
        Args:
            start_indices: (N_objects,) array of start indices
            end_indices: (N_objects,) array of end indices
            pos: (N_objects, 3) array of positions
            quat: (N_objects, 4) array of quaternions (wxyz)
        """
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos).float().cuda()
        if not isinstance(quat, torch.Tensor):
            quat = torch.from_numpy(quat).float().cuda()
            
        # Convert indices to tensor if they are not
        if not isinstance(start_indices, torch.Tensor):
            start_indices = torch.tensor(start_indices, device=pos.device, dtype=torch.long)
        if not isinstance(end_indices, torch.Tensor):
            end_indices = torch.tensor(end_indices, device=pos.device, dtype=torch.long)

        _update_gaussian_properties_kernel(
            self.gau_ori_xyz_all_cu,
            self.gau_ori_rot_all_cu,
            pos,
            quat,
            start_indices,
            end_indices,
            self.gaussians.xyz,
            self.gaussians.rot
        )
        
        self.gau_xyz_all_cu = self.gaussians.xyz
        self.gau_rot_all_cu = self.gaussians.rot

