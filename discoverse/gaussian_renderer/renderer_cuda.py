'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''

from discoverse.gaussian_renderer import util_gau
import numpy as np
import torch
from dataclasses import dataclass

try:
    from gsplat.rendering import rasterization
    GSPLAT_AVAILABLE = True
except ImportError:
    GSPLAT_AVAILABLE = False
    print("Warning: gsplat not available")

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]

def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

class CUDARenderer:
    def __init__(self):
        """
        Initialize CUDA Renderer
        """
        if not GSPLAT_AVAILABLE:
            raise RuntimeError("gsplat backend requested but not available. Please install gsplat.")
        
        self.gaussians = None
        self.gau_env_idx = 0
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
            self.gau_env_idx = gau_xyz[0].shape[0]
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
            
        for i in range(len(start_indices)):
            start = start_indices[i]
            end = end_indices[i]
            
            xyz_ori = self.gau_ori_xyz_all_cu[start:end]
            rot_ori = self.gau_ori_rot_all_cu[start:end]
            
            cur_pos = pos[i]
            cur_quat = quat[i]
            
            cur_quat_expanded = cur_quat.unsqueeze(0).expand(xyz_ori.shape[0], -1)
            
            xyz_new = util_gau.multiple_quaternion_vector3d(cur_quat_expanded, xyz_ori) + cur_pos
            rot_new = util_gau.multiple_quaternions(cur_quat_expanded, rot_ori)
            
            self.gau_xyz_all_cu[start:end] = xyz_new
            self.gau_rot_all_cu[start:end] = rot_new
            
            self.gaussians.xyz[start:end] = xyz_new
            self.gaussians.rot[start:end] = rot_new

