import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class GaussianData:
    def __init__(self, xyz, rot, scale, opacity, sh):
        self.xyz = xyz
        self.rot = rot
        self.scale = scale
        self.opacity = opacity
        self.sh = sh

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

    @property
    def device(self):
        return self.xyz.device

@dataclass
class GaussianBatchData:
    xyz: torch.Tensor      # (B, N, 3)
    rot: torch.Tensor      # (B, N, 4)
    scale: torch.Tensor    # (B, N, 3)
    opacity: torch.Tensor  # (B, N, 1)
    sh: torch.Tensor       # (B, N, K, 3) or (B, N, D)

    def __len__(self):
        return self.xyz.shape[1]
    
    @property
    def batch_size(self):
        return self.xyz.shape[0]

    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

    @property
    def device(self):
        return self.xyz.device
