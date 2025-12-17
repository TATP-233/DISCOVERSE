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
