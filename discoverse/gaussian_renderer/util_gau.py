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
from pathlib import Path
from plyfile import PlyData, PlyElement
from .gaussiandata import GaussianData
from .super_splat_loader import is_super_splat_format, load_super_splat_ply

def load_ply_3dgs(plydata):
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])

    features_dc = np.zeros((xyz.shape[0], 3))
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

    shs_num = (max_sh_degree + 1) ** 2 - 1
    # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = features_extra.reshape((features_extra.shape[0], 3, len(extra_f_names)//3))
    features_extra = features_extra[:, :, :shs_num]
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)

    # if 2dgs
    if len(scale_names) == 2:
        print(f"len(scale_names) = {len(scale_names)} (2dgs ply model)")
        scales = np.hstack([scales, 1e-9 * np.ones_like(scales[:, :1])])

    scales = scales.astype(np.float32)
    opacities = 1. / (1. + np.exp(-opacities))
    opacities = opacities.astype(np.float32)

    shs = np.concatenate([
        features_dc.reshape(-1, 3), 
        features_extra.reshape(features_dc.shape[0], shs_num * 3)
    ], axis=-1).astype(np.float32)

    return GaussianData(xyz, rots, scales, opacities, shs)

def load_ply(path):
    """
    加载PLY格式的3DGS模型
    
    Args:
        path: 模型文件路径（本地路径或相对路径）
    
    Returns:
        GaussianData: 加载的高斯数据
    """
    plydata = PlyData.read(path)

    # 检查是否为 SuperSplat 格式
    if is_super_splat_format(plydata):
        return load_super_splat_ply(plydata)
    
    # 标准 3DGS 格式
    else:
        return load_ply_3dgs(plydata)

def save_ply(gaussian_data: GaussianData, path):
    """
    保存GaussianData为PLY文件
    
    Args:
        gaussian_data: 要保存的高斯数据
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    xyz = to_numpy(gaussian_data.xyz)
    normals = np.zeros_like(xyz)
    
    if len(gaussian_data.sh.shape) > 2:
        gaussian_data.sh = gaussian_data.sh.reshape(gaussian_data.sh.shape[0], -1)
    shs = to_numpy(gaussian_data.sh)
    f_dc = shs[:, :3]
    f_rest = shs[:, 3:]
    
    opacities = to_numpy(gaussian_data.opacity)
    # inverse sigmoid: ln(x / (1-x))
    # clip to avoid nan/inf
    opacities = np.clip(opacities, 1e-6, 1.0 - 1e-6)
    opacities = np.log(opacities / (1. - opacities))
    
    scales = to_numpy(gaussian_data.scale)
    scales = np.log(np.maximum(scales, 1e-8))
    
    rots = to_numpy(gaussian_data.rot)
    
    # Construct dtype
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                  ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    
    # f_rest handling
    # load_ply: 
    #   features_extra (N, 3*k) -> reshape(N, 3, k) -> transpose(0, 2, 1) -> (N, k, 3) -> reshape(N, 3*k) -> shs[:, 3:]
    # save_ply reverse:
    #   f_rest (N, 3*k) [k0c0, k0c1, k0c2, k1c0...] -> reshape(N, k, 3) -> transpose(0, 2, 1) -> (N, 3, k) -> reshape(N, 3*k)
    num_extra = f_rest.shape[1]
    if num_extra > 0:
        k = num_extra // 3
        f_rest = f_rest.reshape(-1, k, 3).transpose(0, 2, 1).reshape(-1, num_extra)
    
    for i in range(num_extra):
        dtype_full.append((f'f_rest_{i}', 'f4'))
        
    dtype_full.append(('opacity', 'f4'))
    
    for i in range(scales.shape[1]):
        dtype_full.append((f'scale_{i}', 'f4'))
        
    for i in range(rots.shape[1]):
        dtype_full.append((f'rot_{i}', 'f4'))
        
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['nx'] = normals[:, 0]
    elements['ny'] = normals[:, 1]
    elements['nz'] = normals[:, 2]
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    for i in range(num_extra):
        elements[f'f_rest_{i}'] = f_rest[:, i]
        
    elements['opacity'] = opacities.flatten()
    
    for i in range(scales.shape[1]):
        elements[f'scale_{i}'] = scales[:, i]
        
    for i in range(rots.shape[1]):
        elements[f'rot_{i}'] = rots[:, i]
        
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def transform_shs(shs_feat: torch.Tensor, rotation_matrix: np.ndarray) -> torch.Tensor:
    """
    Rotate spherical harmonics features.
    
    Args:
        shs_feat (torch.Tensor): (N, M, 3) tensor, where M is the number of SH coefficients excluding DC.
                                 M should be 3 (degree 1), 8 (degree 2), or 15 (degree 3).
                                 The last dimension is RGB.
        rotation_matrix (np.ndarray): (3, 3) rotation matrix.
        
    Returns:
        torch.Tensor: Rotated shs_feat with shape (N, M, 3).
    """

    import einops
    from e3nn import o3

    assert len(shs_feat.shape) == 3, f"shs_feat shape should be (N, M, 3), got {shs_feat.shape}"
    assert shs_feat.shape[2] == 3, f"shs_feat last dimension should be 3 (RGB), got {shs_feat.shape[2]}"
    assert shs_feat.shape[1] in [3, 8, 15], f"shs_feat second dimension should be 3, 8, or 15 (excluding DC), got {shs_feat.shape[1]}"

    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]) # switch axes: yzx -> xyz
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(torch.from_numpy(permuted_rotation_matrix).float())
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einops.einsum(
            D_1,
            one_degree_shs,
            "... i j, ... j -> ... i",
        )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    if shs_feat.shape[1] < 8:
        return shs_feat    
    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einops.einsum(
            D_2,
            two_degree_shs,
            "... i j, ... j -> ... i",
        )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs
    if shs_feat.shape[1] < 15:
        return shs_feat

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einops.einsum(
            D_3,
            three_degree_shs,
            "... i j, ... j -> ... i",
        )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat