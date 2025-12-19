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

import struct
import torch
import numpy as np
from plyfile import PlyData
from typing import Dict, Tuple
from .gaussiandata import GaussianData

# 球谐常数
SH_C0 = 0.28209479177387814


def load_super_splat_ply(plydata: PlyData) -> GaussianData:
    """
    加载 SuperSplat 格式的 PLY 文件
    
    SuperSplat 格式使用压缩的方式存储高斯溅射数据:
    - 每256个顶点对应一个chunk
    - 位置使用 11/10/11 位编码
    - 尺度使用 11/10/11 位编码
    - 颜色和不透明度使用 8/8/8/8 位编码
    - 旋转使用最大分量索引 + 3×10bit 编码
    
    Args:
        plydata: 从 PLY 文件读取的数据
        
    Returns:
        GaussianData: 解码后的高斯数据
    """
    vtx = plydata['vertex'].data  # structured array
    chk = plydata['chunk'].data   # structured array

    # 每256个vertex对应一个chunk(按顺序)
    num_vertex = vtx.shape[0]
    if num_vertex == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return GaussianData(
            empty, 
            np.zeros((0, 4), dtype=np.float32), 
            empty.copy(), 
            np.zeros((0,), dtype=np.float32),
            empty.copy()
        )

    chunk_idx = (np.arange(num_vertex) // 256).astype(np.int64)
    # 防御性裁剪(以防最后一个 chunk 未满或越界情况)
    chunk_idx = np.clip(chunk_idx, 0, chk.shape[0] - 1)

    # 拉取每个点对应 chunk 的标量边界
    def gather_chunk(field):
        return chk[field][chunk_idx]

    # 解码位置(11/10/11)
    positions = _decode_positions(vtx, gather_chunk)
    
    # 解码尺度(11/10/11), 并指数还原
    scales = _decode_scales(vtx, gather_chunk)
    
    # 解码颜色和不透明度(8/8/8/8)
    colors, opacities = _decode_colors_and_opacities(vtx, gather_chunk)
    
    # 解码旋转(最大分量索引 + 3×10bit)
    quats = _decode_rotations(vtx, num_vertex)

    # 处理高阶球谐系数 (SH > 0)
    if 'sh' in plydata:
        sh_elem = plydata['sh'].data
        # 获取所有 f_rest_* 属性名
        prop_names = [p.name for p in plydata['sh'].properties]
        f_rest_names = [n for n in prop_names if n.startswith('f_rest_')]
        
        if f_rest_names:
            # 提取并堆叠数据 (N, num_rest)
            f_rest_uint8 = np.stack([sh_elem[n] for n in f_rest_names], axis=1).astype(np.float32)
            
            # 反量化: value = (uint8 / 256 - 0.5) * 8
            f_rest = (f_rest_uint8 / 256.0 - 0.5) * 8.0
            
            # SuperSplat stores SH as planar (R..., G..., B...), but GaussianData expects interleaved (RGB, RGB...)
            # f_rest is (N, 3 * k) -> (N, 3, k) -> (N, k, 3) -> (N, 3 * k)
            f_rest = f_rest.reshape(f_rest.shape[0], 3, -1)
            f_rest = f_rest.transpose(0, 2, 1)
            f_rest = f_rest.reshape(f_rest.shape[0], -1)
            
            # 合并 DC 和 Rest
            colors = np.concatenate([colors, f_rest], axis=1)

    return GaussianData(positions, quats, scales, opacities, colors)


def _decode_positions(vtx, gather_chunk):
    """解码位置数据 (11/10/11 位编码)"""
    ppos = vtx['packed_position'].astype(np.uint32)
    xbits = (ppos >> 21) & 0x7FF
    ybits = (ppos >> 11) & 0x3FF
    zbits = ppos & 0x7FF
    
    fx = xbits.astype(np.float32) / 2047.0 * (gather_chunk('max_x') - gather_chunk('min_x')) + gather_chunk('min_x')
    fy = ybits.astype(np.float32) / 1023.0 * (gather_chunk('max_y') - gather_chunk('min_y')) + gather_chunk('min_y')
    fz = zbits.astype(np.float32) / 2047.0 * (gather_chunk('max_z') - gather_chunk('min_z')) + gather_chunk('min_z')
    
    return np.stack([fx, fy, fz], axis=1).astype(np.float32)


def _decode_scales(vtx, gather_chunk):
    """解码尺度数据 (11/10/11 位编码), 并指数还原"""
    pscale = vtx['packed_scale'].astype(np.uint32)
    sxb = (pscale >> 21) & 0x7FF
    syb = (pscale >> 11) & 0x3FF
    szb = pscale & 0x7FF
    
    sx = sxb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_x') - gather_chunk('min_scale_x')) + gather_chunk('min_scale_x')
    sy = syb.astype(np.float32) / 1023.0 * (gather_chunk('max_scale_y') - gather_chunk('min_scale_y')) + gather_chunk('min_scale_y')
    sz = szb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_z') - gather_chunk('min_scale_z')) + gather_chunk('min_scale_z')
    
    return np.exp(np.stack([sx, sy, sz], axis=1)).astype(np.float32)


def _decode_colors_and_opacities(vtx, gather_chunk):
    """解码颜色和不透明度 (8/8/8/8 位编码)"""
    pcol = vtx['packed_color'].astype(np.uint32)
    r8 = (pcol >> 24) & 0xFF
    g8 = (pcol >> 16) & 0xFF
    b8 = (pcol >> 8) & 0xFF
    a8 = pcol & 0xFF
    
    fr = r8.astype(np.float32) / 255.0 * (gather_chunk('max_r') - gather_chunk('min_r')) + gather_chunk('min_r')
    fg = g8.astype(np.float32) / 255.0 * (gather_chunk('max_g') - gather_chunk('min_g')) + gather_chunk('min_g')
    fb = b8.astype(np.float32) / 255.0 * (gather_chunk('max_b') - gather_chunk('min_b')) + gather_chunk('min_b')
    
    fr = (fr - 0.5) / SH_C0
    fg = (fg - 0.5) / SH_C0
    fb = (fb - 0.5) / SH_C0
    
    opacities = a8.astype(np.float32) / 255.0
    
    colors = np.stack([fr, fg, fb], axis=1).astype(np.float32)
    
    return colors, opacities


def _decode_rotations(vtx, num_vertex):
    """解码旋转四元数 (最大分量索引 + 3×10bit)"""
    prot = vtx['packed_rotation'].astype(np.uint32)
    largest = (prot >> 30) & 0x3  # 0..3
    v0 = (prot >> 20) & 0x3FF
    v1 = (prot >> 10) & 0x3FF
    v2 = prot & 0x3FF
    
    norm = np.sqrt(2.0) * 0.5
    vals = np.stack([v0, v1, v2], axis=1).astype(np.float32)
    vals = (vals / 1023.0 - 0.5) / norm
    
    # 映射到四元数的非最大分量(顺序依 index 增序, 略过 largest)
    q = np.zeros((num_vertex, 4), dtype=np.float32)

    # Masks for largest index
    m0 = (largest == 0)
    m1 = (largest == 1)
    m2 = (largest == 2)
    m3 = (largest == 3)

    # 对应关系见说明:
    # largest=0: (1,2,3) <= (v0,v1,v2)
    q[m0, 1] = vals[m0, 0]
    q[m0, 2] = vals[m0, 1]
    q[m0, 3] = vals[m0, 2]
    # largest=1: (0,2,3) <= (v0,v1,v2)
    q[m1, 0] = vals[m1, 0]
    q[m1, 2] = vals[m1, 1]
    q[m1, 3] = vals[m1, 2]
    # largest=2: (0,1,3) <= (v0,v1,v2)
    q[m2, 0] = vals[m2, 0]
    q[m2, 1] = vals[m2, 1]
    q[m2, 3] = vals[m2, 2]
    # largest=3: (0,1,2) <= (v0,v1,v2)
    q[m3, 0] = vals[m3, 0]
    q[m3, 1] = vals[m3, 1]
    q[m3, 2] = vals[m3, 2]

    # 复原最大分量
    sum_sq = np.sum(q * q, axis=1)
    max_comp = np.sqrt(np.clip(1.0 - sum_sq, 0.0, 1.0)).astype(np.float32)
    # 写回到对应的 largest 位置(0:w, 1:x, 2:y, 3:z)
    q[m0, 0] = max_comp[m0]
    q[m1, 1] = max_comp[m1]
    q[m2, 2] = max_comp[m2]
    q[m3, 3] = max_comp[m3]
    
    return q.astype(np.float32)


def is_super_splat_format(plydata: PlyData) -> bool:
    """
    检测 PLY 文件是否为 SuperSplat 格式
    
    Args:
        plydata: 从 PLY 文件读取的数据
        
    Returns:
        bool: 如果是 SuperSplat 格式返回 True, 否则返回 False
    """
    try:
        plydata['chunk']
        return True
    except KeyError:
        return False


# ============================================================================
# 压缩/编码函数
# ============================================================================
def _pack_rotations_vectorized(rots: np.ndarray) -> np.ndarray:
    """
    向量化批量压缩四元数为2-10-10-10位格式
    
    Args:
        rots: shape=(N, 4) 四元数数组 (w, x, y, z)
    
    Returns:
        shape=(N,) uint32数组: 压缩后的旋转
    """
    N = len(rots)
    
    # 归一化四元数
    norms = np.linalg.norm(rots, axis=1, keepdims=True)
    rots = rots / norms
    
    # 找到绝对值最大的分量索引
    largest = np.argmax(np.abs(rots), axis=1)  # shape=(N,)
    
    # 确保最大分量为正（翻转符号）
    max_values = rots[np.arange(N), largest]
    flip_mask = max_values < 0
    rots[flip_mask] = -rots[flip_mask]
    
    # 准备压缩
    norm = np.sqrt(2.0) * 0.5
    result = np.zeros(N, dtype=np.uint32)
    
    # 对于每个largest值，打包其他三个分量
    for largest_idx in range(4):
        mask = (largest == largest_idx)
        if not np.any(mask):
            continue
        
        # 获取其他三个分量的索引
        other_indices = [i for i in range(4) if i != largest_idx]
        
        # 提取其他分量并归一化到[0, 1]
        other_values = rots[mask][:, other_indices]  # shape=(M, 3)
        # 注意：这里应该是乘法 (value * norm + 0.5)，不是除法
        other_normalized = other_values * norm + 0.5
        other_normalized = np.clip(other_normalized, 0, 1)
        
        # 打包为10bit整数
        packed = np.round(other_normalized * 1023).astype(np.uint32)
        
        # 组合: (largest_idx << 30) | (v0 << 20) | (v1 << 10) | v2
        result[mask] = (largest_idx << 30) | (packed[:, 0] << 20) | (packed[:, 1] << 10) | packed[:, 2]
    
    return result


class _Chunk:
    """
    处理256个高斯的压缩块
    """
    def __init__(self, size: int = 256):
        self.size = size
        self.data = {
            'x': np.zeros(size, dtype=np.float32),
            'y': np.zeros(size, dtype=np.float32),
            'z': np.zeros(size, dtype=np.float32),
            'scale_0': np.zeros(size, dtype=np.float32),
            'scale_1': np.zeros(size, dtype=np.float32),
            'scale_2': np.zeros(size, dtype=np.float32),
            'f_dc_0': np.zeros(size, dtype=np.float32),
            'f_dc_1': np.zeros(size, dtype=np.float32),
            'f_dc_2': np.zeros(size, dtype=np.float32),
            'opacity': np.zeros(size, dtype=np.float32),
            'rot_0': np.zeros(size, dtype=np.float32),
            'rot_1': np.zeros(size, dtype=np.float32),
            'rot_2': np.zeros(size, dtype=np.float32),
            'rot_3': np.zeros(size, dtype=np.float32),
        }
        
        # 压缩后的数据
        self.position = np.zeros(size, dtype=np.uint32)
        self.rotation = np.zeros(size, dtype=np.uint32)
        self.scale = np.zeros(size, dtype=np.uint32)
        self.color = np.zeros(size, dtype=np.uint32)
    
    def set_data_batch(self, xyz: np.ndarray, rot: np.ndarray, 
                       scale: np.ndarray, opacity: np.ndarray, f_dc: np.ndarray):
        """
        批量设置高斯数据（向量化版本）
        
        Args:
            xyz: shape=(M, 3) 位置
            rot: shape=(M, 4) 四元数 (w, x, y, z)
            scale: shape=(M, 3) 缩放
            opacity: shape=(M,) 不透明度 (logit形式)
            f_dc: shape=(M, 3) DC球谐系数 (RGB)
        """
        M = len(xyz)
        self.data['x'][:M] = xyz[:, 0]
        self.data['y'][:M] = xyz[:, 1]
        self.data['z'][:M] = xyz[:, 2]
        
        self.data['rot_0'][:M] = rot[:, 0]
        self.data['rot_1'][:M] = rot[:, 1]
        self.data['rot_2'][:M] = rot[:, 2]
        self.data['rot_3'][:M] = rot[:, 3]
        
        self.data['scale_0'][:M] = scale[:, 0]
        self.data['scale_1'][:M] = scale[:, 1]
        self.data['scale_2'][:M] = scale[:, 2]
        
        self.data['f_dc_0'][:M] = f_dc[:, 0]
        self.data['f_dc_1'][:M] = f_dc[:, 1]
        self.data['f_dc_2'][:M] = f_dc[:, 2]
        
        self.data['opacity'][:M] = opacity
    
    def pack(self) -> Dict[str, Tuple[float, float]]:
        """
        压缩chunk中的所有数据（向量化版本）
        
        Returns:
            包含各属性min/max的字典
        """
        # 获取数据数组
        x = self.data['x']
        y = self.data['y']
        z = self.data['z']
        scale_0 = self.data['scale_0']
        scale_1 = self.data['scale_1']
        scale_2 = self.data['scale_2']
        rot_0 = self.data['rot_0']
        rot_1 = self.data['rot_1']
        rot_2 = self.data['rot_2']
        rot_3 = self.data['rot_3']
        f_dc_0 = self.data['f_dc_0'].copy()
        f_dc_1 = self.data['f_dc_1'].copy()
        f_dc_2 = self.data['f_dc_2'].copy()
        opacity = self.data['opacity']
        
        # 计算位置的min/max
        px = {'min': float(np.min(x)), 'max': float(np.max(x))}
        py = {'min': float(np.min(y)), 'max': float(np.max(y))}
        pz = {'min': float(np.min(z)), 'max': float(np.max(z))}
        
        # 计算scale的min/max，并限制范围
        sx = {'min': float(np.clip(np.min(scale_0), -20, 20)),
              'max': float(np.clip(np.max(scale_0), -20, 20))}
        sy = {'min': float(np.clip(np.min(scale_1), -20, 20)),
              'max': float(np.clip(np.max(scale_1), -20, 20))}
        sz = {'min': float(np.clip(np.min(scale_2), -20, 20)),
              'max': float(np.clip(np.max(scale_2), -20, 20))}
        
        # 将球谐DC系数转换为颜色 (SH -> RGB)
        f_dc_0 = f_dc_0 * SH_C0 + 0.5
        f_dc_1 = f_dc_1 * SH_C0 + 0.5
        f_dc_2 = f_dc_2 * SH_C0 + 0.5
        
        # 计算颜色的min/max
        cr = {'min': float(np.min(f_dc_0)), 'max': float(np.max(f_dc_0))}
        cg = {'min': float(np.min(f_dc_1)), 'max': float(np.max(f_dc_1))}
        cb = {'min': float(np.min(f_dc_2)), 'max': float(np.max(f_dc_2))}
        
        # === 向量化压缩所有高斯 ===
        
        # 1. 压缩位置 (11-10-11)
        x_norm = self._normalize_array(x, px['min'], px['max'])
        y_norm = self._normalize_array(y, py['min'], py['max'])
        z_norm = self._normalize_array(z, pz['min'], pz['max'])
        
        x_bits = np.round(x_norm * 2047).astype(np.uint32)
        y_bits = np.round(y_norm * 1023).astype(np.uint32)
        z_bits = np.round(z_norm * 2047).astype(np.uint32)
        
        self.position = (x_bits << 21) | (y_bits << 11) | z_bits
        
        # 2. 压缩旋转
        rots = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)  # shape=(256, 4)
        self.rotation = _pack_rotations_vectorized(rots)
        
        # 3. 压缩缩放 (11-10-11)
        sx_norm = self._normalize_array(scale_0, sx['min'], sx['max'])
        sy_norm = self._normalize_array(scale_1, sy['min'], sy['max'])
        sz_norm = self._normalize_array(scale_2, sz['min'], sz['max'])
        
        sx_bits = np.round(sx_norm * 2047).astype(np.uint32)
        sy_bits = np.round(sy_norm * 1023).astype(np.uint32)
        sz_bits = np.round(sz_norm * 2047).astype(np.uint32)
        
        self.scale = (sx_bits << 21) | (sy_bits << 11) | sz_bits
        
        # 4. 压缩颜色和不透明度 (8-8-8-8)
        r_norm = self._normalize_array(f_dc_0, cr['min'], cr['max'])
        g_norm = self._normalize_array(f_dc_1, cg['min'], cg['max'])
        b_norm = self._normalize_array(f_dc_2, cb['min'], cb['max'])
        
        # opacity从logit转换为[0,1]: sigmoid(opacity)
        opacity_normalized = 1.0 / (1.0 + np.exp(-opacity))
        
        r_bits = np.round(r_norm * 255).astype(np.uint32)
        g_bits = np.round(g_norm * 255).astype(np.uint32)
        b_bits = np.round(b_norm * 255).astype(np.uint32)
        a_bits = np.round(opacity_normalized * 255).astype(np.uint32)
        
        self.color = (r_bits << 24) | (g_bits << 16) | (b_bits << 8) | a_bits
        
        return {
            'px': px, 'py': py, 'pz': pz,
            'sx': sx, 'sy': sy, 'sz': sz,
            'cr': cr, 'cg': cg, 'cb': cb
        }
    
    @staticmethod
    def _normalize_array(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        """向量化归一化到[0,1]范围"""
        if max_val - min_val < 1e-7:
            return np.zeros_like(arr)
        result = (arr - min_val) / (max_val - min_val)
        return np.clip(result, 0.0, 1.0)


def save_super_splat_ply(
    gaussian_data: GaussianData,
    output_path: str
) -> None:
    """
    将 GaussianData 压缩保存为 SuperSplat 格式的 PLY 文件
    
    Args:
        gaussian_data: 要压缩的高斯数据
        output_path: 输出文件路径
    """
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    compressed_data = compress_to_super_splat(
        to_numpy(gaussian_data.xyz),
        to_numpy(gaussian_data.rot),
        to_numpy(gaussian_data.scale),
        to_numpy(gaussian_data.opacity),
        to_numpy(gaussian_data.sh)
    )
    
    with open(output_path, 'wb') as f:
        f.write(compressed_data)


def compress_to_super_splat(
    xyz: np.ndarray,
    rot: np.ndarray,
    scale: np.ndarray,
    opacity: np.ndarray,
    sh: np.ndarray
) -> bytes:
    """
    将3DGS模型压缩为SuperSplat compressed PLY格式（向量化优化版）
    
    Args:
        xyz: shape=(N, 3), dtype=float32 - 位置
        rot: shape=(N, 4), dtype=float32 - 四元数 (w, x, y, z)
        scale: shape=(N, 3), dtype=float32 - 缩放 (已exp变换)
        opacity: shape=(N, 1), dtype=float32 - 不透明度 (sigmoid后的值[0,1])
        sh: shape=(N, sh_dim), dtype=float32 - 球谐系数
    
    Returns:
        压缩后的PLY文件字节
    """
    N = len(xyz)
    num_chunks = (N + 255) // 256  # 向上取整
    
    # 将scale和opacity转回原始形式
    # scale: 需要取log
    scale_log = np.log(scale)
    
    # opacity: 需要转回logit形式
    # logit(p) = log(p / (1-p))
    epsilon = 1e-7
    opacity_clamped = np.clip(opacity, epsilon, 1.0 - epsilon)
    opacity_logit = np.log(opacity_clamped / (1.0 - opacity_clamped))
    
    if len(sh.shape) > 2:
        sh = sh.reshape(sh.shape[0], -1)

    # 提取DC分量 (前3个sh系数)
    f_dc = sh[:, :3]  # shape=(N, 3)
    
    # 提取高阶分量
    f_rest = sh[:, 3:] if sh.shape[1] > 3 else None
    num_rest_coeffs = f_rest.shape[1] if f_rest is not None else 0
    
    # 准备头部
    chunk_props = [
        'min_x', 'min_y', 'min_z',
        'max_x', 'max_y', 'max_z',
        'min_scale_x', 'min_scale_y', 'min_scale_z',
        'max_scale_x', 'max_scale_y', 'max_scale_z',
        'min_r', 'min_g', 'min_b',
        'max_r', 'max_g', 'max_b'
    ]
    
    vertex_props = [
        'packed_position',
        'packed_rotation',
        'packed_scale',
        'packed_color'
    ]
    
    header_lines = [
        'ply',
        'format binary_little_endian 1.0',
        'comment compressed by super_splat_loader.py (vectorized)',
        f'element chunk {num_chunks}'
    ]
    header_lines.extend([f'property float {p}' for p in chunk_props])
    header_lines.append(f'element vertex {N}')
    header_lines.extend([f'property uint {p}' for p in vertex_props])
    
    if num_rest_coeffs > 0:
        header_lines.append(f'element sh {N}')
        header_lines.extend([f'property uchar f_rest_{i}' for i in range(num_rest_coeffs)])
        
    header_lines.append('end_header')
    
    header_text = '\n'.join(header_lines) + '\n'
    header_bytes = header_text.encode('ascii')
    
    # 准备输出缓冲区
    output = bytearray(header_bytes)
    
    # 处理每个chunk（向量化）
    chunk = _Chunk(256)
    
    # 预先计算所有chunk需要的数据
    chunk_data_list = []
    vertex_data_list = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * 256
        end_idx = min(start_idx + 256, N)
        num_in_chunk = end_idx - start_idx
        
        # 提取chunk数据
        chunk_xyz = xyz[start_idx:end_idx]
        chunk_rot = rot[start_idx:end_idx]
        chunk_scale_log = scale_log[start_idx:end_idx]
        chunk_opacity_logit = opacity_logit[start_idx:end_idx]
        chunk_f_dc = f_dc[start_idx:end_idx]
        
        # 如果最后一个chunk不足256个，用最后一个高斯填充
        if num_in_chunk < 256:
            pad_size = 256 - num_in_chunk
            last_xyz = chunk_xyz[-1:].repeat(pad_size, axis=0)
            last_rot = chunk_rot[-1:].repeat(pad_size, axis=0)
            last_scale = chunk_scale_log[-1:].repeat(pad_size, axis=0)
            last_opacity = np.full(pad_size, chunk_opacity_logit[-1], dtype=np.float32)
            last_f_dc = chunk_f_dc[-1:].repeat(pad_size, axis=0)
            
            chunk_xyz = np.vstack([chunk_xyz, last_xyz])
            chunk_rot = np.vstack([chunk_rot, last_rot])
            chunk_scale_log = np.vstack([chunk_scale_log, last_scale])
            chunk_opacity_logit = np.concatenate([chunk_opacity_logit, last_opacity])
            chunk_f_dc = np.vstack([chunk_f_dc, last_f_dc])
        
        # 批量设置数据
        chunk.set_data_batch(chunk_xyz, chunk_rot, chunk_scale_log, chunk_opacity_logit, chunk_f_dc)
        
        # 压缩chunk
        ranges = chunk.pack()
        
        # 写入chunk的min/max数据 (18个float32)
        chunk_data = struct.pack('<18f',
            ranges['px']['min'], ranges['py']['min'], ranges['pz']['min'],
            ranges['px']['max'], ranges['py']['max'], ranges['pz']['max'],
            ranges['sx']['min'], ranges['sy']['min'], ranges['sz']['min'],
            ranges['sx']['max'], ranges['sy']['max'], ranges['sz']['max'],
            ranges['cr']['min'], ranges['cg']['min'], ranges['cb']['min'],
            ranges['cr']['max'], ranges['cg']['max'], ranges['cb']['max']
        )
        chunk_data_list.append(chunk_data)
        
        # 保存顶点数据（只保存实际的顶点数，不包括填充）
        vertex_data_list.append((chunk.position[:num_in_chunk], 
                                 chunk.rotation[:num_in_chunk],
                                 chunk.scale[:num_in_chunk],
                                 chunk.color[:num_in_chunk]))
    
    # 写入所有chunk数据
    for chunk_data in chunk_data_list:
        output.extend(chunk_data)
    
    # 写入所有顶点数据（向量化）
    for position, rotation, scale_data, color in vertex_data_list:
        num_vertices = len(position)
        # 将4个uint32数组交织打包
        vertex_array = np.stack([position, rotation, scale_data, color], axis=1)  # shape=(num_vertices, 4)
        vertex_bytes = vertex_array.tobytes()
        output.extend(vertex_bytes)
    
    # 写入高阶SH数据
    if num_rest_coeffs > 0:
        # GaussianData stores SH as interleaved (RGB, RGB...), but SuperSplat expects planar (R..., G..., B...)
        # f_rest is (N, 3 * k) -> (N, k, 3) -> (N, 3, k) -> (N, 3 * k)
        f_rest_reshaped = f_rest.reshape(N, -1, 3)
        f_rest_planar = f_rest_reshaped.transpose(0, 2, 1)
        f_rest_planar = f_rest_planar.reshape(N, -1)

        # 量化: uint8 = (value / 8 + 0.5) * 256
        # 限制在 [0, 255]
        f_rest_quantized = (f_rest_planar / 8.0 + 0.5) * 256.0
        f_rest_quantized = np.clip(f_rest_quantized, 0, 255).astype(np.uint8)
        output.extend(f_rest_quantized.tobytes())
    
    return bytes(output)
