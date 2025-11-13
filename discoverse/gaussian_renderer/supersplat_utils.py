"""
SuperSplat格式压缩和解压缩工具

SuperSplat是一种高效的3D Gaussian Splatting压缩格式:
- 使用chunk-based压缩，每256个顶点对应一个chunk
- 位置: 11/10/11 bit (x/y/z)，相对于chunk边界归一化
- 尺度: 11/10/11 bit (x/y/z)，存储log值
- 颜色+不透明度: 8/8/8/8 bit (r/g/b/a)
- 旋转: 2 bit (最大分量索引) + 3×10 bit (其他三个分量)

压缩比通常可达3-5倍
"""

import numpy as np
from plyfile import PlyData, PlyElement
from typing import Dict, Tuple


# SuperSplat格式常量
CHUNK_SIZE = 256
SH_C0 = 0.28209479177387814  # 球谐系数C0 = sqrt(1/(4*pi))


def decompress_supersplat(plydata: PlyData) -> Dict[str, np.ndarray]:
    """
    解压SuperSplat格式的PLY数据
    
    Args:
        plydata: 已读取的PlyData对象
        
    Returns:
        包含positions, rotations, scales, colors, opacities的字典
    """
    vtx = plydata['vertex'].data  # structured array
    chk = plydata['chunk'].data   # structured array
    
    num_vertex = vtx.shape[0]
    
    if num_vertex == 0:
        return {
            'positions': np.zeros((0, 3), dtype=np.float32),
            'rotations': np.zeros((0, 4), dtype=np.float32),
            'scales': np.zeros((0, 3), dtype=np.float32),
            'colors': np.zeros((0, 3), dtype=np.float32),
            'opacities': np.zeros((0,), dtype=np.float32)
        }
    
    # 计算每个顶点对应的chunk索引
    chunk_idx = (np.arange(num_vertex) // CHUNK_SIZE).astype(np.int64)
    chunk_idx = np.clip(chunk_idx, 0, chk.shape[0] - 1)
    
    def gather_chunk(field):
        """从chunk数组中提取对应字段"""
        return chk[field][chunk_idx]
    
    # 1. 解码位置 (11/10/11 bit)
    ppos = vtx['packed_position'].astype(np.uint32)
    xbits = (ppos >> 21) & 0x7FF  # 11 bits
    ybits = (ppos >> 11) & 0x3FF  # 10 bits
    zbits = ppos & 0x7FF           # 11 bits
    
    fx = xbits.astype(np.float32) / 2047.0 * (gather_chunk('max_x') - gather_chunk('min_x')) + gather_chunk('min_x')
    fy = ybits.astype(np.float32) / 1023.0 * (gather_chunk('max_y') - gather_chunk('min_y')) + gather_chunk('min_y')
    fz = zbits.astype(np.float32) / 2047.0 * (gather_chunk('max_z') - gather_chunk('min_z')) + gather_chunk('min_z')
    positions = np.stack([fx, fy, fz], axis=1).astype(np.float32)
    
    # 2. 解码尺度 (11/10/11 bit)，存储的是log值
    pscale = vtx['packed_scale'].astype(np.uint32)
    sxb = (pscale >> 21) & 0x7FF
    syb = (pscale >> 11) & 0x3FF
    szb = pscale & 0x7FF
    
    sx = sxb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_x') - gather_chunk('min_scale_x')) + gather_chunk('min_scale_x')
    sy = syb.astype(np.float32) / 1023.0 * (gather_chunk('max_scale_y') - gather_chunk('min_scale_y')) + gather_chunk('min_scale_y')
    sz = szb.astype(np.float32) / 2047.0 * (gather_chunk('max_scale_z') - gather_chunk('min_scale_z')) + gather_chunk('min_scale_z')
    scales = np.exp(np.stack([sx, sy, sz], axis=1)).astype(np.float32)
    
    # 3. 解码颜色和不透明度 (8/8/8/8 bit)
    pcol = vtx['packed_color'].astype(np.uint32)
    r8 = (pcol >> 24) & 0xFF
    g8 = (pcol >> 16) & 0xFF
    b8 = (pcol >> 8) & 0xFF
    a8 = pcol & 0xFF
    
    fr = r8.astype(np.float32) / 255.0 * (gather_chunk('max_r') - gather_chunk('min_r')) + gather_chunk('min_r')
    fg = g8.astype(np.float32) / 255.0 * (gather_chunk('max_g') - gather_chunk('min_g')) + gather_chunk('min_g')
    fb = b8.astype(np.float32) / 255.0 * (gather_chunk('max_b') - gather_chunk('min_b')) + gather_chunk('min_b')
    
    # 转换为球谐系数（DC分量）
    fr = (fr - 0.5) / SH_C0
    fg = (fg - 0.5) / SH_C0
    fb = (fb - 0.5) / SH_C0
    colors = np.stack([fr, fg, fb], axis=1).astype(np.float32)
    
    opacities = a8.astype(np.float32) / 255.0
    
    # 4. 解码旋转 (2 bit 最大分量索引 + 3×10 bit)
    prot = vtx['packed_rotation'].astype(np.uint32)
    largest = (prot >> 30) & 0x3  # 最大分量的索引 (0-3)
    v0 = (prot >> 20) & 0x3FF
    v1 = (prot >> 10) & 0x3FF
    v2 = prot & 0x3FF
    
    norm = np.sqrt(2.0) * 0.5
    vals = np.stack([v0, v1, v2], axis=1).astype(np.float32)
    vals = (vals / 1023.0 - 0.5) / norm
    
    q = np.zeros((num_vertex, 4), dtype=np.float32)
    
    # 根据最大分量索引填充四元数
    m0 = (largest == 0)  # w is largest
    m1 = (largest == 1)  # x is largest
    m2 = (largest == 2)  # y is largest
    m3 = (largest == 3)  # z is largest
    
    # largest=0: (x,y,z) <= (v0,v1,v2)
    q[m0, 1] = vals[m0, 0]
    q[m0, 2] = vals[m0, 1]
    q[m0, 3] = vals[m0, 2]
    # largest=1: (w,y,z) <= (v0,v1,v2)
    q[m1, 0] = vals[m1, 0]
    q[m1, 2] = vals[m1, 1]
    q[m1, 3] = vals[m1, 2]
    # largest=2: (w,x,z) <= (v0,v1,v2)
    q[m2, 0] = vals[m2, 0]
    q[m2, 1] = vals[m2, 1]
    q[m2, 3] = vals[m2, 2]
    # largest=3: (w,x,y) <= (v0,v1,v2)
    q[m3, 0] = vals[m3, 0]
    q[m3, 1] = vals[m3, 1]
    q[m3, 2] = vals[m3, 2]
    
    # 恢复最大分量
    sum_sq = np.sum(q * q, axis=1)
    max_comp = np.sqrt(np.clip(1.0 - sum_sq, 0.0, 1.0)).astype(np.float32)
    q[m0, 0] = max_comp[m0]
    q[m1, 1] = max_comp[m1]
    q[m2, 2] = max_comp[m2]
    q[m3, 3] = max_comp[m3]
    rotations = q.astype(np.float32)
    # 尝试读取 sh element 的 f_rest_* 字段（如果存在）
    sh_rest = None
    try:
        sh_el = plydata['sh'].data
        # 收集 f_rest_* 按序
        rest_names = [p.name for p in sh_el.dtype.names if p.name.startswith('f_rest_')]
        rest_names = sorted(rest_names, key=lambda x: int(x.split('_')[-1]))
        if len(rest_names) > 0:
            rest_arr = np.stack([sh_el[name].astype(np.uint8) for name in rest_names], axis=1)
            # 归一化到 0..1
            sh_rest = rest_arr.astype(np.float32) / 255.0
    except Exception:
        sh_rest = None

    # 构造完整 shs：前三个 DC 已放入 colors（球谐系数），其余恢复为归一化值（若存在）
    if sh_rest is None:
        shs = colors.astype(np.float32)
    else:
        shs = np.concatenate([colors.astype(np.float32), sh_rest.astype(np.float32)], axis=1)

    return {
        'positions': positions,
        'rotations': rotations,
        'scales': scales,
        'colors': colors,
        'opacities': opacities,
        'shs': shs
    }


def compress_supersplat(positions: np.ndarray, 
                       rotations: np.ndarray,
                       scales: np.ndarray, 
                       shs: np.ndarray,
                       opacities: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将高斯数据压缩为SuperSplat格式
    
    Args:
        positions: (N, 3) 位置数组
        rotations: (N, 4) 旋转四元数 (w,x,y,z)
        scales: (N, 3) 尺度数组
        colors: (N, 3) 颜色/球谐DC分量
        opacities: (N,) 不透明度
        
    Returns:
        (vertex_array, chunk_array): 压缩后的顶点和chunk数据
    """
    positions = positions.astype(np.float32)
    rotations = rotations.astype(np.float32)
    scales = scales.astype(np.float32)
    # shs: (N, S) 包含 DC (前三个) 以及其余 coeffs
    shs = shs.astype(np.float32)
    colors = shs[:, :3].astype(np.float32)
    opacities = opacities.astype(np.float32)
    
    num_vertex = len(positions)
    num_chunks = (num_vertex + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # 将scale转换为log空间
    log_scales = np.log(np.clip(scales, 1e-8, None))
    
    # 将颜色（球谐DC分量）转换为RGB
    rgb = colors * SH_C0 + 0.5
    rgb = np.clip(rgb, 0.0, 1.0)
    
    # 准备chunk数据
    chunk_data = []
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, num_vertex)
        
        chunk_pos = positions[start_idx:end_idx]
        chunk_scale = log_scales[start_idx:end_idx]
        chunk_rgb = rgb[start_idx:end_idx]
        
        chunk_info = (
            np.min(chunk_pos[:, 0]), np.max(chunk_pos[:, 0]),
            np.min(chunk_pos[:, 1]), np.max(chunk_pos[:, 1]),
            np.min(chunk_pos[:, 2]), np.max(chunk_pos[:, 2]),
            np.min(chunk_scale[:, 0]), np.max(chunk_scale[:, 0]),
            np.min(chunk_scale[:, 1]), np.max(chunk_scale[:, 1]),
            np.min(chunk_scale[:, 2]), np.max(chunk_scale[:, 2]),
            np.min(chunk_rgb[:, 0]), np.max(chunk_rgb[:, 0]),
            np.min(chunk_rgb[:, 1]), np.max(chunk_rgb[:, 1]),
            np.min(chunk_rgb[:, 2]), np.max(chunk_rgb[:, 2]),
        )
        chunk_data.append(chunk_info)
    
    # 编码顶点数据
    packed_positions = np.zeros(num_vertex, dtype=np.uint32)
    packed_scales = np.zeros(num_vertex, dtype=np.uint32)
    packed_colors = np.zeros(num_vertex, dtype=np.uint32)
    packed_rotations = np.zeros(num_vertex, dtype=np.uint32)
    
    for i in range(num_vertex):
        chunk_id = i // CHUNK_SIZE
        chunk_info = chunk_data[chunk_id]
        
        # 编码位置
        px_norm = (positions[i, 0] - chunk_info[0]) / max(chunk_info[1] - chunk_info[0], 1e-8)
        py_norm = (positions[i, 1] - chunk_info[2]) / max(chunk_info[3] - chunk_info[2], 1e-8)
        pz_norm = (positions[i, 2] - chunk_info[4]) / max(chunk_info[5] - chunk_info[4], 1e-8)
        px_norm = np.clip(px_norm, 0.0, 1.0)
        py_norm = np.clip(py_norm, 0.0, 1.0)
        pz_norm = np.clip(pz_norm, 0.0, 1.0)
        
        xbits = int(px_norm * 2047) & 0x7FF
        ybits = int(py_norm * 1023) & 0x3FF
        zbits = int(pz_norm * 2047) & 0x7FF
        packed_positions[i] = (xbits << 21) | (ybits << 11) | zbits
        
        # 编码尺度
        sx_norm = (log_scales[i, 0] - chunk_info[6]) / max(chunk_info[7] - chunk_info[6], 1e-8)
        sy_norm = (log_scales[i, 1] - chunk_info[8]) / max(chunk_info[9] - chunk_info[8], 1e-8)
        sz_norm = (log_scales[i, 2] - chunk_info[10]) / max(chunk_info[11] - chunk_info[10], 1e-8)
        sx_norm = np.clip(sx_norm, 0.0, 1.0)
        sy_norm = np.clip(sy_norm, 0.0, 1.0)
        sz_norm = np.clip(sz_norm, 0.0, 1.0)
        
        sxb = int(sx_norm * 2047) & 0x7FF
        syb = int(sy_norm * 1023) & 0x3FF
        szb = int(sz_norm * 2047) & 0x7FF
        packed_scales[i] = (sxb << 21) | (syb << 11) | szb
        
        # 编码颜色和不透明度
        r_norm = (rgb[i, 0] - chunk_info[12]) / max(chunk_info[13] - chunk_info[12], 1e-8)
        g_norm = (rgb[i, 1] - chunk_info[14]) / max(chunk_info[15] - chunk_info[14], 1e-8)
        b_norm = (rgb[i, 2] - chunk_info[16]) / max(chunk_info[17] - chunk_info[16], 1e-8)
        r_norm = np.clip(r_norm, 0.0, 1.0)
        g_norm = np.clip(g_norm, 0.0, 1.0)
        b_norm = np.clip(b_norm, 0.0, 1.0)
        
        r8 = int(r_norm * 255) & 0xFF
        g8 = int(g_norm * 255) & 0xFF
        b8 = int(b_norm * 255) & 0xFF
        a8 = int(np.clip(opacities[i], 0.0, 1.0) * 255) & 0xFF
        packed_colors[i] = (r8 << 24) | (g8 << 16) | (b8 << 8) | a8
        
        # 编码旋转（找到最大分量）
        q = rotations[i]
        abs_q = np.abs(q)
        largest = np.argmax(abs_q)
        
        # 提取其他三个分量
        other_indices = [j for j in range(4) if j != largest]
        other_vals = q[other_indices]
        
        # 归一化到 [-sqrt(2)/2, sqrt(2)/2]
        norm = np.sqrt(2.0) * 0.5
        other_vals_norm = np.clip(other_vals / norm, -1.0, 1.0)
        other_vals_quant = ((other_vals_norm + 0.5) * 1023).astype(np.int32)
        other_vals_quant = np.clip(other_vals_quant, 0, 1023)
        
        v0 = int(other_vals_quant[0]) & 0x3FF
        v1 = int(other_vals_quant[1]) & 0x3FF
        v2 = int(other_vals_quant[2]) & 0x3FF
        
        packed_rotations[i] = (largest << 30) | (v0 << 20) | (v1 << 10) | v2
    
    # 创建PLY数据结构
    vertex_dtype = [
        ('packed_position', 'u4'),
        ('packed_rotation', 'u4'),
        ('packed_scale', 'u4'),
        ('packed_color', 'u4'),
    ]
    
    vertex_array = np.zeros(num_vertex, dtype=vertex_dtype)
    vertex_array['packed_position'] = packed_positions
    vertex_array['packed_rotation'] = packed_rotations
    vertex_array['packed_scale'] = packed_scales
    vertex_array['packed_color'] = packed_colors
    
    # chunk 字段顺序按示例头：min_x,min_y,min_z,max_x,max_y,max_z,min_scale_x,min_scale_y,min_scale_z,max_scale_x,max_scale_y,max_scale_z,min_r,min_g,min_b,max_r,max_g,max_b
    chunk_dtype = [
        ('min_x', 'f4'), ('min_y', 'f4'), ('min_z', 'f4'),
        ('max_x', 'f4'), ('max_y', 'f4'), ('max_z', 'f4'),
        ('min_scale_x', 'f4'), ('min_scale_y', 'f4'), ('min_scale_z', 'f4'),
        ('max_scale_x', 'f4'), ('max_scale_y', 'f4'), ('max_scale_z', 'f4'),
        ('min_r', 'f4'), ('min_g', 'f4'), ('min_b', 'f4'),
        ('max_r', 'f4'), ('max_g', 'f4'), ('max_b', 'f4'),
    ]

    chunk_array = np.array(chunk_data, dtype=chunk_dtype)

    # 处理 SH 的其余系数（f_rest）: 把 shs 除去前三个 DC，量化为 uchar
    rest = shs[:, 3:]
    if rest.size == 0:
        sh_array = np.zeros((num_vertex, 0), dtype=np.uint8)
    else:
        # 全局按列量化到 0..255，保留每列的 min/max 不写入文件（简单实现）
        mins = rest.min(axis=0)
        maxs = rest.max(axis=0)
        rng = np.where((maxs - mins) > 1e-8, (maxs - mins), 1.0)
        normed = (rest - mins[None, :]) / rng[None, :]
        quant = np.clip((normed * 255.0).round(), 0, 255).astype(np.uint8)
        sh_array = quant

    return vertex_array, chunk_array, sh_array


def save_supersplat_ply(vertex_array: np.ndarray, chunk_array: np.ndarray, sh_array: np.ndarray, output_path: str):
    """
    保存SuperSplat格式的PLY文件
    
    Args:
        vertex_array: 压缩后的顶点数据
        chunk_array: chunk边界数据
        output_path: 输出文件路径
    """
    vertex_el = PlyElement.describe(vertex_array, 'vertex')
    chunk_el = PlyElement.describe(chunk_array, 'chunk')

    # sh_array: (N, K) uint8 -> create element named 'sh' with uchar properties f_rest_0 ...
    sh_els = []
    if sh_array is not None and sh_array.shape[0] > 0 and sh_array.shape[1] > 0:
        ncols = sh_array.shape[1]
        sh_dtype = [(f'f_rest_{i}', 'u1') for i in range(ncols)]
        sh_el_arr = np.zeros(sh_array.shape[0], dtype=sh_dtype)
        for i in range(ncols):
            sh_el_arr[f'f_rest_{i}'] = sh_array[:, i]
        sh_el = PlyElement.describe(sh_el_arr, 'sh')
        PlyData([vertex_el, chunk_el, sh_el], text=False).write(output_path)
    else:
        PlyData([vertex_el, chunk_el], text=False).write(output_path)


def is_supersplat_format(plydata: PlyData) -> bool:
    """
    检查PLY数据是否为SuperSplat格式
    
    Args:
        plydata: PlyData对象
        
    Returns:
        True if SuperSplat format, False otherwise
    """
    try:
        plydata['chunk']
        return True
    except (KeyError, IndexError):
        return False
