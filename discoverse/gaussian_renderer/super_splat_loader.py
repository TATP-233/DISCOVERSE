import numpy as np
from plyfile import PlyData
from .util_gau import GaussianData


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
    
    SH_C0 = 0.28209479177387814
    fr = (fr - 0.5) / SH_C0
    fg = (fg - 0.5) / SH_C0
    fb = (fb - 0.5) / SH_C0
    
    opacity = a8.astype(np.float32) / 255.0
    # opacity = 1.0 / (1.0 + np.exp(-opacity))
    
    colors = np.stack([fr, fg, fb], axis=1).astype(np.float32)
    opacities = opacity.astype(np.float32)
    
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
