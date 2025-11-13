import struct
import numpy as np
from plyfile import PlyData
from typing import Dict, Tuple
from .util_gau import GaussianData

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


# ============================================================================
# 压缩/编码函数
# ============================================================================

def _pack_unorm(value: float, bits: int) -> int:
    """将[0,1]范围的值打包为指定位数的整数"""
    t = (1 << bits) - 1
    return max(0, min(t, int(value * t + 0.5)))


def _pack_111011(x: float, y: float, z: float) -> int:
    """打包三个[0,1]值为11-10-11位格式的uint32"""
    return (_pack_unorm(x, 11) << 21) | (_pack_unorm(y, 10) << 11) | _pack_unorm(z, 11)


def _pack_8888(x: float, y: float, z: float, w: float) -> int:
    """打包四个[0,1]值为8-8-8-8位格式的uint32"""
    return (_pack_unorm(x, 8) << 24) | (_pack_unorm(y, 8) << 16) | \
           (_pack_unorm(z, 8) << 8) | _pack_unorm(w, 8)


def _pack_rotation(rot: np.ndarray) -> int:
    """
    将四元数压缩为2-10-10-10位格式
    
    Args:
        rot: shape=(4,) 四元数 (w, x, y, z)
    
    Returns:
        uint32: 压缩后的旋转
    """
    # 归一化四元数
    rot = rot / np.linalg.norm(rot)
    
    # 找到绝对值最大的分量
    largest = np.argmax(np.abs(rot))
    
    # 确保最大分量为正
    if rot[largest] < 0:
        rot = -rot
    
    # 打包：前2位存储最大分量索引，后30位存储其他三个分量
    norm = np.sqrt(2) * 0.5
    result = largest
    for i in range(4):
        if i != largest:
            # 将[-norm, norm]范围映射到[0, 1]
            value = rot[i] * norm + 0.5
            result = (result << 10) | _pack_unorm(value, 10)
    
    return result


def _normalize(x: float, min_val: float, max_val: float) -> float:
    """将值归一化到[0,1]范围"""
    if x <= min_val:
        return 0.0
    if x >= max_val:
        return 1.0
    if max_val - min_val < 0.00001:
        return 0.0
    return (x - min_val) / (max_val - min_val)


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
    
    def set_data(self, idx: int, xyz: np.ndarray, rot: np.ndarray, 
                 scale: np.ndarray, opacity: float, f_dc: np.ndarray):
        """
        设置单个高斯的数据
        
        Args:
            idx: 在chunk中的索引
            xyz: shape=(3,) 位置
            rot: shape=(4,) 四元数 (w, x, y, z)
            scale: shape=(3,) 缩放
            opacity: 不透明度 (logit形式)
            f_dc: shape=(3,) DC球谐系数 (RGB)
        """
        self.data['x'][idx] = xyz[0]
        self.data['y'][idx] = xyz[1]
        self.data['z'][idx] = xyz[2]
        
        self.data['rot_0'][idx] = rot[0]
        self.data['rot_1'][idx] = rot[1]
        self.data['rot_2'][idx] = rot[2]
        self.data['rot_3'][idx] = rot[3]
        
        self.data['scale_0'][idx] = scale[0]
        self.data['scale_1'][idx] = scale[1]
        self.data['scale_2'][idx] = scale[2]
        
        self.data['f_dc_0'][idx] = f_dc[0]
        self.data['f_dc_1'][idx] = f_dc[1]
        self.data['f_dc_2'][idx] = f_dc[2]
        
        self.data['opacity'][idx] = opacity
    
    def pack(self) -> Dict[str, Tuple[float, float]]:
        """
        压缩chunk中的所有数据
        
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
        
        # 压缩每个高斯
        for i in range(self.size):
            # 压缩位置
            self.position[i] = _pack_111011(
                _normalize(x[i], px['min'], px['max']),
                _normalize(y[i], py['min'], py['max']),
                _normalize(z[i], pz['min'], pz['max'])
            )
            
            # 压缩旋转
            rot = np.array([rot_0[i], rot_1[i], rot_2[i], rot_3[i]])
            self.rotation[i] = _pack_rotation(rot)
            
            # 压缩缩放
            self.scale[i] = _pack_111011(
                _normalize(scale_0[i], sx['min'], sx['max']),
                _normalize(scale_1[i], sy['min'], sy['max']),
                _normalize(scale_2[i], sz['min'], sz['max'])
            )
            
            # 压缩颜色和不透明度
            # opacity从logit转换为[0,1]: sigmoid(opacity)
            opacity_normalized = 1.0 / (1.0 + np.exp(-opacity[i]))
            self.color[i] = _pack_8888(
                _normalize(f_dc_0[i], cr['min'], cr['max']),
                _normalize(f_dc_1[i], cg['min'], cg['max']),
                _normalize(f_dc_2[i], cb['min'], cb['max']),
                opacity_normalized
            )
        
        return {
            'px': px, 'py': py, 'pz': pz,
            'sx': sx, 'sy': sy, 'sz': sz,
            'cr': cr, 'cg': cg, 'cb': cb
        }


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
    compressed_data = compress_to_super_splat(
        gaussian_data.xyz,
        gaussian_data.rot,
        gaussian_data.scale,
        gaussian_data.opacity,
        gaussian_data.sh
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
    将3DGS模型压缩为SuperSplat compressed PLY格式
    
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
    
    # 提取DC分量 (前3个sh系数)
    f_dc = sh[:, :3]  # shape=(N, 3)
    
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
        'comment compressed by super_splat_loader.py',
        f'element chunk {num_chunks}'
    ]
    header_lines.extend([f'property float {p}' for p in chunk_props])
    header_lines.append(f'element vertex {N}')
    header_lines.extend([f'property uint {p}' for p in vertex_props])
    header_lines.append('end_header')
    
    header_text = '\n'.join(header_lines) + '\n'
    header_bytes = header_text.encode('ascii')
    
    # 准备输出缓冲区
    output = bytearray(header_bytes)
    
    # 处理每个chunk
    chunk = _Chunk(256)
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * 256
        end_idx = min(start_idx + 256, N)
        num_in_chunk = end_idx - start_idx
        
        # 填充chunk数据
        for i in range(num_in_chunk):
            global_idx = start_idx + i
            chunk.set_data(
                i,
                xyz[global_idx],
                rot[global_idx],
                scale_log[global_idx],
                opacity_logit[global_idx, 0],
                f_dc[global_idx]
            )
        
        # 如果最后一个chunk不足256个，用最后一个高斯填充
        if num_in_chunk < 256:
            last_idx = end_idx - 1
            for i in range(num_in_chunk, 256):
                chunk.set_data(
                    i,
                    xyz[last_idx],
                    rot[last_idx],
                    scale_log[last_idx],
                    opacity_logit[last_idx, 0],
                    f_dc[last_idx]
                )
        
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
        output.extend(chunk_data)
    
    # 写入顶点数据
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * 256
        end_idx = min(start_idx + 256, N)
        num_in_chunk = end_idx - start_idx
        
        # 重新准备chunk数据
        for i in range(num_in_chunk):
            global_idx = start_idx + i
            chunk.set_data(
                i,
                xyz[global_idx],
                rot[global_idx],
                scale_log[global_idx],
                opacity_logit[global_idx, 0],
                f_dc[global_idx]
            )
        
        if num_in_chunk < 256:
            last_idx = end_idx - 1
            for i in range(num_in_chunk, 256):
                chunk.set_data(
                    i,
                    xyz[last_idx],
                    rot[last_idx],
                    scale_log[last_idx],
                    opacity_logit[last_idx, 0],
                    f_dc[last_idx]
                )
        
        chunk.pack()
        
        # 写入顶点数据 (每个顶点4个uint32)
        for i in range(num_in_chunk):
            vertex_data = struct.pack('<4I',
                chunk.position[i],
                chunk.rotation[i],
                chunk.scale[i],
                chunk.color[i]
            )
            output.extend(vertex_data)
    
    return bytes(output)
