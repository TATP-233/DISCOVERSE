#!/usr/bin/env python3
"""
生成测试用的3DGS模型（ribbon / cube），使用项目内的 GaussianData 和 save_ply 保存。
"""

import numpy as np
import os
from typing import Tuple, List

from gaussian_renderer import GaussianData
from gaussian_renderer.util_gau import save_ply
from gaussian_renderer.super_splat_loader import save_super_splat_ply


def _build_sh_array(num_points: int, rgb: Tuple[float, float, float] = (0.0, 1.0, 0.0), degree: int = 3):
    """
    构建 SH 系数数组（以 DC + 高阶为平面展平形式）：
    - degree: 0..3。默认3（包含到三阶）
    返回形状 (N, 3 + 3 * ((degree+1)^2 - 1)) 的 numpy.float32 数组
    """
    C0 = 0.28209479177387814
    r, g, b = rgb
    dc = np.stack([np.full(num_points, (r - 0.5) / C0),
                   np.full(num_points, (g - 0.5) / C0),
                   np.full(num_points, (b - 0.5) / C0)], axis=1)

    m = (degree + 1) ** 2 - 1
    rest = np.zeros((num_points, m * 3), dtype=np.float32)

    sh = np.concatenate([dc.astype(np.float32), rest], axis=1)
    return sh


def generate_ribbon(filename: str,
                    length: float = 0.04,
                    width: float = 2.0,
                    thickness: float = 1e-6,
                    color: Tuple[float, float, float] = (0.0, 0.5, 1.0),
                    save_supersplat: bool = False,
                    sh_degree: int = 3):
    """
    生成一条带状的3DGS模型并保存为PLY。

    - 长度对应 x 方向，宽度对应 y，厚度在 z 方向接近0。
    - 使用 `GaussianData` 并通过 `save_ply` 保存为标准3DGS PLY。
    - 如果 `save_supersplat=True`，也会同时保存 SuperSplat 压缩版（后缀 .supersplat.ply）。
    """
    # 位置分布：均匀在矩形平面上，加上极小的 z 噪声以避免完全平面
    # xs = (np.random.rand(num_points) - 0.5) * length
    # ys = (np.random.rand(num_points) - 0.5) * width
    # zs = (np.random.randn(num_points) * (thickness / 6.0)).astype(np.float32)
    nx, ny = 5, 100
    xs, ys = np.meshgrid(
        np.linspace(-length / 2, length / 2, nx),
        np.linspace(-width / 2, width / 2, ny)
    )
    zs = np.zeros(nx * ny)
    num_points = nx * ny

    xyz = np.stack([xs.flatten(), ys.flatten(), zs], axis=1).astype(np.float32)

    # 旋转：单位四元数 (w, x, y, z)
    rots = np.zeros((num_points, 4), dtype=np.float32)
    rots[:, 0] = 1.0

    # scales: 保持线性尺度（save_ply 会对其取 log）
    # 对于带状，沿 x 的尺度可以稍大，沿 y 取接近宽度的一小部分，z 非常小
    scale_x = 0.005 
    scale_y = 0.02 
    scale_z = max(1e-9, thickness)
    scales = np.tile(np.array([scale_x, scale_y, scale_z], dtype=np.float32), (num_points, 1))

    # 透明度：使用概率值（save_ply 会转为 logit）
    opacity = np.full((num_points,), 0.99, dtype=np.float32)

    # SH 系数（DC + 高阶）
    shs = _build_sh_array(num_points, rgb=color, degree=sh_degree)

    data = GaussianData(xyz, rots, scales, opacity, shs)

    if save_supersplat:
        save_super_splat_ply(data, filename, save_sh_degree=sh_degree)
    else:
        # 保存标准3DGS PLY
        save_ply(data, filename, save_sh_degree=sh_degree)

    print(f"Generated ribbon saved to {filename} with {num_points} points.")


def generate_cube(filename: str, size: Tuple[float, float, float], rgb: Tuple[float, float, float] = (0.0, 1.0, 0.0), num_points: int = 10000, sh_degree: int = 3):
    """
    兼容旧的 cube 生成器，但使用 GaussianData + save_ply 保存。
    """
    width, depth, height = size

    x = (np.random.rand(num_points) - 0.5) * width
    y = (np.random.rand(num_points) - 0.5) * depth
    z = (np.random.rand(num_points) - 0.5) * height

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    rots = np.zeros((num_points, 4), dtype=np.float32)
    rots[:, 0] = 1.0
    scale_val = 0.001
    scales = np.full((num_points, 3), scale_val, dtype=np.float32)
    opacity = np.full((num_points,), 0.99, dtype=np.float32)
    shs = _build_sh_array(num_points, rgb=rgb, degree=sh_degree)

    data = GaussianData(xyz, rots, scales, opacity, shs)
    save_ply(data, filename, save_sh_degree=sh_degree)
    print(f"Generated {filename} with {num_points} points.")


if __name__ == "__main__":
    output_dir = "assets/3dgs/"
    os.makedirs(output_dir, exist_ok=True)
    out_ribbon = os.path.join(output_dir, "ribbon_blue.ply")
    generate_ribbon(out_ribbon, color=(0.0, 0.5, 1.0), save_supersplat=True)