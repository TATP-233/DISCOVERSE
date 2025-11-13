#!/usr/bin/env python3
"""
SuperSplat格式转换工具
支持标准3DGS PLY格式与SuperSplat压缩格式之间的相互转换

SuperSplat格式说明：
- 使用chunk-based压缩，每256个顶点对应一个chunk
- 位置: 11/10/11 bit (x/y/z)
- 尺度: 11/10/11 bit (x/y/z)，存储log值
- 颜色+不透明度: 8/8/8/8 bit (r/g/b/a)
- 旋转: 2 bit (最大分量索引) + 3×10 bit (其他三个分量)
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from plyfile import PlyData, PlyElement
import argparse
from pathlib import Path

from discoverse.gaussian_renderer import supersplat_utils as ssu

class SuperSplatConverter:
    """SuperSplat格式转换器"""
    
    @staticmethod
    def decompress_ply(path: str) -> dict:
        """
        解压SuperSplat格式的PLY文件
        
        Args:
            path: SuperSplat PLY文件路径
            
        Returns:
            包含positions, rotations, scales, colors, opacities的字典
        """
        print(f"正在解压SuperSplat格式文件: {path}")
        plydata = PlyData.read(path)
        
        # 检查是否为SuperSplat格式
        if not ssu.is_supersplat_format(plydata):
            raise ValueError("不是有效的SuperSplat格式文件（缺少vertex或chunk元素）")

        data = ssu.decompress_supersplat(plydata)
        
        num_vertex = len(data['positions'])
        print(f"顶点数: {num_vertex}")
        print("解压完成!")
        
        return data
    
    @staticmethod
    def compress_ply(data: dict, output_path: str):
        """
        将高斯数据压缩为SuperSplat格式
        
        Args:
            data: 包含positions, rotations, scales, colors, opacities的字典
            output_path: 输出PLY文件路径
        """
        print(f"正在压缩为SuperSplat格式: {output_path}")

        positions = data['positions'].astype(np.float32)
        rotations = data['rotations'].astype(np.float32)
        scales = data['scales'].astype(np.float32)
        shs = data['colors'].astype(np.float32)
        opacities = data['opacities'].astype(np.float32)

        num_vertex = len(positions)
        num_chunks = (num_vertex + 255) // 256

        print(f"顶点数: {num_vertex}, Chunk数: {num_chunks}")

        # 使用supersplat_utils进行压缩
        vertex_array, chunk_array, sh_array = ssu.compress_supersplat(
            positions, rotations, scales, shs, opacities
        )

        # 保存文件
        ssu.save_supersplat_ply(vertex_array, chunk_array, sh_array, output_path)

        print(f"压缩完成! 文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    @staticmethod
    def load_standard_ply(path: str) -> dict:
        """
        加载标准3DGS PLY格式
        
        Args:
            path: 标准PLY文件路径
            
        Returns:
            包含positions, rotations, scales, colors, opacities的字典
        """
        print(f"正在加载标准PLY格式: {path}")
        plydata = PlyData.read(path)
        
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1).astype(np.float32)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])
        opacities = 1.0 / (1.0 + np.exp(-opacities))  # sigmoid
        opacities = opacities.astype(np.float32)
        
        # 读取DC分量（前3个球谐系数）
        colors = np.stack((
            np.asarray(plydata.elements[0]["f_dc_0"]),
            np.asarray(plydata.elements[0]["f_dc_1"]),
            np.asarray(plydata.elements[0]["f_dc_2"])
        ), axis=1).astype(np.float32)
        
        # 读取尺度
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(plydata.elements[0][name]) for name in scale_names], axis=1)
        scales = np.exp(scales).astype(np.float32)
        
        # 读取旋转
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.stack([np.asarray(plydata.elements[0][name]) for name in rot_names], axis=1)
        rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
        rots = rots.astype(np.float32)
        
        print(f"加载完成! 顶点数: {len(xyz)}")
        return {
            'positions': xyz,
            'rotations': rots,
            'scales': scales,
            'colors': colors,
            'opacities': opacities
        }
    
    @staticmethod
    def save_standard_ply(data: dict, output_path: str):
        """
        保存为标准3DGS PLY格式
        
        Args:
            data: 包含positions, rotations, scales, colors, opacities的字典
            output_path: 输出PLY文件路径
        """
        print(f"正在保存为标准PLY格式: {output_path}")
        
        positions = data['positions']
        rotations = data['rotations']
        scales = data['scales']
        colors = data['colors']
        opacities = data['opacities']
        
        # opacity转换为logit
        opacities_logit = -np.log(1.0 / np.clip(opacities, 1e-8, 1.0 - 1e-8) - 1.0)
        
        # scale转换为log
        log_scales = np.log(scales)
        
        num_points = len(positions)
        
        # 构建属性列表
        properties = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]
        
        vertex_data = np.zeros(num_points, dtype=properties)
        vertex_data['x'] = positions[:, 0]
        vertex_data['y'] = positions[:, 1]
        vertex_data['z'] = positions[:, 2]
        vertex_data['f_dc_0'] = colors[:, 0]
        vertex_data['f_dc_1'] = colors[:, 1]
        vertex_data['f_dc_2'] = colors[:, 2]
        vertex_data['opacity'] = opacities_logit
        vertex_data['scale_0'] = log_scales[:, 0]
        vertex_data['scale_1'] = log_scales[:, 1]
        vertex_data['scale_2'] = log_scales[:, 2]
        vertex_data['rot_0'] = rotations[:, 0]
        vertex_data['rot_1'] = rotations[:, 1]
        vertex_data['rot_2'] = rotations[:, 2]
        vertex_data['rot_3'] = rotations[:, 3]
        
        vertex_el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vertex_el], text=False).write(output_path)
        
        print(f"保存完成! 文件大小: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser(
        description='SuperSplat格式转换工具 - 在标准3DGS PLY格式与SuperSplat压缩格式之间转换',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 解压SuperSplat格式到标准格式:
   python supersplat_converter.py -i compressed.ply -o decompressed.ply --decompress

2. 压缩标准格式到SuperSplat格式:
   python supersplat_converter.py -i standard.ply -o compressed.ply --compress

3. 只解压并查看信息（不保存）:
   python supersplat_converter.py -i compressed.ply --decompress --no-save
        """
    )
    
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='输入PLY文件路径')
    parser.add_argument('-o', '--output', type=str,
                        help='输出PLY文件路径')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--compress', action='store_true',
                       help='压缩标准格式为SuperSplat格式')
    group.add_argument('--decompress', action='store_true',
                       help='解压SuperSplat格式为标准格式')
    
    parser.add_argument('--no-save', action='store_true',
                        help='不保存输出文件，仅显示信息')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 检查输出路径
    if not args.no_save and not args.output:
        print("错误: 需要指定输出路径 -o/--output (或使用 --no-save)")
        return
    
    converter = SuperSplatConverter()
    
    try:
        if args.decompress:
            # 解压SuperSplat格式
            data = converter.decompress_ply(args.input)
            
            # 打印统计信息
            print(f"\n=== 解压后的数据统计 ===")
            print(f"顶点数: {len(data['positions'])}")
            print(f"位置范围: [{data['positions'].min(axis=0)}, {data['positions'].max(axis=0)}]")
            print(f"尺度范围: [{data['scales'].min(axis=0)}, {data['scales'].max(axis=0)}]")
            print(f"不透明度范围: [{data['opacities'].min():.4f}, {data['opacities'].max():.4f}]")
            
            if not args.no_save:
                converter.save_standard_ply(data, args.output)
                print(f"\n✓ 解压完成: {args.output}")
        
        elif args.compress:
            # 加载标准格式
            data = converter.load_standard_ply(args.input)
            
            # 打印统计信息
            print(f"\n=== 原始数据统计 ===")
            print(f"顶点数: {len(data['positions'])}")
            print(f"位置范围: [{data['positions'].min(axis=0)}, {data['positions'].max(axis=0)}]")
            print(f"尺度范围: [{data['scales'].min(axis=0)}, {data['scales'].max(axis=0)}]")
            
            if not args.no_save:
                converter.compress_ply(data, args.output)
                
                # 计算压缩比
                input_size = Path(args.input).stat().st_size
                output_size = Path(args.output).stat().st_size
                ratio = input_size / output_size
                print(f"\n✓ 压缩完成: {args.output}")
                print(f"压缩比: {ratio:.2f}x ({input_size/1024/1024:.2f}MB -> {output_size/1024/1024:.2f}MB)")
    
    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()
