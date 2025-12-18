#!/usr/bin/env python3
"""
将标准3DGS PLY模型压缩为SuperSplat格式

使用方法:
    python compress_to_supersplat.py -i input.ply -o output.ply
    python compress_to_supersplat.py --input input.ply --output output.ply
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from discoverse.gaussian_renderer.util_gau import load_ply
from discoverse.gaussian_renderer.super_splat_loader import save_super_splat_ply, is_super_splat_format
from plyfile import PlyData


def main():
    parser = argparse.ArgumentParser(
        description='将标准3DGS PLY模型压缩为SuperSplat格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s models/scene.ply
  %(prog)s models/scene.ply -o models/scene_compressed.ply
  %(prog)s input.ply --output output.ply
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='输入PLY文件路径 (标准3DGS格式)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出PLY文件路径 (SuperSplat压缩格式)。默认: 输入文件名.compressed.ply'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='如果输出文件已存在,强制覆盖'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        return 1
    
    if not input_path.suffix.lower() == '.ply':
        print(f"错误: 输入文件必须是PLY格式: {args.input}", file=sys.stderr)
        return 1
    
    # 设置默认输出路径
    if args.output is None:
        # 将 input.ply 转换为 input.compressed.ply
        output_path = input_path.parent / f"{input_path.stem}.compressed{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    # 检查输出文件
    if output_path.exists() and not args.force:
        print(f"错误: 输出文件已存在: {output_path}", file=sys.stderr)
        print("使用 --force 参数强制覆盖", file=sys.stderr)
        return 1
    
    if not output_path.suffix.lower() == '.ply':
        print(f"错误: 输出文件必须是PLY格式: {output_path}", file=sys.stderr)
        return 1
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在加载模型: {args.input}")
    
    # 检查输入文件是否已经是SuperSplat格式
    try:
        plydata = PlyData.read(str(input_path))
        if is_super_splat_format(plydata):
            print("警告: 输入文件已经是SuperSplat格式,无需压缩", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"错误: 无法读取输入文件: {e}", file=sys.stderr)
        return 1
    
    # 加载模型
    try:
        gaussian_data = load_ply(str(input_path))
    except Exception as e:
        print(f"错误: 加载模型失败: {e}", file=sys.stderr)
        return 1
    
    print(f"模型信息:")
    print(f"  - 高斯数量: {len(gaussian_data.xyz):,}")
    print(f"  - 位置范围: [{gaussian_data.xyz.min():.3f}, {gaussian_data.xyz.max():.3f}]")
    print(f"  - 尺度范围: [{gaussian_data.scale.min():.6f}, {gaussian_data.scale.max():.6f}]")
    print(f"  - 不透明度范围: [{gaussian_data.opacity.min():.3f}, {gaussian_data.opacity.max():.3f}]")
    
    # 计算原始文件大小
    original_size = input_path.stat().st_size
    print(f"\n原始文件大小: {original_size:,} 字节 ({original_size / 1024 / 1024:.2f} MB)")
    
    # 压缩并保存
    print(f"\n正在压缩并保存到: {output_path}")
    try:
        save_super_splat_ply(gaussian_data, str(output_path))
    except Exception as e:
        print(f"错误: 保存失败: {e}", file=sys.stderr)
        return 1
    
    # 计算压缩后文件大小
    compressed_size = output_path.stat().st_size
    compression_ratio = (1 - compressed_size / original_size) * 100
    
    print(f"\n压缩完成!")
    print(f"压缩后文件大小: {compressed_size:,} 字节 ({compressed_size / 1024 / 1024:.2f} MB)")
    print(f"压缩率: {compression_ratio:.2f}% (节省 {original_size - compressed_size:,} 字节)")
    print(f"压缩比: {original_size / compressed_size:.2f}x")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
