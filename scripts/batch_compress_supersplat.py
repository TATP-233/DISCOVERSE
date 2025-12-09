#!/usr/bin/env python3
"""
批量压缩目录中的所有3DGS PLY模型为SuperSplat格式

使用方法:
    python batch_compress_supersplat.py models/
    python batch_compress_supersplat.py --dir models/ --backup
"""

import argparse
import os
import sys
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
from functools import partial
import time

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from discoverse.gaussian_renderer.util_gau import load_ply
from discoverse.gaussian_renderer.super_splat_loader import save_super_splat_ply, is_super_splat_format
from plyfile import PlyData


def find_ply_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    查找目录中的所有PLY文件
    
    Args:
        directory: 搜索目录
        recursive: 是否递归搜索子目录
        
    Returns:
        PLY文件路径列表
    """
    if recursive:
        ply_files = list(directory.rglob("*.ply"))
    else:
        ply_files = list(directory.glob("*.ply"))
    
    return sorted(ply_files)


def check_supersplat_format(ply_path: Path) -> bool:
    """
    检查PLY文件是否已经是SuperSplat格式
    
    Args:
        ply_path: PLY文件路径
        
    Returns:
        True表示已经是SuperSplat格式
    """
    try:
        plydata = PlyData.read(str(ply_path))
        return is_super_splat_format(plydata)
    except Exception:
        return False


def compress_ply_file(
    input_path: Path,
    backup: bool = False,
    gamma: float = 1.0
) -> Tuple[bool, str, int, int, int, Path]:
    """
    压缩单个PLY文件
    
    Args:
        input_path: 输入文件路径
        backup: 是否备份原文件
        gamma: Gamma校正值
        
    Returns:
        (成功标志, 消息, 原始大小, 压缩后大小, 点数, 文件路径)
    """
    try:
        # 检查是否已经是SuperSplat格式
        if check_supersplat_format(input_path):
            return False, "已是SuperSplat格式", 0, 0, 0, input_path
        
        # 记录原始文件大小
        original_size = input_path.stat().st_size
        
        # 备份原文件
        backup_path = None
        if backup:
            backup_path = input_path.with_suffix('.ply.bak')
            shutil.copy2(input_path, backup_path)
        
        # 加载模型
        try:
            gaussian_data = load_ply(str(input_path), gamma=gamma)
            num_points = len(gaussian_data.xyz)
        except Exception as e:
            if backup_path and backup_path.exists():
                backup_path.unlink()
            return False, f"加载失败: {e}", 0, 0, 0, input_path
        
        # 压缩并保存到临时文件
        temp_path = input_path.with_suffix('.ply.tmp')
        try:
            save_super_splat_ply(gaussian_data, str(temp_path))
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            if backup_path and backup_path.exists():
                backup_path.unlink()
            return False, f"压缩失败: {e}", 0, 0, 0, input_path
        
        # 替换原文件
        compressed_size = temp_path.stat().st_size
        temp_path.replace(input_path)
        
        # 如果不需要备份，删除备份文件
        if not backup and backup_path and backup_path.exists():
            backup_path.unlink()
        
        return True, "压缩成功", original_size, compressed_size, num_points, input_path
        
    except Exception as e:
        return False, f"错误: {e}", 0, 0, 0, input_path


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(
        description='批量压缩目录中的所有3DGS PLY模型为SuperSplat格式（原位替换）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s models/
  %(prog)s --dir models/3dgs/ --backup
  %(prog)s models/ --no-recursive --gamma 1.2
  %(prog)s models/ -j 4  # 使用4个并行进程
  %(prog)s models/ -j 0  # 自动使用所有CPU核心
  
注意:
  - 此脚本会直接替换原文件
  - 使用 --backup 选项可以保留原文件备份（.ply.bak）
  - 已经是SuperSplat格式的文件会被跳过
  - 多进程模式可显著加快处理速度
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        nargs='?',
        default=None,
        help='要处理的目录路径'
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        default=None,
        help='要处理的目录路径（可替代位置参数）'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='备份原文件为 .ply.bak'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录'
    )
    
    parser.add_argument(
        '--gamma',
        type=float,
        default=1.0,
        help='Gamma校正值 (默认: 1.0)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅显示将要处理的文件，不实际执行压缩'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='跳过确认提示，直接执行'
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=1,
        help='并行处理的进程数（默认: 1，单进程）。使用 -j 0 自动检测CPU核心数'
    )
    
    args = parser.parse_args()
    
    # 确定并行进程数
    if args.jobs == 0:
        num_workers = mp.cpu_count()
    else:
        num_workers = max(1, args.jobs)
    
    # 确定目录路径
    directory = args.directory or args.dir
    if not directory:
        parser.print_help()
        print("\n错误: 请指定要处理的目录", file=sys.stderr)
        return 1
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"错误: 目录不存在: {directory}", file=sys.stderr)
        return 1
    
    if not dir_path.is_dir():
        print(f"错误: 不是一个目录: {directory}", file=sys.stderr)
        return 1
    
    # 查找所有PLY文件
    print(f"正在搜索PLY文件...")
    recursive = not args.no_recursive
    ply_files = find_ply_files(dir_path, recursive=recursive)
    
    if not ply_files:
        print(f"未找到任何PLY文件")
        return 0
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    
    # 过滤出需要压缩的文件
    files_to_compress = []
    files_skipped = []
    
    print("\n检查文件格式...")
    for i, ply_file in enumerate(ply_files, 1):
        rel_path = ply_file.relative_to(dir_path)
        print(f"  [{i}/{len(ply_files)}] {rel_path}", end=" ... ")
        
        if check_supersplat_format(ply_file):
            print("已是SuperSplat格式，跳过")
            files_skipped.append(ply_file)
        else:
            print("需要压缩")
            files_to_compress.append(ply_file)
    
    print(f"\n总结:")
    print(f"  - 需要压缩: {len(files_to_compress)} 个文件")
    print(f"  - 跳过: {len(files_skipped)} 个文件（已是SuperSplat格式）")
    
    if not files_to_compress:
        print("\n所有文件都已经是SuperSplat格式，无需处理")
        return 0
    
    # Dry run模式
    if args.dry_run:
        print("\n[DRY RUN] 将要压缩的文件:")
        for ply_file in files_to_compress:
            rel_path = ply_file.relative_to(dir_path)
            size = format_size(ply_file.stat().st_size)
            print(f"  - {rel_path} ({size})")
        return 0
    
    # 确认操作
    if not args.yes:
        print(f"\n警告: 此操作将原位替换 {len(files_to_compress)} 个文件")
        if args.backup:
            print("  原文件将备份为 .ply.bak")
        else:
            print("  原文件将被直接覆盖（不保留备份）")
        
        response = input("\n是否继续? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("操作已取消")
            return 0
    
    # 执行压缩
    print("\n开始批量压缩...")
    if num_workers > 1:
        print(f"使用 {num_workers} 个并行进程")
    print("=" * 80)
    
    success_count = 0
    failed_count = 0
    total_original_size = 0
    total_compressed_size = 0
    
    start_time = time.time()
    
    if num_workers > 1:
        # 多进程并行处理
        compress_func = partial(
            compress_ply_file,
            backup=args.backup,
            gamma=args.gamma
        )
        
        # 使用进程池
        with mp.Pool(processes=num_workers) as pool:
            results = []
            for ply_file in files_to_compress:
                result = pool.apply_async(compress_func, (ply_file,))
                results.append((ply_file, result))
            
            # 收集结果
            for i, (ply_file, result) in enumerate(results, 1):
                rel_path = ply_file.relative_to(dir_path)
                print(f"\n[{i}/{len(files_to_compress)}] {rel_path}")
                
                try:
                    success, message, orig_size, comp_size, num_points, _ = result.get(timeout=600)  # 10分钟超时
                    
                    if success:
                        success_count += 1
                        total_original_size += orig_size
                        total_compressed_size += comp_size
                        compression_ratio = (1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
                        print(f"  ✓ {message}")
                        print(f"    点数: {num_points:,}")
                        print(f"    原始大小: {format_size(orig_size)}")
                        print(f"    压缩后: {format_size(comp_size)}")
                        print(f"    压缩率: {compression_ratio:.1f}%")
                    else:
                        failed_count += 1
                        print(f"  ✗ {message}")
                except Exception as e:
                    failed_count += 1
                    print(f"  ✗ 处理失败: {e}")
    else:
        # 单进程顺序处理
        for i, ply_file in enumerate(files_to_compress, 1):
            rel_path = ply_file.relative_to(dir_path)
            print(f"\n[{i}/{len(files_to_compress)}] {rel_path}")
            
            success, message, orig_size, comp_size, num_points, _ = compress_ply_file(
                ply_file,
                backup=args.backup,
                gamma=args.gamma
            )
            
            if success:
                success_count += 1
                total_original_size += orig_size
                total_compressed_size += comp_size
                compression_ratio = (1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
                print(f"  ✓ {message}")
                print(f"    点数: {num_points:,}")
                print(f"    原始大小: {format_size(orig_size)}")
                print(f"    压缩后: {format_size(comp_size)}")
                print(f"    压缩率: {compression_ratio:.1f}%")
            else:
                failed_count += 1
                print(f"  ✗ {message}")
    
    elapsed_time = time.time() - start_time
    
    # 最终统计
    print("\n" + "=" * 80)
    print("批量压缩完成!")
    print(f"\n统计:")
    print(f"  - 成功: {success_count} 个文件")
    print(f"  - 失败: {failed_count} 个文件")
    print(f"  - 跳过: {len(files_skipped)} 个文件")
    print(f"  - 总耗时: {elapsed_time:.1f} 秒")
    
    if success_count > 0:
        avg_time = elapsed_time / success_count
        print(f"  - 平均处理时间: {avg_time:.2f} 秒/文件")
        
        total_saved = total_original_size - total_compressed_size
        overall_ratio = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        print(f"\n空间节省:")
        print(f"  - 原始总大小: {format_size(total_original_size)}")
        print(f"  - 压缩后总大小: {format_size(total_compressed_size)}")
        print(f"  - 节省空间: {format_size(total_saved)} ({overall_ratio:.1f}%)")
        
        if args.backup:
            print(f"\n备份文件保存为 .ply.bak，如确认无误可手动删除")
    
    return 0 if failed_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
