#!/usr/bin/env python3
"""
DISCOVERSE 安装验证脚本

该脚本检查DISCOVERSE各个功能模块的安装状态，
帮助用户快速诊断安装问题。

使用方法:
    python check_installation.py [--verbose]
"""

import sys
import importlib
import argparse
from typing import List, Tuple, Optional

def check_module(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, str]:
    """
    检查模块是否可以导入
    
    Args:
        module_name: 要检查的模块名
        package_name: 显示名称（如果与模块名不同）
    
    Returns:
        (是否成功, 错误信息或版本信息)
    """
    display_name = package_name or module_name
    try:
        module = importlib.import_module(module_name)
        
        # 尝试获取版本信息
        version_attrs = ['__version__', 'version', 'VERSION']
        version = None
        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                break
        
        if version:
            return True, f"{display_name} v{version}"
        else:
            return True, f"{display_name} (版本未知)"
            
    except ImportError as e:
        return False, f"{display_name}: {str(e)}"
    except Exception as e:
        return False, f"{display_name}: 导入错误 - {str(e)}"

def check_core_dependencies() -> List[Tuple[str, bool, str]]:
    """检查核心依赖"""
    core_deps = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"), 
        ("cv2", "OpenCV"),
        ("mujoco", "MuJoCo"),
        ("psutil", "PSUtil"),
        ("screeninfo", "ScreenInfo"),
        ("mediapy", "MediaPy"),
        ("tqdm", "TQDM"),
    ]
    
    results = []
    for module, name in core_deps:
        success, info = check_module(module, name)
        results.append((name, success, info))
    
    return results

def check_optional_dependencies() -> dict:
    """检查可选依赖模块"""
    optional_modules = {
        "激光雷达仿真": [
            ("taichi", "Taichi"),
            ("matplotlib", "Matplotlib"),
            ("pynput", "PyNput"),
        ],
        "3D高斯散射渲染": [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("plyfile", "PLYFile"),
        ],
        "XML场景编辑器": [
            ("PyQt5", "PyQt5"),
            ("OpenGL", "PyOpenGL"),
        ],
        "策略学习": [
            ("torch", "PyTorch"),
            ("einops", "Einops"),
            ("h5py", "H5Py"),
            ("omegaconf", "OmegaConf"),
            ("hydra", "Hydra"),
        ],
        "RealSense支持": [
            ("pyrealsense2", "PyRealSense2"),
        ],
        "ROS支持": [
            ("rospkg", "ROSPkg"),
        ],
        "数据增强": [
            ("transformers", "Transformers"),
            ("PIL", "Pillow"),
        ],
        "可视化": [
            ("matplotlib", "Matplotlib"),
            ("imageio", "ImageIO"),
        ],
    }
    
    results = {}
    for category, modules in optional_modules.items():
        category_results = []
        for module, name in modules:
            success, info = check_module(module, name)
            category_results.append((name, success, info))
        results[category] = category_results
    
    return results

def check_discoverse_modules() -> List[Tuple[str, bool, str]]:
    """检查DISCOVERSE自身模块"""
    discoverse_modules = [
        ("discoverse", "DISCOVERSE核心"),
        ("discoverse.envs", "环境模块"),
        ("discoverse.robots", "机器人模块"),
        ("discoverse.utils", "工具模块"),
    ]
    
    results = []
    for module, name in discoverse_modules:
        success, info = check_module(module, name)
        results.append((name, success, info))
    
    return results

def check_submodules() -> Tuple[int, int, List[str]]:
    """检查submodules状态"""
    from pathlib import Path
    
    submodule_mapping = {
        'gaussian-rendering': ['submodules/diff-gaussian-rasterization'],
        'randomain': ['submodules/ComfyUI'],
        'act': ['policies/act'],
        'lidar': ['submodules/MuJoCo-LiDAR'],
        'rdt': ['submodules/lerobot'],
        'diffusion-policy': ['submodules/lerobot'],
        'xml-editor': ['submodules/XML-Editor'],
    }
    
    all_submodules = set()
    for submodules in submodule_mapping.values():
        all_submodules.update(submodules)
    
    initialized_count = 0
    missing_for_features = []
    
    for submodule in all_submodules:
        submodule_path = Path(submodule)
        if submodule_path.exists() and len(list(submodule_path.iterdir())) > 0:
            initialized_count += 1
        else:
            # Find which features need this submodule
            for feature, feature_subs in submodule_mapping.items():
                if submodule in feature_subs and feature not in missing_for_features:
                    missing_for_features.append(feature)
    
    return initialized_count, len(all_submodules), missing_for_features

def check_gpu_support() -> Tuple[bool, str]:
    """检查GPU支持"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "未知"
            return True, f"检测到 {gpu_count} 个GPU: {gpu_name}"
        else:
            return False, "CUDA不可用，将使用CPU模式"
    except ImportError:
        return False, "PyTorch未安装，无法检查GPU支持"

def print_results(title: str, results: List[Tuple[str, bool, str]], verbose: bool = False):
    """打印检查结果"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    success_count = 0
    total_count = len(results)
    
    for name, success, info in results:
        if success:
            print(f"✓ {info}")
            success_count += 1
        else:
            print(f"✗ {info}")
            if verbose:
                print(f"  建议: pip install -e \".[{name.lower()}]\"")
    
    print(f"\n状态: {success_count}/{total_count} 模块可用")

def print_category_results(results: dict, verbose: bool = False):
    """打印分类结果"""
    for category, category_results in results.items():
        success_count = sum(1 for _, success, _ in category_results if success)
        total_count = len(category_results)
        
        status = "✓" if success_count == total_count else "○" if success_count > 0 else "✗"
        print(f"\n{status} {category} ({success_count}/{total_count})")
        
        if verbose or success_count < total_count:
            for name, success, info in category_results:
                symbol = "  ✓" if success else "  ✗"
                print(f"{symbol} {info}")

def main():
    parser = argparse.ArgumentParser(description="检查DISCOVERSE安装状态")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="显示详细信息")
    args = parser.parse_args()
    
    print("🔍 DISCOVERSE 安装状态检查")
    print("="*60)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("⚠️  警告: 建议使用Python 3.8或更高版本")
    
    # 检查DISCOVERSE核心模块
    discoverse_results = check_discoverse_modules()
    print_results("DISCOVERSE 核心模块", discoverse_results, args.verbose)
    
    # 检查核心依赖
    core_results = check_core_dependencies()
    print_results("核心依赖", core_results, args.verbose)
    
    # 检查可选依赖
    optional_results = check_optional_dependencies()
    print(f"\n{'='*50}")
    print("可选功能模块")
    print(f"{'='*50}")
    print_category_results(optional_results, args.verbose)
    
    # 检查GPU支持
    gpu_available, gpu_info = check_gpu_support()
    print(f"\n{'='*50}")
    print("GPU支持")
    print(f"{'='*50}")
    symbol = "✓" if gpu_available else "○"
    print(f"{symbol} {gpu_info}")
    
    # 检查Submodules
    initialized_count, total_count, missing_features = check_submodules()
    print(f"\n{'='*50}")
    print("Submodules状态")
    print(f"{'='*50}")
    
    if initialized_count == total_count:
        print(f"✓ 所有submodules已初始化 ({initialized_count}/{total_count})")
    else:
        print(f"○ 部分submodules未初始化 ({initialized_count}/{total_count})")
        if missing_features:
            print(f"📦 缺失功能模块的submodules: {', '.join(missing_features)}")
            print(f"💡 运行以下命令来按需下载:")
            print(f"   python scripts/setup_submodules.py --module {' '.join(missing_features)}")
    
    # 生成安装建议
    print(f"\n{'='*50}")
    print("安装建议")
    print(f"{'='*50}")
    
    # 统计各模块可用性
    module_status = {}
    for category, category_results in optional_results.items():
        available = all(success for _, success, _ in category_results)
        module_status[category] = available
    
    if all(module_status.values()):
        print("🎉 所有功能模块都已正确安装！")
    else:
        print("💡 要安装缺失的功能，请使用以下命令：")
        
        missing_modules = [cat for cat, avail in module_status.items() if not avail]
    
        install_map = {
            "激光雷达仿真": "lidar",
            "3D高斯散射渲染": "gaussian-rendering", 
            "XML场景编辑器": "xml-editor",
            "策略学习": "ml",
            "RealSense支持": "realsense",
            "ROS支持": "ros",
            "数据增强": "randomain",
            "可视化": "visualization",
        }
        
        for module in missing_modules:
            if module in install_map:
                print(f"   pip install -e \".[{install_map[module]}]\"  # {module}")
    
    print(f"\n📖 详细安装指南请参考: README_zh.md")
    print(f"🐛 遇到问题请访问: https://github.com/TATP-233/DISCOVERSE/issues")

if __name__ == "__main__":
    main() 