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
from plyfile import PlyData
import os
from pathlib import Path
from .gaussiandata import GaussianData
from .super_splat_loader import is_super_splat_format, load_super_splat_ply
import sys


def gamma_shs(shs, gamma):
    C0 = 0.28209479177387814
    new_shs = ((np.clip(shs * C0 + 0.5, 0.0, 1.0) ** gamma) - 0.5) / C0
    return new_shs

def download_from_huggingface(model_path, hf_repo_id="tatp/DISCOVERSE-models", local_dir=None):
    """
    从Hugging Face下载3DGS模型文件到本地models目录
    
    Args:
        model_path: 模型的相对路径（例如: "scene/lab3/point_cloud.ply"）
        hf_repo_id: Hugging Face仓库ID
        local_dir: 本地目录，默认为None（使用DISCOVERSE_ASSETS_DIR/3dgs）
    
    Returns:
        str: 下载后的文件本地路径
    """
    try:
        from huggingface_hub import hf_hub_download
        from discoverse import DISCOVERSE_ASSETS_DIR
        
        print(f"正在从Hugging Face下载模型: {model_path}")
        
        # 构建完整的HF文件路径（3dgs/相对路径）
        hf_file_path = f"3dgs/{model_path}"

        # 确定本地目录
        if local_dir is None:
            # 如果未指定目录，使用 assets 根目录
            # hf_hub_download 会自动保持 3dgs 目录结构，所以不需要额外指定 3dgs 子目录
            local_dir = DISCOVERSE_ASSETS_DIR
            local_file_path = os.path.join(local_dir, hf_file_path)
        else:
            local_file_path = os.path.join(local_dir, model_path)
        
        # 确保本地目录存在
        os.makedirs(local_dir, exist_ok=True)
        local_file_dir = os.path.dirname(local_file_path)
        
        # 确保本地文件的目录存在
        os.makedirs(local_file_dir, exist_ok=True)
        
        # 下载文件（直接下载到目标位置，不使用HF缓存）
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_file_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            repo_type="model"
        )
        
        # 由于hf_hub_download会在local_dir下创建完整的仓库结构
        # 我们需要将文件移动到正确的位置
        if downloaded_path != local_file_path and os.path.exists(downloaded_path):
            # 检查下载的文件是否在预期位置
            expected_hf_path = os.path.join(local_dir, hf_file_path)
            if os.path.exists(expected_hf_path) and expected_hf_path != local_file_path:
                # 移动文件到正确位置
                import shutil
                shutil.move(expected_hf_path, local_file_path)
                print(f"文件已移动到: {local_file_path}")
                
                # 清理可能创建的空目录
                cleanup_dir = os.path.dirname(expected_hf_path)
                hf_3dgs_dir = os.path.join(local_dir, "3dgs")
                
                # 循环向上删除空目录，直到 3dgs 目录
                while cleanup_dir.startswith(hf_3dgs_dir):
                    try:
                        os.rmdir(cleanup_dir)
                    except OSError:
                        # 目录非空或无法删除，停止清理
                        break
                    
                    if cleanup_dir == hf_3dgs_dir:
                        break
                        
                    cleanup_dir = os.path.dirname(cleanup_dir)
            else:
                local_file_path = downloaded_path
        
        print(f"模型下载成功: {local_file_path}")
        return local_file_path
        
    except ImportError:
        print("错误: 需要安装 huggingface_hub 库")
        print("请运行: pip install huggingface_hub")
        raise

def check_hf_login_or_exit(verbose=True):
    """
    检查当前是否已登录 Hugging Face（huggingface_hub）。
    如果未安装 huggingface_hub，会提示安装并退出；如果未登录，会提示用户登录并安全退出。

    返回:
        True 如果已登录；否则不会返回（调用 sys.exit(1) 退出）。
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        if verbose:
            print("错误: 未安装 huggingface_hub。请运行: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    try:
        info = api.whoami()
        # whoami 返回字典，包含 'name' 或 'email' 等字段
        name = None
        if isinstance(info, dict):
            name = info.get('name') or info.get('email') or info.get('user')
        if verbose:
            print(f"已使用 Hugging Face 登录: {name}")
        return True
    except Exception as e:
        if verbose:
            print("检测到未登录 Hugging Face。请执行 `huggingface-cli login` 或 设置 环境变量 `HUGGINGFACE_HUB_TOKEN` 后重试。")
            print(f"(详细错误: {e})")
        sys.exit(1)
    except Exception as e:
        print(f"从Hugging Face下载模型失败: {e}")
        raise

def load_ply(path, gamma=1, hf_repo_id="tatp/DISCOVERSE-models", local_dir=None):
    """
    加载PLY格式的3DGS模型，支持从本地或Hugging Face下载
    
    Args:
        path: 模型文件路径（本地路径或相对路径）
        gamma: gamma校正值
        hf_repo_id: Hugging Face仓库ID，当本地文件不存在时使用
        local_dir: 下载目标目录，默认为None（使用DISCOVERSE_ASSETS_DIR/3dgs）
    
    Returns:
        GaussianData: 加载的高斯数据
    """
    # 转换为Path对象
    path = Path(path)
    
    # 检查本地文件是否存在
    if not path.exists():
        print(f"本地未找到模型文件: {path}")
        
        # 尝试从Hugging Face下载
        # 获取相对路径（假设路径格式为: .../3dgs/model_name/scene.ply）
        try:
            # 尝试提取相对路径
            path_parts = path.parts
            if "3dgs" in path_parts:
                idx = path_parts.index("3dgs")
                relative_path = "/".join(path_parts[idx+1:])
            else:
                # 如果路径中没有3dgs，使用文件名
                relative_path = path.name
            
            print(f"尝试使用相对路径从HF下载: {relative_path}")
            downloaded_path = download_from_huggingface(
                relative_path, 
                hf_repo_id=hf_repo_id,
                local_dir=local_dir
            )
            path = Path(downloaded_path)
            
        except Exception as e:
            print(f"从Hugging Face下载失败: {e}")
            raise FileNotFoundError(f"本地和远程都未找到模型文件: {path}")
    
    max_sh_degree = 0
    plydata = PlyData.read(path)

    # 检查是否为 SuperSplat 格式
    if is_super_splat_format(plydata):
        return load_super_splat_ply(plydata)
    
    # 标准 3DGS 格式
    else:
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))

        # assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
        features_extra = features_extra.reshape((features_extra.shape[0], 3, len(extra_f_names)//3))
        features_extra = features_extra[:, :, :(max_sh_degree + 1) ** 2 - 1]
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
        opacities = 1/(1 + np.exp(-opacities))
        opacities = opacities.astype(np.float32)

        if abs(gamma - 1.0) > 1e-3:
            features_dc = gamma_shs(features_dc, gamma)
            features_extra[...,:] = 0.0
            opacities *= 0.8

        shs = np.concatenate([features_dc.reshape(-1, 3), 
                            features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
        shs = shs.astype(np.float32)
        return GaussianData(xyz, rots, scales, opacities, shs)
