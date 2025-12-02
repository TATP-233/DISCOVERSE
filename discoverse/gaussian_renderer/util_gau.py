import numpy as np
from plyfile import PlyData
import torch
import glm
import os
from pathlib import Path
from .gaussiandata import GaussianData
from .super_splat_loader import is_super_splat_format, load_super_splat_ply
import sys

def multiple_quaternion_vector3d(qwxyz, vxyz):
    qw = qwxyz[..., 0]
    qx = qwxyz[..., 1]
    qy = qwxyz[..., 2]
    qz = qwxyz[..., 3]
    vx = vxyz[..., 0]
    vy = vxyz[..., 1]
    vz = vxyz[..., 2]
    qvw = -vx*qx - vy*qy - vz*qz
    qvx =  vx*qw - vy*qz + vz*qy
    qvy =  vx*qz + vy*qw - vz*qx
    qvz = -vx*qy + vy*qx + vz*qw
    vx_ =  qvx*qw - qvw*qx + qvz*qy - qvy*qz
    vy_ =  qvy*qw - qvz*qx - qvw*qy + qvx*qz
    vz_ =  qvz*qw + qvy*qx - qvx*qy - qvw*qz
    return torch.stack([vx_, vy_, vz_], dim=-1).cuda().requires_grad_(False)

def multiple_quaternions(qwxyz1, qwxyz2):
    q1w = qwxyz1[..., 0]
    q1x = qwxyz1[..., 1]
    q1y = qwxyz1[..., 2]
    q1z = qwxyz1[..., 3]

    q2w = qwxyz2[..., 0]
    q2x = qwxyz2[..., 1]
    q2y = qwxyz2[..., 2]
    q2z = qwxyz2[..., 3]

    qw_ = q1w * q2w - q1x * q2x - q1y * q2y - q1z * q2z
    qx_ = q1w * q2x + q1x * q2w + q1y * q2z - q1z * q2y
    qy_ = q1w * q2y - q1x * q2z + q1y * q2w + q1z * q2x
    qz_ = q1w * q2z + q1x * q2y - q1y * q2x + q1z * q2w

    return torch.stack([qw_, qx_, qy_, qz_], dim=-1).cuda().requires_grad_(False)

class Camera:
    def __init__(self, h, w):
        self.znear = 1e-6
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = 1.05
        self.position = np.array([0.0, 0.0, -2.0]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, 0.0, 1.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0

        self.is_pose_dirty = True
        self.is_intrin_dirty = True
        
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True
        
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        
        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.
    
    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self, backend="glm"):
        view_matrix = np.array(glm.lookAt(glm.vec3(self.position), glm.vec3(self.target), glm.vec3(self.up)))
        if backend == "gsplat":
            view_matrix[[1,2], :] *= -1
        elif backend == "diff_gaussian":
            view_matrix[[0,2], :] *= -1
        return view_matrix

    def get_project_matrix(self):
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        
    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True
    
    def update_fovy(self, fovy):
        self.fovy = fovy
        self.is_intrin_dirty = True

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
