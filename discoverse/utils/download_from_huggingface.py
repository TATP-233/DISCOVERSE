import os
import sys
from discoverse import DISCOVERSE_ASSETS_DIR

def download_from_huggingface(model_path, hf_repo_id="tatp/DISCOVERSE-models"):
    """
    从Hugging Face下载3DGS模型文件到本地models目录
    
    Args:
        model_path: 模型的相对路径（例如: "scene/lab3/point_cloud.ply"）
        hf_repo_id: Hugging Face仓库ID
    
    Returns:
        str: 下载后的文件本地路径
    """
    check_hf_login_or_exit()

    try:
        from huggingface_hub import hf_hub_download
        
        print(f"正在从Hugging Face下载模型: {model_path}")
        
        # 构建完整的HF文件路径（3dgs/相对路径）
        hf_file_path = f"3dgs/{model_path}"

        # 确定本地目录
        # hf_hub_download 会自动保持 3dgs 目录结构，所以不需要额外指定 3dgs 子目录
        local_dir = DISCOVERSE_ASSETS_DIR
        local_file_path = os.path.join(local_dir, hf_file_path)
        
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
            print("[WARN] 下载文件位置与预期不符，正在调整文件位置...")
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
