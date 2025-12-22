#!/usr/bin/env python3
"""
上传 3DGS 模型到 Hugging Face

使用方法:
    python scripts/upload_to_huggingface.py
    
环境变量:
    HF_TOKEN - Hugging Face 访问令牌
    HF_REPO_ID - 仓库 ID (默认: tatp/DISCOVERSE-models)
    HF_ENDPOINT - HF 端点 (默认使用国内镜像: https://hf-mirror.com)
    USE_MIRROR - 是否使用镜像 (默认: true)
    HF_HUB_DOWNLOAD_TIMEOUT - 下载/上传超时秒数 (默认: 1800 = 30分钟)
    HF_HUB_ETAG_TIMEOUT - ETag 检查超时秒数 (默认: 30)
"""

import os
import sys
from pathlib import Path
import httpx
from huggingface_hub import HfApi, upload_folder, create_repo
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils._http import set_client_factory

# 设置超时环境变量 (30分钟 = 1800秒)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")  # 30分钟
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")  # 30秒

# 配置 httpx client 工厂函数，设置30分钟超时
def custom_client_factory():
    """创建带自定义超时的 httpx.Client"""
    timeout = httpx.Timeout(
        connect=60.0,    # 连接超时 60秒
        read=1800.0,     # 读取超时 30分钟
        write=1800.0,    # 写入超时 30分钟
        pool=60.0        # 连接池超时 60秒
    )
    return httpx.Client(timeout=timeout)

# 设置自定义 client 工厂
set_client_factory(custom_client_factory)

# 配置国内镜像
USE_MIRROR = os.getenv("USE_MIRROR", "true").lower() in ("true", "1", "yes")
if USE_MIRROR and not os.getenv("HF_ENDPOINT"):
    # 使用国内镜像加速
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("🌐 使用 HF-Mirror 国内镜像加速")
    print(f"   端点: {os.environ['HF_ENDPOINT']}")
    print(f"   超时: {os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}秒 (30分钟)")
    print("   如需使用官方源，请设置: export USE_MIRROR=false\n")

# 配置
REPO_ID = os.getenv("HF_REPO_ID", "tatp/DISCOVERSE-models")
DISCOVERSE_ROOT = Path(__file__).parent.parent
MODELS_DIR = DISCOVERSE_ROOT / "models" / "3dgs"

def check_token():
    """检查 HF Token 是否配置"""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ 错误: 未找到 HF_TOKEN 环境变量")
        print("\n请设置 token:")
        print("  export HF_TOKEN=your_token_here")
        print("\n或使用 huggingface-cli login 登录")
        sys.exit(1)
    return token

def ensure_repo_exists(repo_id, token):
    """确保仓库存在，不存在则创建"""
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"✓ 仓库已存在: {repo_id}")
        return True
    except Exception:
        print(f"📦 仓库不存在，准备创建: {repo_id}")
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,
                token=token
            )
            print(f"✓ 仓库创建成功!")
            return True
        except Exception as e:
            print(f"❌ 仓库创建失败: {e}")
            return False

def get_remote_files(repo_id, token):
    """获取远程仓库 3dgs 目录下的文件信息"""
    api = HfApi(token=token)
    remote_files = {}
    try:
        print(f"🔍 正在获取远程仓库文件列表: {repo_id}...")
        # 递归列出 3dgs 目录下的所有文件
        tree = api.list_repo_tree(repo_id=repo_id, path_in_repo="3dgs", repo_type="model", recursive=True)
        for item in tree:
            # 仅处理文件对象 (RepoFile)，跳过目录对象 (RepoFolder)
            if isinstance(item, RepoFile) and item.path.startswith("3dgs/"):
                # 存储相对于 3dgs/ 的路径
                rel_path = os.path.relpath(item.path, "3dgs")
                remote_files[rel_path] = {
                    'size': item.size,
                    'blob_id': item.blob_id
                }
        print(f"✓ 远程仓库共有 {len(remote_files)} 个模型文件")
        return remote_files
    except Exception as e:
        # 如果仓库不存在或目录不存在，返回空字典
        print(f"ℹ️  无法获取远程列表 (可能是新仓库或网络问题): {e}")
        return {}

def get_file_size_mb(path):
    """获取文件大小（MB）"""
    return path.stat().st_size / (1024 * 1024)

def list_files_to_upload(models_dir, remote_files):
    """列出所有要上传的文件，并与远程进行比较"""
    all_files = list(models_dir.rglob("*.ply"))
    
    to_upload_paths = []
    new_count = 0
    mod_count = 0
    unchanged_count = 0
    total_upload_size = 0
    
    by_category = {}
    
    for f in all_files:
        rel_path = str(f.relative_to(models_dir))
        size_mb = get_file_size_mb(f)
        local_size = f.stat().st_size
        
        status = "unchanged"
        if rel_path in remote_files:
            # 简单通过大小判断是否变化，更严谨可以比对 hash
            if remote_files[rel_path]['size'] != local_size:
                status = "modified"
                mod_count += 1
            else:
                unchanged_count += 1
        else:
            status = "new"
            new_count += 1
            
        if status != "unchanged":
            to_upload_paths.append(rel_path)
            total_upload_size += size_mb
            
            # 分类统计 (仅统计待上传的)
            category = str(Path(rel_path).parts[0]) if len(Path(rel_path).parts) > 1 else "other"
            if category not in by_category:
                by_category[category] = {"count": 0, "size": 0, "new": 0, "mod": 0}
            by_category[category]["count"] += 1
            by_category[category]["size"] += size_mb
            if status == "new": by_category[category]["new"] += 1
            else: by_category[category]["mod"] += 1
    
    print(f"\n📋 同步状态报告:")
    print("=" * 80)
    if to_upload_paths:
        print(f"待上传文件统计 (按类别):")
        for category, info in sorted(by_category.items()):
            print(f"  📁 {category:20s} {info['count']:3d} 个 (新:{info['new']}, 改:{info['mod']})  {info['size']:8.2f} MB")
        print("-" * 80)
    
    print(f"总计本地文件: {len(all_files)} 个")
    print(f"  - ✨ 未变更: {unchanged_count}")
    print(f"  - 🚀 待上传: {len(to_upload_paths)} 个 ({total_upload_size:.2f} MB)")
    print("=" * 80)
    
    return to_upload_paths, total_upload_size

def main():
    """主函数"""
    print("🚀 DISCOVERSE 3DGS 模型上传工具 (增量更新模式)")
    print("=" * 80)
    
    # 检查 token
    token = check_token()
    print(f"✓ HF Token 已配置")
    
    # 检查模型目录
    if not MODELS_DIR.exists():
        print(f"❌ 错误: 模型目录不存在: {MODELS_DIR}")
        sys.exit(1)
    
    print(f"📁 本地模型目录: {MODELS_DIR}")
    
    # 确保仓库存在
    if not ensure_repo_exists(REPO_ID, token):
        sys.exit(1)

    # 获取远程文件信息
    remote_files = get_remote_files(REPO_ID, token)
    
    # 列出并比较文件
    to_upload, total_size = list_files_to_upload(MODELS_DIR, remote_files)
    
    if not to_upload:
        print("\n✨ 所有文件已是最新，无需上传。")
        sys.exit(0)
    
    # 确认上传
    print(f"\n📤 准备上传到仓库: {REPO_ID}")
    print(f"📊 上传路径: 3dgs/")
    print(f"\n⚠️  注意:")
    print(f"  - 将仅上传 {len(to_upload)} 个新增或修改的文件")
    print(f"  - 大文件会使用 Git LFS 存储")
    
    response = input("\n是否继续? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ 取消上传")
        sys.exit(0)
    
    # 上传文件夹
    print(f"\n⬆️  开始上传...")
    print(f"提示: 上传过程中可能看起来没有进度，请耐心等待")
    print("=" * 80)
    
    try:
        # 使用环境变量控制的超时 (HF_HUB_DOWNLOAD_TIMEOUT)
        api = HfApi(
            token=token,
            library_name="discoverse",
            library_version="1.0.0"
        )
        
        print(f"⏱️  超时设置: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '10')}秒")
        print(f"📊 开始上传 {len(to_upload)} 个文件，请耐心等待...\n")
        
        api.upload_folder(
            folder_path=str(MODELS_DIR),
            path_in_repo="3dgs",
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=to_upload,  # 关键：只上传变更的文件
            commit_message=f"Update {len(to_upload)} 3DGS models",
            ignore_patterns=[
                "*.pyc",
                "__pycache__",
                ".DS_Store",
                ".git",
                ".gitignore",
                ".cache",
                "*.lock"
            ]
        )
        
        print("\n" + "=" * 80)
        print("✅ 上传成功!")
        print(f"\n🔗 查看仓库: https://huggingface.co/{REPO_ID}")
        print(f"📁 文件路径: https://huggingface.co/{REPO_ID}/tree/main/3dgs")
        print("\n现在可以测试自动下载功能:")
        print(f"  python examples/test_3dgs_download.py")
        
    except Exception as e:
        print(f"\n❌ 上传失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题 - 尝试使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("  2. Token 权限不足（需要 write 权限）")
        print("  3. 仓库不存在或无访问权限")
        print("  4. 文件过大超过限制 - 建议分批上传或使用 Git LFS")
        print("  5. 超时 - 尝试增加超时时间或使用国内镜像")
        print("\n💡 解决方案:")
        print("  - 使用 Git 方法: bash scripts/upload_to_huggingface.sh")
        print("  - 使用镜像: export HF_ENDPOINT=https://hf-mirror.com")
        print("  - 分批上传: 手动上传部分目录")
        print("\n详细错误:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
