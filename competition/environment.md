# 开发环境说明

本文件记录 DG-202612“面向物品识别与搬运的文旅机器人关键技术研究”项目的已验证开发环境和复现步骤。

## 已验证环境

| 项目 | 当前版本 / 状态 |
| --- | --- |
| 操作系统 | Linux（需要可用的图形桌面/X11 以显示 MuJoCo 窗口） |
| Conda 环境 | `discoverse` |
| Python | 3.10.20 |
| DISCOVERSE | 1.9.0，本地可编辑安装 |
| MuJoCo | 3.11.0 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti（已由安装检查识别） |
| 3DGS、LiDAR、ROS、RealSense | 已安装并通过基础安装检查 |

完整的可复现依赖清单见同目录的 [environment.yml](environment.yml)。该文件由当前已验证的 Conda 环境导出，供 Linux 开发机创建同类环境使用。

## 新成员安装

```bash
git clone --recursive https://github.com/lwhhhh13/tourism-robot2026.git
cd tourism-robot2026

conda env create -f competition/environment.yml
conda activate discoverse

# 使 Python 使用当前克隆的源码，而不是已发布的同名包
python -m pip install --no-build-isolation -e .
```

如果仓库不是以 `--recursive` 克隆，补充初始化子模块：

```bash
git submodule update --init --recursive
```

## 基础验证

依次执行下列命令。每条命令均应能启动 MuJoCo 窗口；关闭窗口后再执行下一条。

```bash
python scripts/check_installation.py
python discoverse/robots_env/mmk2_base.py
python examples/tasks_mmk2/box_pick.py
python examples/tasks_mmk2/kiwi_place.py
```

后三个仿真脚本用于验证 MMK2 基础仿真、抓取和放置能力，是比赛任务开发的最低环境验收标准。

## 高保真 3DGS 渲染（可选）

部分示例会从 Hugging Face 下载 3DGS 模型。首次使用前执行：

```bash
hf auth login
```

登录时只需创建并使用具有 `read` 权限的 Hugging Face Access Token。不要把 Token、`.env` 文件或下载后的模型权重提交到 Git。

## 常见说明

- `competition/recordings/`、`data/`、`models/3dgs/` 为本地生成或下载的大文件，已被 `.gitignore` 排除。
- `.idea/` 是个人 PyCharm/IntelliJ 配置，已被忽略。
- 若 `pynput` 在无图形桌面的终端中检测失败，通常是 X11/`DISPLAY` 不可用；这不影响不使用手柄的基础仿真。
- 训练、3DGS 或正式比赛接口变动后如新增依赖，应重新导出 `competition/environment.yml`，并在本文件中记录变更原因。
