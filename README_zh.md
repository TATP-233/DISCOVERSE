# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

<div align="center">

[![论文](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://air-discoverse.github.io/)
[![网站](https://img.shields.io/badge/Website-DISCOVERSE-blue.svg)](https://air-discoverse.github.io/)
[![许可证](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](doc/docker.md)

https://github.com/user-attachments/assets/78893813-d3fd-48a1-8bb4-5b0d87bf900f

*基于3DGS的统一、模块化、开源Real2Sim2Real机器人学习仿真框架*

</div>

## 🌟 一. 核心特性

### 🎯 **高保真Real2Sim生成**
- **分层场景重建**：支持背景环境和交互物体的分层重建
- **先进激光扫描集成**：集成LiDAR传感器进行精确几何捕获
- **AI驱动3D生成**：使用最先进的生成模型
- **基于物理的重新光照**：实现逼真的外观匹配
- **网格-高斯转换技术**：实现无缝资产集成

### ⚡ **卓越性能与效率**
- **650 FPS渲染**：5个相机RGB-D输出（比ORBIT/Isaac Lab快3倍）
- **大规模并行仿真**：GPU加速
- **实时3D高斯散射**：渲染引擎
- **MuJoCo物理集成**：精确接触动力学
- **优化CUDA内核**：最大吞吐量

### 🔧 **通用兼容性与灵活性**
- **多格式资产支持**：3DGS (.ply), 网格 (.obj/.stl), MJCF (.xml)
- **多样化机器人平台**：机械臂、移动操作臂、四旋翼、人形机器人
- **多种传感器模态**：RGB、深度、LiDAR、IMU、触觉传感器
- **ROS2集成**：无缝真实世界部署
- **全面随机化**：包括基于生成的域适应

### 🎓 **端到端学习管道**
- **自动化数据收集**：比真实世界效率提升100倍
- **多种学习算法**：ACT、Diffusion Policy、RDT等
- **零样本Sim2Real迁移**：最先进性能
- **模仿学习工作流**：从演示到部署

## 📦 二. 安装与快速开始

### 先决条件
- **Python 3.8+**
- **CUDA 11.8+**（用于3DGS渲染）
- **NVIDIA GPU**，推荐8GB+显存

### 🚀 快速开始

1. 克隆仓库（推荐按需下载submodules，不使用--recursive）
```bash
git clone https://github.com/TATP-233/DISCOVERSE.git
cd DISCOVERSE
```

2. 选择安装方式
```bash
conda create -n discoverse discoverse python=3.10 # >=3.8即可
conda activate discoverse
pip install -e .              # 仅核心功能（适合于快速上手，推荐）
pip install -e ".[lidar]"     # 激光雷达仿真
pip install -e ".[act_full]"  # 模仿学习算法act, 可替换成[dp_full] [rdt_full]
pip install -e ".[full]"      # 完整功能（不推荐）
```

3. 按需下载submodules（根据安装的功能模块）
```bash
python setup_submodules.py        # 自动检测并下载需要的submodules
# python setup_submodules.py --module lidar act  # 手动指定模块
# python setup_submodules.py --all  # 下载所有submodules
```
> 💡 **按需下载的优势**:
> - ⚡ **下载速度快**: 只下载需要的模块，减少90%下载时间
> - 💾 **节省空间**: 避免下载不需要的大型依赖（如ComfyUI约2GB）
> - 🎯 **精准安装**: 根据实际使用的功能模块智能下载

4. 验证安装
```bash
python check_installation.py
```

5. 更新资产

方式1: Git LFS（推荐）

项目的模型文件通过Git LFS进行版本管理，确保获得最新版本：

```bash
# 安装Git LFS (如果尚未安装)
## Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

## macos 使用Homebrew Homebrew
brew install git-lfs

git lfs install

# 在仓库中拉取LFS文件
git lfs pull
```

方式2: 手动下载

如果Git LFS下载过慢，可从以下地址手动下载，网盘更新可能有延迟：
- [百度网盘](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) 
- [清华云盘](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/)

解压到`models/`目录：
```
models/
├── meshes/          # 网格几何
├── textures/        # 材质纹理  
├── 3dgs/           # 高斯散射模型
│   ├── airbot_play/
│   ├── mmk2/
│   ├── objects/
│   └── scenes/
├── mjcf/           # MuJoCo场景描述
└── urdf/           # 机器人描述
```

### 🎯 按需求选择安装

#### 场景1: 学习机器人仿真基础
```bash
pip install -e .  # 仅核心功能
```
**包含**: MuJoCo、OpenCV、NumPy等基础依赖

#### 场景2: 激光雷达SLAM研究
```bash
pip install -e ".[lidar,visualization]"
```
- **包含**: Taichi GPU加速、LiDAR仿真、可视化工具
- **功能**: 高性能LiDAR仿真，基于Taichi GPU加速
- **依赖**: `taichi>=1.6.0`
- **适用**: 移动机器人SLAM、激光雷达传感器仿真、点云处理

#### 场景3: 机械臂模仿学习
```bash
pip install -e ".[act_full]"
```
- **包含**: ACT算法、数据收集工具、可视化
- **功能**: 模仿学习、机器人技能训练、策略优化
- **依赖**: `torch`, `einops`, `h5py`, `transformers`, `wandb`
- **算法**：其他算法可选[diffusion-policy]和[rdt]"

#### 场景4: 高保真视觉仿真
```bash
pip install -e ".[gaussian-rendering]"
```
- **包含**: 3D高斯散射、PyTorch
- **功能**: 逼真的3D场景渲染，支持实时光照
- **依赖**: `torch>=2.0.0`, `torchvision>=0.14.0`, `plyfile`, `PyGlm`
- **适用**: 高保真视觉仿真、3D场景重建、Real2Sim流程

#### 场景6: 数据处理与增强工具箱 📊
```bash
pip install -e ".[data-collection]"  # 数据收集
pip install -e ".[randomain]"        # 数据增强和AI生成
pip install -e ".[visualization]"    # 可视化工具
```
- **功能**: 数据集构建、域随机化

#### 场景7: 硬件集成 🔌
```bash
pip install -e ".[realsense]"    # RealSense相机支持
pip install -e ".[ros]"          # ROS集成
pip install -e ".[hardware]"     # 硬件集成套件
```
- **功能**: 真实机器人控制、硬件在环仿真、Sim2Real迁移

#### 场景8. XML场景编辑器 🖥️
```bash
pip install -e ".[xml-editor]"
```
- **功能**: 图形化MuJoCo场景编辑工具
- **依赖**: `PyQt5>=5.15.0`, `PyOpenGL>=3.1.0`
- **适用**: 可视化场景设计、MJCF文件编辑、3D模型调整

#### 场景9: 完整研究环境（不推荐，建议根据自身需求安装）
```bash
pip install -e ".[full]"
```
- **包含**: 所有功能模块

### 🔍 安装验证

#### 检查安装状态
```bash
python check_installation.py           # 基础检查
python check_installation.py --verbose # 详细信息
```

#### 输出示例
```
🔍 DISCOVERSE 安装状态检查
============================================================
Python版本: 3.10.16

==================================================
DISCOVERSE 核心模块
==================================================
✓ DISCOVERSE核心 ✓ 环境模块 ✓ 机器人模块 ✓ 工具模块

==================================================
可选功能模块  
==================================================
✓ 激光雷达仿真 (2/2)
✓ 3D高斯散射渲染 (3/3)
○ XML场景编辑器 (1/2)
✓ 策略学习 (5/5)

💡 要安装缺失的功能，请使用以下命令：
   pip install -e ".[xml-editor]"  # XML场景编辑器
```

### 📊 模块功能速览

| 模块 | 安装命令 | 功能 | 适用场景 |
|------|----------|------|----------|
| **基础** | `pip install -e .` | 核心仿真功能 | 学习、基础开发 |
| **激光雷达** | `.[lidar]` | 高性能LiDAR仿真 | SLAM、导航研究 |
| **渲染** | `.[gaussian-rendering]` | 3D高斯散射渲染 | 视觉仿真、Real2Sim |
| **GUI** | `.[xml-editor]` | 可视化场景编辑 | 场景设计、模型调试 |
| **ACT** | `.[act]` | 模仿学习算法 | 机器人技能学习 |
| **扩散策略** | `.[diffusion-policy]` | 扩散模型策略 | 复杂策略学习 |
| **RDT** | `.[rdt]` | 大模型策略 | 通用机器人技能 |
| **硬件集成** | `.[hardware]` | RealSense+ROS | 真实机器人控制 |

## 🐳 三. Docker快速开始

开始使用DISCOVERSE的最快方式：

```bash
# 下载预构建Docker镜像
# 百度网盘：https://pan.baidu.com/s/1mLC3Hz-m78Y6qFhurwb8VQ?pwd=xmp9

- 下载预构建Docker镜像
  
    百度网盘：https://pan.baidu.com/s/1mLC3Hz-m78Y6qFhurwb8VQ?pwd=xmp9
    
    目前更新至v1.8.6，下载.tar文件之后，使用docker load指令加载docker image
    
    将下面的`discoverse_tag.tar`替换为实际下载的镜像tar文件名。

    ```bash
    docker load < discoverse_tag.tar
    ```

- 或者 从`docker file`构建
    ```bash
    git clone https://github.com/TATP-233/DISCOVERSE.git
    cd DISCOVERSE
    python scripts/setup_submodules.py --module gaussian-rendering
    docker build -f discoverse/docker/Dockerfile -t discoverse:latest .
    ```
    `Dockerfile.vnc`是支持 VNC 远程访问的配置版本。它在`discoverse/docker/Dockerfile`的基础上添加了 VNC 服务器支持，允许你通过 VNC 客户端远程访问容器的图形界面。这对于远程开发或在没有本地显示服务器的环境中特别有用。如果需要，将`docker build -f discoverse/docker/Dockerfile -t discoverse:latest .`改为`docker build -f discoverse/docker/Dockerfile.vnc -t discoverse:latest .`


#### 3. 创建Docker容器

# 使用GPU支持运行
docker run -it --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    discoverse:latest
```

详细的Docker设置请参见我们的[Docker部署指南](doc/docker.md)。


## 📷 四. 高保真渲染设置

用于高保真3DGS渲染功能，若无高保真渲染需求，可跳过这一章节。

### 1. CUDA安装
从[NVIDIA官网](https://developer.nvidia.com/cuda-toolkit-archive)安装CUDA 11.8+。

### 2. 3DGS依赖
```bash
# 安装gaussian splatting依赖
pip install -e ".[gaussian-rendering]"

# 构建diff-gaussian-rasterization
cd submodules/diff-gaussian-rasterization/
git checkout 8829d14

# 应用必要补丁
sed -i 's/(p_view.z <= 0.2f)/(p_view.z <= 0.01f)/' cuda_rasterizer/auxiliary.h
sed -i '361s/D += depths\[collected_id\[j\]\] \* alpha \* T;/if (depths[collected_id[j]] < 50.0f)\n        D += depths[collected_id[j]] * alpha * T;/' cuda_rasterizer/forward.cu

# 安装
cd ../..
pip install submodules/diff-gaussian-rasterization
```

### 3. 模型可视化
使用[SuperSplat](https://playcanvas.com/supersplat/editor)在线查看3DGS模型 - 只需拖放`.ply`文件。

## 🔨 Real2Sim管道

<img src="./assets/real2sim.jpg" alt="Real2Sim管道"/>

DISCOVERSE具有全面的Real2Sim管道，用于创建真实环境的数字孪生。详细说明请访问我们的[Real2Sim仓库](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim)。

## 💡 五. 使用示例

### 基础机器人仿真
```bash
# 启动Airbot Play机械臂
python3 discoverse/robots_env/airbot_play_base.py

# 运行操作任务
python3 discoverse/examples/tasks_airbot_play/block_place.py
python3 discoverse/examples/tasks_airbot_play/coffeecup_place.py
python3 discoverse/examples/tasks_airbot_play/cuplid_cover.py
python3 discoverse/examples/tasks_airbot_play/drawer_open.py
```

https://github.com/user-attachments/assets/6d80119a-31e1-4ddf-9af5-ee28e949ea81

### 高级应用

#### 主动SLAM
```bash
python3 discoverse/examples/active_slam/dummy_robot.py
```
<img src="./assets/active_slam.jpg" alt="主动SLAM" style="zoom: 33%;" />

#### 多智能体协作
```bash
python3 discoverse/examples/skyrover_on_rm2car/skyrover_and_rm2car.py
```
<img src="./assets/skyrover.png" alt="多智能体协作" style="zoom: 50%;" />

### 交互式控制
- **'h'** - 显示帮助菜单
- **'F5'** - 重新加载MJCF场景
- **'r'** - 重置仿真状态
- **'['/'']'** - 切换相机视角
- **'Esc'** - 切换自由相机模式
- **'p'** - 打印机器人状态信息
- **'Ctrl+g'** - 切换高斯渲染（需安装gaussian-splatting并制定cfg.use_gaussian_renderer = False）
- **'Ctrl+d'** - 切换深度可视化

## 🎓 学习与训练

### 模仿学习快速开始

DISCOVERSE提供数据收集、训练和推理的完整工作流：

1. **数据收集**：[指南](./doc/imitation_learning/data.md)
2. **模型训练**：[指南](./doc/imitation_learning/training.md) 
3. **策略推理**：[指南](./doc/imitation_learning/inference.md)

### 支持的算法
- **ACT**（Action Chunking with Transformers）
- **Diffusion Policy** 
- **RDT**（Robotics Diffusion Transformer）
- **自定义算法**通过可扩展框架

### 域随机化
<div align="center">

https://github.com/user-attachments/assets/848db380-557c-469d-b274-2c9addf0b6bb

*由生成模型驱动的高级图像随机化*
</div>

DISCOVERSE集成了最先进的随机化技术，包括：
- **生成式图像合成**用于多样化视觉条件
- **物理参数随机化**用于鲁棒策略
- **光照和材质变化**用于逼真适应

详细实现请参见我们的[随机化指南](doc/Randomain.md)。

## 🏆 性能基准

DISCOVERSE展示了卓越的Sim2Real迁移性能：

| 方法 | 关闭笔记本 | 推动鼠标 | 拿起猕猴桃 | **平均** |
|--------|-------------|------------|-----------|-------------|
| MuJoCo | 2% | 48% | 8% | 19.3% |
| SAPIEN | 0% | 24% | 0% | 8.0% |
| SplatSim | 56% | 68% | 26% | 50.0% |
| **DISCOVERSE** | **66%** | **74%** | **48%** | **62.7%** |
| **DISCOVERSE + Aug** | **86%** | **90%** | **76%** | **84.0%** |

*使用ACT策略的零样本Sim2Real成功率*

## ⏩ 最近更新

- **2025.01.13**：🎉 DISCOVERSE开源发布
- **2025.01.16**：🐳 添加Docker支持
- **2025.01.14**：🏁 [S2R2025竞赛](https://sim2real.net/track/track?nav=S2R2025)启动
- **2025.02.17**：📈 集成Diffusion Policy基线
- **2025.02.19**：📡 添加点云传感器支持

## 🤝 社区与支持

### 获取帮助
- 📖 **文档**：`/doc`目录中的全面指南
- 💬 **问题**：通过GitHub Issues报告错误和请求功能
- 🔄 **讨论**：加入社区讨论进行问答和协作

### 贡献
我们欢迎贡献！请查看我们的贡献指南，加入我们不断壮大的机器人研究者和开发者社区。

<div align="center">
<img src="./assets/wechat.jpeg" alt="微信社区" style="zoom:50%;" />

*加入我们的微信社区获取更新和讨论*
</div>

## ❔ 故障排除

有关安装和运行时问题，请参考我们详细的**[故障排除指南](doc/troubleshooting.md)**。

## ⚖️ 许可证

DISCOVERSE在[MIT许可证](LICENSE)下发布。详细信息请参见许可证文件。

## 📜 引用

如果您发现DISCOVERSE对您的研究有帮助，请考虑引用我们的工作：

```bibtex
@misc{discoverse2024,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haizhou Ge and Kairui Ding and Zike Yan and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      url={https://air-discoverse.github.io/},
      year={2024}
}
```

---

<div align="center">

**DISCOVERSE** - *为下一代机器人技术弥合仿真与现实的差距*

[🌐 网站](https://air-discoverse.github.io/) | [📄 论文](https://air-discoverse.github.io/) | [🐳 Docker](doc/docker.md) | [📚 文档](doc/) | [🏆 竞赛](https://sim2real.net/track/track?nav=S2R2025)

</div> 