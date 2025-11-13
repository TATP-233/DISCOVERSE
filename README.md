# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://air-discoverse.github.io/)
[![Website](https://img.shields.io/badge/Website-DISCOVERSE-blue.svg)](https://air-discoverse.github.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](doc/docker.md)

https://github.com/user-attachments/assets/78893813-d3fd-48a1-8bb4-5b0d87bf900f

*A unified, modular, open-source 3DGS-based simulation framework for Real2Sim2Real robot learning*

</div>

[中文文档](README_zh.md)

## 🌟 I. Key Features

### 🎯 **High-Fidelity Real2Sim Generation**
- **Hierarchical scene reconstruction** for both background environments and interactive objects
- **Advanced laser-scanning integration** with LiDAR sensors for precise geometry capture
- **AI-powered 3D generation** using state-of-the-art generative models
- **Physically-based relighting** for photorealistic appearance matching
- **Mesh-Gaussian transfer** technology for seamless asset integration

### ⚡ **Exceptional Performance & Efficiency**
- **650 FPS rendering** for 5 cameras with RGB-D output (3× faster than ORBIT/Isaac Lab)
- **Massively parallel simulation** with GPU acceleration
- **Real-time 3D Gaussian Splatting** rendering engine
- **MuJoCo physics integration** for accurate contact dynamics
- **Optimized CUDA kernels** for maximum throughput

### 🔧 **Universal Compatibility & Flexibility**
- **Multi-format asset support**: 3DGS (.ply), Mesh (.obj/.stl), MJCF (.xml)
- **Diverse robot platforms**: Robotic arms, mobile manipulators, quadcopters, humanoids
- **Multiple sensor modalities**: RGB, Depth, LiDAR, IMU, tactile sensors
- **ROS2 integration** with seamless real-world deployment
- **Comprehensive randomization** including generative-based domain adaptation

### 🎓 **End-to-End Learning Pipeline**
- **Automated data collection** with 100× efficiency improvement over real-world
- **Multiple learning algorithms**: ACT, Diffusion Policy, RDT, and more
- **Zero-shot Sim2Real transfer** with state-of-the-art performance
- **Imitation learning workflows** from demonstration to deployment



## 📦 II. Installation & Quick Start

### Prerequisites
- **Python 3.8+**
- **CUDA 11.8+** (for 3DGS rendering)
- **NVIDIA GPU** with 8GB+ VRAM (recommended)

### 🚀 Quick Start

1. Clone repository (recommended: download submodules on-demand, don't use --recursive)
```bash
git clone https://github.com/TATP-233/DISCOVERSE.git
cd DISCOVERSE
```

2. Choose installation method
```bash
conda create -n discoverse discoverse python=3.10 # >=3.8 is ok
conda activate discoverse

pip install -e .              # Core only (recommended for quick start)
pip install -e ".[lidar]"     # LiDAR simulation
pip install -e ".[act_full]"  # Imitation learning with ACT, can replace with [dp_full] [rdt_full]
pip install -e ".[full]"      # Full features (not recommended)
```

3. Download submodules on-demand (based on installed feature modules)
```bash
python setup_submodules.py        # Auto-detect and download required submodules
# python setup_submodules.py --module lidar act  # Manually specify modules
# python setup_submodules.py --all  # Download all submodules
```

> 💡 **Advantages of On-Demand Download**:
> - ⚡ **Faster download**: Only download needed modules, reduce 90% download time
> - 💾 **Save space**: Avoid downloading unused large dependencies (e.g., ComfyUI ~2GB)
> - 🎯 **Precise installation**: Intelligently download based on actual feature modules used

4. Verify installation
```bash
python check_installation.py
```

5. Download Assets

Method 1: Git LFS (Recommended)

Project model files are managed through Git LFS for version control, ensuring you get the latest versions:

```bash
# Install Git LFS (if not already installed)
## Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

## macos use Homebrew
brew install git-lfs

git lfs install

# pull LFS files in existing repository
git lfs pull
```

Method 2: Manual Download

If Git LFS download is slow, manually download from:
- [Baidu Netdisk](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) 
- [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/)

Extract to the `models/` directory:
```
models/
├── meshes/          # Mesh geometries
├── textures/        # Material textures  
├── 3dgs/           # Gaussian Splatting models
│   ├── airbot_play/
│   ├── mmk2/
│   ├── objects/
│   └── scenes/
├── mjcf/           # MuJoCo scene descriptions
└── urdf/           # Robot descriptions
```

### 🎯 Installation by Use Case

#### Scenario 1: Learning Robot Simulation Basics
```bash
pip install -e .  # Core functionality only
```
**Includes**: MuJoCo, OpenCV, NumPy and other basic dependencies

#### Scenario 2: LiDAR SLAM Research
```bash
pip install -e ".[lidar,visualization]"
```
- **Includes**: Taichi GPU acceleration, LiDAR simulation, visualization tools
- **Function**: High-performance LiDAR simulation with Taichi GPU acceleration
- **Dependencies**: `taichi>=1.6.0`
- **Use Cases**: Mobile robot SLAM, LiDAR sensor simulation, point cloud processing

#### Scenario 3: Robotic Arm Imitation Learning
```bash
pip install -e ".[act_full]"
```
- **Includes**: ACT algorithm, data collection tools, visualization
- **Function**: Imitation learning, robot skill training, policy optimization
- **Dependencies**: `torch`, `einops`, `h5py`, `transformers`, `wandb`
- **Algorithms**: Other algorithms available with [diffusion-policy] and [rdt]

#### Scenario 4: High-Fidelity Visual Simulation
```bash
pip install -e ".[gaussian-rendering]"
```
- **Includes**: 3D Gaussian Splatting, PyTorch
- **Function**: Photorealistic 3D scene rendering with real-time lighting
- **Dependencies**: `torch>=2.0.0`, `torchvision>=0.14.0`, `plyfile`, `PyGlm`
- **Use Cases**: High-fidelity visual simulation, 3D scene reconstruction, Real2Sim pipeline

#### Scenario 6: Data Processing & Augmentation Toolkit 📊
```bash
pip install -e ".[data-collection]"  # Data collection
pip install -e ".[randomain]"        # Data augmentation and AI generation
pip install -e ".[visualization]"    # Visualization tools
```
- **Function**: Dataset construction, domain randomization

#### Scenario 7: Hardware Integration 🔌
```bash
pip install -e ".[realsense]"    # RealSense camera support
pip install -e ".[ros]"          # ROS integration
pip install -e ".[hardware]"     # Hardware integration suite
```
- **Function**: Real robot control, hardware-in-the-loop simulation, Sim2Real transfer

#### Scenario 8: XML Scene Editor 🖥️
```bash
pip install -e ".[xml-editor]"
```
- **Function**: Graphical MuJoCo scene editing tool
- **Dependencies**: `PyQt5>=5.15.0`, `PyOpenGL>=3.1.0`
- **Use Cases**: Visual scene design, MJCF file editing, 3D model adjustment

#### Scenario 9: Complete Research Environment (not recommended, install based on your needs)
```bash
pip install -e ".[full]"
```
- **Includes**: All feature modules



### 🔍 Installation Verification

#### Check Installation Status
```bash
python check_installation.py           # Basic check
python check_installation.py --verbose # Detailed information
```

#### Sample Output
```
🔍 DISCOVERSE Installation Status Check
============================================================
Python version: 3.10.16

==================================================
DISCOVERSE Core Modules
==================================================
✓ DISCOVERSE Core ✓ Environment Module ✓ Robot Module ✓ Utils Module

==================================================
Optional Feature Modules  
==================================================
✓ LiDAR Simulation (2/2)
✓ 3D Gaussian Splatting Rendering (3/3)
○ XML Scene Editor (1/2)
✓ Policy Learning (5/5)

💡 To install missing features, use these commands:
   pip install -e ".[xml-editor]"  # XML Scene Editor
```

### 📊 Module Feature Overview

| Module | Install Command | Function | Use Cases |
|--------|-----------------|----------|-----------|
| **Core** | `pip install -e .` | Core simulation | Learning, basic development |
| **LiDAR** | `.[lidar]` | High-performance LiDAR simulation | SLAM, navigation research |
| **Rendering** | `.[gaussian-rendering]` | 3D Gaussian Splatting rendering | Visual simulation, Real2Sim |
| **GUI** | `.[xml-editor]` | Visual scene editing | Scene design, model debugging |
| **ACT** | `.[act]` | Imitation learning algorithm | Robot skill learning |
| **Diffusion Policy** | `.[diffusion-policy]` | Diffusion model policy | Complex policy learning |
| **RDT** | `.[rdt]` | Large model policy | General robot skills |
| **Hardware Integration** | `.[hardware]` | RealSense+ROS | Real robot control |


## 🐳 III. Docker Quick Start

The fastest way to get started with DISCOVERSE:

    ```bash
    docker load < discoverse_tag.tar
    ```

- Or build from `Dockerfile`
    ```bash
    git clone https://github.com/TATP-233/DISCOVERSE.git
    cd DISCOVERSE
    python scripts/setup_submodules.py --module gaussian-rendering
    docker build -f discoverse/docker/Dockerfile -t discoverse:latest .
    ```
    `Dockerfile.vnc` is a configuration version that supports VNC remote access. It adds VNC server support on top of `discoverse/docker/Dockerfile`, allowing you to remotely access the container's graphical interface through a VNC client. This is particularly useful for remote development or environments without a local display server. If needed, change `docker build -f discoverse/docker/Dockerfile -t discoverse:latest .` to `docker build -f discoverse/docker/Dockerfile.vnc -t discoverse:latest .`

# Or build from source (recommended)
git clone https://github.com/TATP-233/DISCOVERSE.git
cd DISCOVERSE
python setup_submodules.py --all  # Docker image needs all submodules
docker build -t discoverse:latest .

# Run with GPU support
docker run -it --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace \
    discoverse:latest
```

For detailed Docker setup, see our [Docker deployment guide](doc/docker.md).


## 📷 IV. High-Fidelity Rendering Setup

For high-fidelity 3DGS rendering functionality, skip this section if you don't need high-fidelity rendering.

### 1. CUDA Installation
Install CUDA 11.8+ from [NVIDIA's official site](https://developer.nvidia.com/cuda-toolkit-archive).

### 2. 3DGS Dependencies
```bash
# Install Gaussian Splatting requirements
pip install -e ".[gaussian-rendering]"

# Build differential Gaussian rasterization
cd submodules/diff-gaussian-rasterization/
git checkout 8829d14

# Apply required patches
sed -i 's/(p_view.z <= 0.2f)/(p_view.z <= 0.01f)/' cuda_rasterizer/auxiliary.h
sed -i '361s/D += depths\[collected_id\[j\]\] \* alpha \* T;/if (depths[collected_id[j]] < 50.0f)\n        D += depths[collected_id[j]] * alpha * T;/' cuda_rasterizer/forward.cu

# Install
cd ../..
pip install submodules/diff-gaussian-rasterization
```

### 3. Model Visualization
View 3DGS models online using [SuperSplat](https://playcanvas.com/supersplat/editor) - simply drag and drop `.ply` files.

## 🔨 Real2Sim Pipeline

<img src="./assets/real2sim.jpg" alt="Real2Sim Pipeline"/>

DISCOVERSE features a comprehensive Real2Sim pipeline for creating digital twins of real environments. For detailed instructions, visit our [Real2Sim repository](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim).

## 💡 V. Usage Examples

### Basic Robot Simulation
```bash
# Launch Airbot Play robotic arm
python3 discoverse/robots_env/airbot_play_base.py

# Run manipulation tasks
python3 discoverse/examples/tasks_airbot_play/block_place.py
python3 discoverse/examples/tasks_airbot_play/coffeecup_place.py
python3 discoverse/examples/tasks_airbot_play/cuplid_cover.py
python3 discoverse/examples/tasks_airbot_play/drawer_open.py
```

https://github.com/user-attachments/assets/6d80119a-31e1-4ddf-9af5-ee28e949ea81

### Advanced Applications

#### Active SLAM
```bash
python3 discoverse/examples/active_slam/dummy_robot.py
```
<img src="./assets/active_slam.jpg" alt="Active SLAM" style="zoom: 33%;" />

#### Multi-Agent Coordination
```bash
python3 discoverse/examples/skyrover_on_rm2car/skyrover_and_rm2car.py
```
<img src="./assets/skyrover.png" alt="Multi-agent collaboration" style="zoom: 50%;" />

### Interactive Controls
- **'h'** - Show help menu
- **'F5'** - Reload MJCF scene
- **'r'** - Reset simulation state
- **'['/'']'** - Switch camera views
- **'Esc'** - Toggle free camera mode
- **'p'** - Print robot state information
- **'Ctrl+g'** - Toggle Gaussian rendering (requires gaussian-splatting installation and set cfg.use_gaussian_renderer = False)
- **'Ctrl+d'** - Toggle depth visualization

## 🎓 Learning & Training

### Imitation Learning Quickstart

DISCOVERSE provides complete workflows for data collection, training, and inference:

1. **Data Collection**: [Guide](./doc/imitation_learning/data.md)
2. **Model Training**: [Guide](./doc/imitation_learning/training.md) 
3. **Policy Inference**: [Guide](./doc/imitation_learning/inference.md)

### Supported Algorithms
- **ACT** (Action Chunking with Transformers)
- **Diffusion Policy** 
- **RDT** (Robotics Diffusion Transformer)
- **Custom algorithms** via extensible framework

### Domain Randomization
<div align="center">

https://github.com/user-attachments/assets/848db380-557c-469d-b274-2c9addf0b6bb

*Advanced image randomization powered by generative models*
</div>

DISCOVERSE incorporates state-of-the-art randomization techniques including:
- **Generative image synthesis** for diverse visual conditions
- **Physics parameter randomization** for robust policies
- **Lighting and material variations** for photorealistic adaptation

See our [randomization guide](doc/Randomain.md) for implementation details.

## 🏆 Performance Benchmarks

DISCOVERSE demonstrates superior Sim2Real transfer performance:

| Method | Close Laptop | Push Mouse | Pick Kiwi | **Average** |
|--------|-------------|------------|-----------|-------------|
| MuJoCo | 2% | 48% | 8% | 19.3% |
| SAPIEN | 0% | 24% | 0% | 8.0% |
| SplatSim | 56% | 68% | 26% | 50.0% |
| **DISCOVERSE** | **66%** | **74%** | **48%** | **62.7%** |
| **DISCOVERSE + Aug** | **86%** | **90%** | **76%** | **84.0%** |

*Zero-shot Sim2Real success rates using ACT policy*

## ⏩ Recent Updates

- **2025.01.13**: 🎉 DISCOVERSE open source release
- **2025.01.16**: 🐳 Docker support added
- **2025.01.14**: 🏁 [S2R2025 Competition](https://sim2real.net/track/track?nav=S2R2025) launched
- **2025.02.17**: 📈 Diffusion Policy baseline integration
- **2025.02.19**: 📡 Point cloud sensor support added

## 🤝 Community & Support

### Getting Help
- 📖 **Documentation**: Comprehensive guides in `/doc` directory
- 💬 **Issues**: Report bugs and request features via GitHub Issues
- 🔄 **Discussions**: Join community discussions for Q&A and collaboration

### Contributing
We welcome contributions! Please see our contributing guidelines and join our growing community of robotics researchers and developers.

<div align="center">
<img src="./assets/wechat.jpeg" alt="WeChat Community" style="zoom:50%;" />

*Join our WeChat community for updates and discussions*
</div>

## ❔ Troubleshooting

For installation and runtime issues, please refer to our comprehensive **[Troubleshooting Guide](doc/troubleshooting.md)**.

## ⚖️ License

DISCOVERSE is released under the [MIT License](LICENSE). See the license file for details.

## 📜 Citation

If you find DISCOVERSE helpful in your research, please consider citing our work:

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

**DISCOVERSE** - *Bridging the gap between simulation and reality for next-generation robotics*

[🌐 Website](https://air-discoverse.github.io/) | [📄 Paper](https://air-discoverse.github.io/) | [🐳 Docker](doc/docker.md) | [📚 Documentation](doc/) | [🏆 Competition](https://sim2real.net/track/track?nav=S2R2025)

</div>