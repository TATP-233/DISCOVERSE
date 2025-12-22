# GaussianSteamer

GaussianSteamer 是一个高性能的 3D Gaussian Splatting 实时流媒体传输系统原型。

本项目的核心目标是实现**极低延迟**的云端渲染与推流。为此，我们实现了一套基于 **NVIDIA VideoProcessingFramework (VPF)** 的**真·零拷贝 (True Zero-Copy)** 编码管线，允许渲染后的 CUDA Tensor 直接在 GPU 显存内传输给 NVENC 硬件编码器，完全避免了昂贵的 Device-to-Host (GPU->CPU) 内存拷贝。

## ✨ 特性

*   **零拷贝编码**: 使用 NVIDIA VPF 直接处理 PyTorch CUDA Tensors，无需 CPU 参与数据搬运。
*   **高性能**: 在 RTX 30/40 系列显卡上可实现亚毫秒级 (sub-millisecond) 的编码延迟。
*   **双模式支持**:
    *   **GPU 模式 (推荐)**: 依赖 VPF，极致性能。
    *   **CPU 兼容模式**: 依赖 PyAV (FFmpeg)，用于无 NVIDIA 显卡或环境未配置时的回退方案。

## 🛠️ 环境配置

本项目依赖复杂的底层库（特别是 VPF），请严格按照以下步骤配置。

### 1. 基础环境

推荐使用 Conda 管理环境：

```bash
conda create -n gaussian_streamer python=3.10
conda activate gaussian_streamer

# 安装 PyTorch (需带 CUDA 支持) 根据你的 CUDA 版本选择
pip install torch
```

### 2. 安装依赖

```bash
pip install numpy av
```

### 3. 安装 NVIDIA VideoProcessingFramework (VPF)

这是实现零拷贝编码的关键。**VPF 通常需要从源码编译**，且对环境敏感。

**前置要求**:
*   NVIDIA Driver
*   CUDA Toolkit (推荐 11.x 或 12.x)
*   FFmpeg 库 (libavcodec, libavformat, libavutil 等)
*   CMake, C++ 编译器

**编译步骤 (参考)**:

本操作步骤仅供参考，推荐阅读[原仓库](https://github.com/NVIDIA/VideoProcessingFramework.git)进行环境配置。

1.  **克隆 VPF 仓库**:
    ```bash
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git
    cd VideoProcessingFramework
    ```

2.  **处理 ABI 兼容性 (关键)**:
    PyTorch 通常使用旧版 C++ ABI (`_GLIBCXX_USE_CXX11_ABI=0`) 编译。为了让 VPF 生成的扩展能被 PyTorch 加载，编译时必须指定相同的 ABI 标志。
    
    ```bash
    export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" 
    ```

3.  **安装 PytorchNvCodec 扩展**:
    ```bash
    pip install .
    ```
    *注意：如果遇到 `nvcc` 未找到错误，请确保 `/usr/local/cuda/bin` 在你的 `PATH` 中。*

4.  **验证安装**:
    ```bash
    python -c "import PyNvCodec; import PytorchNvCodec; print('VPF Installed Successfully')"
    ```
