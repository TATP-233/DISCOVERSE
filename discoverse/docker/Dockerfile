# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV PYTHONPATH=/workspace

# Install system dependencies and OpenGL dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    git \
    curl \
    software-properties-common \
    libgl1-mesa-dev \
    libglew-dev \ 
    libegl1-mesa-dev \ 
    libgles2-mesa-dev \ 
    libnvidia-egl-wayland1 \
    libosmesa6-dev \
    xvfb \ 
    ffmpeg \ 
    libx11-6 \ 
    libxext6 \ 
    libglfw3-dev \ 
    libglu1-mesa-dev \ 
    libglm-dev \
    pkg-config \ 
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA to install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa

# Install Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure using Python 3.9
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

# Verify pip installation
RUN python3 -m pip --version

# Create working directory
WORKDIR /workspace

# Install base Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch dependencies
RUN python3 -m pip install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# Install base dependencies, numpy1.24.4 supports python3.8-3.11
RUN python3 -m pip install --no-cache-dir \
    numpy==1.24.4\
    scipy \
    mediapy \
    opencv-python \
    mujoco \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install 3DGS related dependencies
RUN python3 -m pip install --no-cache-dir \
    plyfile \
    PyGlm \
    torch>=2.0.0 \
    torchvision>=0.14.0 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set CUDA architecture (supports RTX 30/40 series)
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Copy submodule code
COPY submodules/diff-gaussian-rasterization /workspace/submodules/diff-gaussian-rasterization

# Install diff-gaussian-rasterization
WORKDIR /workspace/submodules/diff-gaussian-rasterization
RUN pip install .

# Return to working directory
WORKDIR /workspace

# Create version check script
COPY <<'EOF' /usr/local/bin/check-versions
#!/bin/bash
echo "=== System Environment Check ==="
echo "Python Version:"
python3 --version
echo -e "\nCUDA Version:"
nvcc --version
echo -e "\nGPU Information:"
nvidia-smi
echo -e "\n=== Python Package Version Check ==="
python3 -c "import torch; print(f'\nPyTorch Information:\n- PyTorch Version: {torch.__version__}\n- CUDA Available: {torch.cuda.is_available()}\n- CUDA Version: {torch.version.cuda}\n- Current Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"}')"
python3 -c "import torchvision; print(f'Torchvision Version: {torchvision.__version__}')"
python3 -c "import numpy; print(f'Numpy Version: {numpy.__version__}')"
python3 -c "import cv2; print(f'OpenCV Version: {cv2.__version__}')"
python3 -c "import mujoco; print(f'Mujoco Version: {mujoco.__version__}')"
EOF

RUN chmod +x /usr/local/bin/check-versions 

RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
   echo '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Add health check
HEALTHCHECK CMD python3 -c "import discoverse, torch, numpy, cv2, mujoco; print('All dependencies installed successfully')" 