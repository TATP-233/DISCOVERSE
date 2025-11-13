FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
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
    libx11-6 \ 
    libxext6 \ 
    libglfw3-dev \ 
    libglu1-mesa-dev \ 
    libglm-dev \
    pkg-config \
    --no-install-recommends && \
    apt-get clean

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade setuptools pip build

COPY . /root/DISCOVERSE

# Install base dependencies
RUN python3 -m pip install --no-cache-dir \
    scipy \
    mediapy \
    opencv-python \
    mujoco -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python3 -m pip install --no-cache-dir torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install 3DGS related dependencies
RUN python3 -m pip install --no-cache-dir \
    plyfile \
    PyGlm -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install diff-gaussian-rasterization
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"
WORKDIR /root/DISCOVERSE/submodules/diff-gaussian-rasterization
RUN git checkout 8829d14
RUN sed -i 's/(p_view.z <= 0.2f)/(p_view.z <= 0.01f)/' cuda_rasterizer/auxiliary.h
RUN sed -i '361s/D += depths\[collected_id\[j\]\] \* alpha \* T;/if (depths[collected_id[j]] < 50.0f)\n        D += depths[collected_id[j]] * alpha * T;/' cuda_rasterizer/forward.cu
RUN python3 -m pip install .

WORKDIR /root/DISCOVERSE
RUN python3 -m pip install -e .

# Fix Rendering Issue
ENV MUJOCO_GL=glfw
RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
   echo '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get install -y \
    x11-apps \
    mesa-utils \
    vim

CMD ["/bin/bash"]