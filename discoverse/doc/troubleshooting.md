# DISCOVERSE Troubleshooting Guide

This guide helps you resolve common issues when installing and using DISCOVERSE. Issues are organized by category for easy navigation.

## Table of Contents

- [Installation Issues](#installation-issues)
  - [CUDA and PyTorch](#cuda-and-pytorch)
  - [Dependencies](#dependencies)
  - [Submodules](#submodules)
- [Runtime Issues](#runtime-issues)
  - [Graphics and Display](#graphics-and-display)
  - [Video Recording](#video-recording)
  - [Server Deployment](#server-deployment)

---

## Installation Issues

### CUDA and PyTorch

#### 1. CUDA/PyTorch Version Mismatch

**Problem**: `diff-gaussian-rasterization` fails to install with error message about mismatched PyTorch and CUDA versions.

**Solution**: Install matching PyTorch version for your CUDA installation:

```bash
# For CUDA 11.8
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
```

> **Tip**: Check your CUDA version with `nvcc --version` or `nvidia-smi`

#### 2. Missing GLM Headers

**Problem**: Compilation error with missing `glm/glm.hpp` header file.

```
fatal error: glm/glm.hpp: no such file or directory
```

**Solution**: Install GLM library and update include path:

```bash
# Using conda (recommended)
conda install -c conda-forge glm
export CPATH=$CONDA_PREFIX/include:$CPATH

# Then reinstall diff-gaussian-rasterization
pip install submodules/diff-gaussian-rasterization
```

### Dependencies

#### 1. Taichi Installation Failure

**Problem**: Taichi fails to install or import properly.

**Solution**: Install specific Taichi version:

```bash
pip install taichi==1.6.0
```

#### 2. PyQt5 Installation Issues

**Problem**: PyQt5 installation fails or GUI components don't work.

**Solution**: Install system packages first:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-pyqt5 python3-pyqt5-dev

# Then install via pip
pip install PyQt5>=5.15.0
```

### Submodules

#### 1. Submodules Not Initialized

**Problem**: Missing submodule content or import errors from submodules.

**Solution**: Initialize submodules using one of these methods:

```bash
# Method 1: On-demand initialization (recommended)
python scripts/setup_submodules.py --list              # Check status
python scripts/setup_submodules.py --module lidar act  # Initialize specific modules
python scripts/setup_submodules.py --all               # Initialize all modules

# Method 2: Traditional Git approach
git submodule update --init --recursive
```

---

## Runtime Issues

### Graphics and Display

Graphics rendering issues in DISCOVERSE typically fall into three categories, each with different root causes and solutions.

#### 1. GLX Configuration Errors

**Problem**: GLFW/OpenGL initialization fails with GLX errors:

```
GLFWError: (65542) b'GLX: No GLXFBConfigs returned'
GLFWError: (65545) b'GLX: Failed to find a suitable GLXFBConfig'
```

**Root Cause**: X11/GLX configuration issues, often due to:
- Dual GPU systems (Intel + NVIDIA) with driver conflicts
- Missing or misconfigured X11 display server
- Incompatible GLX extensions

**Solutions**:

1. **For systems with NVIDIA GPU**: Check and configure graphics driver mode (dual GPU systems):
   ```bash
   # Check EGL vendor:
   eglinfo | grep "EGL vendor"
   
   # If output includes:
   libEGL warning: egl: failed to create dri2 screen
   It indicates a conflict between Intel and NVIDIA drivers.
   
   # Check current driver mode
   prime-select query
   
   # If output is `on-demand`, switch to `nvidia` mode, then reboot or relogin!
   sudo prime-select nvidia
   
   # Force NVIDIA usage
   export __NV_PRIME_RENDER_OFFLOAD=1
   export __GLX_VENDOR_LIBRARY_NAME=nvidia
   
   # Reboot system after switching
   sudo reboot
   ```
   
2. **For systems without NVIDIA GPU** (conda environments):
   
   **Root Cause**: Low version of libstdc++ in conda environment causing GLX compatibility issues.
   
   **Solution 1** - Install newer libstdc++ in conda environment:
   ```bash
   conda install -c conda-forge libstdcxx-ng
   ```

   **Solution 2** - Use system libstdc++ library:
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
   ```
   
3. **Verify GLX support**:
   ```bash
   glxinfo | grep "direct rendering"
   glxgears  # Test basic GLX functionality
   ```

5. **For X11 display issues**:
   
   ```bash
   # Ensure X11 forwarding (if using SSH)
   ssh -X username@hostname
   
   # Check DISPLAY variable
   echo $DISPLAY
   
   # TODO
   ```

#### 2. EGL Initialization Errors

**Problem**: EGL backend fails to initialize, especially in virtual/containerized environments:

```
libEGL warning: MESA-LOADER: failed to open virtio_gpu: /usr/lib/dri/virtio_gpu_dri.so: cannot open shared object file
libEGL warning: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file
GLFWError: (65542) b'EGL: Failed to initialize EGL: EGL is not or could not be initialized'
libGL error: failed to load driver: iris
libGL error: failed to load driver: swrast
```

**Root Cause**: Missing or incompatible Mesa drivers, particularly in:
- Virtual machines (VirtIO GPU driver issues)
- Docker containers without proper GPU passthrough
- ARM-based systems with incomplete driver installations
- Conda environments with conflicting OpenGL libraries

**Solutions**:

1. **Install Mesa drivers**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install mesa-utils libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev
   
   # For virtual environments, also install
   sudo apt-get install mesa-vulkan-drivers mesa-va-drivers
   ```

2. **For conda environment conflicts** (similar to GLX issues):
   
   **Root Cause**: Conda's OpenGL libraries and libstdc++ versions conflict with system Mesa drivers.
   
   **Solution 1** - Fix libstdc++ conflicts (recommended):
   ```bash
   # Step 1: Install latest gcc in conda environment
   conda install libgcc
   
   # Step 2: Check for duplicate libstdc++ files
   sudo find / -wholename "*conda*/**/libstdc++.so*"
   
   # Step 3: Remove conflicting libstdc++ files from conda environment
   # Replace 'your_env_name' with your actual environment name
   rm $CONDA_PREFIX/lib/libstdc++*
   
   # Alternative: Remove specific old versions if you see duplicates
   # rm $CONDA_PREFIX/lib/libstdc++.so.6.0.21  # Example old version
   ```
   
   > **Warning**: After removing libstdc++ files, you may occasionally see `free(): invalid pointer` messages when Python programs terminate. This is generally harmless but indicates library conflicts.
   
   **Solution 2** - Remove conda's conflicting OpenGL packages:
   ```bash
   conda remove --force mesa-libgl-cos6-x86_64 mesa-libgl-devel-cos6-x86_64
   ```
   
   **Solution 3** - Force system OpenGL libraries:
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1:/usr/lib/x86_64-linux-gnu/libEGL.so.1
   ```
   
   **Solution 4** - Install compatible Mesa in conda:
   ```bash
   conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 mesa-dri-drivers-cos7-x86_64
   ```

3. **For VirtIO GPU issues**:
   ```bash
   # Install VirtIO GPU drivers
   sudo apt-get install xserver-xorg-video-qxl
   
   # Or fall back to software rendering
   export LIBGL_ALWAYS_SOFTWARE=1
   ```

4. **Configure EGL for headless rendering**:
   ```bash
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   ```

> **References**: 
> - [Ask Ubuntu discussion](https://askubuntu.com/questions/1352158/libgl-error-failed-to-load-drivers-iris-and-swrast-in-ubuntu-20-04)
> - [StackOverflow libstdc++ solution](https://stackoverflow.com/questions/48453497/anaconda-libstdc-so-6-version-glibcxx-3-4-20-not-found)

#### 3. MuJoCo-Specific Rendering Issues

**Problem**: MuJoCo environments fail to render properly despite working graphics drivers.

**Root Cause**: MuJoCo's specific rendering backend requirements and conflicts with system OpenGL configurations.

**Solutions**:

1. **Set MuJoCo rendering backend**:
   ```bash
   # For headless servers
   export MUJOCO_GL=egl
   
   # For desktop environments with display issues
   export MUJOCO_GL=glfw
   
   # For software rendering (fallback)
   export MUJOCO_GL=osmesa
   ```

2. **Verify MuJoCo installation**:
   ```bash
   python -c "import mujoco; mujoco.MjModel.from_xml_string('<mujoco/>')"
   ```

3. **Test with simple MuJoCo example**:
   ```python
   import mujoco
   import mujoco.viewer
   
   # Simple test model
   xml = """
   <mujoco>
     <worldbody>
       <geom name="floor" type="plane" size="0 0 .05"/>
       <body name="box" pos="0 0 .2">
         <geom name="box" type="box" size=".1 .1 .1"/>
       </body>
     </worldbody>
   </mujoco>
   """
   
   model = mujoco.MjModel.from_xml_string(xml)
   data = mujoco.MjData(model)
   
   # Test rendering
   with mujoco.viewer.launch_passive(model, data) as viewer:
       mujoco.mj_step(model, data)
   ```

> **Reference**: Similar issues reported in [Gymnasium Issue #755](https://github.com/Farama-Foundation/Gymnasium/issues/755#issuecomment-2825928509)

### Video Recording

#### 1. FFmpeg Video Encoding Errors

**Problem**: Video recording fails during task execution with FFmpeg parameter errors:

```
BrokenPipeError: [Errno 32] 断开的管道

RuntimeError: Error writing 'data/coffeecup_place/000/cam_0.mp4': Unrecognized option 'qp'.
Error splitting the argument list: Option not found
```

**Root Cause**: This error occurs when `mediapy` library attempts to write MP4 video files using FFmpeg with incompatible or unrecognized encoding parameters. The issue typically stems from:
- Outdated FFmpeg version that doesn't support the 'qp' (quality parameter) option
- Conflicting FFmpeg installations (system vs conda)
- Missing codec libraries in FFmpeg build

**Solutions**:

1. **Update FFmpeg to latest version**:
   ```bash
   # For conda environments (recommended)
   conda install -c conda-forge ffmpeg
   
   # For system-wide installation (Ubuntu/Debian)
   sudo apt update
   sudo apt install ffmpeg
   
   # Verify installation
   ffmpeg -version
   ```

2. **For conda environment conflicts**:
   ```bash
   # Remove existing FFmpeg installations
   conda remove ffmpeg
   
   # Install latest FFmpeg with full codec support
   conda install -c conda-forge ffmpeg=6.0
   
   # Verify codecs are available
   ffmpeg -codecs | grep h264
   ```

3. **Alternative: Downgrade mediapy to compatible version**:
   ```bash
   pip install mediapy==1.1.0
   ```

4. **Workaround: Use different video format**:
   
   If the issue persists, modify the video recording code to use AVI format instead of MP4:
   
   ```python
   # In airbot_task_base.py or similar files
   # Change from:
   # mediapy.write_video(os.path.join(save_path, f"cam_{id}.mp4"), [...])
   
   # To:
   mediapy.write_video(os.path.join(save_path, f"cam_{id}.avi"), [...], codec='mjpeg')
   ```

5. **For development environments**:
   ```bash
   # Install FFmpeg with specific codecs
   sudo apt install ffmpeg libx264-dev libx265-dev
   
   # Or use static FFmpeg build
   wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
   tar -xf ffmpeg-release-amd64-static.tar.xz
   sudo cp ffmpeg-*-static/ffmpeg /usr/local/bin/
   sudo cp ffmpeg-*-static/ffprobe /usr/local/bin/
   ```

> **Reference**: Similar issue reported in [nerfstudio #1138](https://github.com/nerfstudio-project/nerfstudio/issues/1138)

### Server Deployment

#### 1. Headless Server Setup

**Problem**: Running DISCOVERSE on a server without display.

**Solution**: Configure MuJoCo for headless rendering:

```bash
export MUJOCO_GL=egl
```

Add this to your shell profile (`.bashrc`, `.zshrc`) for permanent effect:

```bash
echo "export MUJOCO_GL=egl" >> ~/.bashrc
source ~/.bashrc
```

---

## Getting Help

If your issue isn't covered here:

1. **Search GitHub Issues**: Check [existing issues](https://github.com/TATP-233/DISCOVERSE/issues) for similar problems
2. **Create New Issue**: Provide detailed error messages and system information
3. **Community Support**: Join our WeChat community for real-time help
4. **Documentation**: Check the `/doc` directory for detailed guides

### Issue Report Template

When reporting issues, please include:

```
**System Information:**
- OS: (e.g., Ubuntu 22.04)
- Python version: 
- CUDA version: 
- GPU model: 

**Error Message:**
[Paste complete error trace here]

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**
[What should happen]

**Additional Context:**
[Any other relevant information]
```

---

> **Note**: This troubleshooting guide is actively maintained. If you find a solution to a problem not listed here, please consider contributing to help other users. 