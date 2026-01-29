# 3DGS转COLMAP数据集工具

## 1. 简介

`gs2colmap.py` 是一个将3D Gaussian Splatting (3DGS) PLY文件转换为COLMAP格式数据集的工具。它通过围绕物体几何中心旋转相机，自动生成多视角图像和标准的COLMAP重建数据格式。

### 1.1 主要功能

- **自动轨迹生成**：根据3DGS模型的几何中心和尺寸，自动生成围绕物体旋转的相机轨迹
- **多仰角支持**：支持从不同仰角（俯视、平视、仰视）拍摄，获得更全面的视角覆盖
- **高质量渲染**：使用gsplat渲染器生成高质量RGB图像
- **COLMAP格式输出**：生成标准的COLMAP稀疏重建格式（cameras.txt, images.txt, points3D.txt）
- **点云提取**：从3DGS高斯点中提取彩色点云，填充points3D.txt

### 1.2 应用场景

- **3DGS数据验证**：将3DGS模型转换回图像数据集，用于验证重建质量
- **数据集生成**：为下游任务（如NeRF训练、MVS重建）生成合成数据集
- **视角采样**：自动化地从3DGS模型采样多视角图像

## 2. 环境要求

### 2.1 依赖项

- CUDA Toolkit（需要nvcc编译器用于gsplat JIT编译）
- gsplat
- mujoco
- plyfile
- opencv-python
- scipy

### 2.2 CUDA环境配置

首次运行前，确保CUDA环境变量已正确设置：

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

可将以上命令添加到 `~/.bashrc` 以永久生效。

## 3. 使用方法

### 3.1 基础用法

```bash
python examples/active_slam/gs2colmap.py --gsply /path/to/model.ply
```

输出目录默认为PLY文件同目录下以PLY文件名命名的文件夹：
- 输入：`/data/3dgs/eggplant.ply`
- 输出：`/data/3dgs/eggplant/`

### 3.2 完整参数示例

```bash
python examples/active_slam/gs2colmap.py \
    --gsply /path/to/model.ply \
    --output /path/to/output \
    --num-views 72 \
    --elevation-angles "0,30,-30,60" \
    --radius-scale 1.5 \
    --fovy 60 \
    --width 1920 \
    --height 1080 \
    --max-points 100000
```

### 3.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gsply` | (必填) | 3DGS PLY文件路径 |
| `--output` | `{PLY目录}/{PLY名称}/` | 输出目录路径 |
| `--num-views` | 36 | 每个仰角的视角数量 |
| `--elevation-angles` | "0,30,-30" | 仰角列表（度），逗号分隔，正值表示相机在物体上方 |
| `--radius-scale` | 1.2 | 相机距离 = 物体尺寸 × 比例 |
| `--radius` | None | 固定相机距离（米），设置后忽略radius-scale |
| `--fovy` | 60.0 | 相机垂直视场角（度） |
| `--width` | 1280 | 输出图像宽度（像素） |
| `--height` | 720 | 输出图像高度（像素） |
| `--max-points` | 50000 | points3D最大点数，设为0则不生成点云 |
| `--show-viewer` | False | 是否显示渲染窗口 |

### 3.4 仰角说明

`--elevation-angles` 参数控制相机在垂直方向的位置：
- `0`：水平视角，相机与物体中心等高
- `30`：俯视30度，相机在物体上方
- `-30`：仰视30度，相机在物体下方

默认值 `"0,30,-30"` 会生成三层视角，每层36个视角，共108张图像。

## 4. 输出格式

### 4.1 目录结构

```
{output}/
├── images/                  # 渲染的RGB图像
│   ├── 000000.png          # 第1个视角
│   ├── 000001.png          # 第2个视角
│   └── ...
├── sparse/0/                # COLMAP稀疏重建格式
│   ├── cameras.txt          # 相机内参
│   ├── images.txt           # 图像外参
│   └── points3D.txt         # 3D点云
└── camera_params.json       # JSON格式相机参数（便于程序读取）
```

### 4.2 COLMAP文件格式详解

#### cameras.txt - 相机内参

使用PINHOLE相机模型，包含焦距和主点：

```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 1280 720 623.538291 623.538291 640.000000 360.000000
```

参数含义：`fx fy cx cy`

#### images.txt - 图像外参

每张图像两行：第一行是外参，第二行是2D点（本工具输出为空）：

```
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 108
1 0.500000 0.500000 0.500000 0.500000 0.0 0.0 0.25 1 000000.png

2 0.454519 0.454519 0.541675 0.541675 0.1 0.0 0.25 1 000001.png

```

**注意**：COLMAP使用world-to-camera变换，四元数格式为(QW, QX, QY, QZ)，平移向量(TX, TY, TZ)表示世界原点在相机坐标系下的位置。

#### points3D.txt - 3D点云

从3DGS高斯点提取的彩色点云：

```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
# Number of points: 50000
1 -0.078756 0.012740 -0.014435 106 56 78 0
2 -0.060464 0.000155 -0.020272 75 25 62 0
```

颜色从3DGS的球谐系数DC分量计算得到。

### 4.3 camera_params.json 格式

```json
{
  "intrinsics": {
    "width": 1280,
    "height": 720,
    "fx": 623.538,
    "fy": 623.538,
    "cx": 640.0,
    "cy": 360.0,
    "fovy": 60.0
  },
  "images": [
    {
      "image_id": 1,
      "camera_id": 1,
      "name": "000000.png",
      "position": [0.252, 0.0, 0.0],
      "quaternion": [0.5, 0.5, 0.5, 0.5]
    },
    ...
  ]
}
```

**注意**：JSON中的位姿是camera-to-world变换（与COLMAP的images.txt相反），四元数格式为(W, X, Y, Z)。

## 5. 技术细节

### 5.1 相机轨迹生成

相机轨迹使用球坐标系生成：

1. 计算3DGS模型的几何中心和边界框
2. 根据边界框大小和radius_scale计算旋转半径
3. 对每个仰角，均匀采样num_views个方位角
4. 计算每个位置的相机朝向，使其始终指向物体中心

### 5.2 相机坐标系

本工具使用的相机坐标系约定：
- **Z轴**：指向相机前方（视线方向）
- **Y轴**：指向相机下方
- **X轴**：指向相机右方

### 5.3 点云提取

从3DGS PLY文件提取点云的过程：

1. 读取高斯点的xyz位置
2. 从球谐系数(f_dc_0, f_dc_1, f_dc_2)计算RGB颜色：
   ```
   color = SH_DC * C0 + 0.5  (C0 = 0.28209479)
   ```
3. 如果点数超过max_points，进行随机采样

## 6. 常见问题

### 6.1 gsplat报错"No CUDA toolkit found"

确保已安装CUDA Toolkit并正确设置环境变量：

```bash
# 检查nvcc是否可用
nvcc --version

# 如果找不到nvcc，设置CUDA路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### 6.2 渲染图像全黑

可能原因：
- 相机距离太远：尝试减小`--radius-scale`值
- 相机朝向错误：确认使用最新版本的脚本

### 6.3 物体在图像中太小

增大物体在画面中的占比：

```bash
# 减小radius-scale让相机更近
--radius-scale 1.0

# 或直接指定较小的半径
--radius 0.2
```

### 6.4 需要更多视角

增加视角数量或仰角层数：

```bash
# 每层72个视角，5个仰角层
--num-views 72 --elevation-angles "0,15,30,-15,-30"
```

## 7. 相关工具

- **camera_view.py**：交互式相机控制工具，支持手动设置关键帧和轨迹插值
- **stereo_camera_zh.md**：双目相机模拟与视角插值工具详细指南
