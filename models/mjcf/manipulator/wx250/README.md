# WX250 机械臂可视化与点云生成工具

基于 [Interbotix WX250](https://www.trossenrobotics.com/widowx-250-robot-arm.aspx) 机械臂的 MuJoCo 仿真模型，通过指定关节角度和夹爪开合度，生成对应姿态的渲染图和表面点云。

## 文件结构

```
wx250/
├── wx250.xml              # MuJoCo MJCF 模型文件
├── wx250_visualize.py     # 主脚本
├── README.md              # 本文档
└── meshes/                # STL 网格文件 (来自 interbotix_ros_manipulators)
    ├── wx250_1_base.stl
    ├── wx250_2_shoulder.stl
    ├── wx250_3_upper_arm.stl
    ├── wx250_4_forearm.stl
    ├── wx250_5_wrist.stl
    ├── wx250_6_gripper.stl
    ├── wx250_7_gripper_prop.stl
    ├── wx250_8_gripper_bar.stl
    └── wx250_9_gripper_finger.stl
```

## 依赖

| 包 | 版本要求 | 用途 | 是否必须 |
|---|---------|------|---------|
| `numpy` | - | 数值计算 | 必须 |
| `mujoco` | >=3.0 | 模型加载、正运动学、渲染 | 必须 |
| `matplotlib` | - | 图像保存 (PNG) | 必须 |
| `trimesh` | - | PLY 导出 | 可选 (有纯文本 fallback) |

安装：

```bash
pip install mujoco numpy matplotlib
```

本工具与 DISCOVERSE 框架无耦合，可独立使用。

## 运动学参数

WX250 具有 5 个手臂旋转关节 + 1 个夹爪旋转关节 + 1 对棱柱手指关节：

| 序号 | 关节名 | 类型 | 旋转轴 | 范围 (rad) | 说明 |
|-----|--------|------|-------|-----------|------|
| 0 | `waist` | 旋转 | Z | [-3.14, 3.14] | 底座旋转 |
| 1 | `shoulder` | 旋转 | Y | [-1.88, 1.99] | 肩部俯仰 |
| 2 | `elbow` | 旋转 | Y | [-2.15, 1.61] | 肘部俯仰 |
| 3 | `wrist_angle` | 旋转 | Y | [-1.75, 2.15] | 腕部俯仰 |
| 4 | `wrist_rotate` | 旋转 | X | [-3.14, 3.14] | 腕部旋转 |
| 5 | `gripper` | 旋转 | X | 无限制 | 夹爪传动螺杆 |
| - | `left/right_finger` | 棱柱 | Y | [0.015, 0.037] m | 由 `--gripper` 参数控制 |

## 使用方法

### 基本用法

```bash
python wx250_visualize.py --joints <6个关节角> --gripper <开合度>
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--joints` | 6个浮点数 | `0 0 0 0 0 0` | 关节角度 (弧度制)，依次为 waist, shoulder, elbow, wrist_angle, wrist_rotate, gripper_rotate |
| `--gripper` | 浮点数 | `0.5` | 夹爪开合比例：`0.0`=完全闭合, `1.0`=完全张开 |
| `--density` | 浮点数 | `50` | 点云采样密度 (点/cm²) |
| `--output` | 路径 | 脚本所在目录 | 输出文件目录 |
| `--prefix` | 字符串 | `wx250` | 输出文件名前缀 |
| `--no-pointcloud` | 开关 | - | 跳过点云生成 |
| `--no-render` | 开关 | - | 跳过渲染图生成 |
| `--show` | 开关 | - | 启动 MuJoCo 交互式 3D 查看器 |

### 示例

```bash
# 使用一组实际关节角数据，夹爪闭合
python wx250_visualize.py \
  --joints 0.08 0.048 0.032 0.0015 1.569 0.0 \
  --gripper 0.0

# 手臂弯曲姿态，夹爪全开，高密度点云
python wx250_visualize.py \
  --joints 0 -0.5 0.8 0 1.57 0 \
  --gripper 1.0 \
  --density 100

# 仅渲染图像（不生成点云）
python wx250_visualize.py --joints 0 0 0 0 0 0 --no-pointcloud

# 仅生成点云（不渲染图像）
python wx250_visualize.py --joints 0 0 0 0 0 0 --no-render

# 打开交互式查看器
python wx250_visualize.py --joints 0 0.5 -0.3 0 0 0 --show
```

## 输出文件

运行后在输出目录生成：

| 文件 | 说明 |
|------|------|
| `{prefix}_pointcloud.ply` | 表面点云 (PLY 格式)，可用 MeshLab / CloudCompare / Open3D 查看 |
| `{prefix}_pointcloud.png` | 点云四视角投影图，按部件着色 |
| `{prefix}_render.png` | MuJoCo 渲染四视角图 (Front-Right, Front-Left, Front, Top-Down) |

## 点云生成原理

点云通过直接在 mesh 三角面上均匀采样生成，而非深度图反投影：

1. MuJoCo 加载 MJCF 模型并执行正运动学 (`mj_forward`)
2. 对每个 mesh geom，从 MuJoCo 内部获取变换后的顶点和面片数据
3. 计算每个三角面的面积，按面积比例分配采样点数
4. 在每个三角面内使用随机重心坐标均匀采样
5. 将所有部件的点合并为完整点云

采样密度由 `--density` 参数控制（单位：点/cm²）。默认 50 pts/cm² 约产生 12 万点。

> 注：`gripper_prop_link`（夹爪内部传动螺杆）在点云中被默认跳过。

## Mesh 来源

STL 文件来自 Interbotix 官方仓库：

```
https://github.com/Interbotix/interbotix_ros_manipulators/tree/main/
  interbotix_ros_xsarms/interbotix_xsarm_descriptions/meshes/wx250_meshes
```

MJCF 模型 (`wx250.xml`) 基于原始 [wx250.urdf.xacro](https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/wx250.urdf.xacro) 手工转换而来。
