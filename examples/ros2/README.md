# ROS2 Jazzy 手柄控制说明

本文档介绍如何使用ROS2 Jazzy版本的手柄控制程序来操作MMK2机器人。

## 前提条件

1. 已安装ROS2 Jazzy
2. 已安装joy包和相关依赖
3. 已连接手柄设备（如Xbox或PS4控制器）
4. python 3.10及以上 [3.8不行（有坑）]

## 安装依赖

```bash
sudo apt update
sudo apt install -y ros-jazzy-joy ros-jazzy-joy-linux
sudo apt install -y python3-pip
pip3 install numpy scipy
```

## 使用方法

### 1. 启动手柄节点

首先，启动ROS2 joy节点，将手柄输入转换为ROS2消息：

```bash
sudo -s
source /opt/ros/jazzy/setup.bash
ros2 run joy joy_node
```

### 2. 运行MMK2手柄控制程序

```bash
cd /DISCOVERSE
python3 discoverse/examples/ros2/mmk2_joy_ros2.py
```

## 控制说明

手柄控制映射如下：

### 基本控制模式（不按L1/R1按钮）

- **左摇杆**：
  - 上/下：控制机器人前进/后退（上推前进，下拉后退）
  - 左/右：控制机器人旋转（左推左转，右推右转）

- **右摇杆**：
  - 上/下：控制头部上下转动（上推头部上抬，下拉头部下垂）
  - 左/右：控制头部左右转动（左推头部左转，右推头部右转）


### 手臂控制模式（要一直按着L1或R1）

- **L1按钮**：单独按下激活左臂控制模式
- **R1按钮**：单独按下激活右臂控制模式

在手臂控制模式下：
- **左摇杆**：
  - 上/下：控制手臂前后移动（上推手臂前伸，下拉手臂后缩）
  - 左/右：控制手臂左右移动（左推手臂左移，右推手臂右移）

- **右摇杆**：
  - 上/下：控制手臂上下移动（上推手臂上升，下拉手臂下降）
  - 左/右：控制手臂旋转（左推手臂左旋，右推手臂右旋）

- **MODE模式**：
- **左摇杆**：
  - 上/下：控制夹爪开合
- **右摇杆**：
    左/右：控制slide

- **按键O**：
  - 保存当前机器人状态到JSON文件

## 保存状态功能

按下键盘上的"O"键可以保存机器人当前状态：

1. 程序会列出场景中的所有物体
2. 输入物体编号选择目标物体
3. 选择要移动的机械臂（a/l/r）
4. 输入延迟时间（秒）
5. 状态将保存到以物体名称命名的JSON文件中

## 故障排除

1. 如果手柄未被识别，请检查设备连接和权限：
   ```bash
   ls -l /dev/input/js0
   sudo chmod a+rw /dev/input/js0
   ```

2. 检查ROS2 joy节点是否正常运行：
   ```bash
   ros2 topic echo /joy
   ```

3. 如果程序崩溃，可能是由于ROS2节点初始化问题，请确保先运行joy_node。
