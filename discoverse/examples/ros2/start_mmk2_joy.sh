#!/bin/bash

# 获取任务名称参数
TASK_NAME=${1:-"task"}  # 如果没有提供参数，默认使用"task"

# 确保脚本以root权限运行
if [ "$EUID" -ne 0 ]; then
  echo "请使用sudo运行此脚本"
  exit 1
fi

# 获取当前用户名
CURRENT_USER=$(logname || echo $SUDO_USER)
echo "当前用户: $CURRENT_USER"

# 设置ROS2环境
source /opt/ros/jazzy/setup.bash

# 确保手柄设备有正确的权限
if [ -e /dev/input/js0 ]; then
  chmod a+rw /dev/input/js0
  echo "已设置手柄设备权限"
else
  echo "警告：未找到手柄设备 /dev/input/js0"
  echo "请确保手柄已连接"
fi

# 在后台启动joy节点
echo "启动ROS2 joy节点..."
ros2 run joy joy_node &
JOY_PID=$!

# 等待joy节点启动
sleep 2

# 检查joy话题是否可用
if ros2 topic list | grep -q "/joy"; then
  echo "Joy节点已成功启动"
else
  echo "警告：Joy节点可能未正确启动，但将继续尝试运行程序"
fi

# 运行MMK2手柄控制程序
echo "启动MMK2手柄控制程序，任务名称: $TASK_NAME..."
cd /home/xhz/DISCOVERSE
sudo -u $CURRENT_USER /home/xhz/anaconda3/envs/imitall/bin/python /home/xhz/DISCOVERSE/discoverse/examples/ros2/mmk2_joy_ros2.py "$TASK_NAME"

# 程序结束后，清理joy节点
echo "程序已退出，正在清理..."
kill $JOY_PID 2>/dev/null

echo "完成"
