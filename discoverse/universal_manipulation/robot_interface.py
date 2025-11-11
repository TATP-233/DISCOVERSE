"""
通用机械臂接口

定义标准的机械臂操作接口，连接抽象原语和实际的机械臂控制。
"""

import time
import mujoco
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from .robot_config import RobotConfigLoader
from .mink_solver import MinkIKSolver
from .gripper_controller import create_gripper_controller

class RobotInterface(ABC):
    """通用机械臂接口基类"""
    
    def __init__(self, robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        """
        初始化机械臂接口
        
        Args:
            robot_config: 机械臂配置
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
        """
        self.robot_config = robot_config
        self.mj_model = mj_model
        self.mj_data = mj_data
        
        # 初始化IK求解器
        self.ik_solver = MinkIKSolver(robot_config, mj_model, mj_data)
        
        # 初始化夹爪控制器
        self.gripper_controller = create_gripper_controller(
            robot_config.gripper, mj_model, mj_data
        )
        
        # 获取关节索引
        self._setup_joint_indices()
        
        # 控制状态 - 使用新的维度配置
        self.ctrl_dim = robot_config.ctrl_dim
        self.qpos_dim = robot_config.qpos_dim
        self.arm_joints = robot_config.arm_joints
        self.joint_pos_sensors = robot_config.joint_pos_sensors
        
        self.target_qpos = np.zeros(self.qpos_dim)
        self.is_moving = False
        self.motion_tolerance = 0.02
        self.velocity_tolerance = 0.1
        
    def _setup_joint_indices(self):
        """设置关节索引"""
        # 设置传感器索引映射
        self._setup_sensor_indices()
        
        # 设置执行器索引映射
        self._setup_actuator_indices()
        
        # 保留关节索引映射（用于兼容性）
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        
        # 机械臂关节
        for joint_name in self.robot_config.arm_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                self.arm_joint_indices.append(joint_id)
            except Exception as e:
                print(f"Warning: Could not find joint {joint_name}: {e}")
        
        # 夹爪关节（兼容性保留）
        self.gripper_joint_indices = []
        if hasattr(self.robot_config, 'gripper') and 'qpos_joints' in self.robot_config.gripper:
            for joint_name in self.robot_config.gripper['qpos_joints']:
                try:
                    joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    self.gripper_joint_indices.append(joint_id)
                except Exception as e:
                    print(f"Warning: Could not find gripper joint {joint_name}: {e}")
    
    def _setup_sensor_indices(self):
        """设置传感器索引映射"""
        self.sensor_indices = {
            'joint_pos': [],
            'joint_vel': [],
            'joint_torque': [],
            'end_effector': {}
        }
        
        # 关节位置传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_pos_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_pos_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_pos'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint position sensor {sensor_name}: {e}")
        
        # 关节速度传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_vel_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_vel_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_vel'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint velocity sensor {sensor_name}: {e}")
        
        # 关节力矩传感器
        if hasattr(self.robot_config, 'sensors') and 'joint_torque_sensors' in self.robot_config.sensors:
            for sensor_name in self.robot_config.sensors['joint_torque_sensors']:
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['joint_torque'].append(sensor_id)
                except Exception as e:
                    print(f"Warning: Could not find joint torque sensor {sensor_name}: {e}")
        
        # 末端执行器传感器
        if hasattr(self.robot_config, 'sensors') and 'end_effector_sensors' in self.robot_config.sensors:
            end_effector_sensors = self.robot_config.sensors['end_effector_sensors']
            for sensor_type, sensor_name in end_effector_sensors.items():
                try:
                    sensor_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    self.sensor_indices['end_effector'][sensor_type] = sensor_id
                except Exception as e:
                    print(f"Warning: Could not find end effector sensor {sensor_name}: {e}")
    
    def _setup_actuator_indices(self):
        """设置执行器索引映射"""
        self.actuator_indices = []
        
        if hasattr(self.robot_config, 'control') and 'actuators' in self.robot_config.control:
            for actuator_config in self.robot_config.control['actuators']:
                try:
                    # 执行器配置可能是字典或字符串
                    if isinstance(actuator_config, dict):
                        actuator_name = actuator_config['name']
                    else:
                        actuator_name = actuator_config
                        
                    actuator_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                    self.actuator_indices.append(actuator_id)
                except Exception as e:
                    print(f"Warning: Could not find actuator {actuator_config}: {e}")
    
    # ============== 调试和状态 ==============
    def __str__(self) -> str:
        """字符串表示"""
        return f"RobotInterface({self.robot_config.robot_name}, ready={self.is_ready()})"
