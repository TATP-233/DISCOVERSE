"""
夹爪控制器 - 统一不同夹爪实现方式的接口

支持三种夹爪实现模式：
1. tendon控制 (如airbot_play)
2. equality约束 (如panda) 
3. 单关节控制 (如ur5e)

支持从传感器数据获取夹爪状态
"""

import mujoco
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod

class GripperController(ABC):
    """夹爪控制器基类"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        self.config = gripper_config
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.ctrl_index = gripper_config["ctrl_index"]
        
        # 从配置文件获取控制范围，但优先使用从MJCF文件中读取的实际关节范围
        self.config_ctrl_range = gripper_config["ctrl_range"]
        
        # 如果配置中指定了关节名称，尝试从MJCF获取真实范围
        # 使用配置中指定的控制范围
        self.ctrl_range = self.config_ctrl_range
        print(f"Gripper control range: {self.ctrl_range}")
        
        # 获取传感器索引（如果配置中有的话）
        self.sensor_index = None
        if "sensor_name" in gripper_config:
            try:
                self.sensor_index = mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_SENSOR, gripper_config["sensor_name"]
                )
            except Exception as e:
                print(f"Warning: Could not find gripper sensor '{gripper_config['sensor_name']}': {e}")
        
    @abstractmethod
    def set_position(self, position: float) -> bool:
        """设置夹爪位置"""
        pass
        
    def open(self) -> float:
        """打开夹爪，返回控制值"""
        return self.config["default_position"]
        
    def close(self) -> float:
        """关闭夹爪，返回控制值"""
        return self.config["close_position"]

class TwoFingerGripper(GripperController):
    """双指tendon夹爪控制器 (如AirBot Play)"""
    
    def __init__(self, gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
        super().__init__(gripper_config, mj_model, mj_data)
        self.qpos_indices = gripper_config.get("qpos_indices", [])
        
    def set_position(self, position: float) -> bool:
        """通过tendon控制器设置夹爪位置"""
        normalized_pos = np.clip(position, self.ctrl_range[0], self.ctrl_range[1])
        self.mj_data.ctrl[self.ctrl_index] = normalized_pos
        return True

def create_gripper_controller(gripper_config: Dict[str, Any], mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> GripperController:
    return TwoFingerGripper(gripper_config, mj_model, mj_data)
