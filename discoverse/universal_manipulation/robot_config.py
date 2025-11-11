"""
机械臂配置加载器

用于加载和解析机械臂配置文件，提供统一的配置接口。
"""

import os
import yaml
import numpy as np
from typing import Dict, List, Any

class RobotConfigLoader:
    """机械臂配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Robot config file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 验证配置文件
            self._validate_config()
            
            # 处理配置
            self._process_config()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse robot config file {config_path}: {e}")
    
    def _validate_config(self):
        """验证配置文件的必要字段"""
        required_fields = [
            'robot_name',
            'kinematics',
            'gripper',
            'sensors'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in robot config: {field}")
        
        # 验证运动学配置
        kinematics = self.config['kinematics']
        kinematics_required = ['base_link', 'end_effector_site', 'arm_joints']
        for field in kinematics_required:
            if field not in kinematics:
                raise ValueError(f"Missing required kinematics field: {field}")
        
        # 验证夹爪配置
        gripper = self.config['gripper']
        gripper_required = ['type', 'ctrl_dim', 'ctrl_index']
        for field in gripper_required:
            if field not in gripper:
                raise ValueError(f"Missing required gripper field: {field}")

        # 验证传感器配置
        sensors = self.config['sensors']
        sensors_required = ['joint_pos_sensors', 'end_effector_sensors']
        for field in sensors_required:
            if field not in sensors:
                raise ValueError(f"Missing required sensor field: {field}")

    def _process_config(self):
        """处理配置，转换数据类型等"""
        # 处理特殊配置
        if self.config['robot_name'] == 'airbot_play' and 'airbot_specific' in self.config:
            airbot_config = self.config['airbot_specific']
            if 'joint_bias' in airbot_config:
                airbot_config['joint_bias'] = np.array(airbot_config['joint_bias'])
            if 'arm_rotation_matrix' in airbot_config:
                airbot_config['arm_rotation_matrix'] = np.array(airbot_config['arm_rotation_matrix'])
    
    # ============== 属性访问方法 ==============
    
    @property
    def robot_name(self) -> str:
        """获取机械臂名称"""
        return self.config['robot_name']
    
    @property
    def end_effector_site(self) -> str:
        """获取末端执行器site名称"""
        return self.config['kinematics']['end_effector_site']
    
    @property
    def qpos_dim(self) -> int:
        """获取qpos维度"""
        return self.config['kinematics']['qpos_dim']
    
    @property  
    def ctrl_dim(self) -> int:
        """获取控制器维度"""
        return self.config['kinematics']['ctrl_dim']
    
    @property
    def arm_joints(self) -> List[str]:
        """获取机械臂关节名称列表"""
        return self.config['kinematics']['arm_joint_names']
    
    @property
    def arm_joints_count(self) -> int:
        """获取机械臂关节数"""
        return self.config['kinematics']['arm_joints']
    
    @property
    def gripper(self) -> Dict:
        """获取夹爪配置"""
        return self.config['gripper']
    
    @property
    def arm_joint_names(self) -> List[str]:
        """获取机械臂关节名称列表"""
        return self.config['kinematics'].get('arm_joint_names', [])
    
    @property
    def joint_pos_sensors(self) -> List[str]:
        """获取关节位置传感器名称列表"""
        return self.config['sensors'].get('joint_pos_sensors', [])

def load_robot_config(config_path: str) -> RobotConfigLoader:
    """
    便利函数：加载机械臂配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置加载器实例
    """
    return RobotConfigLoader(config_path) 