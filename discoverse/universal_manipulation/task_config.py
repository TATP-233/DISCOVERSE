"""
任务配置加载器

用于加载和解析任务配置文件，提供统一的任务定义接口。
"""

import os
import yaml
from typing import Dict, List, Any, Optional
from .config_utils import load_and_resolve_config, replace_variables

class TaskConfigLoader:
    """任务配置加载器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化任务配置加载器
        
        Args:
            config_path: 任务配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.runtime_params = {}
        
        if config_path:
            self.load_config(config_path)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TaskConfigLoader':
        """从字典创建配置加载器
        
        Args:
            config_dict: 配置字典
            
        Returns:
            配置加载器实例
        """
        loader = cls()
        loader.config = config_dict
        loader._validate_config()
        return loader
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载任务配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            任务配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        try:
            # 使用模板化配置解析
            self.config = load_and_resolve_config(config_path)
            self.config = replace_variables(self.config)
            
            # 验证配置文件
            self._validate_config()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse task config file {config_path}: {e}")
    
    def _validate_config(self):
        """验证任务配置文件的必要字段"""
        required_fields = [
            'task_name',
            'description'
        ]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field in task config: {field}")
        
        # 检查状态字段（支持states或task_states）
        if 'states' not in self.config and 'task_states' not in self.config:
            raise ValueError("Task must have 'states' or 'task_states' field")
        
        # 验证状态配置（优先使用states，然后是task_states）
        states = self.config.get('states', self.config.get('task_states', []))
        if not isinstance(states, list) or len(states) == 0:
            raise ValueError("Task must have at least one state")
        
        for i, state in enumerate(states):
            if 'name' not in state:
                raise ValueError(f"State {i} missing required field: name")
            if 'primitive' not in state:
                raise ValueError(f"State {i} ({state['name']}) missing required field: primitive")
    
    def resolve_parameters(self, value: Any) -> Any:
        """
        解析参数化的值，支持 {param} 和 ${param} 格式
        
        Args:
            value: 要解析的值
            
        Returns:
            解析后的值
        """
        if isinstance(value, str):
            # 处理 ${param} 格式的参数替换
            import re
            
            def replace_param(match):
                param_name = match.group(1)
                # 优先使用运行时参数
                if param_name in self.runtime_params:
                    return str(self.runtime_params[param_name])
                # 然后使用配置文件中的运行时参数
                elif 'runtime_parameters' in self.config and param_name in self.config['runtime_parameters']:
                    return str(self.config['runtime_parameters'][param_name])
                # 最后使用旧格式的parameters
                elif param_name in self.config.get('parameters', {}):
                    return str(self.config['parameters'][param_name])
                # 如果找不到参数，保持原样
                return match.group(0)
            
            # 替换 ${param} 格式
            value = re.sub(r'\$\{([^}]+)\}', replace_param, value)
            # 替换 {param} 格式
            value = re.sub(r'\{([^}]+)\}', replace_param, value)
            
        elif isinstance(value, dict):
            return {k: self.resolve_parameters(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve_parameters(item) for item in value]
        
        return value
    
    def get_resolved_states(self) -> List[Dict[str, Any]]:
        """
        获取解析后的状态序列
        
        Returns:
            解析后的状态列表
        """
        resolved_states = []
        
        # 获取状态列表（优先使用states，然后是task_states）
        states = self.config.get('states', self.config.get('task_states', []))
        
        for state in states:
            resolved_state = self.resolve_parameters(state.copy())
            resolved_states.append(resolved_state)
        
        return resolved_states
    
    # ============== 属性访问方法 ==============
    @property
    def task_name(self) -> str:
        """获取任务名称"""
        return self.config.get('task_name', 'unknown_task')

    @property 
    def record_fps(self) -> int:
        """获取记录帧率"""
        return self.config.get('observation', {'fps': 30}).get('fps', 30)
    
    @property
    def camera_configs(self) -> List[Dict[str, Any]]:
        """获取相机配置列表"""
        return self.config.get('observation', {}).get('cameras', [])

    @property 
    def success_check(self) -> Optional[Dict[str, Any]]:
        """获取成功检查配置"""
        return self.config.get('success_check')
    
    @property
    def randomization(self) -> Optional[Dict[str, Any]]:
        """获取随机化配置"""
        return self.config.get('randomization')
    
    # ============== 随机化相关方法 ==============
    def validate_randomization_config(self) -> bool:
        """
        验证随机化配置的有效性
        
        Returns:
            是否有效
        """
        # 检查物体随机化配置
        if 'objects' in self.randomization:
            for i, obj_config in enumerate(self.randomization['objects']):
                if isinstance(obj_config, dict):
                    if 'name' not in obj_config:
                        print(f"❌ 随机化物体配置 {i} 缺少 'name' 字段")
                        return False
        return True

    def __str__(self) -> str:
        """字符串表示"""
        if not self.config:
            return "TaskConfigLoader(not loaded)"
        
        return f"TaskConfigLoader({self.task_name}, {len(self.states)} states)"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def load_task_config(config_path: str) -> TaskConfigLoader:
    """
    便利函数：加载任务配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        任务配置加载器实例
    """
    return TaskConfigLoader(config_path) 