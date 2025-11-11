import os
import yaml

def load_and_resolve_config(config_path: str) -> dict:
    """加载并解析配置文件（支持模板继承）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否有模板继承
    if 'extends' in config:
        template_path = config['extends']
        if not os.path.isabs(template_path):
            # 相对路径，相对于当前配置文件
            base_dir = os.path.dirname(config_path)
            template_path = os.path.join(base_dir, template_path)
        
        print(f"📄 加载模板: {template_path}")
        
        # 递归加载模板
        template_config = load_and_resolve_config(template_path)
        
        # 合并配置（当前配置覆盖模板）
        merged_config = merge_configs(template_config, config)
        return merged_config
    
    return config


def merge_configs(template: dict, override: dict) -> dict:
    """合并配置文件（深度合并，支持状态数组的智能合并）
    
    Args:
        template: 模板配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = template.copy()
    
    for key, value in override.items():
        if key == 'extends':
            continue  # 跳过extends字段
            
        if key == 'states' and isinstance(result.get(key), list) and isinstance(value, list):
            # 特殊处理states数组：按索引合并
            result[key] = merge_states_array(result[key], value)
        elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def merge_states_array(template_states: list, override_states: list) -> list:
    """合并状态数组（按索引覆盖）
    
    Args:
        template_states: 模板中的状态数组
        override_states: 覆盖配置中的状态数组
        
    Returns:
        合并后的状态数组
    """
    # 从模板开始
    result = template_states.copy()
    
    # 按索引覆盖
    for i, override_state in enumerate(override_states):
        if i < len(result):
            # 覆盖已有的状态
            result[i] = override_state
        else:
            # 添加新状态
            result.append(override_state)
    
    return result


def replace_variables(config: dict) -> dict:
    """替换配置中的变量引用
    
    Args:
        config: 原始配置
        
    Returns:
        替换变量后的配置
    """
    import re
    import json
    
    # 获取运行时参数
    runtime_params = config.get('runtime_parameters', {})
    
    # 将配置转换为JSON字符串进行替换
    config_str = json.dumps(config, ensure_ascii=False)
    
    # 替换${variable}格式的变量
    for key, value in runtime_params.items():
        # 1. 替换带引号的变量（保持数据类型）
        quoted_pattern = f"\"${{{key}}}\""
        if isinstance(value, (int, float)):
            quoted_replacement = str(value)  # 数值不加引号
        else:
            quoted_replacement = f'"{value}"'  # 字符串加引号
        config_str = config_str.replace(quoted_pattern, quoted_replacement)
        
        # 2. 替换字符串内的变量（如description中的变量）
        inline_pattern = f"${{{key}}}"
        inline_replacement = str(value)  # 都转为字符串
        config_str = config_str.replace(inline_pattern, inline_replacement)
    
    # 转换回字典
    return json.loads(config_str)
