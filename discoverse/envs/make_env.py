"""
MuJoCo环境生成器

该模块实现了动态组合机械臂和任务的功能，将不同的机械臂模型与任务场景组合成完整的仿真环境。
"""

import os
import mujoco
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Optional

from discoverse import DISCOVERSE_ASSETS_DIR


class MujocoEnv:
    """MuJoCo环境类，用于管理和操作XML配置"""
    
    def __init__(self, xml_root: ET.Element, xml_string: str, xml_path: Optional[str] = None):
        """
        初始化MuJoCo环境
        
        Args:
            xml_root: XML根元素
            xml_string: XML字符串内容
            xml_path: XML文件路径（可选）
        """
        self.xml_root = xml_root
        self.xml_string = xml_string
        self.xml_path = xml_path
    
    def get_xml_root(self) -> ET.Element:
        """返回XML根元素"""
        return self.xml_root
    
    def get_xml_string(self) -> str:
        """返回XML字符串"""
        return self.xml_string
    
    def export_xml(self, output_path: str) -> None:
        """
        导出XML文件到指定路径
        
        Args:
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 只在目录不为空时创建
            os.makedirs(output_dir, exist_ok=True)
        
        # 使用minidom格式化XML，并去除多余空行
        rough_string = ET.tostring(self.xml_root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 去除多余的空行
        lines = pretty_xml.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():  # 只保留非空行
                cleaned_lines.append(line)
        
        # 重新组合，确保文件末尾有一个换行
        final_xml = '\n'.join(cleaned_lines)
        if not final_xml.endswith('\n'):
            final_xml += '\n'
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_xml)
        
        self.xml_path = output_path
    
    def test_mujoco_load(self) -> bool:
        """
        测试生成的XML文件是否能被MuJoCo成功加载
        
        Returns:
            bool: 是否加载成功
        """
        if self.xml_path is None:
            raise ValueError("XML文件路径未设置，请先调用export_xml方法")
        
        try:
            # 只在需要时导入mujoco
            from PIL import Image
            model = mujoco.MjModel.from_xml_path(self.xml_path)
            mj_data = mujoco.MjData(model)
            mujoco.mj_forward(model, mj_data)
            renderer = mujoco.Renderer(model)
            renderer.update_scene(mj_data)
            img = renderer.render()
            model_name = self.xml_root.get("model")
            Image.fromarray(img).save(f"{model_name}.png")
            return True
        except ImportError:
            print("警告: MuJoCo未安装或无法导入，跳过加载测试")
            return False
        except Exception as e:
            print(f"MuJoCo加载失败: {e}")
            return False


def _convert_include_paths_to_absolute(xml_root: ET.Element, xml_file_dir: str) -> None:
    """
    将XML中的include相对路径转换为绝对路径
    
    Args:
        xml_root: XML根元素
        xml_file_dir: 当前XML文件所在目录
    """
    for include in xml_root.findall(".//include"):
        file_attr = include.get("file")
        if file_attr and not os.path.isabs(file_attr):
            # include路径相对于当前XML文件所在目录
            abs_path = os.path.normpath(os.path.join(xml_file_dir, file_attr))
            include.set("file", abs_path)


def _convert_paths_to_absolute(xml_root: ET.Element, base_dir: str) -> None:
    """
    将XML中的相对路径转换为绝对路径
    
    Args:
        xml_root: XML根元素
        base_dir: 基础目录路径（通常是DISCOVERSE_ASSETS_DIR）
    """
    # 更新compiler的meshdir和texturedir为绝对路径
    for compiler in xml_root.findall(".//compiler"):
        meshdir = compiler.get("meshdir")
        if meshdir and not os.path.isabs(meshdir):
            # meshdir相对于base_dir解析
            abs_meshdir = os.path.normpath(os.path.join(base_dir, "meshes"))
            compiler.set("meshdir", abs_meshdir)
        
        texturedir = compiler.get("texturedir") 
        if texturedir and not os.path.isabs(texturedir):
            # texturedir相对于base_dir解析
            abs_texturedir = os.path.normpath(os.path.join(base_dir, "meshes"))
            compiler.set("texturedir", abs_texturedir)

def _merge_robot_and_pure_task(robot_xml_path: str, pure_task_xml_path: str) -> ET.Element:
    """
    合并机械臂和纯任务环境XML文件
    
    Args:
        robot_xml_path: 机械臂XML文件路径  
        pure_task_xml_path: 纯任务环境XML文件路径
    
    Returns:
        ET.Element: 合并后的XML根元素
    """
    # 加载机械臂XML
    robot_tree = ET.parse(robot_xml_path)
    robot_root = robot_tree.getroot()
    
    # 加载纯任务环境XML
    task_tree = ET.parse(pure_task_xml_path)
    task_root = task_tree.getroot()
    
    # 创建新的根元素
    merged_root = ET.Element("mujoco")
    task_name = task_root.get("model", "task")
    robot_name = os.path.basename(robot_xml_path).replace("robot_", "").replace(".xml", "")
    merged_root.set("model", f"{robot_name}_{task_name}")
    
    # 基础目录（用于解析include路径）
    task_xml_dir = os.path.dirname(pure_task_xml_path)
    robot_xml_dir = os.path.dirname(robot_xml_path)
    
    # 转换主XML文件内部的include路径（相对于各自的文件目录）
    _convert_include_paths_to_absolute(robot_root, robot_xml_dir)
    _convert_include_paths_to_absolute(task_root, task_xml_dir)
    
    # 收集所有assets和其他元素
    all_assets = []
    all_others = []
    tail_inlcudes = []
    manipulator_locate = None
    robot_keyframe = None

    # 1. 处理机械臂XML的所有子元素
    for child in robot_root:
        if child.tag in {"worldbody"}:
            continue
        elif child.tag == "keyframe":
            robot_keyframe = child
            continue
        elif child.tag == "include" and "/scene/" in child.get("file"):
            continue
        all_others.append(child)

    # 2. 处理纯任务XML的所有子元素
    for child in task_root:
        if child.tag == "asset":
            for asset_child in child:
                all_assets.append(asset_child)
        elif child.tag == "include" and "/scene/" in child.get("file"):
            tail_inlcudes.append(child)
        elif child.tag != "worldbody":
            all_others.append(child)
    
    # 3. 添加其他元素（智能合并compiler等）
    merged_elements = {}
    
    for elem in all_others:
        tag = elem.tag
        
        if tag == "compiler":
            # 只保留一个compiler，使用任务环境的设置
            if tag not in merged_elements:
                merged_elements[tag] = elem
        else:
            # 其他元素直接添加
            merged_root.append(elem)
    
    # 添加合并后的特殊元素
    for tag, elem in merged_elements.items():
        merged_root.append(elem)
    
    # 5. 创建合并的worldbody
    new_worldbody = ET.SubElement(merged_root, "worldbody")
    
    # 首先添加机械臂的worldbody内容
    arm_name = robot_root.get("model")
    manipulator_locate = task_root.find(".//site[@name='manipulator_locate']")
    robot_worldbody = robot_root.find("worldbody")
    set_arm_pose = False
    if robot_worldbody is not None:
        for child in robot_worldbody:
            print(child.tag, child.get("name"))
            if child.tag == "body" and child.get("name") == f"{arm_name}_pose":
                set_arm_pose = True
                for key in ["pos", "euler", "quat"]:
                    if manipulator_locate is not None and manipulator_locate.get(key):
                        child.set(key, manipulator_locate.get(key))
            new_worldbody.append(child)
    assert set_arm_pose, "未找到机械臂位姿设置节点，请确保机械臂XML中包含正确的body名称"
    
    # 然后添加任务环境的worldbody内容（包括include元素）
    task_worldbody = task_root.find("worldbody")
    if task_worldbody is not None:
        for child in task_worldbody:
            if child is not manipulator_locate:
                new_worldbody.append(child)
    
    for child in tail_inlcudes:
        merged_root.append(child)

    # 添加keyframe
    env_model = mujoco.MjModel.from_xml_path(pure_task_xml_path)
    env_data = mujoco.MjData(env_model)
    mujoco.mj_forward(env_model, env_data)
    env_qpos = env_data.qpos.copy().tolist()

    # 只处理name为"home"的key
    key_elem = None
    for k in robot_keyframe.findall("key"):
        if k.get("name") == "home":
            key_elem = k
            break
    if key_elem is not None:
        # 只保留name为"home"的key，删除其他key
        for k in list(robot_keyframe.findall("key")):
            if k is not key_elem:
                robot_keyframe.remove(k)
        robot_qpos = list(map(float, key_elem.get("qpos").split()))
        keyframe_qpos = [x for x in (robot_qpos + env_qpos)]
        key_elem.set("qpos", " ".join(map(str, keyframe_qpos)))
        merged_root.append(robot_keyframe)

    return merged_root

def make_env(robot_name: str, task_name: str, output_path: Optional[str] = None) -> MujocoEnv:
    """
    创建MuJoCo环境，组合指定的机械臂和任务
    
    Args:
        robot_name: 机械臂名称（如"panda", "ur5e", "airbot_play"等）
        task_name: 任务名称（如"stack_block", "pick_place"等）
        output_path: 输出XML文件路径（可选）
    
    Returns:
        MujocoEnv: 创建的环境对象
    
    Raises:
        FileNotFoundError: 当指定的机械臂或任务文件不存在时
        ValueError: 当参数无效时
    """
    # 构建文件路径
    mjcf_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf")
    
    # 机械臂XML文件路径
    robot_xml_path = os.path.join(mjcf_dir, f"manipulator/robot_{robot_name}.xml")
    if not os.path.exists(robot_xml_path):
        raise FileNotFoundError(f"机械臂文件不存在: {robot_xml_path}")
    
    # 纯任务环境XML文件路径（使用提取的纯环境文件）
    pure_task_xml_path = os.path.join(mjcf_dir, "task_environments", f"{task_name}.xml")
    if not os.path.exists(pure_task_xml_path):
        raise FileNotFoundError(f"纯任务环境文件不存在: {pure_task_xml_path}")
    
    # print(f"正在组合机械臂: {robot_name}")
    # print(f"任务环境: {task_name}")
    # print(f"机械臂文件: {robot_xml_path}")
    # print(f"任务环境文件: {pure_task_xml_path}")
    
    # 合并XML文件
    merged_root = _merge_robot_and_pure_task(robot_xml_path, pure_task_xml_path)
    
    # 转换路径为绝对路径（基于DISCOVERSE_ASSETS_DIR）
    # 注意：include路径已经在合并过程中转换为绝对路径了，这里只处理meshdir和texturedir
    _convert_paths_to_absolute(merged_root, DISCOVERSE_ASSETS_DIR)
    
    # 生成XML字符串，去除多余空行
    rough_string = ET.tostring(merged_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # 去除多余的空行
    lines = pretty_xml.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip():  # 只保留非空行
            cleaned_lines.append(line)
    
    # 重新组合XML字符串
    xml_string = '\n'.join(cleaned_lines)
    if not xml_string.endswith('\n'):
        xml_string += '\n'
    
    # 创建环境对象
    env = MujocoEnv(merged_root, xml_string, output_path)
    
    # 如果指定了输出路径，则导出文件
    if output_path:
        env.export_xml(output_path)
        # print(f"环境XML已导出到: {output_path}")
    
    return env


def list_available_robots() -> list:
    """
    列出所有可用的机械臂
    
    Returns:
        list: 可用机械臂名称列表
    """
    mjcf_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator")
    robots = []
    
    for file in os.listdir(mjcf_dir):
        if file.startswith("robot_") and file.endswith(".xml"):
            robot_name = file.replace("robot_", "").replace(".xml", "")  # 移除"robot_"前缀和".xml"后缀
            robots.append(robot_name)
    
    return sorted(robots)


def list_available_tasks() -> list:
    """
    列出所有可用的任务
    
    Returns:
        list: 可用任务名称列表
    """
    tasks_dir = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "task_environments")
    tasks = []
    
    if os.path.exists(tasks_dir):
        for file in os.listdir(tasks_dir):
            if file.endswith(".xml"):
                task_name = file.replace(".xml", "")  # 移除".xml"后缀
                tasks.append(task_name)
    
    return sorted(tasks)

def test_one(robot_name: str, task_name: str, output_path: str = "test_environment.xml"):
    env = make_env(robot_name, task_name, output_path)
    if not env.test_mujoco_load():
        raise Exception(f"MuJoCo加载测试失败: {robot_name} {task_name}")

if __name__ == "__main__":
    import random
    import argparse
    import traceback

    # 示例用法
    robots_names = list_available_robots()
    tasks_names = list_available_tasks()

    print("可用机械臂:", robots_names)
    print("可用任务:", tasks_names)

    parser = argparse.ArgumentParser(description="测试可用的机械臂和任务环境")
    parser.add_argument("--robot", type=str, default=None, help="指定要测试的机械臂名称")
    parser.add_argument("--task", type=str, default=None, help="指定要测试的任务名称")
    parser.add_argument("--all", action="store_true", help="测试所有机械臂和任务组合")
    args = parser.parse_args()

    if args.all:
        for robot_name in robots_names:
            if args.robot and robot_name != args.robot:
                continue
            for task_name in tasks_names:
                if args.task and task_name != args.task:
                    continue
                try:
                    test_one(robot_name, task_name, f"{robot_name}_{task_name}.xml")
                except Exception as e:
                    print(f"测试失败: {robot_name} {task_name} {e}")
        exit(0)

    else:
        try:
            robot_name = args.robot if args.robot else random.choice(robots_names)
            task_name = args.task if args.task else random.choice(tasks_names)
            test_one(robot_name, task_name, f"{robot_name}_{task_name}.xml")

        except FileNotFoundError as e:
            traceback.print_exc()

        except Exception as e:
            traceback.print_exc()