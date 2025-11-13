import os
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from xml.dom import minidom

class mj_consts:
    """MuJoCo常量类，用于存储与MuJoCo框架相关的常量和映射关系。
    
    包含了支持的坐标系类型及其对应的枚举值、雅可比矩阵计算函数和位置/方向属性。
    """
    # 支持的坐标系类型
    SUPPORTED_FRAMES = ("body", "geom", "site")

    # 坐标系类型到MuJoCo对象枚举的映射
    FRAME_TO_ENUM = {
        "body": mujoco.mjtObj.mjOBJ_BODY,  # 刚体
        "geom": mujoco.mjtObj.mjOBJ_GEOM,  # 几何体
        "site": mujoco.mjtObj.mjOBJ_SITE,  # 站点
    }
    
    # 坐标系类型到对应雅可比矩阵计算函数的映射
    FRAME_TO_JAC_FUNC = {
        "body": mujoco.mj_jacBody,  # 计算刚体的雅可比矩阵
        "geom": mujoco.mj_jacGeom,  # 计算几何体的雅可比矩阵
        "site": mujoco.mj_jacSite,  # 计算站点的雅可比矩阵
    }
    
    # 坐标系类型到位置属性的映射
    FRAME_TO_POS_ATTR = {
        "body": "xpos",       # 刚体位置属性
        "geom": "geom_xpos",  # 几何体位置属性
        "site": "site_xpos",  # 站点位置属性
    }
    
    # 坐标系类型到旋转矩阵属性的映射
    FRAME_TO_XMAT_ATTR = {
        "body": "xmat",       # 刚体旋转矩阵属性
        "geom": "geom_xmat",  # 几何体旋转矩阵属性
        "site": "site_xmat",  # 站点旋转矩阵属性
    }

def mj_quat2mat(quat_wxyz):
    """将四元数转换为旋转矩阵。
    
    Args:
        quat_wxyz: 四元数，格式为[w, x, y, z]
        
    Returns:
        3x3旋转矩阵
    """
    rmat_tmp = np.zeros(9)
    mujoco.mju_quat2Mat(rmat_tmp, quat_wxyz)
    return rmat_tmp.reshape(3, 3)

def add_mocup_body_to_mjcf(mjcf_xml_path, mocap_body_elements, sensor_elements=None, keep_tmp_xml=False):
    """向MJCF模型添加运动捕捉(mocap)刚体和传感器。
    
    Args:
        mjcf_xml_path: MJCF文件路径
        mocap_body_elements: 要添加的mocap刚体的XML元素列表
        sensor_elements: 要添加的传感器的XML元素列表，默认为None
        keep_tmp_xml: 是否保留临时生成的XML文件，默认为False
        
    Returns:
        添加了mocap刚体和传感器的MuJoCo模型
    """
    # 解析原始XML文件
    tree = ET.parse(mjcf_xml_path)
    root = tree.getroot()
    
    # 查找worldbody元素
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("MJCF文件中未找到worldbody元素")
    
    # 添加mocap刚体元素到worldbody
    if mocap_body_elements is not None:
        for body_element in mocap_body_elements:
            worldbody.append(body_element)
    
    # 查找sensor元素，如果不存在则创建
    sensor = root.find('sensor')
    if sensor is None:
        sensor = ET.SubElement(root, 'sensor')
    
    # 添加传感器元素
    if sensor_elements is not None:
        for sensor_element in sensor_elements:
            sensor.append(sensor_element)
    
    # 生成临时XML文件
    tmp_mjcf_xml_path = mjcf_xml_path.replace(".xml", "_tmp.xml")
    
    # 格式化XML并写入文件
    xml_str = ET.tostring(root, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    with open(tmp_mjcf_xml_path, "w", encoding='utf-8') as f:
        f.write(pretty_xml)
    
    # 加载MuJoCo模型
    m = mujoco.MjModel.from_xml_path(tmp_mjcf_xml_path)
    
    # 清理临时文件
    if not keep_tmp_xml:
        os.remove(tmp_mjcf_xml_path)
    
    return m

def move_mocap_to_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_name: str,
    frame_name: str,
    frame_type: str,
) -> None:
    """将mocap刚体移动到目标坐标系的位置和姿态。
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        mocap_name: mocap刚体的名称
        frame_name: 目标坐标系的名称
        frame_type: 目标坐标系的类型，可以是"body"、"geom"或"site"
    
    Raises:
        KeyError: 如果指定的刚体不是mocap刚体
    """
    mocap_id = model.body(mocap_name).mocapid[0]
    if mocap_id == -1:
        raise KeyError(f"Body '{mocap_name}' is not a mocap body.")

    obj_id = mujoco.mj_name2id(model, mj_consts.FRAME_TO_ENUM[frame_type], frame_name)
    xpos = getattr(data, mj_consts.FRAME_TO_POS_ATTR[frame_type])[obj_id]
    xmat = getattr(data, mj_consts.FRAME_TO_XMAT_ATTR[frame_type])[obj_id]

    data.mocap_pos[mocap_id] = xpos.copy()
    mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], xmat)

def generate_mocap_xml(name, box_size=(0.01, 0.01, 0.01), arrow_length=0.025, rgba=(0.3, 0.6, 0.3, 0.2)):
    """生成mocap刚体的XML元素。
    
    创建一个包含盒子和三个箭头（表示XYZ轴）的mocap刚体。
    
    Args:
        name: mocap刚体的名称
        box_size: 盒子的大小，默认为(0.025, 0.025, 0.025)
        arrow_length: 箭头的长度，默认为0.025
        rgba: 盒子的颜色和透明度，默认为(0.3, 0.6, 0.3, 0.2)
        
    Returns:
        包含mocap刚体定义的XML元素
    """
    # 创建body元素
    body = ET.Element('body')
    body.set('name', name)
    body.set('pos', '0 0 0')
    body.set('quat', '1 0 0 0')
    body.set('mocap', 'true')
    
    # 创建inertial元素
    inertial = ET.SubElement(body, 'inertial')
    inertial.set('pos', '0 0 0')
    inertial.set('mass', '1e-4')
    inertial.set('diaginertia', '1e-9 1e-9 1e-9')
    
    # 创建site元素
    site = ET.SubElement(body, 'site')
    site.set('name', f'{name}_site')
    site.set('size', '0.001')
    site.set('type', 'sphere')
    site.set('group', '5')
    
    # 创建box几何体
    box_geom = ET.SubElement(body, 'geom')
    box_geom.set('name', f'{name}_box')
    box_geom.set('type', 'box')
    box_geom.set('size', f'{box_size[0]} {box_size[1]} {box_size[2]}')
    box_geom.set('density', '0')
    box_geom.set('contype', '0')
    box_geom.set('conaffinity', '0')
    box_geom.set('rgba', f'{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}')
    box_geom.set('group', '5')
    
    # 创建X轴箭头（红色）
    x_arrow = ET.SubElement(body, 'geom')
    x_arrow.set('type', 'cylinder')
    x_arrow.set('pos', f'{arrow_length} 0 0')
    x_arrow.set('euler', '0 1.5708 0')
    x_arrow.set('size', f'{arrow_length/10.} {arrow_length}')
    x_arrow.set('density', '0')
    x_arrow.set('contype', '0')
    x_arrow.set('conaffinity', '0')
    x_arrow.set('rgba', '1 0 0 .2')
    x_arrow.set('group', '5')
    
    # 创建Y轴箭头（绿色）
    y_arrow = ET.SubElement(body, 'geom')
    y_arrow.set('type', 'cylinder')
    y_arrow.set('pos', f'0 {arrow_length} 0')
    y_arrow.set('euler', '1.5708 0 0')
    y_arrow.set('size', f'{arrow_length/10.} {arrow_length}')
    y_arrow.set('density', '0')
    y_arrow.set('contype', '0')
    y_arrow.set('conaffinity', '0')
    y_arrow.set('rgba', '0 1 0 .2')
    y_arrow.set('group', '5')
    
    # 创建Z轴箭头（蓝色）
    z_arrow = ET.SubElement(body, 'geom')
    z_arrow.set('type', 'cylinder')
    z_arrow.set('pos', f'0 0 {arrow_length}')
    z_arrow.set('euler', '0 0 0')
    z_arrow.set('size', f'{arrow_length/10.} {arrow_length}')
    z_arrow.set('density', '0')
    z_arrow.set('contype', '0')
    z_arrow.set('conaffinity', '0')
    z_arrow.set('rgba', '0 0 1 .2')
    z_arrow.set('group', '5')

    return body

def generate_mocap_sensor_xml(mocap_name, ref_name, ref_type="body"):
    """生成用于获取mocap刚体位置和方向的传感器XML元素。
    
    Args:
        mocap_name: mocap刚体的名称
        ref_name: 参考坐标系的名称
        ref_type: 参考坐标系的类型，默认为"body"
        
    Returns:
        包含传感器定义的XML元素列表
    """
    sensors = []
    
    # 创建位置传感器
    pos_sensor = ET.Element('framepos')
    pos_sensor.set('name', f'{mocap_name}_pos')
    pos_sensor.set('objtype', 'site')
    pos_sensor.set('objname', f'{mocap_name}_site')
    pos_sensor.set('reftype', ref_type)
    pos_sensor.set('refname', ref_name)
    sensors.append(pos_sensor)
    
    # 创建方向传感器
    quat_sensor = ET.Element('framequat')
    quat_sensor.set('name', f'{mocap_name}_quat')
    quat_sensor.set('objtype', 'site')
    quat_sensor.set('objname', f'{mocap_name}_site')
    quat_sensor.set('reftype', ref_type)
    quat_sensor.set('refname', ref_name)
    sensors.append(quat_sensor)
    
    return sensors
