import os
import json
import glfw
import threading
import numpy as np
import xml.etree.ElementTree as ET

import rclpy

from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.examples.ros2.mmk2_ros2_joy import MMK2ROS2JoyCtl
from discoverse.utils import get_site_tmat, get_body_tmat

def read_object_positions(xml_path):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 用于存储物体信息（name, pos）
    objects = []

    # 遍历所有的<worldbody>下的<body>标签
    for idx, body in enumerate(root.findall(".//worldbody/body")):
        name = body.get("name")  # 获取物品名称
        pos = body.get("pos")  # 获取位置属性
        if name and pos:
            # 打印物品名称及其位置，并存储物品信息
            objects.append((idx + 1, name, pos))  # 物品编号从1开始
            print(f"{idx + 1}. Object: {name}, Position: {pos}")

    return objects

class MMK2ROS2GP(MMK2ROS2JoyCtl):

    def on_key(self, window, key, scancode, action, mods):
        super().on_key(window, key, scancode, action, mods)
        print(f"按键: {key}, 动作: {action}")  # 添加调试信息
        if action == glfw.PRESS:
            if key == glfw.KEY_O:
                print("left_end position:", get_site_tmat(self.mj_data, "lft_endpoint")[:3, 3])
                print("right_end position:", get_site_tmat(self.mj_data, "rgt_endpoint")[:3, 3])
                self.save_state_to_json()

    def save_state_to_json(self):
        """保存机器人的当前状态到 JSON 文件"""
        self.objects = read_object_positions(self.mjcf_file)

        # 提示用户选择物体编号
        if self.objects:
            print("\n请选择一个物体的编号:")
            try:
                selected_idx = int(input(f"请输入编号（1 到 {len(self.objects)}）：")) - 1
            except ValueError:
                print("无效编号，保存失败。")
                return
            if 0 <= selected_idx < len(self.objects):
                selected_object = self.objects[selected_idx]
                selected_name = selected_object[1]
                selected_pos = get_body_tmat(self.mj_data, selected_name)[:3, 3]  # 物体的位置信息
                print(f"选择了物体: {selected_name}, 位置: {selected_pos}")
            else:
                print("无效编号，保存失败。")
                return
        else:
            print("未找到任何物体。")
            return

        select_arm = input("请选择要移动的机械臂（a/l/r）：")
        if not select_arm in {"a", "l", "r"}:
            print(("无效输入，保存失败。"))
            return

        try:
            delay_time_s = float(input("请输入延迟时间（秒）："))
        except ValueError:
            delay_time_s = 0.0

        # 获取机器人的当前状态，并调整为相对于物体的坐标
        state_data = {
            "object_name": selected_name,  # 使用选择的物体名称
            "left_arm": {
                "position_object_local": [round(coord - selected_pos[idx], 3) for idx, coord in enumerate(get_site_tmat(self.mj_data, "lft_endpoint")[:3, 3])],
                "rotation_robot_local": [round(val, 3) for val in self.lft_end_euler],
                "gripper": round(self.tctr_lft_gripper[0], 3),
                "movement": "stop" if select_arm == "r" else "move"
            },
            "right_arm": {
                "position_object_local": [round(coord - selected_pos[idx], 3) for idx, coord in enumerate(get_site_tmat(self.mj_data, "rgt_endpoint")[:3, 3])],
                "rotation_robot_local": [round(val, 3) for val in self.rgt_end_euler],
                "gripper": round(self.tctr_rgt_gripper[0], 3),
                "movement": "stop" if select_arm == "l" else "move"
            },
            "slide": [round(self.tctr_slide[0], 3)],
            "head": [round(self.tctr_head[0], 3), round(self.tctr_head[1], 3)],
            "delay_s": delay_time_s
        }

        # 定义 JSON 文件保存路径
        json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"scheme_{os.path.basename(self.mjcf_file).split('.')[0]}.json")
        
        # 如果文件已存在，则加载已有数据并追加新状态，否则创建一个新文件
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        # 添加当前状态到列表中
        data.append(state_data)

        # 将更新后的数据写入 JSON 文件
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

        print("当前状态已保存到 JSON 文件。")


if __name__ == "__main__":
    rclpy.init()

    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    # 创建MMK2配置
    cfg = MMK2Cfg()
    
    cfg.init_key = "pick"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = None
    cfg.obs_depth_cam_id = None
    
    cfg.render_set = {
        "fps"    : 30,
        "width"  : 1920,
        "height" : 1080
    }
    
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/throw_rubbish.xml"
    
    exec_node = MMK2ROS2GP(cfg)
    exec_node.reset()

    spin_thread = threading.Thread(target=lambda:rclpy.spin(exec_node))
    spin_thread.start()

    pubtopic_thread = threading.Thread(target=exec_node.thread_pubros2topic, args=(30,))
    pubtopic_thread.start()

    while exec_node.running:
        exec_node.teleopProcess()
        exec_node.step(exec_node.target_control)
    
    exec_node.destroy_node()
    rclpy.shutdown()
    pubtopic_thread.join()
    spin_thread.join()

