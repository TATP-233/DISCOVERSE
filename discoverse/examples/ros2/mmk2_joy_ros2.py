import os
import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
import json
import glfw
import xml.etree.ElementTree as ET
import time

from discoverse.envs.mmk2_base import MMK2Base, MMK2Cfg
from discoverse.airbot_play import AirbotPlayFIK
from discoverse.mmk2 import MMK2FIK
from discoverse.utils.joy_stick_ros2 import JoyTeleopRos2
from discoverse.utils import get_site_tmat, get_body_tmat

from discoverse import DISCOVERSE_ASSERT_DIR

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

class MMK2JOY(MMK2Base):
    arm_action_init_position = {
        "pick" : {
            "l" : np.array([0.254,  0.216,  1.069]),
            "r" : np.array([0.254, -0.216,  1.069]),
        },
        "carry" : {
            "l" : np.array([0.254,  0.216,  1.069]),
            "r" : np.array([0.254, -0.216,  1.069]),
        },
    }

    target_control = np.zeros(19)
    
    def __init__(self, config: MMK2Cfg):
        self.arm_action = config.init_key
        self.tctr_base = self.target_control[:2]
        self.tctr_slide = self.target_control[2:3]
        self.tctr_head = self.target_control[3:5]
        self.tctr_left_arm = self.target_control[5:11]
        self.tctr_lft_gripper = self.target_control[11:12]
        self.tctr_right_arm = self.target_control[12:18]
        self.tctr_rgt_gripper = self.target_control[18:19]

        super().__init__(config)

        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action]["l"].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action]["r"].copy()
        self.rgt_end_euler = np.zeros(3)

        self.arm_fik = AirbotPlayFIK(urdf = os.path.join(DISCOVERSE_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

        # 初始化ROS2节点和手柄控制
        # 注意：rclpy.init()只在main()函数中调用一次
        self.teleop = JoyTeleopRos2()
        self.ros_executor = rclpy.executors.SingleThreadedExecutor()
        self.ros_executor.add_node(self.teleop)

        self.objects = []  # 用于存储物体信息
        
        # 添加关键点记录相关属性
        self.keypoints = []  # 用于存储关键点
        self.task_name = "task"  # 默认任务名称
        
    def __del__(self):
        # 如果有记录的关键点但尚未保存，在退出时自动保存
        if hasattr(self, 'keypoints') and self.keypoints:
            self.save_keypoints_scheme()
            
        # 清理ROS2资源
        if hasattr(self, 'ros_executor'):
            self.ros_executor.shutdown()
        if hasattr(self, 'teleop') and self.teleop is not None:
            self.teleop.destroy_node()
        # 检查共享内存是否存在再进行删除
        if hasattr(self, 'shm') and self.shm is not None:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                print("共享内存文件未找到，无法删除。")
        # 不在这里调用rclpy.shutdown()，因为它应该在main()函数结束时调用

    def resetState(self):
        super().resetState()
        self.target_control[:] = self.init_joint_ctrl[:]
        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action]["l"].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action]["r"].copy()
        self.rgt_end_euler = np.zeros(3)
        self.teleop.reset()

    def teleopProcess(self):
        # 处理ROS2消息
        self.ros_executor.spin_once(timeout_sec=0)
        
        # 打印手柄输入信息，帮助调试
        # print(f"手柄输入 - axes: {self.teleop.joy_cmd.axes}, buttons: {self.teleop.joy_cmd.buttons}")
        
        # 获取手柄axes的长度
        axes_len = len(self.teleop.joy_cmd.axes)
        buttons_len = len(self.teleop.joy_cmd.buttons)
        
        # 确保有足够的按钮
        if buttons_len < 6:
            print("警告：手柄按钮数量不足，需要至少6个按钮")
            return
        
        # 检查是否处于手臂控制模式
        left_arm_mode = self.teleop.joy_cmd.buttons[4] and not self.teleop.joy_cmd.buttons[5]  # 只按L1
        right_arm_mode = self.teleop.joy_cmd.buttons[5] and not self.teleop.joy_cmd.buttons[4]  # 只按R1
        
        # 如果同时按下L1和R1，则退出手臂控制模式
        if self.teleop.joy_cmd.buttons[4] and self.teleop.joy_cmd.buttons[5]:
            left_arm_mode = False
            right_arm_mode = False
            print("退出手臂控制模式")
        
        # 判断当前是否处于任何手臂控制模式
        arm_control_mode = left_arm_mode or right_arm_mode
        
        # 基本移动控制（默认为0）
        linear_vel = 0.0
        angular_vel = 0.0
        
        # =============== 普通模式（不控制手臂） ===============
        if not arm_control_mode:
            # 确保有足够的axes进行基本控制
            if axes_len >= 2:
                # 左摇杆控制机器人前进/后退和旋转
                # 注意：axes[1]是前后方向，上推(负值)代表前进，下拉(正值)代表后退
                linear_vel = 1.0 * self.teleop.joy_cmd.axes[1]**2 * np.sign(self.teleop.joy_cmd.axes[1])
                # 注意：axes[0]是左右方向，左推(负值)代表左转，右推(正值)代表右转
                angular_vel = 2.0 * self.teleop.joy_cmd.axes[0]**2 * np.sign(self.teleop.joy_cmd.axes[0])
            
            # 控制头部
            if axes_len >= 4:
                # 右摇杆控制头部
                # 注意：axes[2]是右摇杆上下，上推(负值)代表头部上抬，下拉(正值)代表头部下垂
                self.tctr_head[0] += self.teleop.joy_cmd.axes[2] * 1. / self.render_fps
                # 注意：axes[3]是右摇杆左右，左推(负值)代表头部左转，右推(正值)代表头部右转
                self.tctr_head[1] -= self.teleop.joy_cmd.axes[3] * 1. / self.render_fps
                
                # 确保头部角度在有效范围内
                self.tctr_head[1] = np.clip(self.tctr_head[1], 
                                           self.mj_model.joint("head_pitch_joint").range[0], 
                                           self.mj_model.joint("head_pitch_joint").range[1])
                self.tctr_head[0] = np.clip(self.tctr_head[0], 
                                           self.mj_model.joint("head_yaw_joint").range[0], 
                                           self.mj_model.joint("head_yaw_joint").range[1])
            
            
            # 控制身体高度
            delta_height = 0
            if axes_len >= 6:  # 使用L2/R2控制高度
                # L2(负值)降低高度，R2(负值)增加高度
                delta_height = (self.teleop.joy_cmd.axes[5] - self.teleop.joy_cmd.axes[4]) * 0.1 / self.render_fps
            elif axes_len >= 5:  # 只有一个触发器
                delta_height = -self.teleop.joy_cmd.axes[4] * 0.1 / self.render_fps
            
            # 确保高度在有效范围内
            if self.tctr_slide[0] + delta_height < self.mj_model.joint("slide_joint").range[0]:
                delta_height = self.mj_model.joint("slide_joint").range[0] - self.tctr_slide[0]
            elif self.tctr_slide[0] + delta_height > self.mj_model.joint("slide_joint").range[1]:
                delta_height = self.mj_model.joint("slide_joint").range[1] - self.tctr_slide[0]
            
            # 应用高度变化
            self.tctr_slide[0] += delta_height
            # 同时调整手臂目标位置，保持相对位置不变
            self.lft_arm_target_pose[2] -= delta_height
            self.rgt_arm_target_pose[2] -= delta_height
        
        # =============== 左臂控制模式 ===============
        elif left_arm_mode:
            print("左臂控制模式")
            tmp_lft_arm_target_pose = self.lft_arm_target_pose.copy()

            # 使用左右摇杆控制手臂
            if axes_len >= 4:
                # 左摇杆控制手臂前后左右
                # 注意：axes[1]是前后方向，上推(负值)代表手臂向前，下拉(正值)代表手臂向后
                tmp_lft_arm_target_pose[1] += self.teleop.joy_cmd.axes[0] * 0.1 / self.render_fps
                # 注意：axes[0]是左右方向，左推(负值)代表手臂向左，右推(正值)代表手臂向右
                tmp_lft_arm_target_pose[0] += self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps
                
                # 右摇杆控制手臂上下
                # 注意：axes[2]是右摇杆上下，上推(负值)代表手臂向上，下拉(正值)代表手臂向下
                tmp_lft_arm_target_pose[2] += self.teleop.joy_cmd.axes[2] * 0.1 / self.render_fps
            
            # 夹爪控制
            delta_gripper = 0
            if axes_len >= 6:  # 使用L2/R2控制夹爪
                # L2(负值)关闭夹爪，R2(负值)打开夹爪
                delta_gripper = (self.teleop.joy_cmd.axes[5] - self.teleop.joy_cmd.axes[4]) * 1. / self.render_fps
            elif axes_len >= 5:  # 只有一个触发器
                delta_gripper = -self.teleop.joy_cmd.axes[4] * 1. / self.render_fps
            
            # 应用夹爪控制
            self.tctr_lft_gripper[0] += delta_gripper
            self.tctr_lft_gripper[0] = np.clip(self.tctr_lft_gripper[0], 0, 1)
            
            # 手臂旋转控制
            el = self.lft_end_euler.copy()
            if axes_len >= 4:
                # 右摇杆左右控制手臂旋转
                # 注意：axes[3]是右摇杆左右，左推(负值)代表手臂左旋，右推(正值)代表手臂右旋
                el[0] -= self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
                # 其他旋转轴
                if axes_len >= 6:
                    # 注意：axes[2]是右摇杆上下，上推(负值)代表手臂上旋，下拉(正值)代表手臂下旋
                    el[1] -= self.teleop.joy_cmd.axes[2] * 0.35 / self.render_fps
                    # 注意：axes[0]是左摇杆左右，左推(负值)代表手臂左旋，右推(正值)代表手臂右旋
                    el[2] -= self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            
            # 应用手臂控制
            try:
                self.tctr_left_arm[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(
                    tmp_lft_arm_target_pose, self.arm_action, "l", 
                    self.tctr_slide[0], self.tctr_left_arm, 
                    Rotation.from_euler('zyx', el).as_matrix()
                )
                self.lft_arm_target_pose[:] = tmp_lft_arm_target_pose
                self.lft_end_euler[:] = el
            except ValueError:
                print("左臂目标位置无效:", tmp_lft_arm_target_pose)
        
        # =============== 右臂控制模式 ===============
        elif right_arm_mode:
            print("右臂控制模式")
            tmp_rgt_arm_target_pose = self.rgt_arm_target_pose.copy()

            # 使用左右摇杆控制手臂
            if axes_len >= 4:
                # 左摇杆控制手臂前后左右
                # 注意：axes[1]是前后方向，上推(负值)代表手臂向前，下拉(正值)代表手臂向后
                tmp_rgt_arm_target_pose[1] += self.teleop.joy_cmd.axes[0] * 0.1 / self.render_fps
                # 注意：axes[0]是左右方向，左推(负值)代表手臂向左，右推(正值)代表手臂向右
                tmp_rgt_arm_target_pose[0] += self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps
                
                # 右摇杆控制手臂上下
                # 注意：axes[2]是右摇杆上下，上推(负值)代表手臂向上，下拉(正值)代表手臂向下
                tmp_rgt_arm_target_pose[2] += self.teleop.joy_cmd.axes[2] * 0.1 / self.render_fps
            
            # 夹爪控制
            delta_gripper = 0
            if axes_len >= 6:  # 使用L2/R2控制夹爪
                # L2(负值)关闭夹爪，R2(负值)打开夹爪
                delta_gripper = (self.teleop.joy_cmd.axes[5] - self.teleop.joy_cmd.axes[4]) * 1. / self.render_fps
            elif axes_len >= 5:  # 只有一个触发器
                delta_gripper = -self.teleop.joy_cmd.axes[4] * 1. / self.render_fps
            
            # 应用夹爪控制
            self.tctr_rgt_gripper[0] += delta_gripper
            self.tctr_rgt_gripper[0] = np.clip(self.tctr_rgt_gripper[0], 0, 1)
            
            # 手臂旋转控制
            el = self.rgt_end_euler.copy()
            if axes_len >= 4:
                # 右摇杆左右控制手臂旋转
                # 注意：axes[3]是右摇杆左右，左推(负值)代表手臂左旋，右推(正值)代表手臂右旋
                el[0] -= self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
                # 其他旋转轴
                if axes_len >= 6:
                    # 注意：axes[2]是右摇杆上下，上推(负值)代表手臂上旋，下拉(正值)代表手臂下旋
                    el[1] -= self.teleop.joy_cmd.axes[2] * 0.35 / self.render_fps
                    # 注意：axes[0]是左摇杆左右，左推(负值)代表手臂左旋，右推(正值)代表手臂右旋
                    el[2] -= self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            
            # 应用手臂控制
            try:
                self.tctr_right_arm[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(
                    tmp_rgt_arm_target_pose, self.arm_action, "r", 
                    self.tctr_slide[0], self.tctr_right_arm, 
                    Rotation.from_euler('zyx', el).as_matrix()
                )
                self.rgt_arm_target_pose[:] = tmp_rgt_arm_target_pose
                self.rgt_end_euler[:] = el
            except ValueError:
                print("右臂目标位置无效:", tmp_rgt_arm_target_pose)
        
        # 应用基本移动控制
        self.base_move(linear_vel, angular_vel)

    def base_move(self, linear_vel, angular_vel):
        self.tctr_base[0] = linear_vel
        self.tctr_base[1] = angular_vel

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

    def printMessage(self):
        super().printMessage()
        print("    lta local = {}".format(self.lft_arm_target_pose))
        print("    rta local = {}".format(self.rgt_arm_target_pose))
        print("       euler  = {}".format(self.lft_end_euler))
        print("       euler  = {}".format(self.rgt_end_euler))
        print("       head   = {}".format(self.tctr_head))
        

def main():
    # 初始化ROS2节点
    rclpy.init()

    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    
    # 创建MMK2配置
    cfg = MMK2Cfg()
    
    cfg.init_key = "pick"
    cfg.use_gaussian_renderer = False
    cfg.obs_rgb_cam_id = None
    cfg.obs_depth_cam_id = None
    
    cfg.render_set = {
        "fps"    : 25,
        # "fps"    : 30,
        "width"  : 1920,
        "height" : 1080
    }
    
    # 获取任务名称作为命令行参数
    import sys
    task_name = "task"
    if len(sys.argv) > 1:
        task_name = sys.argv[1]
    
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_main_copy.xml"
    cfg.render_mode = "window"
    
    # 创建并运行MMK2JOY实例
    env = MMK2JOY(cfg)
    env.resetState()
    
    # 设置任务名称
    env.task_name = task_name
    
    # 使用与ROS1版本相同的运行方式
    while env.running:
        env.teleopProcess()
        obs, _, _, _, _ = env.step(env.target_control)
    
    # # 清理ROS2资源
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
