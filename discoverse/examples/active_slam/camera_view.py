import os
import argparse

import cv2
import time
import json
import glfw
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import tkinter as tk
from tkinter import ttk

from dataclasses import dataclass
from typing import List, Tuple

from discoverse.envs import SimulatorBase
from discoverse.utils.base_config import BaseConfig
from discoverse.utils import interpolate_camera_poses
from discoverse import DISCOVERSE_ASSETS_DIR

@dataclass
class CameraViewpoint:
    position: Tuple[float, float, float]  # xyz
    quaternion: Tuple[float, float, float, float]  # wxyz
    name: str = ""  # 可选的视角名称
    
    # 使用下标[]访问
    def __getitem__(self, index):
        if index == 0:
            return self.position
        elif index == 1:
            return self.quaternion
        else:
            raise IndexError(f"Index {index} is out of bounds for CameraViewpoint")

    def to_dict(self):
        return {
            "position": list(self.position),
            "quaternion": list(self.quaternion),
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            position=tuple(data["position"]),
            quaternion=tuple(data["quaternion"]),
            name=data["name"]
        )

class ViewpointGUI:
    """视角管理GUI类"""
    def __init__(self, parent):
        self.parent = parent
        self.selected_index = -1
        self.setup_gui()
        
    def setup_gui(self):
        """初始化GUI界面"""
        # 创建主框架
        self.frame = ttk.Frame(self.parent)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建表格框架
        table_frame = ttk.Frame(self.frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建表格
        columns = ('Index', 'Position', 'Quaternion')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # 设置列标题和宽度比例
        self.tree.heading('Index', text='Index')
        self.tree.heading('Position', text='Position')
        self.tree.heading('Quaternion', text='Quaternion')
        
        # 设置列宽比例 1:6:8
        total_width = 600  # 窗口总宽度
        index_width = int(total_width * 0.067)  # 1/15
        position_width = int(total_width * 0.4)  # 6/15
        quaternion_width = int(total_width * 0.533)  # 8/15
        
        self.tree.column('Index', width=index_width, minwidth=40)
        self.tree.column('Position', width=position_width, minwidth=200)
        self.tree.column('Quaternion', width=quaternion_width, minwidth=300)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局表格和滚动条
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.tree.bind('<<TreeviewSelect>>', self.on_select)
        
        # 绑定DELETE键事件到Tkinter窗口
        self.parent.bind('<Delete>', self.on_delete)
        
        # 添加控制说明（放在底部）
        control_frame = ttk.LabelFrame(self.frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=5, side=tk.BOTTOM)
        
        controls = [
            "Space: Save current viewpoint",
            "Del: Delete selected viewpoint",
            "Left click: Select viewpoint"
        ]
        
        for control in controls:
            ttk.Label(control_frame, text=control).pack(anchor=tk.W, padx=5, pady=2)
    
    def update_viewpoints(self, viewpoints):
        """更新视角列表"""
        # 保存当前选中的索引
        current_selection = self.selected_index
        
        # 清除现有项目
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 添加新项目
        for i, vp in enumerate(viewpoints):
            pos_str = f"({vp.position[0]:.3f}, {vp.position[1]:.3f}, {vp.position[2]:.3f})"
            quat_str = f"({vp.quaternion[0]:.3f}, {vp.quaternion[1]:.3f}, {vp.quaternion[2]:.3f}, {vp.quaternion[3]:.3f})"
            item_id = self.tree.insert('', tk.END, values=(i, pos_str, quat_str))
            
            # 如果这个项目是之前选中的，重新选中它
            if i == current_selection:
                self.tree.selection_set(item_id)
    
    def on_select(self, event):
        """处理选择事件"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            if int(item['values'][0]) != self.selected_index:
                self.selected_index = int(item['values'][0])
                print(f"selected_index = {self.selected_index}")
                if hasattr(self.parent, 'env') and hasattr(self.parent.env, 'mj_model'):
                    self.parent.env.mj_model.body("stereo").pos[:] = self.parent.env.viewpoints[self.selected_index].position
                    self.parent.env.mj_model.body("stereo").quat[:] = self.parent.env.viewpoints[self.selected_index].quaternion

    def on_delete(self, event):
        """处理DELETE键事件"""
        if self.selected_index >= 0:
            # 通知环境删除选中的视角
            if hasattr(self.parent, 'env') and hasattr(self.parent.env, 'delete_selected_viewpoint'):
                self.parent.env.delete_selected_viewpoint()
            else:
                print("环境对象未找到或不支持删除视角操作")

class CamEnv(SimulatorBase):
    """相机环境类，用于控制和显示相机视角
    
    继承自SimulatorBase类，实现了一个可交互的相机环境，允许用户通过键盘和鼠标控制相机的移动和旋转。
    主要功能包括：
    - 支持双目相机设置
    - 实时渲染RGB和深度图像
    - 通过WASD键控制相机移动
    - 通过鼠标控制相机旋转
    """
    
    # 相机的偏航角和俯仰角
    camera_yaw = 0.0      # 偏航角(绕垂直轴旋转)
    camera_pitch = 0.0    # 俯仰角(抬头/低头)

    # 是否触发空格键
    key_space_trigger = False
    key_c_trigger = False

    mouse_in_tk = False

    def __init__(self, config: BaseConfig):
        """初始化相机环境
        
        Args:
            config: 包含环境配置参数的BaseConfig对象
        """
        super().__init__(config)
        self.viewpoints: List[CameraViewpoint] = []
        self.selected_viewpoint_index = -1
        self.show_gui = config.show_gui if hasattr(config, 'show_gui') else False
        
        if self.show_gui:
            self.setup_gui()

    def setup_gui(self):
        """初始化GUI界面"""
        self.root = tk.Tk()
        self.root.title("Camera Viewpoints")
        self.root.geometry("600x400")
        self.gui = ViewpointGUI(self.root)
        
        # 设置环境引用，让GUI可以访问环境方法
        self.root.env = self
        
        # 设置窗口置顶
        self.root.attributes('-topmost', True)
        
        # 绑定关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 添加鼠标事件处理
        self.root.bind('<Enter>', self.on_enter_window)
        self.root.bind('<Leave>', self.on_leave_window)

    def on_closing(self):
        """处理窗口关闭事件"""
        self.show_gui = False
        self.root.destroy()
    
    def update_gui(self):
        """更新GUI显示"""
        if not self.show_gui:
            return
            
        try:
            # 更新视角列表
            self.gui.update_viewpoints(self.viewpoints)
            
            # 更新选中的视角索引
            self.selected_viewpoint_index = self.gui.selected_index
            
            # 处理GUI事件
            self.root.update()
            
            # 如果鼠标在Tkinter窗口内，确保窗口获得焦点
            if self.mouse_in_tk:
                self.root.focus_force()
            
        except Exception as e:
            print(f"GUI更新错误: {e}")
            self.show_gui = False
            if hasattr(self, 'root'):
                self.root.destroy()
    
    def save_current_viewpoint(self):
        """保存当前相机视角"""
        position = tuple(self.mj_model.body("stereo").pos)
        quaternion = tuple(self.mj_model.body("stereo").quat) # wxyz
        
        viewpoint = CameraViewpoint(
            position=position,
            quaternion=quaternion,
            name=f"Viewpoint_{len(self.viewpoints)}"
        )
        
        self.viewpoints.append(viewpoint)
        self.update_gui()
        print(f"Saved viewpoint {len(self.viewpoints)-1}:")
        print(f"Position: {position}")
        print(f"Quaternion: {quaternion}")
    
    def delete_selected_viewpoint(self):
        """删除选中的视角"""
        print(self.selected_viewpoint_index)
        if 0 <= self.selected_viewpoint_index < len(self.viewpoints):
            deleted = self.viewpoints.pop(self.selected_viewpoint_index)
            print(f"Deleted viewpoint {self.selected_viewpoint_index}:")
            print(f"Position: {deleted.position}")
            print(f"Quaternion: {deleted.quaternion}")
            self.selected_viewpoint_index = -1
            self.update_gui()
    
    def updateControl(self, action):
        pass

    def post_load_mjcf(self):
        pass

    def getObservation(self):
        """获取环境观测信息
        
        Returns:
            dict: 包含以下键值对的字典：
                - rgb_cam_posi: RGB相机位姿列表 (position:(x,y,z), quaternion:(w,x,y,z))
                - depth_cam_posi: 深度相机位姿列表 (position:(x,y,z), quaternion:(w,x,y,z))
                - rgb_img: RGB图像字典，键为相机ID
                - depth_img: 深度图像字典，键为相机ID
        """
        rgb_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_rgb_cam_id]
        depth_cam_pose_lst = [self.getCameraPose(id) for id in self.config.obs_depth_cam_id]
        self.obs = {
            "rgb_cam_posi"   : rgb_cam_pose_lst,
            "depth_cam_posi" : depth_cam_pose_lst,
            "rgb_img"        : self.img_rgb_obs_s,
            "depth_img"      : self.img_depth_obs_s,
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs    

    def checkTerminated(self):
        return False
    
    def getReward(self):
         return None
    
    def on_enter_window(self, event):
        """鼠标进入窗口时"""
        self.mouse_in_tk = True

    def on_leave_window(self, event):
        """鼠标离开窗口时"""
        self.mouse_in_tk = False

    def on_mouse_move(self, window, xpos, ypos):
        """处理OpenGL窗口的鼠标移动事件"""
        # 如果鼠标在Tkinter窗口内，不处理OpenGL窗口的鼠标事件
        if self.mouse_in_tk:
            return
        
        if self.cam_id == -1:
            super().on_mouse_move(window, xpos, ypos)
        else:
            if self.mouse_pressed['left']:
                self.camera_pose_changed = True
                height = self.config.render_set["height"]
                dx = float(xpos) - self.mouse_pos["x"]
                dy = float(ypos) - self.mouse_pos["y"]
                # 根据鼠标移动更新相机角度
                self.camera_yaw -= 2. * dx / height    # 左右旋转
                self.camera_pitch += 2. * dy / height  # 上下俯仰

                # 将欧拉角转换为四元数并更新相机姿态
                quat_wxyz = Rotation.from_euler("xyz", [0.0, self.camera_pitch, self.camera_yaw]).as_quat()[[3,0,1,2]]
                self.mj_model.body("stereo").quat[:] = quat_wxyz

            self.mouse_pos['x'] = xpos
            self.mouse_pos['y'] = ypos

    def on_key(self, window, key, scancode, action, mods):
        """处理键盘事件
        
        实现了以下键盘控制：
        - WASD: 在水平面上移动相机
        - QE: 垂直方向移动相机
        - Shift: 按住可以增加移动速度
        
        Args:
            window: GLFW窗口对象
            key: 按键代码
            scancode: 扫描码
            action: 按键动作（按下/释放）
            mods: 修饰键状态
        """
        super().on_key(window, key, scancode, action, mods)

        # 检查Shift键是否按下，用于调整移动速度
        is_shift_pressed = (mods & glfw.MOD_SHIFT)
        move_step_ratio = 10.0 if is_shift_pressed else 3.0  # Shift按下时移动速度更快
        step = move_step_ratio / float(self.config.render_set["fps"])

        if action == glfw.PRESS or action == glfw.REPEAT:
            # 初始化移动方向
            dxlocal, dylocal, dz = 0.0, 0.0, 0.0
            
            # 检测WASDQE按键状态并更新移动方向
            if glfw.get_key(window, glfw.KEY_W):
                dxlocal += step  # 前进
            if glfw.get_key(window, glfw.KEY_S):
                dxlocal -= step  # 后退
            if glfw.get_key(window, glfw.KEY_A):
                dylocal += step  # 左移
            if glfw.get_key(window, glfw.KEY_D):
                dylocal -= step  # 右移
            if glfw.get_key(window, glfw.KEY_Q):
                dz += step      # 上升
            if glfw.get_key(window, glfw.KEY_E):
                dz -= step      # 下降

            # 根据当前相机朝向计算实际移动方向并更新相机位置
            self.mj_model.body("stereo").pos[0] += dxlocal * np.cos(self.camera_yaw) - dylocal * np.sin(self.camera_yaw)
            self.mj_model.body("stereo").pos[1] += dxlocal * np.sin(self.camera_yaw) + dylocal * np.cos(self.camera_yaw)
            self.mj_model.body("stereo").pos[2] += dz

            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                self.key_space_trigger = True
                if self.show_gui:
                    self.save_current_viewpoint()

            if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS:
                self.key_c_trigger = True

            if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
                if len(self.viewpoints) > 0:
                    with open(os.path.join(os.path.dirname(self.config.gsply), "camera_list.json"), "w") as f:
                        json.dump([vp.to_dict() for vp in self.viewpoints], f)
                    print(f"Saved camera list to {os.path.join(os.path.dirname(self.config.gsply), 'camera_list.json')}")

    def printHelp(self):
        """打印帮助信息，显示控制说明"""
        print("Camera View")
        print("W/S/A/D: 移动相机")
        print("Q/E: 升降相机")
        print("Shift: 加快移动速度")
        print("左键按住拖动: 旋转相机视角")
        print("ESC: 切换到自由视角")
        print("]/[: 切换相机")
        print("Space: 保存当前视角和图像")
        print("Ctrl+G: 切换高斯渲染")
        print("Ctrl+D: 切换深度渲染")
        if self.show_gui:
            print("在视角管理窗口中，选中视角后按Delete键可删除视角")

if __name__ == "__main__":
    # 设置numpy打印格式
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='相机环境程序')
    parser.add_argument('--gsply', type=str, help='高斯渲染模型文件路径(.ply)', default=os.path.join(DISCOVERSE_ASSETS_DIR, "3dgs/scene/drobotics_canteen/point_cloud.ply"))
    parser.add_argument('--mesh', type=str, default=None, help='场景网格文件路径(.obj)')
    parser.add_argument('--max-depth', type=float, default=5.0, help='最大渲染深度')
    parser.add_argument('--camera-distance', type=float, default=0.1, help='双目相机基线距离')
    parser.add_argument('--fovy', type=float, default=75.0, help='相机视场角(度)')
    parser.add_argument('--width', type=int, default=1280, help='渲染图像宽度')
    parser.add_argument('--height', type=int, default=720, help='渲染图像高度')
    parser.add_argument('--show-gui', action='store_true', help='是否显示视角管理GUI界面')
    parser.add_argument('-cp', '--camera-pose-path', type=str, default=None, help='相机位姿文件路径')
    parser.add_argument('-ni', '--num-interpolate', type=int, default=0, help='插值相机位姿的总数')
    args = parser.parse_args()

    # 检查高斯渲染模型文件是否存在
    if not os.path.exists(args.gsply):
        raise FileNotFoundError(f"gsply文件不存在: {args.gsply}")

    # 准备场景网格的XML描述
    asset_xml = ''
    geom_xml = ''
    if args.mesh is None:
        # 如果未指定mesh文件，尝试使用默认的scene.obj
        obj_path = os.path.join(os.path.dirname(args.gsply), "scene.obj")
        if os.path.exists(obj_path):
            asset_xml = f'  <asset>\n    <mesh name="scene" file="{obj_path}"/>\n  </asset>'
            geom_xml = f'    <geom type="mesh" rgba="0.5 0.5 0.5 1" mesh="scene"/>'
    elif os.path.exists(args.mesh):
        # 使用指定的mesh文件
        asset_xml = f'  <asset>\n    <mesh name="scene" file="{args.mesh}"/>\n  </asset>'
        geom_xml = f'    <geom type="mesh" rgba="0.5 0.5 0.5 1" mesh="scene"/>'

    camera_poses = None
    camera_xml_str = ''
    if args.camera_pose_path is not None:
        with open(args.camera_pose_path, 'r') as f:
            camera_poses = json.load(f)
            for i, pose in enumerate(camera_poses):
                position = pose["position"]
                quaternion = pose["quaternion"]
                camera_xml_str += f'    <site name="traj_{i}" pos="{position[0]} {position[1]} {position[2]}" quat="{quaternion[0]} {quaternion[1]} {quaternion[2]} {quaternion[3]}"/>\n' 
            
    # 构建MuJoCo场景XML
    camera_env_xml = f"""
    <mujoco model="camera_env">
      <option integrator="RK4" solver="Newton" gravity="0 0 0"/>
      {asset_xml}
      <worldbody>
        {geom_xml}
        <body name="stereo" pos="0 0 1" quat="1 0 0 0">
          <camera name="camera_left" fovy="{args.fovy}" pos="0 {args.camera_distance/2.} 0" quat="0.5 0.5 -0.5 -0.5"/>
          <site pos="0 {args.camera_distance/2.} 0" quat="0.5 -0.5 0.5 -0.5"/>
          <camera name="camera_right" fovy="{args.fovy}" pos="0 {-args.camera_distance/2.} 0" quat="0.5 0.5 -0.5 -0.5"/>
          <site pos="0 {-args.camera_distance/2.} 0" quat="0.5 -0.5 0.5 -0.5"/>
          <geom type="box" size="0.05 0.2 0.05" rgba="1 1 1 0."/>
        </body>
        {camera_xml_str}
      </worldbody>
    </mujoco>"""

    # 临时保存场景XML文件
    xml_save_path = os.path.join(os.path.dirname(args.gsply), "_camera_env.xml")

    # 配置环境参数
    cfg = BaseConfig()
    cfg.gsply = args.gsply
    cfg.render_set["fps"] = 30                    # 渲染帧率
    cfg.render_set["width"] = args.width          # 渲染宽度
    cfg.render_set["height"] = args.height        # 渲染高度
    cfg.timestep = 1./cfg.render_set["fps"]       # 时间步长
    cfg.decimation = 1                            # 仿真步数
    cfg.mjcf_file_path = xml_save_path           # MuJoCo场景文件路径
    cfg.max_render_depth = args.max_depth        # 最大渲染深度
    cfg.obs_rgb_cam_id = [0, 1]                  # 启用的RGB相机ID（左右相机）
    cfg.obs_depth_cam_id = [0, 1]                # 启用的深度相机ID（左右相机）
    cfg.show_gui = args.show_gui

    # 配置高斯渲染器
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = args.gsply

    # 检查并加载环境点云
    env_ply_path = os.path.join(os.path.dirname(args.gsply), "environment.ply")
    if os.path.exists(env_ply_path):
        cfg.gs_model_dict["background_env"] = env_ply_path

    # 创建并配置相机环境
    with open(xml_save_path, "w") as f:
        f.write(camera_env_xml)

    robot = CamEnv(cfg)
    robot.options.label = mujoco.mjtLabel.mjLABEL_CAMERA.value
    robot.options.frame = mujoco.mjtFrame.mjFRAME_SITE.value
    robot.options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
    os.remove(xml_save_path)

    if camera_poses is not None:
        for pose in camera_poses:
            robot.viewpoints.append(CameraViewpoint(**pose))

    # 重置环境并获取初始观测
    obs = robot.reset()
    rgb_cam_posi = obs["rgb_cam_posi"]
    depth_cam_posi = obs["depth_cam_posi"]
    rgb_img_0 = obs["rgb_img"][0]
    rgb_img_1 = obs["rgb_img"][1]
    depth_img_0 = obs["depth_img"][0]
    depth_img_1 = obs["depth_img"][1]

    # 打印左相机信息
    print('>>>>>> camera_left:')
    print("rgb_cam_posi    = ", rgb_cam_posi[0])
    print("depth_cam_posi  = ", depth_cam_posi[0])
    print("rgb_img_0.shape = ", rgb_img_0.shape, "rgb_img_0.dtype = ", rgb_img_0.dtype)
    print("depth_img_0.shape = ", depth_img_0.shape, "depth_img_0.dtype = ", depth_img_0.dtype)

    # 打印右相机信息
    print('>>>>>> camera_right:')
    print("rgb_cam_posi    = ", rgb_cam_posi[1])
    print("depth_cam_posi  = ", depth_cam_posi[1])
    print("rgb_img_1.shape = ", rgb_img_1.shape, "rgb_img_1.dtype = ", rgb_img_1.dtype)
    print("depth_img_1.shape = ", depth_img_1.shape, "depth_img_1.dtype = ", depth_img_1.dtype)

    # 设置初始相机和视角参数
    robot.cam_id = 0                  # 默认使用左相机视角
    robot.free_camera.distance = 1.   # 设置自由视角相机距离

    if args.num_interpolate > 0 and len(robot.viewpoints) > 1:
        robot.config.sync = False
        viewpoints = interpolate_camera_poses(robot.viewpoints, args.num_interpolate)
        os.mkdir(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints"))
        camera_poses_cam1 = []
        camera_poses_cam2 = []
        for i, vp in enumerate(viewpoints):
            robot.mj_model.body("stereo").pos[:] = vp[0]
            robot.mj_model.body("stereo").quat[:] = vp[1]
            obs, _, _, _, _ = robot.step()

            # 保存 图像和相机外参
            rgb_img_0 = obs["rgb_img"][0]
            rgb_img_1 = obs["rgb_img"][1]
            depth_img_0 = obs["depth_img"][0]
            depth_img_1 = obs["depth_img"][1]
            cv2.imwrite(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", f"rgb_img_0_{i}.png"), rgb_img_0)
            cv2.imwrite(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", f"rgb_img_1_{i}.png"), rgb_img_1)
            np.save(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", f"depth_img_0_{i}.npy"), depth_img_0)
            np.save(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", f"depth_img_1_{i}.npy"), depth_img_1)

            # 相机外参
            camera_poses_cam1.append({
                "name": f"traj_{i}_cam0",
                "position": list(obs["rgb_cam_posi"][0][0]),
                "quaternion": list(obs["rgb_cam_posi"][0][1])
            })
            camera_poses_cam2.append({
                "name": f"traj_{i}_cam1",
                "position": list(obs["depth_cam_posi"][1][0]),
                "quaternion": list(obs["depth_cam_posi"][1][1])
            })
        with open(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", "camera_poses_cam1.json"), "w") as f:
            json.dump(camera_poses_cam1, f)
        with open(os.path.join(os.path.dirname(args.gsply), "interpolate_viewpoints", "camera_poses_cam2.json"), "w") as f:
            json.dump(camera_poses_cam2, f)
        exit()

    # 打印控制说明
    print("-" * 50)
    robot.printHelp()
    print("-" * 50)

    robot.config.obs_rgb_cam_id = []
    robot.config.obs_depth_cam_id = []

    # 主循环
    while robot.running:
        obs, _, _, _, _ = robot.step()
        if robot.key_c_trigger:
            robot.key_c_trigger = False
            img_vis = robot.getRgbImg(robot.cam_id)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            cv2.imwrite(f"camera_view_{stamp}.png", cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        if robot.show_gui:
            try:
                robot.update_gui()
            except Exception as e:
                print(f"GUI更新错误: {e}")
                robot.show_gui = False
                if hasattr(robot, 'root'):
                    robot.root.destroy()

    # 在程序结束前清理GUI
    if robot.show_gui and hasattr(robot, 'root'):
        try:
            robot.root.destroy()
        except Exception as e:
            print(f"GUI清理错误: {e}")
