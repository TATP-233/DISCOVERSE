import os
import sys
import time
import json
import platform
import argparse
import traceback
import threading
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import mujoco.viewer
np.set_printoptions(precision=5, suppress=True, linewidth=500)

import mink
import discoverse
from discoverse.envs import make_env
from discoverse.examples.mocap_ik.mocap_ik_utils import \
    add_mocup_body_to_mjcf, \
    generate_mocap_xml
from discoverse.examples.force_control.impedance_control import ImpedanceController
from discoverse.utils import (
    get_mocap_tmat, 
    get_site_tmat, 
    get_body_tmat,
    get_control_idx,
    get_sensor_idx
)
from discoverse.universal_manipulation.recorder import PyavImageEncoder
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR

class Manipulator:
    MINK_SOLVER = "quadprog"
    MINK_POS_THRESHOLD = 1e-3
    MINK_ORI_THRESHOLD = 1e-3
    MINK_MAX_ITERS = 10

    #################################################
    # inference interface
    target_position = np.zeros(3)
    target_quat_wxyz = np.zeros(4)

    #################################################
    def __init__(self, args):
        self.robot_name = args.robot
        self.task_name = args.task
        self.inference_mode = args.inference
        self.inference_hz = args.infer_hz
        if self.inference_mode:
            args.mouse_3d = False

        self.mjcf_path = self._prepare_mjcf(args)
        self.arm_dof = self._prepare_dof(self.mjcf_path)
        self.mj_model, self.mj_data = self._prepare_m_d(self.mjcf_path)
        self._prepare_viewer(args)
        if args.mouse_3d:
            self._prepare_3dmouse()
            self.use_3dmouse = True
        else:
            self.use_3dmouse = False

        self.impedance_control = ("force" in self.mjcf_path)
        self.ID_controller = self._prepare_impedance_controller()
        self._prepare_recorder(args)
        self._prepare_control_idx()
        self._prepare_sensors_idx()
        self._prepare_mink()

        self.reset()

    def run(self):
        self.last_select = self.viewer.perturb.select
        try:
            while self.viewer.is_running():
                step_start = time.time()
                if self.enable_record and self.task_success:
                    self.total_record_cnt += 1
                    self.recoder_state_file(self.save_dir, self.obs_lst)
                    self.reset_sig = True
                
                if self.reset_sig:
                    self.reset()
                
                if self.inference_mode:
                    self._upate_policy()

                self.step()

                if self.enable_record and len(self.obs_lst) <= self.mj_data.time * self.record_frequency:
                    self.record_once()

                if self.ID_controller and self.ID_controller.enable_plot:
                    ee_force = self.ID_controller.run_step(dt=self.mj_model.opt.timestep)
                    # print(self.mj_model.opt.timestep)

                # 计算下一步开始前需要等待的时间，保证帧率稳定
                time_until_next_step = (1. / self.render_fps) - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


        except KeyboardInterrupt:
            print("用户中断，退出程序")

        finally:
            self.close()
            print("程序结束")

    def step(self):
        self.pre_step()

        # 执行渲染间隔次数的物理仿真步骤
        for _ in range(self.render_gap):
            if self.impedance_control: 
                self.ID_controller.update_state(
                    self.mj_data.sensordata[self.joint_q_sensor_idx], 
                    self.mj_data.sensordata[self.joint_dq_sensor_idx], 
                    self.mj_data.sensordata[self.joint_tau_sensor_idx]
                )
                # ee_force = self.ID_controller.get_ext_force()
                torque = self.ID_controller.compute_torque()
                self.mj_data.ctrl[:self.arm_dof] = torque[:]

            # 执行物理仿真步骤
            mujoco.mj_step(self.mj_model, self.mj_data)
        

        

        t1 = self.mj_data.time
        self.viewer.sync()
        t2 = self.mj_data.time
        if t1 > t2:
            self.reset_sig = True
        self.post_step()

    def reset(self):
        self.reset_sig = False
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._reset_mink()
        if self.enable_record:
            self._reset_recorder()

    def close(self):
        if self.viewer:
            self.viewer.close()
            del self.viewer
        if self.enable_record and len(self.camera_names):
            for ec in self.camera_encoders.values():
                try:
                    ec.close()
                except Exception:
                    pass
            for k in tuple(self.camera_encoders.keys()):
                del self.camera_encoders[k]
        if self.ID_controller:
            self.ID_controller.close()

    def record_once(self):
        obs = self.get_observation()
        # imgs = obs.pop('img')
        imgs = obs['img']
        for cam_id, img in imgs.items():
            self.camera_encoders[cam_id].encode(img, obs["time"])
        self.obs_lst.append(obs)

    def get_observation(self):
        if hasattr(self, "obs") and np.abs(self.obs["time"] - self.mj_data.time) < 1e-6:
            return self.obs

        tmat_target = get_mocap_tmat(self.mj_data, self.mocap_id)
        tmat_arm_base = get_site_tmat(self.mj_data, "armbase")
        tmat_target_local = np.linalg.inv(tmat_arm_base) @ tmat_target
        target_position = tmat_target_local[:3, 3]
        target_quat = Rotation.from_matrix(tmat_target_local[:3, :3]).as_quat()[[3,0,1,2]]
        self.obs = {
            "time": self.mj_data.time,
            "jq"  : self.mj_data.sensordata[self.joint_q_sensor_idx].tolist(),
            "jv"  : self.mj_data.sensordata[self.joint_dq_sensor_idx].tolist(),
            "tau" : self.mj_data.sensordata[self.joint_tau_sensor_idx].tolist(),
            "eef_pos" : self.mj_data.sensordata[self.eef_sensor_idx["endpoint_pos"]:self.eef_sensor_idx["endpoint_pos"]+3].tolist(),
            "eef_quat": self.mj_data.sensordata[self.eef_sensor_idx["endpoint_quat"]:self.eef_sensor_idx["endpoint_quat"]+4].tolist(),
            "eef_vel" : self.mj_data.sensordata[self.eef_sensor_idx["endpoint_vel"]:self.eef_sensor_idx["endpoint_vel"]+3].tolist(),
            "eef_gyro": self.mj_data.sensordata[self.eef_sensor_idx["endpoint_gyro"]:self.eef_sensor_idx["endpoint_gyro"]+3].tolist(),
            "eef_acc" : self.mj_data.sensordata[self.eef_sensor_idx["endpoint_acc"]:self.eef_sensor_idx["endpoint_acc"]+3].tolist(),
            "end_force" : self.ID_controller.get_ext_force().tolist() if self.impedance_control else [],
            "action" : np.hstack([target_position, target_quat]).tolist(),
            "img" : {}
        }
        for camera_name in self.camera_names:
            self.renderer.update_scene(self.mj_data, camera_name)
            self.obs["img"][camera_name] = self.renderer.render()
        return self.obs

    def converge_ik(self, dt):
        for _ in range(self.MINK_MAX_ITERS):
            vel = mink.solve_ik(self.configuration, self.mink_tasks, dt, self.MINK_SOLVER, 1e-3)
            self.configuration.integrate_inplace(vel, dt)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.MINK_POS_THRESHOLD
            ori_achieved = np.linalg.norm(err[3:]) <= self.MINK_ORI_THRESHOLD

            if pos_achieved and ori_achieved:
                return True
        return False

    def pre_step(self):
        if self.use_3dmouse:
            self._proc_3dmouse()
        
        self._proc_mink_ik()
        self.task_success = self._proc_task_referee()

    def post_step(self):
        self._proc_perturb()

    def key_press_callback(self, key):
        pass

    def _prepare_mjcf(self, args):
        if args.robot is not None and args.task is not None:
            mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "tmp", f"{args.robot}_{args.task}.xml")
            env = make_env(args.robot, args.task, mjcf_path)
            env.export_xml(mjcf_path)
        elif args.robot is not None:
            mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", f"robot_{args.robot}.xml")
        elif args.mjcf is not None:
            mjcf_path = args.mjcf
            # 加载机器人模型的MJCF文件
            if not os.path.exists(mjcf_path):
                paths = [
                    os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", mjcf_path),
                    os.path.join(DISCOVERSE_ASSETS_DIR, mjcf_path),
                    os.path.join(os.getcwd(), mjcf_path),
                ]
                for path in paths:
                    if os.path.exists(path) and os.path.isfile(path) and (path.endswith(".xml") or path.endswith(".mjb")):
                        mjcf_path = path
                        break
                else:
                    raise FileNotFoundError(f"MJCF file not found: {mjcf_path}")
        else:
            mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", "robot_airbot_play.xml")        

        return mjcf_path

    def _prepare_dof(self, mjcf_path):
        if "airbot_play" in mjcf_path:
            arm_dof = 6
        elif "arx_l5" in mjcf_path:
            arm_dof = 6
        elif "arx_x5" in mjcf_path:
            arm_dof = 6
        elif "piper" in mjcf_path:
            arm_dof = 6
        elif "rm65" in mjcf_path:
            arm_dof = 6
        elif "ur5e" in mjcf_path:
            arm_dof = 6
        elif "panda" in mjcf_path:
            arm_dof = 7
        elif "iiwa14" in mjcf_path:
            arm_dof = 7
        elif "xarm7" in mjcf_path:
            arm_dof = 7
        else:
            raise ValueError(f"Unsupported robot: {self.robot_name}")
        return arm_dof

    def _prepare_m_d(self, mjcf_path):
        # 设置末端执行器目标（mocap）名称
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mocap_name = "target"
        self.mocap_box_name = self.mocap_name + "_box"
        try:
            mid = mj_model.body(self.mocap_name).mocapid[0]
            if mid == -1:
                raise KeyError(f"Mocap body '{self.mocap_name}' not found")
        except KeyError:
            # 生成mocap刚体XML元素
            mocap_body_element = generate_mocap_xml(self.mocap_name)
            # 将mocap刚体添加到模型中
            mj_model = add_mocup_body_to_mjcf(mjcf_path, [mocap_body_element])
        self.mocap_id = mj_model.body(self.mocap_name).mocapid[0]

        mj_data = mujoco.MjData(mj_model)
        return mj_model, mj_data

    def _prepare_impedance_controller(self):
        if self.impedance_control:
            mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf", "manipulator", f"robot_{self.robot_name}.xml")
            mj_model_idc = mujoco.MjModel.from_xml_path(mjcf_path)
            kp = [100, 100, 100, 5, 100, 5]
            kd = [5, 5, 5, 0.5, 0.5, 0.1]
            enable_plot = False
            # enable_plot=args.plot
            return ImpedanceController(mj_model_idc, kp, kd, enable_plot)
        else:
            return None

    def _prepare_control_idx(self):
        # get_control_idx(self.mj_model)
        pass

    def _prepare_sensors_idx(self):
        self.joint_q_sensor_idx = np.array(list(get_sensor_idx(self.mj_model, [f"joint{i+1}_pos" for i in range(self.arm_dof)], check=self.enable_record).values()))
        self.joint_dq_sensor_idx = np.array(list(get_sensor_idx(self.mj_model, [f"joint{i+1}_vel" for i in range(self.arm_dof)], check=self.enable_record).values()))
        self.joint_tau_sensor_idx = np.array(list(get_sensor_idx(self.mj_model, [f"joint{i+1}_torque" for i in range(self.arm_dof)], check=self.enable_record).values()))
        self.eef_sensor_idx = get_sensor_idx(self.mj_model, ["endpoint_pos", "endpoint_quat", "endpoint_vel", "endpoint_gyro", "endpoint_acc"], check=self.enable_record)

    def _prepare_mink(self):
        # Create a Mink configuration
        self.configuration = mink.Configuration(self.mj_model)
        # Define tasks
        self.end_effector_task = mink.FrameTask(
            frame_name="endpoint",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.mj_model, cost=1e-2)
        self.mink_tasks = [self.end_effector_task, self.posture_task]

    def _prepare_recorder(self, args):
        self.enable_record = args.record
        self.camera_names = args.camera_names
        self.record_frequency = args.record_frequency or 24
        self.total_record_cnt = 0
        self.renderer = mujoco.Renderer(self.mj_model) if len(self.camera_names) else None
        if self.renderer:
            self.img_width = 640
            self.img_height = 480
            self._set_renderer_size(self.img_width, self.img_height)

        self.task_success = False
        self.obs_lst = []
        self.save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", f"{self.robot_name}_{self.task_name}", f"{self.total_record_cnt:03d}")
        self._set_encoders(self.camera_names)

    def _prepare_viewer(self, args):
        # 设置渲染帧率
        self.render_fps = 125.0
        # 计算渲染间隔，确保按照指定帧率渲染
        self.render_gap = int(1.0 / self.render_fps / self.mj_model.opt.timestep)

        try:
            # 启动MuJoCo查看器
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, 
                show_left_ui=False, 
                show_right_ui=False,
                key_callback=self.key_press_callback
            )
            if not args.hide_mocap:
                self.viewer.opt.geomgroup[5] = 1  # 显示group 5中的几何体
        except RuntimeError as e:
            if "mjpython" in str(e) and platform.system() == "Darwin":
                print("\n错误: 在macOS上必须使用mjpython运行此脚本")
                print("请使用以下命令:")
                print(f"mjpython {' '.join(sys.argv)}")
            else:
                print(f"运行时错误: {e}")
            sys.exit(1)

    def _prepare_3dmouse(self):
        import pyspacemouse
        self.smouse = pyspacemouse
        success = self.smouse.open()
        if not success:
            print("3D鼠标打开失败，请检查设备连接")
            sys.exit(1)

    def _reset_recorder(self):
        self.task_success = False
        self.obs_lst = []
        self.save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", f"{self.robot_name}_{self.task_name}", f"{self.total_record_cnt:03d}")
        self._set_encoders(self.camera_names)
    
    def _set_encoders(self, camera_names):
        if hasattr(self, 'camera_encoders'):
            for ec in self.camera_encoders.values():
                try:
                    ec.close()
                except Exception:
                    pass
        self.camera_encoders = {}
        for cam_name in camera_names:
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
                print(f"Camera '{cam_name}' not found in the MJCF model.")
            else:
                self.camera_encoders[cam_name] = PyavImageEncoder(
                    self.img_width,
                    self.img_height,
                    self.save_dir,
                    cam_name,
                    fps=self.record_frequency,
                )
    
    def _reset_mink(self):
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, self.mocap_name, "endpoint", "site")
        self.configuration.update(self.mj_data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
    
    def _proc_perturb(self):
        if self.last_select != self.viewer.perturb.select:
            body_name = self.mj_model.body(self.viewer.perturb.select).name
            print(f"选择物体: {body_name}")
            tmat_body = get_body_tmat(self.mj_data, body_name)
            if body_name != self.mocap_name:
                print(">>> select object:")
                print(tmat_body)
            print(">>> target_body: ")
            tmat_mocap = get_mocap_tmat(self.mj_data, self.mocap_id)
            print(tmat_mocap)
            print(">>> object2target")
            print(np.linalg.inv(tmat_body) @ tmat_mocap)
            self.last_select = self.viewer.perturb.select
    
    def _proc_3dmouse(self):
        state = self.smouse.read()
        delta_position = np.array([state.y, -state.x, state.z])
        delta_position *= (np.abs(delta_position) > 0.01)
        delta_position = np.pow(delta_position, 2) * np.sign(delta_position)
        # delta_euler = np.array([-state.roll, -state.pitch, state.yaw])
        # tmat_mocap = get_mocap_tmat(mj_data, mocap_id)
        rmat_base = get_site_tmat(self.mj_data, "armbase")[:3,:3]
        self.mj_data.mocap_pos[self.mocap_id] += (rmat_base @ delta_position) * 0.2 / self.render_fps
    
    def _proc_mink_ik(self):
        if self.inference_mode:
            mink_target = get_site_tmat(self.mj_data, "armbase") @ \
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(self.target_quat_wxyz),
                    translation=self.target_position
                ).as_matrix()
            self.mj_data.mocap_pos[self.mocap_id] = mink_target[:3,3]
            self.mj_data.mocap_quat[self.mocap_id] = Rotation.from_matrix(mink_target[:3,:3]).as_quat()[[3,0,1,2]]

        mink_target_se3 = mink.SE3.from_mocap_name(self.mj_model, self.mj_data, self.mocap_name)
        self.end_effector_task.set_target(mink_target_se3)
        res = self.converge_ik(self.mj_model.opt.timestep)
        if not self.inference_mode:
            if res:
                # 设置目标框为绿色（表示IK计算成功）
                self.mj_model.geom(self.mocap_box_name).rgba = (0.3, 0.6, 0.3, 0.2)
            else:
                # 设置目标框为红色（表示IK计算失败）
                self.mj_model.geom(self.mocap_box_name).rgba = (0.6, 0.3, 0.3, 0.2)

        if self.impedance_control:
            q_desired = self.configuration.q[:self.arm_dof]
            self.ID_controller.set_target(q_desired)
        else:
            self.mj_data.ctrl[:self.arm_dof] = self.configuration.q[:self.arm_dof]

    def _proc_task_referee(self):
        success = False
        if self.task_name == "peg_in_hole":
            tmat_endpoint = get_site_tmat(self.mj_data, "endpoint")
            tmat_hole = get_body_tmat(self.mj_data, "hole")
            dist_xy = np.linalg.norm(tmat_endpoint[:2,3] - tmat_hole[:2,3])
            dist_z = tmat_endpoint[2,3] - tmat_hole[2,3]
            if dist_xy < 0.005 and dist_z < 0.001 and np.abs(tmat_endpoint[0,2]) > 0.9996573249755573:
                self.mj_model.geom("peg_box").rgba = (0, 1, 0, 1)
                success = True
            else:
                self.mj_model.geom("peg_box").rgba = (1, 0, 0, 1)
        return success
                
    def _set_renderer_size(self, width, height):
        if self.renderer is None:
            raise ValueError("Renderer is not initialized.")
        else:
            self.renderer._width = width
            self.renderer._height = height
            self.renderer._rect.width = width
            self.renderer._rect.height = height

    def recoder_state_file(self, save_path, obs_lst):
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
            save_dict = {
                "time" : [],
                "obs"  : {
                    "jq" : [],      # 6个关节角度
                    "jv" : [],      # 6个关节速度
                    "tau": [],      # 6个关节力矩
                    "eef_pos" : [], # 末端执行器位置xyz  armbase坐标系
                    "eef_quat": [], # 末端执行器姿态wxyz armbase坐标系
                    "eef_vel" : [], # 末端执行器速度xyz  armbase坐标系
                    "eef_gyro": [], # 末端imu角速度
                    "eef_acc" : [], # 末端imu加速度
                    "end_force" : [], # 动力学算出来的6维度wrench 训练只用前三维
                },
                "act"  : [] # 末端位姿控制 xyz wxyz armbase坐标系
            }
            for obs in obs_lst:
                save_dict["time"].append(obs['time'])
                save_dict["obs"]["jq"].append(obs['jq'])
                save_dict["obs"]["jv"].append(obs['jv'])
                save_dict["obs"]["tau"].append(obs['tau'])
                save_dict["obs"]["eef_pos"].append(obs['eef_pos'])
                save_dict["obs"]["eef_quat"].append(obs['eef_quat'])
                save_dict["obs"]["eef_vel"].append(obs['eef_vel'])
                save_dict["obs"]["eef_gyro"].append(obs['eef_gyro'])
                save_dict["obs"]["eef_acc"].append(obs['eef_acc'])
                save_dict["obs"]["end_force"].append(obs['end_force'])
                save_dict["act"].append(obs['action'])
            json.dump(save_dict, fp)

    def _upate_policy(self):
        if not hasattr(self, "last_infer_time"):
            self.last_infer_time = self.mj_data.time
            to_update_policy = True
        elif self.mj_data.time - self.last_infer_time >= 1./self.inference_hz:
            to_update_policy = True
        else:
            to_update_policy = False
        if to_update_policy:
            self.policy_function()

    def policy_function(self):
        """
        :TO: GHZ
        obs = self.get_observation()
        action = your_policy_network(obs)
        tart_position = xxxx
        tart_orientation = xxx
        self.target_position = tart_position
        self.target_quat_wxyz = tart_orientation
        """
        raise NotImplementedError

def parse_args():
    parser = argparse.ArgumentParser(
        description="Airbot Play 机器人MuJoCo仿真主程序\n"
                    "用法示例：\n"
                    "  python mocap_ik_airbot_play.py [--mjcf]\n"
                    "参数说明：\n"
                    "  mjcf  (可选) 指定MJCF模型文件路径，若不指定则使用默认模型。"
    )
    parser.add_argument(
        "-r", "--robot",
        type=str,
        default=None,
        help="输入机器人模型名称",
        choices=["airbot_play", "airbot_play_force", "arx_l5", "arx_x5", "iiwa14", "panda", "piper", "rm65", "ur5e", "xarm7"],
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default=None,
        help="输入目标名称",
        choices=["block_bridge_place", "close_laptop", "cover_cup", "open_drawer", "peg_in_hole", "pick_jujube", "place_block", "place_coffeecup", "place_jujube", "place_jujube_coffeecup", "place_kiwi_fruit", "push_mouse", "stack_block"],
    )
    parser.add_argument(
        "-m", "--mjcf",
        type=str,
        default=None,
        help="输入MJCF文件的路径（可选）。如未指定，则使用默认的robot_airbot_play.xml"
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="在macOS上跳过mjpython提示，直接尝试启动查看器",
    )
    parser.add_argument(
        "--mouse-3d",
        action="store_true",
        help="启用3D鼠标进行机械臂控制（需要3D鼠标硬件支持）"
    )
    parser.add_argument(
        "--hide-mocap",
        action="store_true",
        help="隐藏运动捕捉目标"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="启用记录功能"
    )
    parser.add_argument(
        "--record-frequency",
        type=int, default=24,
        help="设置记录频率（单位：Hz）"
    )
    parser.add_argument(
        "--camera-names",
        type=str, nargs='*', default=[],
        help="指定需要渲染的相机名称列表（可选）"
    )
    parser.add_argument(
        '--inference',
        action='store_true',
        help='启用推理模式'
    )
    parser.add_argument(
        '--infer-hz',
        type=float, default=25,
        help="推理频率"
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help="开启画图"
    )

    return parser.parse_args()

if __name__ == "__main__":
    """
    机械臂的仿真主程序
    
    该程序创建一个机械臂模型的MuJoCo仿真环境，添加运动捕捉(mocap)目标，
    并使用逆运动学(IK)控制机器人的单臂跟踪目标位置和姿态。
    """
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)
    args = parse_args()

    # 检查是否在macOS上运行并给出适当的提示
    if platform.system() == "Darwin" and not args.y:
        print("\n===================================================")
        print("注意: 在macOS上运行MuJoCo查看器需要使用mjpython")
        print("请使用以下命令运行此脚本:")
        print(f"mjpython {' '.join(sys.argv)}")
        print("===================================================\n")
        
        user_input = input("是否继续尝试启动查看器? (y/n): ")
        if user_input.lower() != 'y':
            print("退出程序。")
            sys.exit(0)

    # 打印MuJoCo查看器的使用提示信息
    print("\n===================== MuJoCo 查看器使用说明 =====================")
    print("1. 双击选中机械臂末端的绿色方块，按下ctrl和鼠标左键，拖动鼠标可以旋转机械臂")
    print("   按下ctrl和鼠标右键，拖动鼠标可以平移机械臂末端")
    print("2. 按 Tab 键切换左侧 UI 的可视化界面。")
    print("   按 Shift+Tab 键切换右侧 UI 的可视化界面。")
    print("3. 在左侧 UI 中点击 'Copy State' 可以当前的机器人状态复制到剪贴板。")
    print("4. 在右侧 UI 的 control 面板中可以调节 gripper 滑块控制夹爪的开合。")
    print("================================================================\n")

    exec_node = Manipulator(args)
    try:
        exec_node.run()
    except Exception as e:
        traceback.print_exc()
    finally:
        pass