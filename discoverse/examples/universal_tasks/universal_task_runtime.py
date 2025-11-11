import os
import time
import shutil
import argparse
import traceback

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import discoverse
from discoverse.envs import make_env
from discoverse import DISCOVERSE_ROOT_DIR, DISCOVERSE_ASSETS_DIR

from discoverse.universal_manipulation import UniversalTaskBase, PyavImageEncoder, recoder_single_arm

from discoverse.utils import (
    SimpleStateMachine, step_func, get_body_tmat
)

class UniversalRuntimeTaskExecutor:
    """通用运行时任务执行器
    
    集成了utils模块、简化的错误处理、模板化配置支持
    """

    def __init__(self, task: UniversalTaskBase, viewer, mj_model: mujoco.MjModel, 
                 mj_data: mujoco.MjData, robot_name: str, sync: bool = False):
        """初始化运行时执行器
        
        Args:
            task: UniversalTaskBase任务实例
            viewer: MuJoCo viewer
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据
            robot_name: 机械臂名称
            sync: 是否启用实时同步
        """
        self.task = task
        self.viewer = viewer
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.renderer = mujoco.Renderer(mj_model)
        self.robot_name = robot_name
        self.sync = sync

        # 时间和频率控制
        self.sim_timestep = mj_model.opt.timestep
        self.viewer_fps = 60
        
        # 任务配置 - 支持模板化配置
        self.resolved_states = task.task_config.get_resolved_states()
        self.total_states = len(self.resolved_states)

        # 从任务配置获取机械臂维度信息
        self.n_arm_joints = len(task.robot_interface.arm_joints)  # 机械臂关节数
        self.gripper_ctrl_idx = self.n_arm_joints  # 夹爪控制索引在机械臂关节之后

        # 关节sensor索引
        self.joint_pos_sensor_idx = [
            mujoco.mj_name2id(
                mj_model, 
                mujoco.mjtObj.mjOBJ_SENSOR, 
                sensor_name
            ) 
            for sensor_name in task.robot_interface.joint_pos_sensors
        ]
        print(f"🔍 关节位置传感器: {task.robot_interface.joint_pos_sensors}")
        print(f"🔍 关节位置传感器索引: {self.joint_pos_sensor_idx}")

        self.mujoco_ctrl_dim = mj_model.nu
        self.move_speed = 1.5  # 控制速度
        self.max_time = 20.0  # 最大执行时间（仿真时间，非真实时间）

        self.task.randomizer.set_viewer(viewer)
        self.task.randomizer.set_renderer(self.renderer)

        self.record_frq = self.task.task_config.record_fps
        self.camera_cfgs = {cam['name']: cam for cam in self.task.task_config.camera_configs}

        self.camera_encoders = {}

        # self.reset(random=False)
        self.reset()

    def get_current_qpos(self):
        """获取当前关节位置"""
        return self.mj_data.qpos.copy()
    
    def check_action_done(self):
        """检查动作是否完成"""
        current_qpos = self.get_current_qpos()
        # 只检查机械臂关节
        position_error = np.linalg.norm(current_qpos[:self.n_arm_joints] - self.target_control[:self.n_arm_joints])
        position_done = position_error < 0.02  # 2cm容差
        
        # 检查延时条件
        if self.current_delay > 0 and self.delay_start_sim_time is not None:
            delay_elapsed = self.mj_data.time - self.delay_start_sim_time
            delay_done = delay_elapsed >= self.current_delay
            if not delay_done:
                return False  # 延时未完成，动作未完成
            
        return position_done

    def get_observation(self):
        """获取当前观测"""
        obs = {
            "time": self.mj_data.time,
            "jq" : self.mj_data.sensordata[self.joint_pos_sensor_idx].tolist(),
            "action": self.action[:self.mujoco_ctrl_dim].tolist(),
            "img" : {}
        }
        for camera_name in self.camera_cfgs.keys():
            self.set_renderer_size(self.camera_cfgs[camera_name]['width'], self.camera_cfgs[camera_name]['height'])
            obs["img"][camera_name] = self.get_rgb_image(camera_name).copy()
        return obs

    def get_rgb_image(self, camera_name):
        self.renderer.update_scene(self.mj_data, camera_name)
        return self.renderer.render()

    def set_renderer_size(self, width, height):
        self.renderer._width = width
        self.renderer._height = height
        self.renderer._rect.width = width
        self.renderer._rect.height = height

    def set_target_from_primitive(self, state_config):
        """使用原语设置目标控制信号"""
        try:
            primitive = state_config["primitive"]
            params = state_config.get("params", {})
            gripper_state = state_config.get("gripper_state", "open")
            
            if primitive == "move_to_object":
                # 使用原语计算目标位置
                object_name = params.get("object_name", "")
                offset = np.array(params.get("offset", [0, 0, 0]))
                
                if object_name:
                    # 获取物体位置
                    object_tmat = get_body_tmat(self.mj_data, object_name)
                    target_pos = object_tmat[:3, 3] + offset
                    
                    # 获取当前末端执行器姿态矩阵（从MuJoCo数据直接读取）
                    site_name = self.task.robot_interface.robot_config.end_effector_site
                    site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                    
                    # 获取完整的qpos (包含所有自由度)
                    full_current_qpos = self.mj_data.qpos.copy()

                    # 求解IK
                    solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                        target_pos, target_rmat, full_current_qpos
                    )
                    
                    if converged:
                        # IK求解器返回机械臂关节解
                        self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                        self.set_mocap_target("target", target_pos, Rotation.from_matrix(target_rmat).as_quat()[[3,0,1,2]])
                    else:
                        return False
                        
            elif primitive == "move_relative":
                offset = np.array(params.get("offset", [0, 0, 0]))
               
                site_name = self.task.robot_interface.robot_config.end_effector_site
                site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                current_pos = self.mj_data.site_xpos[site_id].copy()
                target_rmat = self.mj_data.site_xmat[site_id].reshape(3, 3).copy()
                
                target_pos = current_pos + offset
                
                full_current_qpos = self.mj_data.qpos.copy()
                
                solution, converged, solve_info = self.task.robot_interface.ik_solver.solve_ik(
                    target_pos, target_rmat, full_current_qpos
                )
                
                if converged:
                    self.target_control[:self.n_arm_joints] = solution[:self.n_arm_joints]
                    self.set_mocap_target("target", target_pos, Rotation.from_matrix(target_rmat).as_quat()[[3,0,1,2]])
                else:
                    return False
           
            if gripper_state == "open":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.open()
            elif gripper_state == "close":
                self.target_control[self.gripper_ctrl_idx] = self.task.robot_interface.gripper_controller.close()
            
            current_ctrl = self.mj_data.ctrl[:self.mujoco_ctrl_dim].copy()
            dif = np.abs(current_ctrl - self.target_control)
            self.joint_move_ratio = dif / (np.max(dif) + 1e-6)
            
            return True
            
        except Exception as e:
            print(f"   ❌ 原语执行失败: {e}")
            traceback.print_exc()
            return False

    def set_mocap_target(self, target_name, target_pos, target_quat, box_color=(0,1,0,0.1)):
        """设置Mocap目标位置和姿态"""
        mocap_id = self.mj_model.body(target_name).mocapid
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = target_pos
            self.mj_data.mocap_quat[mocap_id] = target_quat
            self.mj_model.geom(f'{target_name}_box').rgba = box_color

    def step(self, decimation=5):
        """单步执行 - 高频主循环"""
        try:
            if self.stm.trigger():
                if self.stm.state_idx < self.total_states:
                    state_config = self.resolved_states[self.stm.state_idx]
                    self.current_delay = state_config.get("delay", 0.0)
                    
                    if not self.set_target_from_primitive(state_config):
                        print(f"   ❌ 状态 {self.stm.state_idx} 设置失败")
                        return False
                        
                    if self.current_delay > 0:
                        self.delay_start_sim_time = self.mj_data.time
                else:
                    self.success = self.check_task_success()
                    self.running = False
                    return True
                    
            elif self.mj_data.time > self.max_time:
                self.running = False
                return False

            else:
                self.stm.update()
            
            if self.check_action_done():
                self.current_delay = 0.0
                self.delay_start_sim_time = None
                self.stm.next()
            
            for i in range(self.n_arm_joints):
                self.action[i] = step_func(
                    self.action[i], 
                    self.target_control[i], 
                    self.move_speed * float(decimation) * self.joint_move_ratio[i] * self.mj_model.opt.timestep
                )
            self.action[self.gripper_ctrl_idx] = self.target_control[self.gripper_ctrl_idx]
            
            self.mj_data.ctrl[:self.mujoco_ctrl_dim] = self.action[:self.mujoco_ctrl_dim]
            
            for _ in range(decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)

            return True

        except Exception as e:
            print(f"❌ 步进失败: {e}")
            self.running = False
            return False
    
    def check_task_success(self):
        """检查任务成功条件"""
        return self.task.check_success()
    
    def run(self):
        """运行任务主循环"""
        step_count = 0
        obs_lst = []
        last_report_time = time.time()
        
        if self.sync:
            real_start_time = time.time()
            expected_sim_time = 0.0
        
        last_render_time = 0.0
        
        while self.running:
            if not self.step():
                break
                
            step_count += 1

            # 实时同步控制
            if self.sync:
                expected_sim_time = self.mj_data.time
                real_elapsed = time.time() - real_start_time
                sim_elapsed = expected_sim_time
                
                # 如果仿真跑得太快，等待实际时间追上
                if sim_elapsed > real_elapsed:
                    sleep_time = sim_elapsed - real_elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            # 检查viewer是否被关闭 - 使用官方API
            if self.viewer is not None:
                if not self.viewer.is_running():
                    print("🎬 查看器已关闭，退出程序")
                    self.viewer_closed = True
                    self.running = False
                    return False
                
                # 定期同步显示（降低频率避免性能问题）
                if self.mj_data.time - last_render_time > (1.0 / self.viewer_fps):
                    self.viewer.sync()
                    last_render_time = self.mj_data.time
            
            if len(obs_lst) < self.mj_data.time * self.record_frq:
                obs = self.get_observation()
                imgs = obs.pop('img')
                for cam_id, img in imgs.items():
                    self.camera_encoders[cam_id].encode(img, obs["time"])
                obs_lst.append(obs)

            # 每秒报告一次进度
            if time.time() - last_report_time > 1.0:
                elapsed = time.time() - self.start_time
                sim_time_info = f", 仿真时间: {self.mj_data.time:.1f}s" if self.sync else ""
                print(f"   ⏱️  运行时间: {elapsed:.1f}s, 步数: {step_count}, 当前状态: {self.stm.state_idx+1}/{self.total_states}{sim_time_info}")
                last_report_time = time.time()

        # 报告结果
        elapsed_time = time.time() - self.start_time
        print(f"\\n📊 {self.robot_name.upper()}运行架构执行完成!")
        print(f"   总时间: {elapsed_time:.2f}s")
        print(f"   仿真时间: {self.mj_data.time:.2f}s")
        print(f"   总步数: {step_count}")
        print(f"   完成状态: {self.stm.state_idx}/{self.total_states}")
        print(f"   任务成功: {'✅ 是' if self.success else '❌ 否'}")
        if self.sync:
            time_ratio = self.mj_data.time / elapsed_time if elapsed_time > 0 else 0
            print(f"   时间比例: {time_ratio:.2f} (仿真时间/真实时间)")

        for ec in self.camera_encoders.values():
            ec.close()
        
        if not self.success:
            shutil.rmtree(self.save_dir, ignore_errors=True)
            print(f"   ❌ 任务未成功，已删除保存目录: {self.save_dir}")
        else:
            # 保存观测数据
            recoder_single_arm(self.save_dir, obs_lst)
            print(f"   观测数据已保存到: {self.save_dir}")

        return self.success
    
    def reset(self, random=True):
        """重置环境和执行器状态"""
        # 重置到home位置
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key(0).id)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # 重新初始化mocap target
        mink.move_mocap_to_frame(self.mj_model, self.mj_data, "target", "endpoint", "site")

        # 应用场景随机化
        if random:
            self.task.randomize_scene()
        
        # 重置状态机
        self.stm = SimpleStateMachine()
        self.stm.max_state_cnt = self.total_states
        
        # 重置控制状态
        self.target_control = np.zeros(self.mujoco_ctrl_dim)
        self.action = np.zeros(self.mujoco_ctrl_dim)
        self.joint_move_ratio = np.ones(self.mujoco_ctrl_dim)
        
        # 重置运行时状态
        self.running = True
        self.start_time = time.time()
        self.success = False
        self.viewer_closed = False  # 重置viewer关闭标志
        
        # 重置延时状态
        self.current_delay = 0.0
        self.delay_start_sim_time = None
        
        # 重新初始化动作
        self.action[:] = self.get_current_qpos()[:self.mujoco_ctrl_dim]
        
        self.save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data", f"{self.robot_name}_{self.task.task_config.task_name}")
        os.makedirs(self.save_dir, exist_ok=True)

        self.camera_encoders = {}
        for cam_name in self.camera_cfgs.keys():
            if mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
                # raise ValueError(f"Camera '{cam_name}' not found in the MJCF model.")
                print(f"Camera '{cam_name}' not found in the MJCF model.")
            else:
                # fovy = self.camera_cfgs[cam_name].get("fovy", None)
                # if fovy:
                #     mj_model.cam(cam_name).fovy = fovy
                self.camera_encoders[cam_name] = PyavImageEncoder(
                    self.camera_cfgs[cam_name]["width"], 
                    self.camera_cfgs[cam_name]["height"], 
                    self.save_dir, 
                    cam_name
                )

def generate_robot_task_model(robot_name, task_name):
    """生成指定机械臂的任务模型"""
    xml_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/tmp", f"{robot_name}_{task_name}.xml")
    make_env(robot_name, task_name, xml_path)
    return xml_path

def create_simple_visualizer(mj_model, mj_data):
    """创建MuJoCo内置可视化器"""
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
    if mj_model.ncam > 0:
        viewer.cam.fixedcamid = 0  # 使用id=0的相机
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    return viewer

def main(robot_name="airbot_play", task_name="place_block", sync=False, once=False, headless=False):
    """
    Args:
        robot_name: 机械臂名称
        task_name: 任务名称
        sync: 实时同步
        once: 单次执行
        headless: 无头模式
    """
    print(f"Welcome to discoverse {discoverse.__version__} !")
    print(discoverse.__logo__)

    xml_path = generate_robot_task_model(robot_name, task_name)
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    # 创建查看器（除非是无头模式）
    viewer = None if headless else create_simple_visualizer(mj_model, mj_data)

    # 创建通用任务 - 使用预处理的配置
    configs_root = os.path.join(DISCOVERSE_ROOT_DIR, "discoverse", "configs")
    robot_config_path = os.path.join(configs_root, "robots", f"{robot_name}.yaml")
    task_config_path = os.path.join(configs_root, "tasks", f"{task_name}.yaml")
    
    # 直接创建任务实例，传递预处理的配置
    task = UniversalTaskBase(
        robot_config_path=robot_config_path,
        task_config_path=task_config_path,
        mj_model=mj_model,
        mj_data=mj_data
    )

    # 创建通用运行时执行器
    try:
        executor = UniversalRuntimeTaskExecutor(task, viewer, mj_model, mj_data, robot_name, sync)

        task_count = 0
       
        while True:
            task_count += 1
            print(f"\n{'='*50}")
            print(f"🎯 第 {task_count} 轮任务开始")
            print(f"{'='*50}")
            
            # 运行任务
            success = executor.run()
            
            if success:
                print(f"\n🎉 第 {task_count} 轮任务成功完成!")
                print(f"   任务目标已达成")
            else:
                print(f"\n⚠️ 第 {task_count} 轮任务未完全成功")
            
            # 单次执行模式下直接退出
            if once:
                break
            
            # 检查是否需要退出循环
            if executor.viewer_closed:
                break
            
            # 重置环境准备下一轮
            executor.reset()
        
    except Exception as e:
        print(f"❌ 运行时执行失败: {e}")
        traceback.print_exc()

    finally:
        # 关闭查看器
        if viewer is not None:
            try:
                viewer.close()
                print("🎬 查看器已关闭")
            except:
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="通用机械臂任务演示")
    parser.add_argument("-r", "--robot", type=str, default="airbot_play", help="选择机械臂类型", 
                       choices=["airbot_play", "arx_x5", "arx_l5", "piper", "panda", "rm65", "xarm7", "iiwa14", "ur5e"])
    parser.add_argument("-t", "--task", type=str, default="place_block", help="选择任务类型",
                       choices=["place_block", "cover_cup", "stack_block", "place_kiwi_fruit", "place_coffeecup", "close_laptop"])
    parser.add_argument("-s", "--sync", action="store_true", help="启用实时同步模式（仿真时间与真实时间一致）")
    parser.add_argument("-1", "--once", action="store_true", help="单次执行模式（默认为循环执行）")
    parser.add_argument("--headless", action="store_true", help="无头模式运行（CICD测试用）")
    args = parser.parse_args()

    main(args.robot, args.task, sync=args.sync, once=args.once, headless=args.headless)
