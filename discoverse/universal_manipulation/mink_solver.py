"""
Mink逆运动学求解器

基于mink库的通用逆运动学求解器，支持多种机械臂。
"""

import time
import mink
import mujoco
import numpy as np
from typing import Optional, Tuple, Dict, Any
from scipy.spatial.transform import Rotation

from .robot_config import RobotConfigLoader

class MinkIKSolver:
    """基于Mink的通用逆运动学求解器"""
    
    def __init__(self, robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData = None):
        """
        初始化Mink IK求解器
        
        Args:
            robot_config: 机械臂配置加载器
            mj_model: MuJoCo模型
            mj_data: MuJoCo数据（用于调试）
            
        Raises:
            ImportError: mink库未安装
            ValueError: 配置无效
        """
        self.robot_config = robot_config
        self.mj_model = mj_model
        self.mj_data = mj_data  # 添加这个用于调试
        
        # 初始化mink配置
        self.configuration = mink.Configuration(mj_model)
        
        # 获取求解器配置 - 支持多种配置键名
        ik_solver_config = robot_config.config.get('ik_solver', {})
        if not ik_solver_config:
            raise ValueError("IK solver configuration is missing or invalid.")

        self.solver_config = ik_solver_config
        self.solver_type = self.solver_config.get('solver_type', 'quadprog')
        self.position_tolerance = self.solver_config.get('position_tolerance', 1e-4)
        self.orientation_tolerance = self.solver_config.get('orientation_tolerance', 1e-4)
        self.max_iterations = self.solver_config.get('max_iterations', 50)
        self.damping = float(self.solver_config.get('damping', 1e-3))
        self.dt = self.solver_config.get('dt', 2e-3)
        
        # 设置IK任务
        self._setup_ik_tasks()
        
    def _setup_ik_tasks(self):
        """设置IK任务"""
        self.end_effector_task = mink.FrameTask(
            frame_name=self.robot_config.end_effector_site,
            frame_type="site",
            position_cost=self.solver_config.get('position_cost', 100.0),
            orientation_cost=self.solver_config.get('orientation_cost', 10.0),
            lm_damping=self.damping,
        )
        
        self.posture_task = mink.PostureTask(
            model=self.mj_model,
            cost=self.solver_config.get('posture_cost', 1e-2)
        )
        
        self.tasks = [self.end_effector_task, self.posture_task]
        
        if self.mj_model.nkey > 0:
            home_qpos = self.mj_model.key(0).qpos.copy()
            self.configuration.update(home_qpos)
            self.posture_task.set_target_from_configuration(self.configuration)
        else:
            raise ValueError("MuJoCo model does not contain keyframes. Cannot set home posture.")
    
    def solve_ik(self, 
                 target_pos: np.ndarray, 
                 target_ori: np.ndarray, 
                 current_qpos: np.ndarray,
                 reference_qpos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        求解逆运动学
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_ori: 目标姿态矩阵 (3x3) 或四元数 [qw, qx, qy, qz]
            current_qpos: 当前关节位置
            reference_qpos: 参考关节位置（用于选择最优解）
            
        Returns:
            Tuple[关节位置, 是否收敛, 求解信息]
        """
        start_time = time.time()
        
        # 更新当前配置
        self.configuration.update(current_qpos)
        
        # 处理目标姿态
        if target_ori.shape == (4,):
            # 四元数转旋转矩阵
            target_rot_matrix = Rotation.from_quat(target_ori[[1, 2, 3, 0]]).as_matrix()
        elif target_ori.shape == (3, 3):
            target_rot_matrix = target_ori
        else:
            raise ValueError(f"Invalid target orientation shape: {target_ori.shape}")
        
        # 构建目标变换矩阵
        T_target = np.eye(4)
        T_target[:3, :3] = target_rot_matrix
        T_target[:3, 3] = target_pos
        
        # 设置目标
        target_SE3 = mink.SE3.from_matrix(T_target)
        self.end_effector_task.set_target(target_SE3)
        
        # 如果提供了参考位置，更新姿态任务
        if reference_qpos is not None:
            temp_config = mink.Configuration(self.mj_model)
            temp_config.update(reference_qpos)
            self.posture_task.set_target_from_configuration(temp_config)
        
        # 迭代求解
        dt = 1e-3
        converged = False
        iteration = 0
        errors = []
        
        for iteration in range(self.max_iterations):
            # 计算速度
            velocity = mink.solve_ik(
                self.configuration, 
                self.tasks, 
                dt, 
                self.solver_type, 
                self.damping
            )
            
            # 积分更新配置
            self.configuration.integrate_inplace(velocity, dt)
            
            # 检查收敛
            error = self.end_effector_task.compute_error(self.configuration)
            position_error = np.linalg.norm(error[:3])
            orientation_error = np.linalg.norm(error[3:])
            
            errors.append({
                'position_error': position_error,
                'orientation_error': orientation_error,
                'total_error': position_error + orientation_error
            })
            
            # 检查收敛条件
            if (position_error < self.position_tolerance and 
                orientation_error < self.orientation_tolerance):
                converged = True
                break
        
        solution = self.configuration.q[:self.robot_config.arm_joints_count].copy()
        
        is_valid = self._validate_solution(solution)
        
        solve_time = time.time() - start_time
        
        # 构建求解信息
        solve_info = {
            'converged': converged and is_valid,
            'iterations': iteration + 1,
            'solve_time': solve_time,
            'final_position_error': errors[-1]['position_error'] if errors else float('inf'),
            'final_orientation_error': errors[-1]['orientation_error'] if errors else float('inf'),
            'is_valid_solution': is_valid,
            'error_history': errors
        }
        
        return solution, converged and is_valid, solve_info

    
    def _validate_solution(self, joint_positions: np.ndarray) -> bool:
        """
        验证IK解的有效性
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            是否有效
        """
        # 检查是否存在NaN或无穷大
        if np.any(np.isnan(joint_positions)) or np.any(np.isinf(joint_positions)):
            return False
        
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MinkIKSolver({self.robot_config.robot_name}")
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


def create_mink_solver(robot_config: RobotConfigLoader, mj_model: mujoco.MjModel, mj_data: mujoco.MjData = None) -> MinkIKSolver:
    """
    便利函数：创建Mink IK求解器
    
    Args:
        robot_config: 机械臂配置
        mj_model: MuJoCo模型
        mj_data: MuJoCo数据（可选，用于调试）
        
    Returns:
        Mink IK求解器实例
    """
    return MinkIKSolver(robot_config, mj_model, mj_data) 