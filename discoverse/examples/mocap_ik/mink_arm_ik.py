import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, Dict, Any
from discoverse.utils import get_site_tmat

class Mink_IK:
    def __init__(self, mjcf_path, arm_dof):
        self.arm_dof = arm_dof
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.configuration = mink.Configuration(self.mj_model)
        self.end_effector_task = mink.FrameTask(
            frame_name="endpoint",
            frame_type="site",
            position_cost=100.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.mj_model, cost=1e-2)
        self.mink_tasks = [self.end_effector_task, self.posture_task]

        self.posture_task.set_target_from_configuration(self.configuration)

        self.solver = "quadprog"
        self.pos_threshold = 5e-3
        self.ori_threshold = 1e-2
        self.max_iters = 200

    def converge_ik(self, dt=0.0):
        dt = dt or self.mj_model.opt.timestep * 2
        for _ in range(self.max_iters):
            vel = mink.solve_ik(self.configuration, self.mink_tasks, dt, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, dt)
            err = self.end_effector_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                return True
        return False
    
    def fk(self, current_qpos):
        # 更新当前配置
        tmp_q = self.configuration.data.qpos.copy()
        tmp_q[:len(current_qpos)] = current_qpos[:]
        self.configuration.update(tmp_q)
        tmat_base = get_site_tmat(self.configuration.data, "armbase")
        tmat_endpoint = get_site_tmat(self.configuration.data, "endpoint")
        tmat_local = np.linalg.inv(tmat_base) @ tmat_endpoint
        target_position = tmat_local[:3,3]
        target_quat_wxyz = Rotation.from_matrix(tmat_local[:3,:3]).as_quat()[[3,0,1,2]]
        return target_position, target_quat_wxyz

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
        
        # 更新当前配置
        tmp_q = self.configuration.data.qpos.copy()
        tmp_q[:len(current_qpos)] = current_qpos[:]
        self.configuration.update(tmp_q)

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
        tmat_base = get_site_tmat(self.configuration.data, "armbase")
        
        # 设置目标
        target_SE3 = mink.SE3.from_matrix(tmat_base @ T_target)
        self.end_effector_task.set_target(target_SE3)
        
        # 如果提供了参考位置，更新姿态任务
        if reference_qpos is not None:
            temp_config = mink.Configuration(self.mj_model)
            temp_config.update(reference_qpos)
            self.posture_task.set_target_from_configuration(temp_config)
        
        converged = self.converge_ik(0.005)
        solution = self.configuration.data.qpos[:self.arm_dof]

        return solution, converged

if __name__ == "__main__":
    import os
    from discoverse import DISCOVERSE_ASSETS_DIR

    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    mjcf_path = os.path.join(DISCOVERSE_ASSETS_DIR, "mjcf/manipulator", "robot_airbot_play_force.xml")
    mik = Mink_IK(mjcf_path, arm_dof=6)

    solution, converged = mik.solve_ik(
        target_pos = np.array([0.205, 0.0, 0.22]),
        target_ori = np.array([1.0, 0.0, 0.0, 0.0]),
        current_qpos = np.zeros(6)
    )

    print(converged)
    print(solution)