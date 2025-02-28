import numpy as np
import gymnasium
import mujoco
from gymnasium import spaces
from discoverse.examples.tasks_mmk2.pick_kiwi import SimNode, cfg
from discoverse.task_base import MMK2TaskBase
from discoverse.utils import get_body_tmat


class Env(gymnasium.Env):
    def __init__(self, task_base=None, render=False):
        super(Env, self).__init__()

        # 环境配置
        cfg.use_gaussian_renderer = False  # 关闭高斯渲染器
        cfg.init_key = "pick"  # 初始化模式为“抓取”
        cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义白色盘子的模型路径
        cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义奇异果的模型路径
        cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景的模型路径
        cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo 环境文件路径
        cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
        cfg.sync = True  # 是否同步更新
        cfg.headless = not render  # 根据render参数决定是否显示渲染画面

        # 创建基础任务环境
        if task_base is None:
            self.task_base = MMK2TaskBase(cfg)  # 使用给定配置初始化基础任务环境
        else:
            self.task_base = task_base
        self.mj_model = self.task_base.mj_model  # 获取MuJoCo模型
        self.mj_data = self.task_base.mj_data  # 获取MuJoCo数据

        # 动作空间：机械臂关节角度控制
        # 使用actuator_ctrlrange来确定动作空间范围
        ctrl_range = self.mj_model.actuator_ctrlrange  # 获取控制器范围
        self.action_space = spaces.Box(  # 定义动作空间
            low=ctrl_range[:, 0],
            high=ctrl_range[:, 1],
            dtype=np.float32
        )

        # 观测空间：包含机械臂关节位置、速度和目标物体位置
        self.observation_space = spaces.Box(  # 定义观测空间
            low=-np.inf,
            high=np.inf,
            shape=(self.mj_model.nq + self.mj_model.nv + 6,),  # 加上目标物体的位置和方向
            dtype=np.float32
        )

        self.max_steps = 1000  # 最大时间步数
        self.current_step = 0  # 当前时间步数

        # 初始化奖励信息字典
        self.reward_info = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0  # 重置当前时间步数

        try:
            # 重置环境
            self.task_base.reset()  # 重置任务环境
            self.task_base.domain_randomization()  # 域随机化

            observation = self._get_obs()  # 获取初始观测
            info = {}
            return observation, info  # 返回观察值和信息
        except Exception as e:
            print(f"重置环境失败: {str(e)}")
            raise e

    def step(self, action):
        try:
            self.current_step += 1  # 更新当前时间步数

            # 执行动作
            # 确保动作的形状正确
            action_array = np.array(action, dtype=np.float32)
            if action_array.shape != self.action_space.shape:
                raise ValueError(f"动作形状不匹配: 期望 {self.action_space.shape}, 实际 {action_array.shape}")

            # 将动作限制在合法范围内
            clipped_action = np.clip(
                action_array,
                self.action_space.low,
                self.action_space.high
            )

            # 直接更新控制信号，不通过task_base
            self.mj_data.ctrl[:] = clipped_action  # 更新控制器信号
            mujoco.mj_step(self.mj_model, self.mj_data)  # 模拟物理引擎一步

            # 获取新的状态
            observation = self._get_obs()  # 获取新的观察值
            reward = self._compute_reward()  # 计算奖励
            terminated = self._check_termination()  # 检查是否终止
            truncated = self.current_step >= self.max_steps  # 检查是否超出最大步数
            info = {}  # 信息字典

            # 将奖励信息添加到info中
            info.update(self.reward_info)
            return observation, reward, terminated, truncated, info
        except Exception as e:
            print(f"执行动作失败: {str(e)}")
            raise e

    def _get_obs(self):
        # 获取机械臂状态
        qpos = self.mj_data.qpos.copy()  # 关节位置
        qvel = self.mj_data.qvel.copy()  # 关节速度

        # 获取猕猴桃和盘子的位置
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 获取奇异果的变换矩阵
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 获取白色盘子的变换矩阵

        kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
        plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置

        # 组合观测
        obs = np.concatenate([
            qpos,
            qvel,
            kiwi_pos,
            plate_pos
        ]).astype(np.float32)  # 将关节状态和目标物体位置组合为观测值

        return obs

    def _compute_reward(self):
        # 获取位置信息
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵
        tmat_rgt_arm = get_body_tmat(self.mj_data, "rgt_arm_link6")  # 右臂末端效应器的变换矩阵

        kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
        plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置
        rgt_arm_pos = np.array([tmat_rgt_arm[1, 3], tmat_rgt_arm[0, 3], tmat_rgt_arm[2, 3]])  # 右臂末端的位置

        # 计算距离
        distance_to_kiwi = np.linalg.norm(rgt_arm_pos - kiwi_pos)  # 右臂末端到奇异果的距离
        kiwi_to_plate = np.linalg.norm(kiwi_pos - plate_pos)  # 奇异果到盘子的距离

        # 计算各种奖励
        # 接近奖励：鼓励机械臂靠近奇异果
        approach_reward = 0.0
        if distance_to_kiwi < 0.05:
            approach_reward = 2.0
        else:
            approach_reward = -distance_to_kiwi

        # 放置奖励：鼓励机械臂将奇异果放置到盘子
        place_reward = 0.0
        if kiwi_to_plate < 0.02:  # 成功放置
            place_reward = 10.0
        elif kiwi_to_plate < 0.1:  # 比较接近
            place_reward = 2.0
        else:
            place_reward = -kiwi_to_plate

        # 步数惩罚：每一步都有一定的惩罚
        step_penalty = -0.01 * self.current_step

        # 动作幅度惩罚：惩罚较大的控制信号
        action_magnitude = np.mean(np.abs(self.mj_data.ctrl))
        action_penalty = -0.1 * action_magnitude

        # 总奖励
        total_reward = (
                approach_reward +
                place_reward +
                step_penalty +
                action_penalty
        )

        # 记录详细的奖励信息供日志使用
        self.reward_info = {
            "rewards/total": total_reward,
            "rewards/approach": approach_reward,
            "rewards/place": place_reward,
            "rewards/step_penalty": step_penalty,
            "rewards/action_penalty": action_penalty,
            "info/distance_to_kiwi": distance_to_kiwi,
            "info/kiwi_to_plate": kiwi_to_plate,
            "info/action_magnitude": action_magnitude
        }

        return total_reward

    def _check_termination(self):
        # 检查是否完成任务
        tmat_kiwi = get_body_tmat(self.mj_data, "kiwi")  # 奇异果的变换矩阵
        tmat_plate = get_body_tmat(self.mj_data, "plate_white")  # 盘子的变换矩阵

        kiwi_pos = np.array([tmat_kiwi[1, 3], tmat_kiwi[0, 3], tmat_kiwi[2, 3]])  # 奇异果的位置
        plate_pos = np.array([tmat_plate[1, 3], tmat_plate[0, 3], tmat_plate[2, 3]])  # 盘子的位置

        # 如果奇异果成功放置在盘子上
        if np.linalg.norm(kiwi_pos - plate_pos) < 0.02:
            return True  # 任务完成，终止环境
        return False

    def render(self):
        pass  # 使用MMK2TaskBase的渲染功能

    def close(self):
        """关闭环境并释放资源"""
        if hasattr(self, 'task_base'):
            del self.task_base
            self.task_base = None
