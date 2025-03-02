import numpy as np
import gymnasium
import mujoco
from gymnasium import spaces
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.utils import get_body_tmat


class Env(gymnasium.Env):
    """
    奇异果抓取环境，用于RLPD算法训练
    """
    def __init__(self, render=False):
        super(Env, self).__init__()

        # 环境配置
        self.cfg = MMK2Cfg()
        self.cfg.use_gaussian_renderer = False  # 关闭高斯渲染器
        self.cfg.init_key = "pick"  # 初始化模式为"抓取"
        self.cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义白色盘子的模型路径
        self.cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义奇异果的模型路径
        self.cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景的模型路径
        self.cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo 环境文件路径
        self.cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
        self.cfg.sync = True  # 是否同步更新
        self.cfg.headless = not render  # 根据render参数决定是否显示渲染画面

        # 创建基础任务环境
        from simulator import Simulator
        self.sim = Simulator(self.cfg)
        self.mj_model = self.sim.mj_model  # 获取MuJoCo模型
        self.mj_data = self.sim.mj_data  # 获取MuJoCo数据

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
        
        # 初始化机器人的固定位姿
        # 这些值应该是一个合适的初始姿态，可以根据需要调整
        self.init_qpos = np.zeros(self.mj_model.nq)  # 初始化为零向量
        # 设置一个合理的初始姿态，例如机械臂在中间位置
        # 可以根据需要调整这些值
        
        # 键盘控制相关参数
        self.key_control_mode = False  # 是否启用键盘控制
        self.control_idx = 0  # 当前控制的关节索引
        self.control_step = 0.05  # 每次键盘控制的步长

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0  # 重置当前时间步数

        try:
            # 重置环境状态，避免调用sim.reset()，因为它会调用未实现的getObservation方法
            self.sim.resetState()  # 重置状态
            
            # 设置固定的初始位置
            self.mj_data.qpos[:] = self.init_qpos[:]
            
            # 前向传播更新物理状态
            mujoco.mj_forward(self.mj_model, self.mj_data)
            
            # 如果不在键盘控制模式下，才进行域随机化
            if not self.key_control_mode:
                self._domain_randomization()  # 域随机化

            observation = self._get_obs()  # 获取初始观测
            info = {}
            return observation, info  # 返回观察值和信息
        except Exception as e:
            print(f"重置环境失败: {str(e)}")
            raise e

    def _domain_randomization(self):
        """
        对环境进行域随机化，增加训练的鲁棒性
        """
        # 随机化奇异果的位置
        kiwi_pos = self.mj_data.body("kiwi").xpos.copy()
        kiwi_pos[0] += np.random.uniform(-0.05, 0.05)  # x方向随机偏移
        kiwi_pos[1] += np.random.uniform(-0.05, 0.05)  # y方向随机偏移
        self.mj_data.body("kiwi").xpos[:] = kiwi_pos
        
        # 随机化盘子的位置
        plate_pos = self.mj_data.body("plate_white").xpos.copy()
        plate_pos[0] += np.random.uniform(-0.03, 0.03)  # x方向随机偏移
        plate_pos[1] += np.random.uniform(-0.03, 0.03)  # y方向随机偏移
        self.mj_data.body("plate_white").xpos[:] = plate_pos
        
        # 前向传播更新物理状态
        mujoco.mj_forward(self.mj_model, self.mj_data)

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

            # 更新控制信号并执行仿真步骤
            self.mj_data.ctrl[:] = clipped_action
            for _ in range(self.sim.decimation):
                mujoco.mj_step(self.mj_model, self.mj_data)

            # 获取新的状态
            observation = self._get_obs()  # 获取新的观察值
            reward = self._compute_reward()  # 计算奖励
            terminated = self._check_termination()  # 检查是否终止
            truncated = self.current_step >= self.max_steps  # 检查是否超出最大步数
            info = {}  # 信息字典

            # 将奖励信息添加到info中
            info.update(self.reward_info)
            
            # 如果在键盘控制模式下，添加相关信息
            if self.key_control_mode:
                info["key_control"] = True
                info["control_idx"] = self.control_idx
                info["joint_name"] = f"Joint {self.control_idx}"
                info["joint_value"] = self.mj_data.qpos[self.control_idx]
                
                # 添加所有关节的位置信息
                joint_positions = {}
                for i in range(min(19, self.mj_model.nq)):
                    joint_positions[f"joint_{i}"] = self.mj_data.qpos[i]
                info["joint_positions"] = joint_positions
            
            # 如果启用了渲染，则渲染环境
            if not self.cfg.headless:
                self.sim.render()
                
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
        """渲染环境"""
        if not self.cfg.headless:
            self.sim.render()
            
    def close(self):
        """关闭环境"""
        pass
        
    def _handle_keyboard_input(self):
        """
        处理键盘输入，控制机器人的关节
        这个方法需要在外部调用并传入键盘事件
        """
        # 这里的实现依赖于外部代码传入键盘事件
        # 在实际应用中，需要在主循环中捕获键盘事件并调用这个方法
        pass
        
    def toggle_key_control_mode(self):
        """
        切换键盘控制模式
        """
        self.key_control_mode = not self.key_control_mode
        print(f"键盘控制模式: {'\u5f00\u542f' if self.key_control_mode else '\u5173\u95ed'}")
        return self.key_control_mode
        
    def set_control_joint(self, idx):
        """
        设置当前控制的关节索引
        
        Args:
            idx: 关节索引
        """
        if 0 <= idx < self.mj_model.nq:
            self.control_idx = idx
            print(f"当前控制关节: {self.control_idx}")
        else:
            print(f"无效的关节索引: {idx}, 有效范围: 0-{self.mj_model.nq-1}")
        
    def control_joint(self, direction):
        """
        控制当前关节的运动
        
        Args:
            direction: 方向，1表示正方向，-1表示负方向
        """
        if not self.key_control_mode:
            print("当前不在键盘控制模式下")
            return
        
        # 获取当前控制范围
        ctrl_range = self.mj_model.actuator_ctrlrange
        
        # 如果关节索引在控制器范围内，直接控制控制器
        if self.control_idx < len(self.mj_data.ctrl):
            # 更新控制信号
            current_ctrl = self.mj_data.ctrl[self.control_idx]
            new_ctrl = current_ctrl + direction * self.control_step
            
            # 限制在控制范围内
            if self.control_idx < len(ctrl_range):
                low = ctrl_range[self.control_idx, 0]
                high = ctrl_range[self.control_idx, 1]
                new_ctrl = np.clip(new_ctrl, low, high)
            
            self.mj_data.ctrl[self.control_idx] = new_ctrl
            print(f"控制器 {self.control_idx} 值: {new_ctrl:.4f}")
        else:
            # 如果关节索引超出控制器范围，直接控制关节位置
            # 更新关节位置
            self.mj_data.qpos[self.control_idx] += direction * self.control_step
            print(f"关节 {self.control_idx} 位置: {self.mj_data.qpos[self.control_idx]:.4f}")
        
        # 前向传播更新物理状态
        mujoco.mj_forward(self.mj_model, self.mj_data)


# 用于收集演示数据的辅助函数
def collect_demonstrations(env, num_episodes=10, max_steps_per_episode=1000, use_keyboard=False):
    """
    收集专家演示数据
    
    Args:
        env: 环境实例
        num_episodes: 收集的轨迹数量
        max_steps_per_episode: 每个轨迹的最大步数
        use_keyboard: 是否使用键盘控制收集数据
        
    Returns:
        demonstrations: 包含状态、动作、奖励、下一状态、终止标志的字典
    """
    # 如果使用键盘控制，则调用键盘控制器收集数据
    if use_keyboard:
        try:
            import os
            import sys
            # 添加pick_kiwi目录到系统路径
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_kiwi"))
            from keyboard_control_mmk2 import MMK2KeyboardControl
            
            print("初始化键盘控制器...")
            controller = MMK2KeyboardControl(env)
            print("请使用键盘控制机器人，并按S开始记录数据，完成后按C保存数据，按Q退出")
            controller.run()
            
            # 检查是否有保存的演示数据
            if os.path.exists("manual_demos.npy"):
                print("加载手动收集的演示数据...")
                demonstrations = np.load("manual_demos.npy", allow_pickle=True).item()
                return demonstrations
            else:
                print("未找到手动收集的演示数据，将使用启发式策略收集数据")
        except Exception as e:
            print(f"键盘控制收集数据失败: {str(e)}")
            print("将使用启发式策略收集数据")
    
    # 使用启发式策略收集数据
    demonstrations = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': []
    }
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        for step in range(max_steps_per_episode):
            # 使用启发式策略代替随机动作
            action = heuristic_policy(state, env)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            demonstrations['states'].append(state)
            demonstrations['actions'].append(action)
            demonstrations['rewards'].append(reward)
            demonstrations['next_states'].append(next_state)
            demonstrations['dones'].append(done)
            
            state = next_state
            
            if done:
                break
                
        print(f"已收集 {episode+1}/{num_episodes} 个轨迹")
    
    # 转换为numpy数组
    for key in demonstrations:
        demonstrations[key] = np.array(demonstrations[key])
        
    return demonstrations


def heuristic_policy(state, env):
    """
    启发式策略用于生成演示数据
    
    Args:
        state: 当前状态
        env: 环境实例
        
    Returns:
        action: 生成的动作
    """
    # 获取机械臂、奇异果和盘子的位置
    qpos = state[:env.mj_model.nq]
    qvel = state[env.mj_model.nq:env.mj_model.nq+env.mj_model.nv]
    
    # 奇异果和盘子的位置在状态的最后6个元素
    kiwi_pos = state[-6:-3]
    plate_pos = state[-3:]
    
    # 获取机械臂末端位置
    tmat_rgt_arm = get_body_tmat(env.mj_data, "rgt_arm_link6")
    rgt_arm_pos = np.array([tmat_rgt_arm[1, 3], tmat_rgt_arm[0, 3], tmat_rgt_arm[2, 3]])
    
    # 计算机械臂到奇异果的距离
    distance_to_kiwi = np.linalg.norm(rgt_arm_pos - kiwi_pos)
    
    # 计算奇异果到盘子的距离
    kiwi_to_plate = np.linalg.norm(kiwi_pos - plate_pos)
    
    # 初始化动作
    action = np.zeros(env.action_space.shape[0])
    
    # 启发式策略：
    # 1. 如果机械臂距离奇异果还较远，则移动到奇异果位置
    # 2. 如果机械臂接近奇异果，则尝试拿起奇异果
    # 3. 如果已经拿起奇异果，则移动到盘子位置
    
    if distance_to_kiwi > 0.1:
        # 移动到奇异果位置
        direction = (kiwi_pos - rgt_arm_pos) / distance_to_kiwi
        # 生成一个指向奇异果的动作
        action = np.random.uniform(-0.1, 0.1, env.action_space.shape[0])
        # 添加一些偏置使动作更倾向于移动到奇异果
        action[:3] += direction * 0.5
    elif distance_to_kiwi <= 0.1 and kiwi_to_plate > 0.1:
        # 尝试拿起奇异果
        # 生成一个拿取动作
        action = np.random.uniform(-0.1, 0.1, env.action_space.shape[0])
        # 添加一些偏置使动作更倾向于拿取
        action[-1] = 0.8  # 假设最后一个动作维度是拿取
    else:
        # 移动到盘子位置
        direction = (plate_pos - rgt_arm_pos) / np.linalg.norm(plate_pos - rgt_arm_pos)
        # 生成一个指向盘子的动作
        action = np.random.uniform(-0.1, 0.1, env.action_space.shape[0])
        # 添加一些偏置使动作更倾向于移动到盘子
        action[:3] += direction * 0.5
    
    # 将动作限制在合法范围内
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    return action
