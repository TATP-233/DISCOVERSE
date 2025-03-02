# RLPD 框架（细节待开发）

本项目实现了基于RLPD（Reinforcement Learning with Prior Data）算法的机器人任务学习框架。当前实现了奇异果抓取任务，并计划扩展到更多任务场景。RLPD算法结合了离线数据和在线强化学习，能够显著提高样本效率和训练稳定性。


## 项目结构

- `env.py`: 实现任务环境，包括观察空间、动作空间、奖励计算等
- `simulator.py`: 实现基于MuJoCo的物理仿真器
- `rlpd.py`: 实现RLPD算法核心组件，包括策略网络、Q网络、经验回放缓冲区等
- `train.py`: 实现训练和评估RLPD算法的脚本
- `README.md`: 项目说明文档

## 功能模块

### 1. 强化学习训练

基于RLPD算法实现了机器人任务的强化学习训练。RLPD的核心设计包括：

- **对称采样**：从在线和离线缓冲区各采样一半数据，平衡新旧经验
- **LayerNorm**：在价值网络中使用LayerNorm，防止过度外推和提高训练稳定性
- **最大熵强化学习**：通过熵正则化项鼓励探索
- **集成Q网络**：使用多个Q网络进行集成，减少过估计偏差

### 2. 键盘控制功能（待开发）

计划实现键盘控制MMK2机器人的功能，允许用户通过键盘直接控制机器人的各个关节。

### 3. 手柄控制功能（待开发）

## 任务场景适配指南

当需要将框架适配到新的任务场景时（不仅限于奇异果抓取），需要修改以下组件：

### 1. 环境配置 (`env.py`)

#### 修改内容：

- **任务物体定义**：
  ```python
  # 修改或添加新的任务物体
  self.cfg.gs_model_dict["new_object"] = "object/new_object.ply"  # 添加新物体的模型路径
  self.cfg.obj_list = ["plate_white", "new_object"]  # 更新环境中包含的对象列表
  ```

- **MuJoCo环境文件**：
  ```python
  # 更新MuJoCo环境文件路径
  self.cfg.mjcf_file_path = "mjcf/tasks_mmk2/new_task.xml"  # 新任务的MuJoCo环境文件
  ```

- **观察空间**：
  ```python
  # 根据新任务调整观察空间的维度和内容
  self.observation_space = spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(self.mj_model.nq + self.mj_model.nv + task_specific_dims,),  # 添加任务特定的维度
      dtype=np.float32
  )
  ```

### 2. 奖励函数 (`env.py`)

#### 修改内容：

- **奖励计算**：
  ```python
  def _compute_reward(self):
      # 初始化奖励信息字典
      self.reward_info = {}
      total_reward = 0.0
      
      # 1. 任务特定奖励项（根据新任务修改）
      # 例如：距离目标的奖励
      target_pos = self._get_target_position()  # 获取目标位置
      end_effector_pos = self._get_end_effector_position()  # 获取末端执行器位置
      distance = np.linalg.norm(target_pos - end_effector_pos)
      distance_reward = -distance * self.distance_weight
      self.reward_info["distance_reward"] = distance_reward
      total_reward += distance_reward
      
      # 2. 任务完成奖励
      if self._check_task_success():
          success_reward = self.success_reward
          self.reward_info["success_reward"] = success_reward
          total_reward += success_reward
      
      # 3. 能量惩罚（可选）
      energy_penalty = -np.sum(np.abs(self.mj_data.ctrl)) * self.energy_weight
      self.reward_info["energy_penalty"] = energy_penalty
      total_reward += energy_penalty
      
      return total_reward
  ```

### 3. 成功条件判定 (`env.py`)

#### 修改内容：

- **任务成功条件**：
  ```python
  def _check_task_success(self):
      # 根据新任务定义成功条件
      # 例如：物体被成功放置在目标位置
      object_pos = self.mj_data.body("new_object").xpos.copy()
      target_pos = self.mj_data.body("target_location").xpos.copy()
      
      # 计算物体与目标位置的距离
      distance = np.linalg.norm(object_pos - target_pos)
      
      # 判断是否满足成功条件
      if distance < self.success_threshold:
          return True
      
      return False
  ```

- **终止条件**：
  ```python
  def _check_termination(self):
      # 检查是否达到终止条件
      # 1. 任务成功
      if self._check_task_success():
          return True
      
      # 2. 任务失败条件（根据新任务定义）
      # 例如：物体掉落到地面
      object_pos = self.mj_data.body("new_object").xpos.copy()
      if object_pos[2] < self.floor_height + 0.02:  # 物体高度低于地面一定高度
          return True
      
      # 3. 其他终止条件
      # ...
      
      return False
  ```

### 4. 观察值获取 (`env.py`)

#### 修改内容：

- **观察值计算**：
  ```python
  def _get_obs(self):
      # 获取机械臂状态
      qpos = self.mj_data.qpos.copy()  # 关节位置
      qvel = self.mj_data.qvel.copy()  # 关节速度
      
      # 获取任务相关物体的位置和姿态
      object_pos = self.mj_data.body("new_object").xpos.copy()
      object_quat = self.mj_data.body("new_object").xquat.copy()
      
      # 获取目标位置
      target_pos = self.mj_data.body("target_location").xpos.copy()
      
      # 计算物体与目标之间的相对位置（可选）
      relative_pos = object_pos - target_pos
      
      # 组合观察值
      obs = np.concatenate([
          qpos,                # 关节位置
          qvel,                # 关节速度
          object_pos,          # 物体位置
          object_quat,         # 物体姿态
          target_pos,          # 目标位置
          relative_pos         # 相对位置（可选）
      ])
      
      return obs
  ```

### 5. 域随机化 (`env.py`)

#### 修改内容：

- **随机化参数**：
  ```python
  def _domain_randomization(self):
      # 随机化新任务中的物体位置
      object_pos = self.mj_data.body("new_object").xpos.copy()
      object_pos[0] += np.random.uniform(-0.05, 0.05)  # x方向随机偏移
      object_pos[1] += np.random.uniform(-0.05, 0.05)  # y方向随机偏移
      self.mj_data.body("new_object").xpos[:] = object_pos
      
      # 随机化目标位置
      target_pos = self.mj_data.body("target_location").xpos.copy()
      target_pos[0] += np.random.uniform(-0.03, 0.03)  # x方向随机偏移
      target_pos[1] += np.random.uniform(-0.03, 0.03)  # y方向随机偏移
      self.mj_data.body("target_location").xpos[:] = target_pos
      
      # 随机化物理参数（可选）
      # 例如：摩擦系数
      friction = self.mj_model.geom_friction.copy()
      friction[:, 0] *= np.random.uniform(0.8, 1.2)  # 随机缩放摩擦系数
      self.mj_model.geom_friction[:] = friction
      
      # 前向传播更新物理状态
      mujoco.mj_forward(self.mj_model, self.mj_data)
  ```

## 使用方法

### 训练模型

```bash
cd /home/xhz/DISCOVERSE/policies/RLPD/pick_kiwi
python train.py --num_episodes 1000 --render
```

主要参数说明：
- `--num_episodes`: 训练的总回合数
- `--render`: 是否渲染环境
- `--num_demos`: 收集的演示数据数量
- `--demo_path`: 演示数据保存/加载路径
- `--batch_size`: 批次大小
- `--utd_ratio`: 更新频率比例（每个环境步骤更新网络的次数）

### 收集演示数据（不可用）

```bash
python train.py --num_episodes 10 --num_demos 20 --demo_path demos.npy --render
```

### 评估模型

```bash
python train.py --eval --model_path results/model_final --render
```

## 性能指标

训练过程中会记录以下指标：
- 每个回合的累积奖励
- 每个回合的步数
- 任务成功率
- 学习曲线（保存为图像）

## 注意事项

1. 训练前需要确保MuJoCo环境已正确安装
2. 演示数据对于RLPD算法的性能至关重要，可以使用高质量的专家演示
3. 可以调整奖励函数中的各项权重以改变任务的难度和学习曲线
4. 在适配新任务时，需要仔细设计奖励函数和成功条件，这对算法性能有重大影响
5. 键盘和手柄控制功能正在开发中，将在后续版本中提供
