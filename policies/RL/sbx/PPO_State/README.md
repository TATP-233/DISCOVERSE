# SBX PPO 强化学习训练框架

本目录包含使用 PPO (Proximal Policy Optimization) 算法进行强化学习训练的框架代码。这个框架专为 DISCOVERSE 环境设计，特别是针对机械臂抓取任务。

## 安装
参考https://github.com/araffin/sbx，用的是v0.20.0

## 文件结构

- `env.py`: 环境类定义，包含观察空间、动作空间、奖励计算等
- `train.py`: 训练脚本，用于创建和训练 PPO 模型

## 环境配置与自定义

### 导入说明

在 `env.py` 中，我们从示例目录导入了 `SimNode` 和 `cfg`：

```python
from discoverse.examples.tasks_mmk2.pick_kiwi import SimNode, cfg
```

这里的 `pick_kiwi.py` 是位于 `discoverse/examples/tasks_mmk2/` 目录下的示例文件。您可以根据自己的任务需求，替换为其他任务文件。

### 环境配置

环境配置主要在 `Env` 类的 `__init__` 方法中设置：

```python
# 环境配置
cfg.use_gaussian_renderer = False  # 关闭高斯渲染器
cfg.init_key = "pick"  # 初始化模式为"抓取"
cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义白色盘子的模型路径
cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义奇异果的模型路径
cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景的模型路径
cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo 环境文件路径
cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
cfg.sync = True  # 是否同步更新
cfg.headless = not render  # 根据render参数决定是否显示渲染画面
```

您可以根据自己的任务需求修改这些配置，例如更改模型路径、对象列表等。

### 重点自定义部分

#### 1. 观察空间 (Observation Space)

观察空间定义了智能体能够感知到的环境状态。在 `_get_obs` 方法中自定义：

```python
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
```

根据您的任务，您可能需要包含不同的状态信息，例如：
- 机械臂的关节角度和速度
- 目标物体的位置和姿态
- 障碍物的位置
- 传感器读数等

#### 2. 奖励函数 (Reward Function)

奖励函数是强化学习中最关键的部分之一，它定义了任务的目标。在 `_compute_reward` 方法中自定义：

```python
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
```

设计奖励函数时，请考虑：
- 稀疏奖励 vs 密集奖励：稀疏奖励只在完成任务时给予，而密集奖励在每一步都提供反馈
- 奖励分量：将奖励分解为多个组成部分，如接近目标的奖励、完成任务的奖励等
- 惩罚项：对不希望的行为施加惩罚，如过大的动作、过多的步数等

#### 3. 终止条件 (Termination Condition)

终止条件定义了何时结束一个回合。在 `_check_termination` 方法中自定义：

```python
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
```

## 训练与测试

### 训练模型

使用 `train.py` 脚本训练模型：

```bash
python train.py --render
```

参数说明：
- `--render`: 在训练过程中显示渲染画面（可选）

### 测试模型

训练完成后，使用以下命令测试模型：

```bash
python train.py --test --model_path /path/to/your/model.zip
```

参数说明：
- `--test`: 启用测试模式
- `--model_path`: 指定要加载的模型路径

## 自定义任务的步骤

1. **创建任务环境**：
   - 复制 `env.py` 并根据您的任务需求进行修改
   - 调整观察空间、奖励函数和终止条件

2. **配置环境**：
   - 修改 `cfg` 配置，包括模型路径、对象列表等
   - 根据需要调整渲染设置

3. **调整训练参数**：
   - 在 `train.py` 中修改 PPO 算法的超参数
   - 调整训练步数、批次大小等

4. **训练与评估**：
   - 运行训练脚本并监控训练进度
   - 定期评估模型性能并保存最佳模型

## 提示与技巧

1. **奖励设计**：
   - 奖励函数是强化学习中最关键的部分
   - 尝试不同的奖励组合，找到最有效的设计
   - 使用奖励分量来引导智能体学习复杂任务

2. **观察空间**：
   - 只包含任务相关的信息，避免无关信息干扰学习
   - 考虑使用相对位置而非绝对位置

3. **超参数调优**：
   - 学习率、批次大小、更新频率等超参数对训练效果有显著影响
   - 使用网格搜索或贝叶斯优化来寻找最佳超参数

4. **可视化与调试**：
   - 使用 TensorBoard 等工具可视化训练过程
   - 记录详细的奖励分量，便于分析和调试

5. **域随机化**：
   - 在训练中引入随机性，提高模型的泛化能力
   - 随机化初始状态、物体位置、物理参数等
