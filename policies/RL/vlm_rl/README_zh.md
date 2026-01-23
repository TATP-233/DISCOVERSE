# VLM-RL：视觉-语言模型引导的强化学习

本模块通过视觉-语言模型（VLM）自动生成奖励，实现强化学习。无需手工设计奖励函数，系统流程如下：

1. **从 MuJoCo 场景几何中提出候选关键点**
2. **渲染带编号关键点的标注图像**
3. **使用 VLM（GPT-4o）根据任务指令生成约束函数**
4. **将约束转换为奖励**用于 PPO 训练

## 架构

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      VLM-RL 训练系统                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   阶段 1：约束生成（只需运行一次）                                          │
│   ┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐     │
│   │ MuJoCo 场景   │ ──▶ │  关键点          │ ──▶ │ VLM (GPT-4o)      │     │
│   │ + 任务指令    │     │  提议器          │     │  约束生成         │     │
│   └───────────────┘     └──────────────────┘     └───────────────────┘     │
│                                │                           │               │
│                                ▼                           ▼               │
│                         ┌──────────────┐           ┌──────────────┐        │
│                         │ 标注图像     │           │ 约束函数     │        │
│                         └──────────────┘           └──────────────┘        │
│                                                                            │
│   阶段 2：强化学习训练                                                     │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                         训练循环                                   │    │
│   │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐    │    │
│   │  │ PPO 智能体  │◀──▶│  VLMRLEnv       │◀──▶│  MuJoCo         │    │    │
│   │  │ (SBX)       │    │  (Gymnasium)    │    │  仿真           │    │    │
│   │  └─────────────┘    └─────────────────┘    └─────────────────┘    │    │
│   │                              │                                    │    │
│   │                              ▼                                    │    │
│   │                     ┌─────────────────┐                           │    │
│   │                     │  奖励适配器     │                           │    │
│   │                     │  (约束→奖励)    │                           │    │
│   │                     └─────────────────┘                           │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## 安装

```bash
# 确保已安装 DISCOVERSE
cd /path/to/DISCOVERSE
pip install -e .

# 安装额外依赖
pip install sbx-rl opencv-python openai pyyaml
```

## 快速开始

### 步骤 1：生成约束

```bash
cd policies/RL/vlm_rl

# 设置 OpenAI API Key
export OPENAI_API_KEY="your-api-key"

# 为 pick_kiwi 任务生成约束
python scripts/generate_constraints.py \
    --config configs/tasks/pick_kiwi.yaml \
    --instruction "Pick up the kiwi and place it on the white plate"
```

这将会：
- 加载 MuJoCo 场景
- 从物体表面生成候选关键点
- 渲染标注图像
- 调用 GPT-4o 生成约束函数
- 将结果保存到 `outputs/pick_kiwi/constraints/`

### 步骤 2：训练策略

```bash
python scripts/train.py \
    --config configs/tasks/pick_kiwi.yaml \
    --constraints_dir outputs/pick_kiwi/constraints \
    --total_timesteps 1000000
```

### 步骤 3：评估

```bash
python scripts/inference.py \
    --config configs/tasks/pick_kiwi.yaml \
    --model_path outputs/pick_kiwi/train_*/models/final_model \
    --constraints_dir outputs/pick_kiwi/constraints \
    --episodes 100 \
    --render
```

## 项目结构

```
vlm_rl/
├── src/
│   ├── __init__.py
│   ├── keypoint_proposer.py    # 从 MuJoCo 几何生成关键点
│   ├── annotated_renderer.py   # 渲染带关键点标注的场景
│   ├── constraint_generator.py # VLM 约束生成
│   ├── keypoint_tracker.py     # 仿真中跟踪关键点
│   ├── reward_adapter.py       # 将约束转换为奖励
│   └── env.py                  # Gymnasium 环境
│
├── configs/
│   ├── default.yaml            # 默认配置
│   └── tasks/
│       └── pick_kiwi.yaml      # 任务配置
│
├── prompts/
│   └── constraint_prompt.txt   # VLM 提示词模板
│
├── scripts/
│   ├── generate_constraints.py # 步骤 1：生成约束
│   ├── train.py                # 步骤 2：训练策略
│   └── inference.py            # 步骤 3：评估
│
├── outputs/                    # 生成输出
│   └── <task_name>/
│       ├── constraints/        # VLM 生成的约束
│       └── train_*/            # 训练日志与模型
│
└── README.md
```

## 关键概念

### 关键点提议

不同于预定义语义关键点（如“把手”“壶嘴”），我们：
1. 从 MuJoCo 几何表面采样候选点
2. 使用最远点采样（FPS）保证多样性
3. 让 VLM 决定哪些点具有语义意义

该方法（受 [ReKep](https://rekep-robot.github.io/) 启发）无需人工标注。

### 约束函数

VLM 生成的 Python 约束函数具有如下签名：

```python
def stage{N}_{type}_constraint{M}(end_effector, keypoints):
    """
    Args:
        end_effector: np.ndarray [3] - 末端执行器位置
        keypoints: np.ndarray [K, 3] - 所有关键点位置

    Returns:
        cost: float - 负值/零表示满足，正值表示违反
    """
    return cost
```

类型：
- **subgoal**：阶段结束时必须满足
- **path**：在该阶段全过程中必须满足

### 多阶段任务

任务被分解为多个阶段（如接近 → 抓取 → 运输 → 放置）。
每个阶段都有自己的约束，奖励适配器负责跟踪进度。

## 配置

### 任务配置（`configs/tasks/*.yaml`）

```yaml
task_name: "my_task"
task_module: "discoverse.examples.tasks_mmk2.my_task"

instruction: "自然语言任务描述"

object_bodies:
  - "object1"
  - "object2"

end_effector_body: "gripper_link"
points_per_object: 5

reward_config:
  reward_type: "negative"  # negative, exponential, sparse, tanh
  subgoal_weight: 2.0
  path_weight: 1.0
```

### 添加新任务

1. 创建 MuJoCo 模型（`.xml`）与任务模块
2. 在 `configs/tasks/` 中创建配置文件
3. 使用任务指令运行 `generate_constraints.py`
4. 使用 `train.py` 训练

## 与手工奖励设计的对比

| 方面 | 手工 | VLM-RL |
|------|------|--------|
| 配置成本 | 高（需要设计奖励） | 低（只需写指令） |
| 新任务 | 需要工程工作 | 只需改指令 |
| 可解释性 | 取决于设计 | 约束可读 |
| 性能 | 可优化 | 依赖 VLM 质量 |

## 故障排查

### VLM 生成问题

- 确认已设置 `OPENAI_API_KEY`
- 查看 `outputs/*/constraints/raw_output.txt` 中的 VLM 输出
- 必要时调整 `prompts/constraint_prompt.txt`

### 训练问题

- 检查约束文件是否有语法错误
- 使用 `--render` 可视化行为
- 调整配置中的奖励权重

## 参考

- [ReKep: Spatio-Temporal Reasoning of Relational Keypoint Constraints for Robotic Manipulation](https://rekep-robot.github.io/)
- [DISCOVERSE Simulation Framework](https://github.com/DISCOVER-Robotics/DISCOVERSE)
- [Stable Baselines3 (SBX)](https://github.com/araffin/sbx)
