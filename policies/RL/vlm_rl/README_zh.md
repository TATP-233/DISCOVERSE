# VLM-RL：视觉-语言模型引导的强化学习

本模块通过视觉-语言模型（VLM）自动生成奖励，实现强化学习。无需手工设计奖励函数，系统流程如下：

1. **从分割图 + 深度中提出候选关键点**
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
- 从物体 mask 采样关键点（中心点 + FPS）
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
│   ├── keypoint_proposer.py    # 从分割 + 深度生成关键点
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
│   ├── convert_dataset.py      # 数据集转换为 MuJoCo 格式
│   ├── generate_constraints.py # 步骤 1：生成约束
│   ├── train.py                # 步骤 2：训练策略
│   └── inference.py            # 步骤 3：评估
│
├── assets/                     # 场景资源
│   ├── meshes/                 # 物体和机器人网格文件
│   └── mjcf/                   # MuJoCo 场景文件
│
├── outputs/                    # 生成输出
│   └── <task_name>/
│       ├── constraints/        # VLM 生成的约束
│       └── train_*/            # 训练日志与模型
│
└── README.md
```

## 网格转换（凸分解）

将 3D 扫描的数据集转换为 MuJoCo 场景，支持 **凸分解** 生成精确的碰撞几何体。

### 为什么需要凸分解？

3D 扫描得到的网格（OBJ/STL）只有视觉几何，没有物理属性：

| 原始数据 | 物理属性 |
|----------|----------|
| `.obj` 网格 | 只有顶点/面，无碰撞体 |
| `.ply` 点云 | 无物理信息 |

MuJoCo 默认使用网格的**凸包**进行碰撞检测，这会"填满"凹陷部分（如锅的内部），导致碰撞不准确。

**凸分解**将复杂的非凸网格分解为多个凸部件，每个部件都能精确参与碰撞检测。

### 使用方法

```bash
cd policies/RL/vlm_rl

# 基本转换（输出到 assets/<dataset_name>/）
python scripts/convert_dataset.py /path/to/dataset

# 自定义输出目录和参数
python scripts/convert_dataset.py /path/to/dataset \
  --output ./my_scene \
  --name custom_scene_name \
  --table-width 1.0 \
  --table-depth 0.7 \
  --table-height 0.75 \
  --test
```

### 参数选项

| 参数 | 说明 |
|------|------|
| `dataset_dir` | 数据集目录 |
| `--output, -o` | 输出目录（默认：`assets/<dataset_name>`） |
| `--name, -n` | 场景名称（默认：数据集目录名） |
| `--no-robot` | 不生成机械臂 |
| `--table-width` | 桌子宽度（默认：1.0） |
| `--table-depth` | 桌子深度（默认：0.7） |
| `--table-height` | 桌子高度（默认：0.75） |
| `--test` | 测试加载生成的场景 |

### 输出结构

```
my_scene/
├── meshes/
│   ├── bottle.obj
│   ├── bottle_part_0.obj
│   ├── franka/
│   ├── robotiq/
│   └── ...
├── mjcf/
│   └── custom_scene_name.xml    # 完整场景（含机器人）
└── metadata.json
```

### 依赖安装

```bash
# 凸分解需要 coacd 和 trimesh
pip install coacd trimesh

# 预览需要 mujoco
pip install mujoco
```

### 数据集转换说明

`scripts/convert_dataset.py` 内部会调用 `mesh2mjcf.py` 的凸分解能力为每个物体生成碰撞网格，通常无需直接运行 `mesh2mjcf.py`。  
当前碰撞只使用凸分解结果，若分解失败会直接报错终止（不再回退到 box/cylinder）。

机器人资产说明：`assets/meshes/franka`、`assets/meshes/robotiq` 与 `assets/mjcf/panda_robotiq.xml` 已从 roboArena 同步，供场景直接引用。

### 查看生成的场景

```bash
python -m mujoco.viewer --mjcf=assets/mjcf/pick_<scene_name>.xml
```

### 示例：转换 new-desk 数据集

```bash
python scripts/convert_dataset.py /home/zoyo/Projects/DISCOVERSE/data/new-desk \
  --output /home/zoyo/Projects/DISCOVERSE/policies/RL/vlm_rl/assets/new_desk \
  --test
```

## 关键概念

### 关键点提议

不同于预定义语义关键点（如"把手""壶嘴"），我们：
1. 渲染分割图并在 2D mask 上采样（中心点 + FPS）
2. 通过深度与相机参数反投影到 3D
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
keypoint_include_center: true
keypoint_depth_search_radius: 6

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

---

## 开发进度记录

### 2026-01-26: Put Bottle Into Pot 任务开发

#### 已完成

1. **物理验证通过**
   - 场景加载正常 (`assets/mjcf/new_desk_scene.xml`)
   - 机械臂 7 个关节可控制
   - 夹爪可开合 (0-0.82 rad)
   - 物体 (bottle, pot, duster) 都有 free joint，可被移动

2. **创建独立约束生成脚本**
   - `scripts/generate_constraints_standalone.py` - 不依赖 DISCOVERSE SimNode
   - 直接用 MuJoCo 加载 MJCF 场景

3. **自定义相机支持**
   - 在 `AnnotatedRenderer` 中添加了自由相机参数支持
   - 配置文件支持 `camera_config`: lookat, distance, azimuth, elevation
   - 最终视角: 从机械臂对面俯瞰桌面

4. **VLM 约束生成基本工作**
   - GPT-4o 能正确理解场景和任务
   - 生成 3 阶段约束: 抓取 → 抬起 → 放入

5. **2D 关键点采样流程（分割+深度）**
   - 使用 MuJoCo segmentation 渲染得到对象 mask
   - 在 2D mask 上采样关键点（中心点 + FPS）
   - 通过深度与相机参数反投影到 3D
   - 解决“锅中心点缺失/点漂移”的问题

6. **渲染与脚本更新**
   - AnnotatedRenderer 支持分割渲染与 2D 关键点可视化
   - `generate_constraints.py` 和 `generate_constraints_standalone.py` 支持 2D 采样
   - 配置新增：`keypoint_include_center`、`keypoint_depth_search_radius`
   - 移除旧的 3D 表面采样提议器（默认使用 2D 流程）

### 2026-01-26 晚间更新: RL训练框架构建完成

#### 新增完成

1. **Prompt 改进**
   - 修复 `RELEASE_KEYPOINTS` 语义混淆问题（VLM 误解为目标位置）
   - 添加明确说明：RELEASE_KEYPOINTS 指释放**被抓取的点**，而非目标位置
   - 强制要求每个 stage 都必须有 subgoal constraint（用于 stage transition）

2. **独立训练环境** (`src/standalone_env.py`)
   - `StandaloneVLMRLEnv`: 纯 MuJoCo 实现，不依赖 DISCOVERSE
   - 自动检测 Franka Panda 的 7 个 arm joints
   - 正确处理 home keyframe reset
   - 修复 `mj_data.time` 未重置导致 episode 提前终止的 bug

3. **独立训练脚本** (`scripts/train_standalone.py`)
   - 使用 SBX PPO (JAX-based) 或 Stable Baselines3
   - 支持 VecNormalize 归一化
   - 详细的 episode 日志（reward, length, stage）
   - 自动保存 checkpoint 和最终模型

4. **Bug 修复**
   - `reward_adapter.py`: 修复第一步 reward=inf 的问题（prev_cost 初始化）
   - `constraint_generator.py`: 修复 "open" 误报（"opening" 触发 blacklist）
   - `standalone_env.py`: 修复 reset 不重置仿真时间

#### 小规模训练测试结果

```bash
python scripts/train_standalone.py \
    --config configs/tasks/put_bottle_into_pot.yaml \
    --constraints_dir outputs/put_bottle_into_pot/constraints \
    --total_timesteps 10000
```

结果:
- Episode 长度: 500 (max_steps，正常)
- Reward: ~-20 (stage 1)
- 训练速度: ~500 it/s (CPU)

#### 当前约束结构

| Stage | Type | Constraint | 说明 |
|-------|------|------------|------|
| 1 | subgoal | `distance(ee, kp2) < 2cm` | 接近瓶子抓取点 |
| 2 | subgoal | `bottle_height > 0.6m` | 抬起瓶子 |
| 2 | path | `bottle_vertical` | 保持瓶子竖直 |
| 3 | subgoal1 | `horizontal_dist < 3cm` | 对齐锅上方 |
| 3 | subgoal2 | `bottle_bottom ≤ pot_height` | 放入锅中 |

#### 待解决问题

1. **Stage 2 高度 0.6m 是 hardcoded**
   - 可能需要根据场景动态计算

2. **JAX 使用 CPU 而非 GPU**
   - 警告: "An NVIDIA GPU may be present but CUDA-enabled jaxlib is not installed"
   - 需要安装 `jaxlib[cuda]` 以使用 GPU

#### 下一步

1. **大规模训练测试**
   - 使用 GPU 训练 100k+ timesteps
   - 观察是否能进入 Stage 2/3

2. **可能的改进**
   - 调整 reward shaping 参数
   - 添加 gripper 控制（当前只控制 arm joints）

#### 相关文件

```
新增/修改的文件:
├── src/
│   ├── standalone_env.py      # [新增] 独立 Gym 环境
│   ├── reward_adapter.py      # [修复] inf reward bug
│   └── constraint_generator.py # [修复] blacklist 误报
├── scripts/
│   └── train_standalone.py    # [新增] 独立训练脚本
├── prompts/
│   └── constraint_prompt.txt  # [改进] RELEASE_KEYPOINTS 说明
└── outputs/put_bottle_into_pot/
    └── train_*/               # 训练输出
```

### 2026-01-27: 训练问题诊断与修复

#### 100k训练结果分析

首次100k训练完成，发现问题：
- **Avg Stage: 1.02** - 几乎一直停在Stage 1
- **仿真不稳定警告** - "Nan, Inf or huge value in QACC at DOF 17/28/29"
- Reward从-20提升到-0.57（有学习，但未进阶）

#### 问题诊断

1. **End Effector位置错误（关键问题）**
   ```
   robotiq_base (之前使用): z = 0.982m → 到keypoint 2距离 0.521m
   gripper site (正确位置): z = 0.833m → 到keypoint 2距离 0.388m

   差距: 13cm！使用错误的位置导致距离计算偏大
   ```

2. **Action过大导致仿真爆炸**
   - 原来action直接映射到关节位置全范围
   - 突变的关节位置导致巨大加速度

3. **缺少引导性reward**
   - 只有subgoal满足时才有正向信号
   - 远离目标时没有方向引导

4. **可达性问题（待解决）**
   ```
   目标位置 (keypoint 2):  [0.450, 0.141, 0.501]
   最佳可达位置:           [0.429, 0.172, 0.550]
   最小可达距离:           0.062m (6.2cm)
   Subgoal阈值:           0.02m (2cm)

   机械臂物理上无法到达瓶子位置到2cm以内！
   ```

#### 已完成修复

1. **End Effector位置修复** (`standalone_env.py`)
   ```python
   # 优先使用gripper site（指尖位置）而非robotiq_base（基座）
   def _get_end_effector_pos(self):
       site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
       if site_id >= 0:
           return self.mj_data.site_xpos[site_id].copy()
   ```

2. **增量控制** (`standalone_env.py`)
   ```python
   # 每步最大变化量限制为0.05rad（约3度）
   max_delta = 0.05
   delta = action[i] * max_delta
   target = current_pos + delta  # 增量而非绝对位置
   ```

3. **Shaped Reward** (`standalone_env.py`)
   ```python
   def _compute_shaped_reward(self, ee_pos, keypoints):
       # Stage 1: 引导末端执行器接近抓取点
       distance = np.linalg.norm(ee_pos - keypoints[grasp_idx])
       return weight * (np.exp(-distance * 5) - 0.5)
   ```

#### 修复后测试结果

20k训练测试：
- **无仿真爆炸警告** ✓
- Reward: -14.77 → -12.64（稳定）
- Stage仍为1.0（因可达性问题）

#### 待决策问题

机械臂无法到达keypoint 2到2cm以内，需要选择：
1. **放宽阈值** - `subgoal_threshold: 0.02 → 0.08`
2. **调整场景** - 移动瓶子位置
3. **修改关键点** - 使用更易到达的关键点
4. **调整机械臂位置** - 修改MJCF

#### 相关文件更新

```
修改的文件:
├── src/
│   └── standalone_env.py
│       - _get_end_effector_pos(): 使用gripper site
│       - _apply_action(): 增量控制
│       - _compute_shaped_reward(): shaped reward引导
│       - step(): 集成shaped reward
└── outputs/put_bottle_into_pot/
    ├── train_100k/            # 首次100k训练（有问题）
    └── train_fixed_20k/       # 修复后20k测试
```

### 2026-01-27: Stage 1 抓取奖励增强（夹爪包围约束）

#### 新增

1. **夹爪包围奖励（软约束）**
   - 使用左右夹板中点作为参考，判断抓取点是否落在夹爪间隙内
   - 条件：轴向投影在夹爪宽度内 + 径向距离小于半径阈值
   - 可选要求夹爪闭合到一定程度（open fraction 阈值）
   - 满足条件给正奖励，否则给惩罚（可关闭惩罚）

2. **配置项新增**
   - `grasp_enclosure_enable`
   - `grasp_enclosure_weight`
   - `grasp_enclosure_radius`
   - `grasp_enclosure_margin`
   - `grasp_enclosure_max_open`
   - `grasp_enclosure_penalty`

#### 相关文件

```
修改的文件:
├── src/
│   └── standalone_env.py      # [新增] grasp_enclosure_reward
├── scripts/
│   └── train_standalone.py    # [新增] grasp_enclosure_reward 日志
└── configs/tasks/
    └── put_bottle_into_pot.yaml  # [新增] grasp_enclosure_* 配置
```

### 2026-01-27 下午: 约束函数阈值重构 + EE位置修复

#### 问题诊断

1. **jaw_center 模式位置计算错误**
   - `left_pad`/`right_pad` body 的世界坐标位置不正确（比 gripper site 高 14cm）
   - 导致距离计算偏大，Stage 1 无法完成

   | 方法 | Z 高度 | 到目标距离 |
   |------|--------|-----------|
   | gripper site | 0.833m | 0.388m |
   | jaw center | 0.971m | 0.511m |

2. **约束函数硬编码阈值**
   - VLM 生成的约束中 `return distance - 0.02` 硬编码了 2cm 阈值
   - 配置中 `subgoal_threshold: 0.03` 是另一个阈值
   - 两个阈值叠加，语义混乱，调参困难

#### 修复内容

1. **EE 位置修复**
   - 将 `end_effector_mode` 从 `jaw_center` 改为 `site`
   - 直接使用 `gripper` site 位置（更准确的指尖位置）

2. **约束函数阈值重构**
   - 修改 prompt，要求 VLM 不硬编码阈值
   - 约束函数只返回原始距离/误差值
   - 阈值完全由配置控制

   ```python
   # 旧设计（有问题）
   return distance - 0.02  # 硬编码阈值

   # 新设计
   return distance  # 返回原始值，阈值由配置决定
   ```

3. **reward_adapter 更新**
   - 新增 `path_threshold` 参数（默认 0.1）
   - path constraint 判断改为 `cost <= path_threshold`

#### 重新生成的约束

| Stage | Type | 返回值 | 配置阈值 |
|-------|------|-------|---------|
| 1 | subgoal | `distance` | 0.03m |
| 2 | subgoal | `max(0, height_error)` | 0.03m |
| 2 | path | `alignment_error` | 0.1 |
| 3 | subgoal | `horizontal_distance` | 0.03m |

#### 可达性验证

```
目标位置 (keypoint 2): [0.450, 0.141, 0.501]
最小可达距离:          2.3cm
配置阈值:             3cm
结论: ✓ 目标可达
```

#### 修改的文件

```
├── prompts/
│   └── constraint_prompt.txt     # [修改] 不硬编码阈值说明
├── src/
│   └── reward_adapter.py         # [新增] path_threshold 参数
├── configs/tasks/
│   └── put_bottle_into_pot.yaml  # [修改] end_effector_mode: site, path_threshold
└── outputs/put_bottle_into_pot/
    └── constraints/              # [重新生成] 无硬编码阈值
```

### 2026-01-27 晚间: 训练调试与优化

#### 训练测试结果

| 训练 | Timesteps | Stage 2 达成率 | 问题 |
|------|-----------|---------------|------|
| train_50k_refactored | 50k | 1/102 (1.0%) | 训练量不足 |
| train_50k_v2 | 50k | 1/102 (1.0%) | 增大 max_delta, shaped_weight |
| train_50k_v3 | 50k | 1/102 (1.0%) | 改进 shaped reward |
| train_200k | 200k | 6/401 (1.5%) | 训练量增加有帮助 |

#### 发现的问题

1. **left_pad/right_pad body 位置不正确**
   ```
   gripper site Z: 0.833m
   jaw center Z:   0.971m  (差距 14cm！)
   ```
   导致 grasp_enclosure 永远无法满足（一直返回惩罚）

2. **物理可达性验证**
   - 用启发式搜索，93 步可达 0.006m（远小于 3cm 阈值）
   - 说明问题不在物理，而在 RL 学习

#### 已做的优化

1. **禁用 grasp_enclosure** - body 位置计算有问题
2. **改进 shaped reward**：
   ```python
   # 旧: 指数衰减（梯度太平缓）
   return shaped_weight * (np.exp(-distance * 5) - 0.5)

   # 新: 线性惩罚 + 接近奖励
   if distance < 0.05:
       return shaped_weight * (1.0 - distance * 10)
   else:
       return shaped_weight * (-distance)
   ```
3. **增大控制参数**：
   - `max_action_delta`: 0.05 → 0.1 rad
   - `shaped_reward_weight`: 0.1 → 0.5

#### 当前训练

`train_200k_v2`: 200k timesteps，禁用 grasp_enclosure

#### 修改的文件

```
├── src/
│   └── standalone_env.py           # [修改] shaped reward 改进
├── configs/tasks/
│   └── put_bottle_into_pot.yaml    # [修改] 禁用 grasp_enclosure
└── outputs/put_bottle_into_pot/
    ├── train_50k_*/                # 50k 测试训练
    └── train_200k*/                # 200k 训练
```

### 2026-01-27 深夜: 1M 训练突破 + 抓取逻辑探索

#### 1M 训练结果

```
Total episodes: 2002
Final avg reward: -1.95
Stage 2 rate: 33.4% (669/2002)
Stage 3 rate: 0%
```

**关键发现**：
- Stage 2（抬起瓶子）达成率 33.4%
- Stage 3（放入锅中）达成率 0%
- **根本问题**：Stage 1 只是让 EE 接近目标，但没有真正抓住瓶子

#### 抓取逻辑修复

1. **Geom 位置修复**
   - 之前用 `left_pad`/`right_pad` body 位置（与 gripper site 差 14cm）
   - 改用 `left_finger_pad`/`right_finger_pad` **geom** 位置（只差 0.9cm）

   ```python
   # 新增 _get_finger_pad_positions() 方法
   # 优先使用 geom_xpos，fallback 到 body xpos
   ```

2. **Top-down approach（实验）**
   - 尝试添加"从上方接近"约束
   - 结果：反而影响了学习效率，Stage 2 rate 下降

3. **Grasp enclosure（实验）**
   - 启用后 Stage 2 rate 从 33.4% 下降到 3.2%
   - 可能干扰了原有的学习信号

#### 当前状态

**核心问题仍未解决**：需要让策略学会：
1. 接近目标
2. 张开夹爪
3. 对准位置
4. 闭合夹爪抓住物体
5. 检测是否成功抓住

**待探索方向**：
1. 分阶段训练（curriculum learning）
2. 接触检测作为抓取成功的信号
3. 物体位置变化作为抓取成功的判断

#### 修改的文件

```
├── src/
│   └── standalone_env.py
│       - _get_finger_pad_positions(): 使用 geom 位置
│       - _compute_grasp_enclosure_reward(): 更新使用 geom
│       - 简化 shaped reward（移除 top-down 约束）
└── outputs/put_bottle_into_pot/
    ├── train_1M/                   # 最佳结果: 33.4% Stage 2
    ├── train_100k_grasp/           # grasp enclosure 测试
    └── train_200k_grasp_v2/        # 简化 grasp 测试
```

### 2026-01-27 深夜续: 抓取确认门控机制

#### 问题分析

之前的尝试：
- **grasp_enclosure_reward**：作为额外奖励添加，干扰了学习信号
- **结果**：Stage 2 rate 从 33.4% 下降到 3.2%

核心问题：Stage 1 完成只需要 EE 接近目标，不需要真正抓住物体。

#### 新方案：require_grasp_for_stage1

不再把抓取检测作为额外奖励，而是作为 **Stage 1 完成的必要条件**：

```python
# standalone_env.py step() 函数
if self.require_grasp_for_stage1 and stage_before == 1:
    if reward_info.stage_complete and not self._grasp_confirmed:
        # Subgoal 满足但抓取未确认 - 撤销 stage 进阶
        self.reward_adapter.current_stage = 1
        reward -= self.reward_adapter.stage_bonus
```

**工作原理**：
1. 策略移动 EE 到目标点（满足 Stage 1 subgoal）
2. Auto-grasp 激活，开始闭合夹爪
3. 当夹爪停止闭合（碰到物体）且物体跟随夹爪移动时，`_grasp_confirmed = True`
4. 只有 `_grasp_confirmed` 时才允许进入 Stage 2

#### 训练测试

```
train_50k_grasp_gate:
- 所有 100 个 episode 停留在 Stage 1
- 平均奖励改善：-1050 → -506（EE 在学习接近）
- grasp_confirmed 从未触发（Avg Enclosure: 0.000）
```

#### 新问题识别

策略需要：
1. 移动 EE 到目标附近 ✓（学到了）
2. **稳定停留**足够长时间让 auto_grasp 激活 ✗
3. 夹爪关闭并确认抓取 ✗

当前奖励只鼓励"接近"，没有鼓励"稳定停留"。策略学会了快速经过目标点，但没有学会停在那里。

#### 待探索方向

1. **停留奖励**：EE 在目标附近停留时给予奖励
2. **速度惩罚**：当 EE 接近目标时惩罚高速度
3. **分阶段目标**：先学接近，再学停留，再学抓取
4. **降低 auto_grasp 激活条件**：更短的 align_steps，更大的 align_tol

#### 新增配置项

```yaml
# 要求抓取确认才能完成 Stage 1
require_grasp_for_stage1: true
```

#### 修改的文件

```
├── src/
│   └── standalone_env.py
│       - [新增] require_grasp_for_stage1 配置项
│       - [修改] step(): 在 compute_reward 前调用 _update_grasp_state
│       - [修改] step(): 门控 Stage 1 完成条件
├── configs/tasks/
│   └── put_bottle_into_pot.yaml
│       - [新增] require_grasp_for_stage1: true
└── outputs/put_bottle_into_pot/
    └── train_50k_grasp_gate/       # 门控机制测试
```

#### 训练结果汇总

| 训练 | Timesteps | grasp_enclosure | require_grasp | Stage 2 Rate |
|------|-----------|-----------------|---------------|--------------|
| train_200k | 200k | disabled | N/A | 1.7% |
| train_1M | 1M | disabled | N/A | **33.4%** |
| train_500k_baseline | 500k | disabled | N/A | 6.5% |
| train_200k_grasp_v2 | 200k | enabled | N/A | 3.2% |
| train_50k_grasp_gate | 50k | disabled | enabled | 0% (预期) |

**关键发现**：
- 更长训练时间有显著帮助（200k 1.7% → 1M 33.4%）
- grasp_enclosure 作为额外奖励会干扰学习
- require_grasp 门控正确阻止了"假进阶"，但需要配合其他机制让策略学会停留
