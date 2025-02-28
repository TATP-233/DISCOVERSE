# SBX PPO 基于视觉的强化学习训练框架

本目录包含使用 PPO (Proximal Policy Optimization) 算法进行基于视觉的强化学习训练的框架代码。这个框架专为 DISCOVERSE 环境设计，特别是针对机械臂抓取任务，使用视觉输入代替状态输入。

## 安装
参考 https://github.com/araffin/sbx，使用的是 v0.20.0 版本

## 文件结构

- `env.py`: 环境类定义，包含基于视觉的观察空间、动作空间、奖励计算等
- `train.py`: 训练脚本，用于创建和训练基于视觉的 PPO 模型

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

#### 1. 基于视觉的观察空间 (Observation Space)

与基于状态的PPO不同，基于视觉的PPO使用图像作为输入：

```python
# 观测空间：基于视觉的观察空间
self.observation_space = spaces.Box(
    low=0,
    high=1,
    shape=(3, 84, 84),  # RGB图像，调整为84x84大小
    dtype=np.float32
)
```

#### 2. 视觉观察获取方法

在 `_get_obs` 方法中，我们获取并处理图像数据：

```python
def _get_obs(self):
    # 获取摄像头图像
    action = np.zeros_like(self.mj_data.ctrl)  # 创建空动作
    obs_dict = self.task_base.step(action)  # 获取观察字典
    
    # 提取图像数据
    if 'img' in obs_dict[0]:
        img = obs_dict[0]['img'][0]  # 获取第一个相机的图像
        img = img.astype(np.float32) / 255.0  # 归一化到[0,1]范围
        img = img.transpose(2, 0, 1)  # 转换为(C, H, W)格式
        img = resize(img, (3, 84, 84), anti_aliasing=True)  # 调整大小为84x84
        return img
    else:
        # 如果没有图像，返回零矩阵
        return np.zeros((3, 84, 84), dtype=np.float32)
```

#### 3. 自定义CNN特征提取器

在 `train.py` 中，我们定义了一个自定义的CNN特征提取器，用于处理图像输入：

```python
class CNNFeatureExtractor(torch.nn.Module):
    def __init__(self, observation_space):
        super(CNNFeatureExtractor, self).__init__()
        # 输入是(3, 84, 84)的RGB图像
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        
        # 计算CNN输出特征的维度
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, 512),
            torch.nn.ReLU()
        )
        
        self._features_dim = 512
```

## 训练与测试

### 训练模型

使用 `train.py` 脚本训练模型：

```bash
sudo python train.py --render
```

参数说明：
- `--render`: 在训练过程中显示渲染画面（可选）

### 测试模型

训练完成后，使用以下命令测试模型：

```bash
sudo python train.py --test --model_path /path/to/your/model.zip
```

参数说明：
- `--test`: 启用测试模式
- `--model_path`: 指定要加载的模型路径

## 与基于状态的PPO的区别

基于视觉的PPO与基于状态的PPO主要有以下区别：

1. **输入类型**：基于视觉的PPO使用图像作为输入，而基于状态的PPO使用状态向量作为输入。
2. **网络结构**：基于视觉的PPO使用CNN来处理图像输入，而基于状态的PPO使用MLP来处理状态向量。
3. **计算复杂度**：基于视觉的PPO计算复杂度更高，训练时间更长，但可以直接从原始像素中学习策略。
4. **泛化能力**：基于视觉的PPO通常具有更好的泛化能力，可以适应不同的视觉环境。

## 性能优化建议

1. **图像预处理**：调整图像大小、归一化等预处理步骤对性能影响很大。
2. **CNN架构**：尝试不同的CNN架构，如ResNet、EfficientNet等。
3. **数据增强**：使用随机裁剪、旋转等数据增强技术提高泛化能力。
4. **多帧输入**：考虑使用多帧图像作为输入，捕捉时间信息。
5. **混合输入**：结合视觉输入和状态输入，可能获得更好的性能。
