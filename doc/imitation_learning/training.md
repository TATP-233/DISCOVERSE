## 命令

```bash
python3 policies/train.py <policy> [args]
```

解释：
- `policy`: 位置参数，指定策略的类型，目前支持的选项：act、dp
- `[args]`: 不同的策略有不同的命令行参数，请参考下面对应策略的说明

## act

### 依赖安装

```bash
pip install -r policies/act/requirements/train_eval.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 训练配置
参考的训练配置文件位于`policies/act/configurations/task_configs/example_task.py`中，其中主要参数解释如下：
- `camera_names`: 训练数据中相机的序号
- `state_dim`: 训练数据中观测向量的维度
- `action_dim`: 训练数据中动作向量的维度
- `batch_size_train`: 训练时的batch_size大小
- `batch_size_validate`: 验证时的batch_size大小
- `chunk_size`: 单次预测的动作数量
- `num_epochs`: 训练的总步数
- `learning_rate`: 学习率

训练特定任务时，需要复制一份配置文件并重命名为任务名，后续将通过任务名索引相关配置文件。


### 数据集位置
仿真采集的数据默认位于discoverse仓库根目录的data文件夹中，而训练时默认从policies/act/data/hdf5中寻找数据。因此，建议使用软连接的方式将前者链接到后者，命令如下（注意修改命令中的路径，并且需要绝对路径）：

```bash
ln -sf /absolute/path/to/discoverse/data /absolute/path/to/discoverse/policies/act/data
```

### 训练命令

```bash
python3 policies/train.py act -tn <task_name>
```

其中`-tn`参数指定任务名，程序会根据任务名分别在`task_configs`和`act/data/hdf5`目录下寻找同名的配置文件和数据集。

### 训练结果

训练结果保存在`policies/act/my_ckpt`目录下。

## dp

### 依赖安装

```bash
pip install -r policies/dp/requirements.txt 
cd policies/dp
pip install -e .
```

### 训练配置
参考的训练配置文件位于`policies/dp/configs/block_place.yaml`中，其中主要参数解释如下：
- `task_path`: 推理时，程序会加载其中的`SimNode`类和实例`cfg`来创建仿真环境
- `max_episode_steps`: 推理时动作执行总步数
- `obs_keys`: 模型输入的obs名称，若有多个视角的图像，则在`image`后加上对应`cam_id`
- `shape_meta`: 输入obs的形状及类型，注意img的尺寸需要和生成的图像尺寸一致
- `action_dim`: 动作维度
- `obs_steps`: 输入`obs`时间步长
- `action_steps`: 输出`action`时间步长

训练特定任务时，可以复制一份配置文件并重命名为任务名，作为该任务特定的配置文件。


### 数据集位置
仿真采集的数据默认位于discoverse仓库根目录的data文件夹中，而训练时默认从policies/dp/data/zarr中寻找数据。因此，建议使用软连接的方式将前者链接到后者，命令如下（注意修改命令中的路径，并且需要绝对路径）：

```bash
ln -sf /absolute/path/to/discoverse/data /absolute/path/to/discoverse/policies/dp/data
```

### 训练命令

```bash
python3 policies/train.py dp --config-path=configs --config-name=block_place mode=train
```

其中:
- `--config-path`: 配置文件所在路径
- `--config-name`: 配置文件名
- `mode`: 指定训练或是推理

### 训练结果

训练结果保存在`policies/dp/logs`目录下。

## diffusion policy

### 安装依赖:
```bash
cd policies/Diffusion-Policy
pip install -e .
cd ../..
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
```
### 数据集位置
```bash
cd DISCOVERSE
mv data/zarr/block_place.zarr policies/Diffusion-Policy/data
```

### 训练命令
```bash
cd policies/Diffusion-Policy
bash train.sh ${robot} ${task_name} ${gpu_id}
# As example: bash train.sh airbot block_place 0
# As example: bash train.sh mmk2 mmk2_pick_kiwi 0
```
配置文件: policies/Diffusion-Policy/diffusion_policy/config

### Note
1. 建议在 MMK2 任务中使用 96×72 大小的图像，并在数据生成时采用该尺寸。MMK2 的 Diffusion Policy 配置文件中，图像大小默认为 96×72，可以根据需要进行调整。
2. 配置文件中的 checkpoint_note 用于在 ckpt 文件名后附加额外的信息。通过修改该变量，可以为不同的任务配置保存具有区分度的 ckpt 文件名。

## RDT

### GPU
训练至少需要25G内存（batch size = 4），
推理需要0.5G内存

### 环境
```bash
conda create -n rdt python=3.10.0
conda activate rdt
cd DISCOVERSE
pip install -r requirements.txt
pip install -e .
pip install torch==2.1.0 torchvision==0.16.0 packaging==24.0 ninja 
pip install flash-attn==2.7.2.post1 --no-build-isolation
# 如果安装flash-attn失败，可以从官方下载对应的.whl安装: https://github.com/Dao-AILab/flash-attention/releases
# 安装flash_attn-*.whl:
# pip install flash_attn-*.whl
cd DISCOVERSE/policies/RDT
pip install -r requirements.txt
pip install huggingface_hub==0.25.2
```
### 下载模型
```bash
cd DISCOVERSE/policies/RDT
mkdir -p weights/RDT && cd cd weights/RDT
huggingface-cli download google/t5-v1_1-xxl --local-dir t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir siglip-so400m-patch14-384
huggingface-cli download robotics-diffusion-transformer/rdt-1b --local-dir rdt-1b
```

### 生成language embeding
```bash
cd DISCOVERSE
python3 policies/RDT/scripts/encode_lang_batch_once.py ${task_name} ${gpu_id}
# for example:
python3 policies/RDT/scripts/encode_lang_batch_once.py block_place 0
```

### 配置文件
复制policies/RDT/model_config/model_name.yml，并重命名model_name

### 训练微调
```bash
cd DISCOVERSE/policies/RDT
python3 scripts/encode_lang_batch_once.py {task_name} {gpu_id}
# for example:
python3 scripts/encode_lang_batch_once.py block_place 0
```

### 推理
```bash
cd DISCOVERSE/policies/RDT
bash eval.sh {robot} {task_name} {model_name} {ckpt_id}
# for example:
bash eval.sh airbot block_place model_name 20000
```

## openpi

### 环境

```bash
conda create -n pi python=3.11.0
conda activate pi
cd DISCOVERSE
pip install -r requirements.txt
pip install -e .
cd policies/openpi/packages/openpi-client/
pip install -e .
cd ../..
pip install -e .
cd ../../submodules/lerobot
pip install -e .
```

### 配置文件

- 在`policies/openpi/src/openpi/training/config.py`中有一个名为`_CONFIGS`的字典。你可以修改预设的PI0配置项：
`pi0_base_aloha_robotwin_lora`
`pi0_fast_aloha_robotwin_lora`
`pi0_base_aloha_robotwin_full`
`pi0_fast_aloha_robotwin_full`

- 如果你的GPU显存不足，可以设置`fsdp_devices`，相关配置可参考`policies/openpi/src/openpi/training/config.py`。

- 当你需要更换机器人，或更改机器人的观测和动作时，可以修改`policies/openpi/src/openpi/training/config.py`中`_CONFIGS`下的`RepackTransform`。

### 设置缓存目录

如果你的 `~/.cache` 路径下磁盘空间不足，请使用以下命令将缓存目录设置为有足够空间的其他路径：
```bash
export HF_LEROBOT_HOME=/path/to/your/cache
# for example: 
mkdir -p ~/openpi_cache
export HF_LEROBOT_HOME=~/openpi_cache
```

### 处理数据

```bash
bash generate.sh ./training_data training_data
```
```bash
python3 scripts/compute_norm_stats.py --config-name ${train_config_name}
# for example:
python3 scripts/compute_norm_stats.py --config-name pi0_base_aloha_full
```

### 训练微调

```bash
export HF_LEROBOT_HOME=/path/to/your/cache
# for example: 
export HF_LEROBOT_HOME=~/openpi_cache
```
```bash
bash finetune.sh ${train_config_name} ${model_name}
# for example:
bash finetune.sh pi0_base_aloha_full model_a
```