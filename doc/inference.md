## 命令

```bash
python3 policies/infer.py <policy> [args]
```

解释：
- `policy`: 位置参数，指定策略的类型，目前支持的选项：act
- [args]: 不同的策略有不同的命令行参数，请参考下面对应策略的说明

## act

### 依赖安装

```bash
pip install -r policies/act/requirements/train_eval.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 推理配置
推理配置文件可基于训练配置文件修改，其中主要参数解释如下：
- `max_timesteps`: 动作执行总步数，动作达到总步数后自动结束本次推理

### 推理命令

```bash
python3 policies/infer.py act -tn <task_name> -mts 100 -ts 20241125-110709 -rn discoverse/examples/<tasks_folder>/<task_script>
```

其中：
- `-tn` 任务名，程序会根据任务名分别在`task_configs`和`data`目录下寻找同名的配置文件和数据集
- `-mts` 动作执行总步数，该命令行参数会覆盖配置文件中的`max_timesteps`
- `-ts` 时间戳，对应训练得到的模型文件所在的以时间戳命名的文件夹，程序会根据任务名和时间戳在policies/act/my_ckpt目录下寻找对应的模型文件
- `-rn` 数据采集时使用的脚本文件路径，例如`discoverse/examples/tasks_airbot_play/drawer_open.py`，程序会加载其中的`SimNode`类和`AirbotPlayCfg`的实例`cfg`来创建仿真环境

## dp

### 推理配置

推理配置文件与训练配置文件相同

### 推理命令

```bash
python3 policies/infer.py dp --config-path=configs --config-name=block_place mode=eval model_path=path/to/model
```

其中:
- `--config-path`: 配置文件所在路径
- `--config-name`: 配置文件名
- `mode`: 指定训练或是推理
- `model_path`: 模型权重路径

### 真机推理

```bash
python3 policies/dp/infer_real.py --config-path=configs --config-name=block_place
```
其中:
- `--config-path`: 配置文件所在路径
- `--config-name`: 配置文件名
- 需要注意，真机推理的`config.yaml`相较于`sim`中的`config.yaml`，需要增加`global_camid`和`wrist_camid`，分别指向对应的相机编号