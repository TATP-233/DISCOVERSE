### Domain Randomization

* 该部分代码位于`discoverse/randomain`下
* 采样及生成结果均位于`data/randomain`下
* 实现对样本视频每一帧的场景与细节的随机化



### 生成模型

* **依赖安装**

```
cd submodules
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install -r requirements.txt
```

* **模型部署**
- `checkpoints`:[sd_xl_turbo_1.0_fp16](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/sd_xl_turbo_1.0_fp16.safetensors)

- `controlnet`:[controlnet_depth_sdxl_1.0](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/blob/main/diffusion_pytorch_model.safetensors)

- `vae`:[sdxl_vae](https://huggingface.co/stabilityai/sdxl-vae/blob/main/diffusion_pytorch_model.safetensors)

分别部署于本目录下`models`下的同名文件夹中，并在`models/extra_model_paths.yaml`中添加模型所在目录路径，修改后如下：

```bash
randoma
├── models
│   ├── checkpoints
│   │   └── sd_xl_turbo_1.0_fp16.safetensors
│   ├── controlnet
│   │   └── controlnet_depth_sdxl_1.0.safetensors
│   ├── extra_model_paths.yaml
│   └── vae
│       └── sdxl_vae.safetensors
```



* **路径链接**

```
# 在conda环境的.env配置中添加ComfyUI，以及指向模型位置的extra_model_paths.yaml
export PYTHONPATH=/path/to/ComfyUI:$PYTHONPATH
export COMFYUI_CONFIG_PATH=/path/to/extra_model_paths.yaml
```



### 光流估计

目前支持`Farneback方法`与`RAFT方法`

* 若选用`RAFT方法`，可下载预训练权重[raft](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)于`models\flow\`

* 若希望使用其他方法，可仿照我们对`FlowCompute\raft`的处理形式进行迁移



### 使用流程

* **样本采集**

在轨迹运行过程中，采集彩色图像、深度图像，以及各个对象和背景的掩码图像，保存格式为`.mp4`

1. 实例化采样对象

```python
from discoverse.randomain.utils import SampleforDR
samples = SampleforDR(objs=objs, robot_parts=robot_parts,
                      cam_ids=cfg.obs_rgb_cam_id, save_dir=save_dir,
                      fps=cfg.render_set['fps'], max_vis_dis=max_vis_dis)
# objs 操作对象 e.g.['block_green', 'bowl_pink']
# robot_parts 机器人部件名称 e.g. cfg.rb_link_list
# max_vis_dis 采样depth时的最大可视距离(m)，默认值为1，影响depth保存时的归一化
```

2. 在线采样

```python
# 添加到轨迹运行过程中
samples.sampling(sim_node)
```

3. 采样结束，保存

```python
# 得到采样结果 [cam.mp4 depth.mp4 obj1.mp4 ... objn.mp4 background.mp4]
samples.save()
```

* **提示词生成**

在`augment.py`中，支持：

1. 根据预输入批量生成提示词

```python
# e.g. for task of block_place
mode = 'input'
fore_objs = {
        "block_green": "A green block",
        "bowl_pink": "A pink bowl",
        "robot": "A black robot arm"
    }
background = 'A table'
scene = 'In one room, a robotic arm is doing the task of clipping a block into a bowl'
negative = "No extra objects in the scene"
num_prompts = 50
```

2. 基于`example.jsonl`中提示词增广

```python
# e.g. for task of block_place
mode = 'example'
input_path = 'path_to_example'
num_prompts = 50
```

* **随机化生成**

```python
cd discoverse/randomain
python generate.py  [--arg]
```

下面是一些需要针对特定需求更改的参数，完整参数见`generate.py`

`task_name`:任务名称

`work_dir`:格式类似于`000`，为采样的某一个轨迹的保存路径序号

`cam_id`:相机编号

`fore_objs`:除`background`外的所有物体名称，e.g.`['block_green', 'bowl_pink', 'robot']`

`wide`、`height`:输入输出的共同宽高，推荐`1280*768`

`num_steps`:生成步数

`flow_interval`:生成图像的帧间隔，中间的帧用光流方法计算

`flow_method`:`rgb`-Farneback方法；`raft`-RAFT方法；对应方法部署后可自行拓展
