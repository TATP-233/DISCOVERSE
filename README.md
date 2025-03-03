# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

https://github.com/user-attachments/assets/78893813-d3fd-48a1-8bb4-5b0d87bf900f

Yufei Jia†, Guangyu Wang†, Yuhang Dong, Junzhe Wu, Yupei Zeng, Haizhou Ge, Kairui Ding, Zike Yan, Weibin Gu, Chuxuan Li, Ziming Wang, Yunjie Cheng, Wei Sui, Ruqi Huang‡, Guyue Zhou‡

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FTATP-233%2FDISCOVERSE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23FF0000&title=Repo+Viewers&edge_flat=false)](https://hits.seeyoufarm.com)

[Webpage](https://air-discoverse.github.io/) | [PDF](https://drive.google.com/file/d/1637XPqWMajfC_ZqKfCGxDxzRMrsJQA1g/view?usp=drive_link)

## 🌟 Features

+ High-fidelity, hierarchical Real2Sim generation for both background node and interactive scene nodes in various complex real-world scenarios, leveraging advanced laser-scanning, generative models, physically-based re-lighting, and Mesh-Gaussian transfer.
+ Efficient simulation and user-friendly configuration. By seamlessly integrating 3DGS rendering engine, MuJoCo physical engine, and ROS2 robotic interface, we provide an easy-to-use, massively parallel implementation for rapid deployment and flexible extension. The overall throughput of DISCOVERSE can achieve 650 FPS for 5 cameras rendering RGB-D frames, which is ∼3× faster than ORBIT (Issac Lab).
+ Compatibilities with existing 3D assets and inclusive supports for robot models (robotic arm, mobile manipulator, quadrocopter, etc.), sensor modalities (RGB, depth, LiDAR), ROS plugins, and a variety of Sim&Real data mixing schemes. DISCOVERSE lays a solid foundation for developing a comprehensive set of Sim2Real robotic benchmarks for end-to-end robot learning, with real-world tasks including manipulation, navigation, multi-agent collaboration, etc., to stimulate further research and practical applications in the related fields.

## 🐳 Docker

Please refer to [docker deployment](doc/docker.md), or directly download [v1.6.1 docker images](https://pan.baidu.com/s/1mLC3Hz-m78Y6qFhurwb8VQ?pwd=xmp9). If docker is used, the `📦 Install` and `📷 Photorealistic/Preparation 1-3` parts can be skipped.

## 📦 Installation

```bash
git clone https://github.com/TATP-233/DISCOVERSE.git --recursive
cd DISCOVERSE
pip install -r requirements.txt
pip install -e .
```

### Download Resource Files

Download the `meshes` and `textures` folders from [Baidu Netdisk](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) or [Tsinghua Netdisk](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/) and place them under the `models` directory. After downloading the model files, the `models` directory will contain the following contents.

```
models
├── meshes
├── mjcf
├── textures
└── urdf
```

## 📷 Photorealistic Rendering

![photorealistic simulation](./assets/img2.png)

### Preparation

The physical engine of `DISCOVERSE` is [mujoco](https://github.com/google-deepmind/mujoco). If the user does not need the high-fidelity rendering function based on [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), this section can be skipped. If photorealistic rendering is required, please follow the instructions in this subsection.

1. Install CUDA. Please install the corresponding version of CUDA according to your graphics card model from the [download link](https://developer.nvidia.com/cuda-toolkit-archive).
2. pip install -r requirements_gs.txt
3. Install `diff-gaussian-rasterization`

    ```bash
    cd submodules/diff-gaussian-rasterization/
    git checkout 8829d14
    ```
    Modify line 154 of `submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h`,
    change `(p_view.z <= 0.2f)` to `(p_view.z <= 0.01f)`.

    ```bash
    cd ../..
    pip install submodules/diff-gaussian-rasterization
    ```

4. Prepare 3DGS model files. The high-fidelity visual effect of `DISCOVERSE` depends on 3DGS technology and corresponding model files. The pre-reconstructed robot, object, and scene models are placed on Baidu Netdisk [link](https://pan.baidu.com/s/1y4NdHDU7alCEmjC1ebtR8Q?pwd=bkca) and Tsinghua Netdisk [link](https://cloud.tsinghua.edu.cn/d/0b92cdaeb58e414d85cc/). After downloading the model files, the `models` directory will contain the following contents. (Note: Not all models are necessary. Users can download according to their own needs. It is recommended to download all ply models except those in the `scene` directory, and for the models in the `scene` folder, only download the ones that will be used.)

```
models
├── 3dgs
│   ├── airbot_play
│   ├── mmk2
│   ├── tok2
│   ├── skyrover
│   ├── hinge
│   ├── object
│   └── scene
├── meshes
├── mjcf
├── textures
└── urdf
```

### Online Viewing of 3DGS Models

If you want to view a single ply model, you can open [SuperSplat](https://playcanvas.com/supersplat/editor) in the browser, drag the ply model into the webpage, and you can view and perform simple editing. The webpage effect is as follows.

<img src="./assets/supersplat.png" alt="supersplat" style="zoom:50%;" />

## 🔨 Real2Sim

<img src="./assets/real2sim.jpg" alt="real2sim"/>

Please refer to our Real2Sim repository [DISCOVERSE-Real2Sim](https://github.com/GuangyuWang99/DISCOVERSE-Real2Sim) for this part of the content.

## 💡 Usage

+ airbot_play robotic arm

```shell
python3 discoverse/envs/airbot_play_base.py
```

+ Robotic arm desktop manipulation tasks

```shell
python3 discoverse/examples/tasks_airbot_play/block_place.py
python3 discoverse/examples/tasks_airbot_play/coffeecup_place.py
python3 discoverse/examples/tasks_airbot_play/cuplid_cover.py
python3 discoverse/examples/tasks_airbot_play/drawer_open.py
```

https://github.com/user-attachments/assets/6d80119a-31e1-4ddf-9af5-ee28e949ea81

There are many examples under the `discoverse/examples` path, including ros1, ros2, grpc, imitation learning, active mapping, etc.

+ Active SLAM

```shell
python3 discoverse/examples/active_slam/dummy_robot.py
```
<img src="./assets/active_slam.jpg" alt="active slam" style="zoom: 33%;" />

+ Collision Detection

```shell
python3 discoverse/examples/collision_detection/mmk2_collision_detection.ipynb
```

+ Vehicle and Drone Collaboration

```bash
python3 discoverse/examples/skyrover_on_rm2car/skyrover_and_rm2car.py
```

<img src="./assets/skyrover.png" alt="Drone and car" style="zoom: 50%;" />

### Imitation Learning Quick Start

We currently provide the entire process of data collection, model training, and inference of the act algorithm in the simulator. You can refer to [Data Collection and Format Conversion](./doc/data.md), [Training](./doc/training.md), [Inference](./doc/inference.md), and refer to the corresponding tutorials.

### Diverse images

https://github.com/user-attachments/assets/848db380-557c-469d-b274-2c9addf0b6bb

Thanks to the open-sourcing of the [lucidsim](https://github.com/lucidsim/lucidsim), we have incorporated the image randomization feature into the emulator. Please follow the content in [document](doc/Randomain.md) to implement this feature.

### Keyboard Operations

- Press 'h' to print help
- Press 'F5' to reload the mjcf file
- Press 'r' to reset the state
- Press '[' or ']' to switch camera view
- Press 'Esc' to set free camera
- Press 'p' to print the robot state
- Press 'g' to toggle Gaussian rendering
- Press 'd' to toggle depth rendering

## ⏩ Updates

+   2025.01.13: DISCOVERSE is open source
+   2025.01.16: add docker file
+   2025.01.14: The [S2R2025](sim2real.net/track/track?nav=S2R2025) competition is officially launched. We welcome everyone to sign up and participate, the [repository](https://github.com/DISCOVER-Robotics/SIM2REAL-2025.git).
+   2025.02.17: add dp baseline, thanks to [CleanDiffuserTeam](https://github.com/CleanDiffuserTeam/CleanDiffuser)
+   2025.02.19: add point cloud sensor

## ❔ Frequently Asked Questions

1. `diff-gaussian-rasterization` fails to install due to mismatched pytorch and cuda versions: Please install the specified version of pytorch.

2. If you want to use it on a server, please specify the environment variable:

    ```bash
    export MUJOCO_GL=egl
    ```

## 📬 Communication

You are welcome to add the author's contact information. Please add a note when adding.

<img src="./assets/wechat.jpeg" alt="wechat" style="zoom:50%;" />

## ⚖️ License

DISCOVERSE is licensed under the MIT License. See [LICENSE](https://github.com/TATP-233/DISCOVERSE/blob/main/LICENSE) for additional details.

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@misc{discoverse2024,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haizhou Ge and Kairui Ding and Zike Yan and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      url={https://air-discoverse.github.io/},
      year={2024}
    }
```
