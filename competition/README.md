# DG-202612 文旅机器人竞赛代码

本目录是“DG-202612 面向物品识别与搬运的文旅机器人关键技术研究”的团队开发区，基于 DISCOVERSE 平台实现比赛要求的移动双臂机器人任务。

## 比赛目标

机器人在虚拟景区场景中接收自然语言指令，完成物品识别、抓取、安全搬运和指定位置放置。

- 任务一：从货架二、三、四层中找到长方体包装盒，并搬运至桌面。
- 任务二：识别指定颜色的包装盒，将其放到指定参照物的机器人视角左侧或右侧。

最终交付为 Docker 镜像、`client_task_1.py`、`client_task_2.py`、部署说明及作品介绍材料。

## 快速开始

完整环境说明见 [environment.md](environment.md)。首次克隆建议使用：

```bash
git clone --recursive https://github.com/lwhhhh13/tourism-robot2026.git
cd tourism-robot2026
conda env create -f competition/environment.yml
conda activate discoverse
python -m pip install --no-build-isolation -e .
```

基础平台验证：

```bash
python discoverse/robots_env/mmk2_base.py
python examples/tasks_mmk2/box_pick.py
python examples/tasks_mmk2/kiwi_place.py
```

## 目录职责

```text
competition/
├── client_task_1.py       # 任务一最终比赛入口（当前为抓取原型）
├── client_task_2.py       # 任务二最终比赛入口
├── environment.yml        # Conda/Pip 依赖快照
├── environment.md         # 环境复现、验证与排错说明
├── src/
│   ├── common/            # 共享数据结构与平台接口适配
│   ├── perception/        # RGB-D、颜色/道具识别、2D→3D 定位
│   ├── manipulation/      # IK、抓取、放置与安全姿态
│   ├── navigation/        # 底盘运动与目标点停靠
│   └── planning/          # 指令解析、左右关系与高层状态机
├── config/                # 可调参数：相机、抓取、导航、规划
├── tests/                 # 单元测试与批量回归测试
├── scripts/               # 录制、评测、数据检查工具
└── recordings/            # 本地录制数据（不提交 Git）
```

## 当前开发原则

1. 先以 MuJoCo 真值完成稳定的“抓取—搬运—放置—结束”动作闭环。
2. 再用 RGB-D 感知替换真值定位；最终代码不得依赖正式比赛环境未承诺提供的物体真值。
3. 通过 `platform_adapter` 隔离 DISCOVERSE 本地接口与后续比赛 Server/Client 接口差异。
4. 机械臂动作时底盘保持静止；底盘移动时机械臂保持安全收回姿态，优先避免碰撞。

## 协作约定

- 日常开发从 `main` 新建个人功能分支，通过 Pull Request 合并；不要直接修改他人的功能分支。
- `origin` 是团队 GitHub 仓库；`upstream` 是 DISCOVERSE 官方仓库。日常代码推送仅使用 `origin`。
- 不提交模型权重、视频、录制数据、IDE 配置、Token 或密钥。
- 修改依赖、接口或公共数据结构时，须同步更新本 README 和 `environment.md`。

## 当前状态

- DISCOVERSE、MMK2 基础仿真、抓取示例和放置示例均已在 Linux 开发机上验证运行。
- 任务一将以 `examples/tasks_mmk2/box_pick.py` 的双臂抓取状态机为原型。
- 任务二将复用抓取模块，并增加颜色识别、目标参照物定位和机器人视角左右关系推理。
