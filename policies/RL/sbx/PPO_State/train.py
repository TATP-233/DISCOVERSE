import os
import argparse
import numpy as np
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from env import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, BaseCallback
from sbx import PPO
from tqdm import tqdm


def make_env(render=True):
    """创建环境的工厂函数"""

    def _init():
        try:
            env = Env(render=render)
            return env
        except Exception as e:
            print(f"环境创建失败: {str(e)}")
            raise e

    return _init


def train(render=True):
    # 设置随机种子，保证结果可复现
    np.random.seed(42)

    try:
        # 创建环境
        env = make_env(render=render)()
        # 添加Monitor包装器来记录训练数据
        log_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/PPO/logs")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        # 使用DummyVecEnv包装单个环境
        env = DummyVecEnv([lambda: env])

        # 创建评估环境
        eval_env = make_env(render=render)()
        eval_env = Monitor(eval_env, log_dir)
        eval_env = DummyVecEnv([lambda: eval_env])

        # 创建评估回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=log_dir,
            eval_freq=10000,  # 每10000时间步评估一次
            deterministic=True,
            render=render
        )

        # 创建PPO模型
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,  # 每次更新所收集的轨迹长度
            batch_size=2,  # 批次大小
            n_epochs=10,  # 每次更新迭代次数
            gamma=0.99,  # 折扣因子
            learning_rate=3e-4,  # 学习率
            clip_range=0.2,  # PPO策略裁剪范围
            ent_coef=0.01,  # 熵正则化系数
            tensorboard_log=log_dir,
            verbose=1  # 输出详细程度
        )

        # 训练模型
        total_timesteps = 1000000
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=10,  # 每10次更新后记录日志
        )

        # 保存最终模型
        save_path = os.path.join(log_dir, "final_model")
        model.save(save_path)
        print(f"模型已保存到: {save_path}")

    except Exception as e:
        print(f"训练过程发生错误: {str(e)}")
        raise e
    finally:
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()


def test(model_path):
    # 创建测试环境
    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = False  # 关闭高斯渲染器
    cfg.init_key = "pick"  # 初始化模式
    cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义"白色盘子"模型路径
    cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义"奇异果"模型路径
    cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景模型路径
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo环境文件路径
    cfg.obj_list = ["plate_white", "kiwi"]  # 环境中包含的对象列表
    cfg.sync = True  # 是否同步更新
    cfg.headless = False  # 是否启用无头模式（显示渲染画面）

    # 创建环境
    task_base = MMK2TaskBase(cfg)
    env = Env(task_base=task_base, render=True)

    # 加载模型
    model = PPO.load(model_path)

    # 测试循环
    obs, info = env.reset()  # 重置环境，获取初始观察值
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)  # 预测动作
        obs, reward, terminated, truncated, info = env.step(action)  # 执行动作，获取反馈
        # 如果环境结束或截断，重新初始化
        if terminated or truncated:
            obs, info = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--model_path", type=str, help="模型路径，用于测试模式")
    parser.add_argument("--render", action="store_true", help="在训练过程中显示渲染画面")
    args = parser.parse_args()

    if args.test:
        if not args.model_path:
            print("测试模式需要指定模型路径 --model_path")
        else:
            test(args.model_path)
    else:
        train(render=args.render)
