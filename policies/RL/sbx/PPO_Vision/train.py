import os
import argparse
import numpy as np
import torch
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.envs.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from env import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sbx import PPO
from tqdm import tqdm

# 自定义特征提取器，用于处理图像输入
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
        
    def forward(self, observations):
        return self.linear(self.cnn(observations))
    
    @property
    def features_dim(self):
        return self._features_dim


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
    torch.manual_seed(42)

    try:
        print("开始创建环境...")
        env = make_env(render=render)()
        print("环境创建完成")
        # 添加Monitor包装器来记录训练数据
        log_dir = os.path.join(DISCOVERSE_ROOT_DIR, "data/PPO_Vision/logs")
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        print("Monitor包装器添加完成")
        # 使用DummyVecEnv包装单个环境
        env = DummyVecEnv([lambda: env])
        print("DummyVecEnv包装完成")

        # 创建评估环境
        print("开始创建评估环境...")
        eval_env = make_env(render=render)()
        print("评估环境创建完成")
        eval_env = Monitor(eval_env, log_dir)
        print("评估环境Monitor包装器添加完成")
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
        print("评估回调创建完成")

        # 自定义进度条回调
        class TqdmCallback(BaseCallback):
            def __init__(self, total_timesteps, verbose=0):
                super(TqdmCallback, self).__init__(verbose)
                self.pbar = None
                self.total_timesteps = total_timesteps
                
            def _on_training_start(self):
                self.pbar = tqdm(total=self.total_timesteps, desc="训练进度")

            def _on_step(self):
                self.pbar.update(1)
                return True
                
            def _on_training_end(self):
                self.pbar.close()
                self.pbar = None

        # 创建特征提取器的策略关键字参数
        policy_kwargs = {
            "features_extractor_class": CNNFeatureExtractor,
            "features_extractor_kwargs": {"observation_space": env.observation_space}
        }
        print("特征提取器参数设置完成")

        # 创建PPO模型
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,  # 使用自定义特征提取器
            n_steps=2048,  # 每次更新所收集的轨迹长度
            batch_size=64,  # 批次大小
            n_epochs=10,  # 每次更新迭代次数
            gamma=0.99,  # 折扣因子
            learning_rate=3e-4,  # 学习率
            clip_range=0.2,  # PPO策略裁剪范围
            ent_coef=0.01,  # 熵正则化系数
            tensorboard_log=log_dir,
            verbose=1  # 输出详细程度
        )
        
        print("PPO模型创建完成，开始收集经验...")

        # 训练模型
        total_timesteps = 1000000
        print("开始训练模型，总时间步数:", total_timesteps)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, TqdmCallback(total_timesteps=total_timesteps)],
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
