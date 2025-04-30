import os
import argparse
import numpy as np
import time
from datetime import datetime
import torch
import gymnasium as gym
from discoverse import DISCOVERSE_ROOT_DIR
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from env import Env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from sbx import PPO
from tqdm import tqdm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.frame_stack_size = observation_space.shape[3]  # 应该是4
        n_input_channels = observation_space.shape[0] * self.frame_stack_size  # 3 * 4 = 12
        self.cnn = torch.nn.Sequential(
            # 第一层卷积，处理原始尺寸的输入
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            # 添加自适应平均池化层，将特征图调整为固定大小 (7x7)
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten()
        )
        
        # 计算CNN输出特征的维度
        with torch.no_grad():
            # 创建一个示例输入，形状为(1, 3, 84, 84, 4)
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            # 重塑为(1, 12, 84, 84)
            sample_input_reshaped = self._reshape_input(sample_input)
            n_flatten = self.cnn(sample_input_reshaped).shape[1]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU()
        )
    
    def _reshape_input(self, observations):
        # 输入形状: (batch_size, channels, height, width, stack_size)
        batch_size = observations.shape[0]
        channels = observations.shape[1]
        height = observations.shape[2]
        width = observations.shape[3]
        stack_size = observations.shape[4]
        
        # 重塑为 (batch_size, channels*stack_size, height, width)
        return observations.permute(0, 1, 4, 2, 3).reshape(batch_size, channels*stack_size, height, width)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 重塑输入以适应CNN
        reshaped_obs = self._reshape_input(observations)
        return self.linear(self.cnn(reshaped_obs))

def make_env(render=True, seed=0):
    """创建环境的工厂函数
    
    Args:
        render (bool): 是否渲染环境
        seed (int): 随机种子
        
    Returns:
        callable: 创建环境的函数
    """

    def _init():
        try:
            env = Env(render=render)
            return env
        except Exception as e:
            print(f"环境创建失败: {str(e)}")
            raise e

    return _init


def train(render=True, seed=42, total_timesteps=1000000, batch_size=64, n_steps=2048, 
learning_rate=3e-4, log_dir=None, model_path=None, eval_freq=10000, log_interval=10):
    """训练PPO模型
    
    Args:
        render (bool): 是否渲染环境
        seed (int): 随机种子
        total_timesteps (int): 总训练步数
        batch_size (int): 批次大小
        n_steps (int): 每次更新所收集的轨迹长度
        learning_rate (float): 学习率
        log_dir (str): 日志目录，如果为None则使用默认目录
        model_path (str): 预训练模型路径，用于继续训练
        eval_freq (int): 评估频率
        log_interval (int): 日志记录间隔
    """

    # 设置日志目录
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(DISCOVERSE_ROOT_DIR, f"data/PPO_Vision/logs_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    print(f"日志目录: {log_dir}")

    try:
        print("开始创建环境...")
        # 创建环境
        env = make_env(render=render, seed=seed)()
        # 添加Monitor包装器来记录训练数据
        env = Monitor(env, log_dir)
        print("Monitor包装器添加完成")
        # 使用DummyVecEnv包装单个环境
        env = DummyVecEnv([lambda: env])
        print("DummyVecEnv包装完成")

        # 创建评估环境
        print("开始创建评估环境...")
        eval_env = make_env(render=render, seed=seed+1)()
        eval_env = Monitor(eval_env, log_dir)
        print("评估环境Monitor包装器添加完成")
        eval_env = DummyVecEnv([lambda: eval_env])
        print("评估环境创建完成")

        # 创建评估回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, "best_model"),
            log_path=log_dir,
            eval_freq=eval_freq,  # 每eval_freq时间步评估一次
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
                self.start_time = None
                
            def _on_training_start(self):
                self.start_time = time.time()
                self.pbar = tqdm(total=self.total_timesteps, desc="训练进度")

            def _on_step(self):
                self.pbar.update(1)
                # 更新进度条描述，显示已用时间
                elapsed_time = time.time() - self.start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                self.pbar.set_description(
                    f"训练进度 [{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}]"
                )
                return True
                
            def _on_training_end(self):
                self.pbar.close()
                self.pbar = None

        # 创建PPO模型或加载预训练模型
        if model_path is not None and os.path.exists(model_path):
            print(f"加载预训练模型: {model_path}")
            model = PPO.load(model_path, env=env)
            # 更新学习率
            model.learning_rate = learning_rate
            print("预训练模型加载完成")
        else:
            print("创建新的PPO模型")
            # 自定义特征提取器
            policy_kwargs = {
                "features_extractor_class": CNNFeatureExtractor,
                "features_extractor_kwargs": {"features_dim": 256}
            }
            
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=n_steps,  # 每次更新所收集的轨迹长度
                batch_size=batch_size,  # 批次大小
                n_epochs=10,  # 每次更新迭代次数
                gamma=0.99,  # 折扣因子
                learning_rate=learning_rate,  # 学习率
                clip_range=0.2,  # PPO策略裁剪范围
                ent_coef=0.01,  # 熵正则化系数
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs,  # 传递自定义特征提取器
                verbose=1  # 输出详细程度
            )
        
        print("PPO模型创建完成，开始收集经验...")

        # 训练模型
        print(f"开始训练模型，总时间步数: {total_timesteps}")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, TqdmCallback(total_timesteps=total_timesteps)],
            log_interval=log_interval,  # 每log_interval次更新后记录日志
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


def test(model_path, render=True, episodes=10, deterministic=True, seed=42):
    """测试训练好的模型
    
    Args:
        model_path (str): 模型路径
        render (bool): 是否渲染环境
        episodes (int): 测试回合数
        deterministic (bool): 是否使用确定性策略
        seed (int): 随机种子
    """
    
    print(f"加载模型: {model_path}")
    
    try:
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
        cfg.headless = not render  # 是否启用无头模式（显示渲染画面）
        cfg.obs_rgb_cam_id = [0]  # 使用第一个摄像头

        # 创建环境
        task_base = MMK2TaskBase(cfg)
        env = Env(task_base=task_base, render=render)
        env.seed(seed)

        # 加载模型
        model = PPO.load(model_path)
        print("模型加载完成，开始测试...")

        # 测试循环
        total_rewards = []
        
        for episode in tqdm(range(episodes), desc="测试进度"):
            episode_reward = 0
            obs, info = env.reset()  # 重置环境，获取初始观察值
            done = False
            step_count = 0
            
            while not done and step_count < 1000:
                action, _states = model.predict(obs, deterministic=deterministic)  # 预测动作
                obs, reward, terminated, truncated, info = env.step(action)  # 执行动作，获取反馈
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
            print(f"回合 {episode+1}/{episodes} 完成，奖励: {episode_reward:.2f}")
        
        # 输出测试结果统计
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print(f"\n测试完成! {episodes}个回合的平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        
    except Exception as e:
        print(f"测试过程发生错误: {str(e)}")
        raise e
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于视觉的PPO强化学习训练与测试脚本")
    # 通用参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--render", action="store_true", default=False, help="在训练/测试过程中显示渲染画面 (默认: False)")
    
    # 模式选择
    parser.add_argument("--test", action="store_true", help="测试模式")
    
    # 训练参数
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="总训练步数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--n_steps", type=int, default=2048, help="每次更新所收集的轨迹长度")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--log_dir", type=str, default=None, help="日志目录")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径，用于继续训练或测试")
    parser.add_argument("--eval_freq", type=int, default=10000, help="评估频率")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    
    # 测试参数
    parser.add_argument("--episodes", type=int, default=10, help="测试回合数")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略进行测试")
    
    args = parser.parse_args()

    if args.test:
        if not args.model_path:
            print("测试模式需要指定模型路径 --model_path")
        else:
            test(
                model_path=args.model_path,
                render=args.render,
                episodes=args.episodes,
                deterministic=args.deterministic,
                seed=args.seed
            )
    else:
        train(
            render=args.render,
            seed=args.seed,
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            learning_rate=args.learning_rate,
            log_dir=args.log_dir,
            model_path=args.model_path,
            eval_freq=args.eval_freq,
            log_interval=args.log_interval
        )
