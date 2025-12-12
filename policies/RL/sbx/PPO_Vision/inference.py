import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import gymnasium as gym
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from env import Env
from sbx import PPO
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
        cfg.gs_model_dict["plate_white"] = "object/plate_white.ply"  # 定义"白色盘子"模型路径
        cfg.gs_model_dict["kiwi"] = "object/kiwi.ply"  # 定义"奇异果"模型路径
        cfg.gs_model_dict["background"] = "scene/tsimf_library_1/point_cloud.ply"  # 定义背景模型路径
        cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  # MuJoCo环境文件路径
        cfg.sync = True  # 是否同步更新
        cfg.headless = not render  # 是否启用无头模式（显示渲染画面）
        cfg.obs_rgb_cam_id = [0]  # 使用第一个摄像头

        # 创建环境
        task_base = MMK2TaskBase(cfg)
        env = Env(task_base=task_base, render=render)

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
    parser = argparse.ArgumentParser(description="基于视觉的PPO强化学习推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--render", action="store_true", default=True, help="在测试过程中显示渲染画面 (默认: True)")
    parser.add_argument("--episodes", type=int, default=10, help="测试回合数")
    parser.add_argument("--deterministic", action="store_true", help="使用确定性策略进行测试")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    test(
        model_path=args.model_path,
        render=args.render,
        episodes=args.episodes,
        deterministic=args.deterministic,
        seed=args.seed
    )
