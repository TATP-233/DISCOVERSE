import os
import argparse
import numpy as np
from tqdm import tqdm
from discoverse.robots_env.mmk2_base import MMK2Cfg
from discoverse.task_base import MMK2TaskBase
from env import Env
from sbx import PPO

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
    parser = argparse.ArgumentParser(description="PPO强化学习推理脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--render", action="store_true", default=True, help="在测试过程中显示渲染画面 (默认: True)")
    parser.add_argument("--episodes", type=int, default=10000, help="测试回合数")
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
