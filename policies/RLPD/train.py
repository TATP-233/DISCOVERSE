import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import Env, collect_demonstrations
from rlpd import RLPD
from discoverse import DISCOVERSE_ROOT_DIR


def train(args):
    """
    训练RLPD算法
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建保存目录
    save_dir = os.path.join(DISCOVERSE_ROOT_DIR, "policies/RLPD/results")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 创建环境
        print("正在创建环境...")
        env = Env(render=args.render)
        env = RecordEpisodeStatistics(env)  # 记录统计信息
        
        # 打印环境信息以进行调试
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 获取环境信息
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
        
        # 测试环境重置
        print("测试环境重置...")
        state, info = env.reset()
        print(f"初始状态形状: {state.shape}")
        
        # 收集或加载离线演示数据
        if args.demo_path and os.path.exists(args.demo_path):
            # 加载已有的演示数据
            print(f"正在加载演示数据: {args.demo_path}")
            demonstrations = np.load(args.demo_path, allow_pickle=True).item()
        else:
            # 收集新的演示数据
            print("正在收集演示数据...")
            demonstrations = collect_demonstrations(env, num_episodes=args.num_demos)
            if args.demo_path:
                # 保存演示数据
                np.save(args.demo_path, demonstrations)
                print(f"演示数据已保存到: {args.demo_path}")
                
        # 打印演示数据的统计信息
        for key, value in demonstrations.items():
            print(f"{key} 形状: {value.shape}, 类型: {value.dtype}")
    
    except Exception as e:
        print(f"初始化环境或收集演示数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # 初始化RLPD算法
    agent = RLPD(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        offline_demonstrations=demonstrations,
        hidden_dims=(256, 256),
        discount=args.discount,
        tau=args.tau,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        alpha_lr=args.alpha_lr,
        num_critics=args.num_critics,
        critic_ensemble_size=args.critic_ensemble_size,
        batch_size=args.batch_size,
        utd_ratio=args.utd_ratio
    )
    
    # 训练循环
    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 添加到经验回放缓冲区
            agent.add_to_buffer(state, action, reward, next_state, float(done))
            
            # 训练代理
            agent.train(env_steps=1)
            
            # 更新状态
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # 渲染环境
            if args.render:
                env.render()
        
        # 记录统计信息
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 打印进度
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_length = np.mean(episode_lengths[-args.log_interval:])
            print(f"Episode {episode+1}/{args.num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f}")
        
        # 保存模型
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(save_dir, f"model_ep{episode+1}"))
    
    # 保存最终模型
    agent.save(os.path.join(save_dir, "model_final"))
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    plt.subplot(1, 2, 2)
    window_size = min(10, len(episode_rewards))
    avg_rewards = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(len(avg_rewards)), avg_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.title(f'Average Rewards (Window Size: {window_size})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    
    return agent


def evaluate(args):
    """
    评估训练好的RLPD模型
    
    Args:
        args: 命令行参数
    """
    # 创建环境
    env = Env(render=True)
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 初始化代理
    agent = RLPD(
        state_dim=state_dim,
        action_dim=action_dim,
        action_space=env.action_space,
        offline_demonstrations=None
    )
    
    # 加载模型
    agent.load(args.model_path)
    
    # 评估循环
    total_rewards = []
    
    for episode in range(args.eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # 选择动作（评估模式）
            action = agent.select_action(state, evaluate=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 渲染环境
            env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}/{args.eval_episodes} | Reward: {episode_reward:.2f}")
    
    # 打印评估结果
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"Evaluation over {args.eval_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLPD算法训练与评估")
    
    # 通用参数
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--render", action="store_true", help="是否渲染环境")
    
    # 训练参数
    parser.add_argument("--num_episodes", type=int, default=1000, help="训练的总回合数")
    parser.add_argument("--discount", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--tau", type=float, default=0.005, help="软更新系数")
    parser.add_argument("--policy_lr", type=float, default=3e-4, help="策略网络学习率")
    parser.add_argument("--q_lr", type=float, default=3e-4, help="Q网络学习率")
    parser.add_argument("--alpha_lr", type=float, default=3e-4, help="熵系数学习率")
    parser.add_argument("--batch_size", type=int, default=256, help="批次大小")
    parser.add_argument("--utd_ratio", type=int, default=20, help="更新频率比例")
    parser.add_argument("--num_critics", type=int, default=2, help="Q网络数量")
    parser.add_argument("--critic_ensemble_size", type=int, default=2, help="用于计算目标值的Q网络数量")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    
    # 演示数据参数
    parser.add_argument("--num_demos", type=int, default=10, help="收集的演示数据数量")
    parser.add_argument("--demo_path", type=str, default="demos.npy", help="演示数据保存/加载路径")
    
    # 评估参数
    parser.add_argument("--eval", action="store_true", help="评估模式")
    parser.add_argument("--model_path", type=str, help="要评估的模型路径")
    parser.add_argument("--eval_episodes", type=int, default=10, help="评估的回合数")
    
    args = parser.parse_args()
    
    if args.eval:
        if not args.model_path:
            print("评估模式需要指定模型路径 --model_path")
        else:
            evaluate(args)
    else:
        train(args)
