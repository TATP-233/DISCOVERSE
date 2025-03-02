import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
import gymnasium as gym


# 使用LayerNorm的MLP网络
class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super(MLPWithLayerNorm, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 使用LayerNorm防止过度外推
            layers.append(activation())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


# 策略网络（Actor）
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), activation=nn.ReLU,
                 log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.trunk = MLPWithLayerNorm(state_dim, 2 * action_dim, hidden_dims, activation)
        
    def forward(self, state):
        mu_log_std = self.trunk(state)
        mu, log_std = mu_log_std.chunk(2, dim=-1)
        
        # 限制log_std的范围，防止数值不稳定
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        
        return mu, std
        
    def sample(self, state):
        mu, std = self.forward(state)
        normal = Normal(mu, std)
        
        # 使用重参数化技巧采样
        x = normal.rsample()
        
        # 计算tanh变换后的动作和对应的log概率
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, mu, std


# 价值网络（Critic）
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super(QNetwork, self).__init__()
        
        self.q1 = MLPWithLayerNorm(state_dim + action_dim, 1, hidden_dims, activation)
        self.q2 = MLPWithLayerNorm(state_dim + action_dim, 1, hidden_dims, activation)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
    
    def load_demonstrations(self, demonstrations):
        """加载预先收集的演示数据"""
        n = len(demonstrations['states'])
        for i in range(n):
            self.add(
                demonstrations['states'][i],
                demonstrations['actions'][i],
                demonstrations['rewards'][i],
                demonstrations['next_states'][i],
                demonstrations['dones'][i]
            )
        print(f"已加载 {n} 个演示样本到离线缓冲区")


# RLPD算法实现
class RLPD:
    def __init__(
        self,
        state_dim,
        action_dim,
        action_space,
        offline_demonstrations=None,
        hidden_dims=(256, 256),
        discount=0.99,
        tau=0.005,
        policy_lr=3e-4,
        q_lr=3e-4,
        alpha_lr=3e-4,
        target_entropy=None,
        num_critics=2,
        critic_ensemble_size=2,
        batch_size=256,
        utd_ratio=20,  # 更新频率比例
        device=None
    ):
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        self.utd_ratio = utd_ratio
        self.action_space = action_space
        
        # 设备配置
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化策略网络
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        
        # 初始化多个Q网络（集成）
        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []
        
        for _ in range(num_critics):
            critic = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.critics.append(critic)
            self.critic_targets.append(deepcopy(critic))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=q_lr))
        
        # 设置目标熵
        if target_entropy is None:
            self.target_entropy = -action_dim  # 默认为动作维度的负值
        else:
            self.target_entropy = target_entropy
            
        # 初始化熵系数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp()
        
        # 初始化经验回放缓冲区
        self.online_buffer = ReplayBuffer(state_dim, action_dim)
        self.offline_buffer = ReplayBuffer(state_dim, action_dim)
        
        # 加载离线演示数据
        if offline_demonstrations is not None:
            self.offline_buffer.load_demonstrations(offline_demonstrations)
            
        # 集成参数
        self.critic_ensemble_size = critic_ensemble_size
        
        # 训练步数计数
        self.train_steps = 0
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if evaluate:
                # 评估模式下使用均值作为动作
                mu, _ = self.policy(state)
                return torch.tanh(mu).cpu().numpy().flatten()
            else:
                # 训练模式下使用采样动作
                action, _, _, _ = self.policy.sample(state)
                return action.cpu().numpy().flatten()
    
    def train(self, env_steps=1):
        """
        训练RLPD算法
        
        Args:
            env_steps: 环境交互步数
        """
        # 对每个环境步骤进行多次更新（UTD比例）
        for _ in range(self.utd_ratio * env_steps):
            self._update_networks()
            self.train_steps += 1
    
    def _update_networks(self):
        # 对称采样：从在线和离线缓冲区各采样一半数据
        online_batch_size = min(self.batch_size // 2, self.online_buffer.size)
        offline_batch_size = self.batch_size - online_batch_size
        
        # 如果在线数据不足，则全部使用离线数据
        if online_batch_size == 0:
            state, action, reward, next_state, done = self.offline_buffer.sample(self.batch_size)
        else:
            # 分别从在线和离线缓冲区采样
            online_state, online_action, online_reward, online_next_state, online_done = \
                self.online_buffer.sample(online_batch_size)
            
            offline_state, offline_action, offline_reward, offline_next_state, offline_done = \
                self.offline_buffer.sample(offline_batch_size)
            
            # 合并样本
            state = torch.cat([online_state, offline_state], dim=0)
            action = torch.cat([online_action, offline_action], dim=0)
            reward = torch.cat([online_reward, offline_reward], dim=0)
            next_state = torch.cat([online_next_state, offline_next_state], dim=0)
            done = torch.cat([online_done, offline_done], dim=0)
        
        # 更新Q网络
        with torch.no_grad():
            # 从下一状态采样动作
            next_action, next_log_prob, _, _ = self.policy.sample(next_state)
            
            # 计算目标Q值
            next_q_values = []
            for critic_target in self.critic_targets:
                q1, q2 = critic_target(next_state, next_action)
                next_q_values.append(torch.min(q1, q2))
            
            # 从集成中选择子集计算目标值
            if self.critic_ensemble_size < len(next_q_values):
                indices = torch.randperm(len(next_q_values))[:self.critic_ensemble_size]
                next_q_values = [next_q_values[i] for i in indices]
            
            # 取最小值作为目标Q值
            next_q_value = torch.min(torch.cat(next_q_values, dim=1), dim=1, keepdim=True)[0]
            
            # 添加熵项
            next_q_value = next_q_value - self.alpha * next_log_prob
            
            # 计算目标值
            target_q_value = reward + (1 - done) * self.discount * next_q_value
        
        # 更新所有Q网络
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            # 当前Q值
            current_q1, current_q2 = critic(state, action)
            
            # 计算损失
            critic_loss = F.mse_loss(current_q1, target_q_value) + F.mse_loss(current_q2, target_q_value)
            
            # 反向传播和优化
            optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
        
        # 更新策略网络
        new_action, log_prob, _, _ = self.policy.sample(state)
        
        # 计算所有Q网络的Q值
        q_values = []
        for critic in self.critics:
            q1 = critic.q1_forward(state, new_action)
            q_values.append(q1)
        
        # 取平均值作为策略梯度的基础
        q_value = torch.mean(torch.cat(q_values, dim=1), dim=1, keepdim=True)
        
        # 计算策略损失（最大化Q值减去熵正则项）
        policy_loss = (self.alpha * log_prob - q_value).mean()
        
        # 反向传播和优化
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        clip_grad_norm_(self.policy.parameters(), 1.0)  # 梯度裁剪
        self.policy_optimizer.step()
        
        # 更新熵系数
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        for critic, critic_target in zip(self.critics, self.critic_targets):
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        """添加经验到在线缓冲区"""
        self.online_buffer.add(state, action, reward, next_state, done)
    
    def save(self, directory):
        """保存模型"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.policy.state_dict(), os.path.join(directory, "policy.pth"))
        for i, critic in enumerate(self.critics):
            torch.save(critic.state_dict(), os.path.join(directory, f"critic_{i}.pth"))
        torch.save(self.log_alpha, os.path.join(directory, "alpha.pth"))
        
        print(f"模型已保存到 {directory}")
    
    def load(self, directory):
        """加载模型"""
        self.policy.load_state_dict(torch.load(os.path.join(directory, "policy.pth"), map_location=self.device))
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(torch.load(os.path.join(directory, f"critic_{i}.pth"), map_location=self.device))
            self.critic_targets[i] = deepcopy(critic)
        self.log_alpha = torch.load(os.path.join(directory, "alpha.pth"), map_location=self.device)
        self.alpha = self.log_alpha.exp()
        
        print(f"模型已从 {directory} 加载")
