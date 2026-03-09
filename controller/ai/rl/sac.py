# -*- coding: utf-8 -*-
"""
Soft Actor-Critic 算法

@author: https://github.com/zhaohaojie1998
"""

''' SAC '''
# model free controller
from typing import Optional
import pathlib
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv

from .utils import build_mlp, ContinuousEnvWrapper, ReplayBuffer

__all__ = ['SAC']


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, mlp_sizes=[128, 128, 128], log_std_min=-20, log_std_max=2):
        super().__init__()
        mlp_sizes = [obs_dim] + mlp_sizes
        self.mlp = build_mlp(mlp_sizes, activation="ReLU", output_activation="ReLU")
        self.mean_head = nn.Linear(mlp_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(mlp_sizes[-1], act_dim)
        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float32))

    def forward(self, obs, deterministic=False, compute_log_prob=True):
        feature = self.mlp(obs)
        mean = self.mean_head(feature)
        log_std = self.log_std_head(feature)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # 构造正态分布, 采样出无约束动作
        dist = Normal(mean, std)
        if deterministic:
            u = mean
        else:
            u = dist.rsample() # 需要rsample确保梯度回传

        # 分布变换, 转换为有约束动作, 并对对数概率进行雅可比修正
        a = torch.tanh(u)
        if compute_log_prob:
            # SAC论文公式: log_prob(a) = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # 由于 a=tanh(u) 会导致梯度消失, 将 tanh(u)^2 按照tanh定义展开, 得到如下公式:
            log_prob = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True) # (batch, 1)
        else:
            log_prob = None
        
        return a, log_prob


class CriticQ(nn.Module):
    def __init__(self, obs_dim, act_dim, mlp_sizes=[128, 128, 128]):
        super().__init__()
        mlp_sizes = [obs_dim + act_dim] + mlp_sizes + [1]
        self.q1 = build_mlp(mlp_sizes, activation="ReLU", output_activation=None)
        self.q2 = build_mlp(mlp_sizes, activation="ReLU", output_activation=None)
    
    def forward(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=-1)), self.q2(torch.cat([obs, action], dim=-1))


class SAC:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        alpha: float = 0.2,
        batch_size: int = 64,
        buffer_size: int = 10000,
        target_entropy: Optional[float] = None,
        lr_alpha: float = 1e-3,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        tau: float = 0.005,
        actor_mlp_sizes: list[int] = [128, 128, 128],
        critic_mlp_sizes: list[int] = [128, 128, 128],
    ):
        """
        Args:
            env (gym.Env): 环境
            gamma (float, optional): 折扣因子, 默认值为0.99
            alpha (float, optional): 初始温度系数, 默认值为0.2
            batch_size (int, optional): 批次大小, 默认值为64
            buffer_size (int, optional): 经验池大小, 默认值为10000
            target_entropy (Optional[float], optional): 策略熵优化目标值, 默认None取-dim(A)
            lr_alpha (float, optional): 温度系数学习率, 默认值为1e-3
            lr_actor (float, optional): 策略网络学习率, 默认值为1e-3
            lr_critic (float, optional): Q网络学习率, 默认值为1e-3
            tau (float, optional): 目标Q软更新系数, 默认值为0.005
            actor_mlp_sizes (list[int], optional): 策略网络MLP层大小, 默认值为[128, 128, 128]
            critic_mlp_sizes (list[int], optional): Q网络MLP层大小, 默认值为[128, 128, 128]
        """
        assert isinstance(env, gym.Env), "env must be a gym.Env instance"
        assert not isinstance(env, VectorEnv) and not hasattr(env, "num_envs"), "only support single env"
        assert isinstance(env.action_space, gym.spaces.Box), "only support continuous act space"
        assert isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 1, "only support 1-d obs space"
        assert isinstance(env.action_space, gym.spaces.Box) and len(env.action_space.shape) == 1, "only support 1-d act space"
        
        # Environment
        self.env = ContinuousEnvWrapper(env)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high
        
        # SAC Parameters
        self.gamma = gamma

        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Networks
        self.actor = Actor(self.obs_dim, self.act_dim, actor_mlp_sizes).to(self.device)
        self.critic = CriticQ(self.obs_dim, self.act_dim, critic_mlp_sizes).to(self.device)
        self.critic_target = CriticQ(self.obs_dim, self.act_dim, critic_mlp_sizes).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.tau = tau
        
        # Temperature
        self.alpha = alpha
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(self.act_dim) # SAC论文: -dim(A)
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        
        # Tensorboard
        self.writer = SummaryWriter()
        self.global_step = 0
    
    @cached_property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @torch.no_grad
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        与环境交互，返回动作, 范围 [-1, 1]
        
        Args:
            obs: 观测 (obs_dim,)
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作 (act_dim,)
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1) # (1, obs_dim)
        action, _ = self.actor(obs_tensor, deterministic=deterministic, compute_log_prob=False)
        action_np = action.numpy(force=True)
        return action_np.ravel() # (act_dim,)
    
    def update(self) -> dict:
        """执行一次SAC更新"""
        if not self.buffer.should_update():
            return {}
        
        self.global_step += 1
        
        ## 1.经验回放
        batch = self.buffer.replay(device=self.device)
        obs = batch["obs"] # (batch_size, obs_dim)
        action = batch["action"] # (batch_size, act_dim)
        reward = batch["reward"] # (batch_size, 1)
        next_obs = batch["next_obs"] # (batch_size, obs_dim)
        terminated = batch["terminated"] # (batch_size, 1)
        
        ## 2.更新critic网络
        # 计算目标Q值
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_obs, deterministic=False, compute_log_prob=True)
            next_q1, next_q2 = self.critic_target(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - terminated) * self.gamma * (next_q - self.alpha * next_log_prob)

        # 计算critic loss
        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # 更新critic网络
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        ## 3.更新actor网络
        # 冻结critic网络参数
        for param in self.critic.parameters():
            param.requires_grad = False
        
        # 计算actor loss
        new_action, log_prob = self.actor(obs, deterministic=False, compute_log_prob=True)
        q1, q2 = self.critic(obs, new_action)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - q).mean()
        
        # 更新actor网络
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        # 解冻critic网络参数
        for param in self.critic.parameters():
            param.requires_grad = True
        
        ## 4.更新alpha参数
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        #alpha_loss = -(self.log_alpha.exp() * (log_prob.detach() + self.target_entropy)).mean() # SAC原始公式, 但计算慢
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()
        self.alpha = self.log_alpha.exp().item()
        
        ## 5.更新target critic网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        ## 6.记录指标
        with torch.no_grad():
            metrics = {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "soft_q1": current_q1.mean().item(),
                "soft_q2": current_q2.mean().item(),
                "log_prob": log_prob.mean().item(),
                "alpha": self.alpha,
            }
        
        self.writer.add_scalar("Loss/actor", metrics["actor_loss"], self.global_step)
        self.writer.add_scalar("Loss/critic", metrics["critic_loss"], self.global_step)
        self.writer.add_scalar("Loss/alpha", metrics["alpha_loss"], self.global_step)
        self.writer.add_scalar("Value/soft_q1", metrics["soft_q1"], self.global_step)
        self.writer.add_scalar("Value/soft_q2", metrics["soft_q2"], self.global_step)
        self.writer.add_scalar("Policy/log_prob", metrics["log_prob"], self.global_step)
        self.writer.add_scalar("Policy/temperature(alpha)", metrics["alpha"], self.global_step)
        
        print(
            f"Global Step: {self.global_step} "
            f"| actor_loss: {metrics['actor_loss']:.4f} "
            f"| critic_loss: {metrics['critic_loss']:.4f} "
            f"| alpha_loss: {metrics['alpha_loss']:.4f} "
            f"| log_prob: {metrics['log_prob']:.4f} "
            f"| temperature(alpha): {metrics['alpha']:.4f} "
            f"| soft_q1: {metrics['soft_q1']:.4f} "
            f"| soft_q2: {metrics['soft_q2']:.4f} "
        )
        
        return metrics
    
    def train(self, max_env_steps: int = 100000, random_steps: int = 1000, update_freq: int = 1, update_times: int = 1):
        """
        SAC训练

        Args:
            max_env_steps: 最大环境步数
            random_steps: 随机探索环境的步数
            update_freq: 每多少环境步更新网络
            update_times: 每次更新网络的次数
        """
        env_step = 0
        episode = 0

        obs, _ = self.env.reset()
        episode_return, episode_length = 0, 0
        
        while env_step < max_env_steps:
            # 环境交互
            if env_step < random_steps:
                action = np.random.uniform(-1, 1, size=self.act_dim)
            else:
                action = self.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = self.env.step(action)

            episode_return += reward
            episode_length += 1
            env_step += 1
            
            # 保存transition
            self.buffer.store(obs, action, reward, next_obs, terminated)
            
            # 回合结束
            if terminated or truncated:
                print(f"Episode {episode}: Return={episode_return:.2f}, Length={episode_length}")

                # 记录指标
                self.writer.add_scalar("Env/episode_return", episode_return, episode)
                self.writer.add_scalar("Env/episode_length", episode_length, episode)

                # 重置环境
                episode += 1
                obs, _ = self.env.reset()
                episode_return, episode_length = 0, 0
                
            else:
                obs = next_obs
            
            # SAC更新
            if env_step % update_freq == 0:
                for _ in range(update_times):
                    self.update()
        
        # 关闭tensorboard
        self.writer.close()
    
    def save_onnx(self, onnx_path: str, deterministic: bool = True, device: str = "cpu"):
        """
        导出ONNX模型
        
        Args:
            onnx_path: ONNX模型保存路径
            deterministic: 是否使用确定性策略, 默认值为True
            device: 导出设备, 默认值为"cpu"
        """
        onnx_path = pathlib.Path(onnx_path).with_suffix(".onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device(device)
        self.actor.to(device)
        
        class ActorWrapper(nn.Module):
            def __init__(this, actor, u_min, u_max):
                super().__init__()
                this.actor = actor
                this.u_min = torch.tensor(u_min, device=device, dtype=torch.float32)
                this.u_max = torch.tensor(u_max, device=device, dtype=torch.float32)
            
            def forward(this, obs):
                action, _ = this.actor(obs, deterministic=deterministic, compute_log_prob=False)
                scaled_action = action * (this.u_max - this.u_min) + this.u_min
                return scaled_action
        
        wrapper = ActorWrapper(self.actor, self.u_min, self.u_max)
        dummy_input = torch.randn(1, self.obs_dim, device=device)
        torch.onnx.export(
            wrapper,
            dummy_input,  # 输入：obs
            onnx_path,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "action": {0: "batch_size"}
            }
        )
        print(f"ONNX模型已保存到: {onnx_path}")
        
        self.actor.to(self.device)
    
    @staticmethod
    def get_controller(onnx_path: str, dt: float):
        """
        获取RL控制器
        
        Args:
            onnx_path: ONNX模型路径
            dt: 控制器步长
            
        Returns:
            RLController: 强化学习控制器
        """
        from .rl_controller import RLController
        return RLController(onnx_path, dt)
    
    def __repr__(self):
        return f"SAC(gamma={self.gamma}, alpha={self.alpha}, target_entropy={self.target_entropy}, tau={self.tau})"
