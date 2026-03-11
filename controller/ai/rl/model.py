# -*- coding: utf-8 -*-
"""
强化学习模型

@author: https://github.com/zhaohaojie1998
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

from .utils import build_mlp

__all__ = [
    "GaussianActor",
    "CriticQ",
    "CriticV",
]

# ============== Actor Network ==============

class GaussianActor(nn.Module):
    """高斯分布Actor网络

    约束处理方法:
        构造 μ∈[-∞, ∞] 的正态分布, 采样得到无约束动作u
        通过tanh变换得到有约束动作 a∈[-1, 1], 同时对logprob(u)进行雅可比修正得到logprob(a)
    """
    def __init__(self, obs_dim, act_dim, mlp_sizes=[128, 128, 128], log_std_min=-20, log_std_max=2):
        super().__init__()
        mlp_sizes = [obs_dim] + mlp_sizes
        self.mlp = build_mlp(mlp_sizes, activation="ReLU", output_activation="ReLU")
        self.mean_head = nn.Linear(mlp_sizes[-1], act_dim)
        self.log_std_head = nn.Linear(mlp_sizes[-1], act_dim)
        self.register_buffer("log_std_min", torch.tensor(log_std_min, dtype=torch.float32))
        self.register_buffer("log_std_max", torch.tensor(log_std_max, dtype=torch.float32))

    def forward(self, obs, action=None, deterministic=False, compute_log_prob=True):
        """
        Args:
            obs: 观测 (batch_size, obs_dim)
            action: 可选, 给定的动作 (batch_size, act_dim), 计算动作的对数概率
            deterministic: 是否使用确定性策略 (action=None时生效), 默认False
            compute_log_prob: 是否计算对数概率 (action=None时生效), 默认True
            
        Returns:
            action: 动作 (batch_size, act_dim)
            log_prob: 对数概率 (batch_size, 1)
        """
        feature = self.mlp(obs)
        mean = self.mean_head(feature)
        log_std = self.log_std_head(feature)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # 构造正态分布
        dist = Normal(mean, std)
        
        # 采样模式：根据策略网络输出的均值和标准差，从正态分布中采样动作，并使用tanh变换限制动作范围到[-1, 1]
        if action is None:
            if deterministic:
                u = mean
            else:
                u = dist.rsample() # 需要rsample确保梯度回传
            
            # 使用tanh变换限制动作范围, 雅可比修正对数概率
            a = torch.tanh(u)
            if compute_log_prob:
                # SAC论文公式: log_prob(a) = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
                # 由于 a=tanh(u) 会导致梯度消失, 将 tanh(u)^2 按照tanh定义展开, 得到如下公式:
                log_prob = torch.sum(
                    (dist.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u)))), # (batch_size, act_dim)
                    dim=1, keepdim=True
                ) # (batch_size, 1)
            else:
                log_prob = None
        
        # 评估模式：计算给定动作的对数概率
        else:
            a = action
            # 逆变换动作a到u
            u = torch.atanh(torch.clamp(action, -0.999999, 0.999999)) # 防止atanh数值溢出
            # 计算雅可比修正后的log_prob
            log_prob = torch.sum(
                (dist.log_prob(u) - (2 * (np.log(2) - u - F.softplus(-2 * u)))), # (batch_size, act_dim)
                dim=1, keepdim=True
            ) # (batch_size, 1)
        
        return a, log_prob


# ============== Critic Network ==============

class CriticQ(nn.Module):
    """Q函数, 输出 Q(s, a)"""

    def __init__(self, obs_dim, act_dim, mlp_sizes=[128, 128, 128]):
        super().__init__()
        mlp_sizes = [obs_dim + act_dim] + mlp_sizes + [1]
        self.q1 = build_mlp(mlp_sizes, activation="ReLU", output_activation=None)
        self.q2 = build_mlp(mlp_sizes, activation="ReLU", output_activation=None)
    
    def forward(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=-1)), self.q2(torch.cat([obs, action], dim=-1))


class CriticV(nn.Module):
    """V函数, 输出 V(s)"""
    
    def __init__(self, obs_dim, mlp_sizes=[128, 128, 128]):
        super().__init__()
        self.obs_dim = obs_dim
        mlp_sizes = [obs_dim] + mlp_sizes + [1]
        self.value = build_mlp(mlp_sizes, activation="ReLU", output_activation=None)
    
    def forward(self, obs):
        return self.value(obs)