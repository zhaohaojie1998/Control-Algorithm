# -*- coding: utf-8 -*-
"""
强化学习工具

@author: https://github.com/zhaohaojie1998
"""
import torch
import torch.nn as nn

import numpy as np
import gymnasium as gym

__all__ = [
    'build_mlp',
    'lr_schedule',
    'ContinuousEnvWrapper',
    'ReplayBuffer',
    'RolloutBuffer'
]


def build_mlp(mlp_sizes: list[int], activation: str = "ReLU", output_activation: str = None) -> nn.Sequential:
    layers = []
    for i in range(len(mlp_sizes) - 1):
        layers.append(nn.Linear(mlp_sizes[i], mlp_sizes[i + 1]))
        
        # 中间层激活函数
        if i < len(mlp_sizes) - 2:
            activation_class = getattr(nn, activation)
            layers.append(activation_class())

    # 输出层激活函数
    if output_activation is not None:
        activation_class = getattr(nn, output_activation)
        layers.append(activation_class())
    
    return nn.Sequential(*layers)


def lr_schedule(optimizer: torch.optim.Optimizer, current_step: int, max_steps: int, final_lr_ratio: float = 0.1):
    """
    学习率线性衰减到final_lr_ratio * lr_init
    
    Args:
        optimizer: 优化器
        current_step: 当前训练步数
        max_steps: 最大训练步数
        final_lr_ratio: 最终学习率占初始学习率的比例, 默认值为0.1
    
    Returns:
        lr: 当前学习率
    """
    lr_init = optimizer.defaults["lr"]
    lr = (1 - final_lr_ratio) * lr_init * max(0, 1 - current_step / max_steps) + final_lr_ratio * lr_init
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class ContinuousEnvWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box) and len(env.action_space.shape) == 1, "only support 1-d act space"
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high

    def action(self, action):
        """将神经网络输出的 [-1, 1] 映射到 [u_min, u_max]"""
        return (action + 1) * (self.u_max - self.u_min) / 2 + self.u_min

    @property
    def env_name(self) -> str:
        """环境的名称"""
        if self.spec is None:
            return self.unwrapped.__class__.__name__
        return self.spec.id


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer_size = int(buffer_size)
        self.batch_size = int(batch_size)
        self.buffer = []
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)
    
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, terminated: bool):
        """
        存储经验
        
        Args:
            obs: 观测 (obs_dim,)
            action: 动作 (act_dim,)
            reward: 奖励 float
            next_obs: 下一个观测 (obs_dim,)
            terminated: 终止状态 bool
        """        
        if len(self.buffer) < self.buffer_size: # 这里是<, 不然会多一个数据
            self.buffer.append({
                "obs": obs.ravel(),
                "action": action.ravel(),
                "reward": reward,
                "next_obs": next_obs.ravel(),
                "terminated": terminated
            })
        else:
            self.buffer[self.position] = {
                "obs": obs.ravel(),
                "action": action.ravel(),
                "reward": reward,
                "next_obs": next_obs.ravel(),
                "terminated": terminated
            }
        self.position = (self.position + 1) % self.buffer_size
    
    def replay(self, device: str = "cpu") -> dict:
        """
        采样经验
        
        Args:
            device: 设备
        
        Returns:
            dict[str: torch.Tensor]: 采样的经验, 包含obs, action, reward, next_obs, terminated
        """
        assert len(self) >= self.batch_size, "buffer current size must >= batch_size"
        batch_indices = np.random.choice(len(self), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in batch_indices]
        
        obs_batch = np.stack([item["obs"] for item in batch])
        action_batch = np.stack([item["action"] for item in batch])
        reward_batch = np.array([item["reward"] for item in batch])
        next_obs_batch = np.stack([item["next_obs"] for item in batch])
        terminated_batch = np.array([item["terminated"] for item in batch])
        
        return {
            "obs": torch.tensor(obs_batch, device=device, dtype=torch.float32),
            "action": torch.tensor(action_batch, device=device, dtype=torch.float32),
            "reward": torch.tensor(reward_batch, device=device, dtype=torch.float32).view(-1, 1),
            "next_obs": torch.tensor(next_obs_batch, device=device, dtype=torch.float32),
            "terminated": torch.tensor(terminated_batch, device=device, dtype=torch.float32).view(-1, 1)
        }
    
    def should_update(self) -> bool:
        return len(self) >= self.batch_size


class RolloutBuffer:
    def __init__(self, rollout_length: int):
        self.rollout_length = int(rollout_length)
        self.buffer = {"obs": [], "action": [], "reward": [], "value": [], "log_prob": [], "terminated": []}
        self.position = 0

    def __len__(self):
        return len(self.buffer["obs"])
    
    def reset(self):
        for key in self.buffer:
            self.buffer[key].clear()
        self.position = 0
    
    def store(self, obs: np.ndarray, action: np.ndarray, reward: float, terminated: bool, value: np.ndarray, log_prob: np.ndarray):
        """
        存储轨迹数据
        
        Args:
            obs: 观测 (obs_dim,)
            action: 动作 (act_dim,)
            reward: 奖励 float
            value: 价值 (1,)
            log_prob: 对数概率 (1,)
            terminated: 终止标志 bool
        """
        self.buffer["obs"].append(obs.ravel())
        self.buffer["action"].append(action.ravel())
        self.buffer["reward"].append(reward)
        self.buffer["terminated"].append(terminated)
        self.buffer["value"].append(value.ravel())
        self.buffer["log_prob"].append(log_prob.ravel())
        self.position += 1
    
    def replay(self, device: str = "cpu") -> dict:
        """
        获取轨迹数据, 并清空缓冲区
        
        Args:
            device: 设备
        
        Returns:
            dict[str: torch.Tensor]: 完整的轨迹数据, 包含obs, action, reward, value, log_prob, terminated
        """
        assert len(self) == self.rollout_length, "buffer is not full"
        
        result = {}
        result["obs"] = torch.tensor(np.stack(self.buffer["obs"]), device=device, dtype=torch.float32)
        result["action"] = torch.tensor(np.stack(self.buffer["action"]), device=device, dtype=torch.float32)
        result["reward"] = torch.tensor(np.array(self.buffer["reward"]), device=device, dtype=torch.float32).view(-1, 1)
        result["terminated"] = torch.tensor(np.array(self.buffer["terminated"]), device=device, dtype=torch.float32).view(-1, 1)
        result["value"] = torch.tensor(np.stack(self.buffer["value"]), device=device, dtype=torch.float32).view(-1, 1)
        result["log_prob"] = torch.tensor(np.stack(self.buffer["log_prob"]), device=device, dtype=torch.float32).view(-1, 1)
        
        self.reset()
        return result
    
    def should_update(self) -> bool:
        return len(self) >= self.rollout_length