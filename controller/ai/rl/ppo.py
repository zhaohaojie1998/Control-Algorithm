# -*- coding: utf-8 -*-
"""
Proximal Policy Optimization 算法, 参考sb3的实现

@author: https://github.com/zhaohaojie1998
"""

''' PPO '''
# model free controller
from typing import Optional
import pathlib
from datetime import datetime
from functools import cached_property

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv

from .model import GaussianActor, CriticV
from .utils import ContinuousEnvWrapper, RolloutBuffer, lr_schedule

__all__ = ['PPO']


class PPO:
    """Proximal Policy Optimization 算法(clip版)"""

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        rollout_length: int = 2048,
        micro_batch_size: int = 64,
        ppo_epochs: int = 10,
        target_kl: Optional[float] = None,

        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        normalize_advantage: bool = True,

        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        decay_lr: bool = False,
        actor_mlp_sizes: list = [128, 128, 128],
        critic_mlp_sizes: list = [128, 128, 128],
    ):
        """
        Args:
            env (gym.Env): 环境
            gamma (float, optional): 折扣因子, 默认值为0.99.
            gae_lambda (float, optional): GAE参数, 默认值为0.95.
            rollout_length (int, optional): 环境rollout长度, 必须为micro_batch_size的整数倍, 默认值为2048.
            micro_batch_size (int, optional): 小批量大小, 必须能被rollout_length整除, 默认值为64.
            ppo_epochs (int, optional): 一次rollout的数据复用多少轮, 复用过多可能导致训练不稳定, 复用过少收敛慢, 默认值为10.
            target_kl (Optional[float], optional): 优化过程中KL超过此值时跳出ppo_epochs, 防止失稳, 默认None跑完所有的ppo_epochs.
            clip_range (float, optional): Surrogate Objective裁剪阈值, 默认值为0.2.
            clip_range_vf (float, optional): Value Function裁剪阈值, 默认None不裁剪.
            value_coef (float, optional): V损失系数, 默认值为0.5.
            entropy_coef (float, optional): 熵损失系数, 默认值为0.0.
            max_grad_norm (float, optional): 最大梯度范数, 默认值为0.5.
            normalize_advantage (bool, optional): 是否对优势函数归一化, 默认值为True.
            lr_actor (float, optional): 策略学习率, 默认值为3e-4.
            lr_critic (float, optional): V学习率, 默认值为3e-4.
            decay_lr (bool, optional): 是否衰减学习率, 默认值为False.
            actor_mlp_sizes (list, optional): 策略网络的MLP层大小, 默认值为[128, 128, 128].
            critic_mlp_sizes (list, optional): V网络的MLP层大小, 默认值为[128, 128, 128].
        """
        assert isinstance(env, gym.Env), "env must be a gym.Env instance"
        assert not isinstance(env, VectorEnv) and not hasattr(env, "num_envs"), "only support single env"
        assert isinstance(env.action_space, gym.spaces.Box), "only support continuous act space"
        assert isinstance(env.observation_space, gym.spaces.Box) and len(env.observation_space.shape) == 1, "only support 1-d obs space"
        assert isinstance(env.action_space, gym.spaces.Box) and len(env.action_space.shape) == 1, "only support 1-d act space"
        assert rollout_length >= micro_batch_size and rollout_length % micro_batch_size == 0, f"rollout_length ({rollout_length}) must be divisible by micro_batch_size ({micro_batch_size})"

        # Environment
        self.env = ContinuousEnvWrapper(env)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high

        # PPO Parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Rollout
        self.buffer = RolloutBuffer(rollout_length)
        self.rollout_length = int(rollout_length)
        self.micro_batch_size = int(micro_batch_size)
        self.ppo_epochs = int(ppo_epochs)
        self.target_kl = target_kl

        # Objective
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_advantage = normalize_advantage
        
        # Networks
        self.actor = GaussianActor(self.obs_dim, self.act_dim, actor_mlp_sizes).to(self.device)
        self.critic = CriticV(self.obs_dim, critic_mlp_sizes).to(self.device)
        
        # Optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.decay_lr = decay_lr
        
        # Tensorboard
        self.writer = SummaryWriter(f"runs/PPO_{self.env.env_name}_{datetime.now().strftime('%b%d_%H-%M-%S')}")
        self.global_step = 0

    @cached_property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """
        与环境交互, 返回动作、对数概率、价值, 动作范围 [-1, 1]
        
        Args:
            obs: 观测 (obs_dim,)
            deterministic: 是否使用确定性策略
        
        Returns:
            action: 动作 (act_dim,)
            log_prob: 对数概率 (1,)
            value: 价值 (1,)
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, -1) # (1, obs_dim)
        action, log_prob = self.actor(obs_tensor, deterministic=deterministic, compute_log_prob=True)
        value = self.critic(obs_tensor)

        action_np = action.numpy(force=True).ravel()
        log_prob_np = log_prob.numpy(force=True).ravel()
        value_np = value.numpy(force=True).ravel()
        return action_np, log_prob_np, value_np # (act_dim,), (1,), (1)
    
    @torch.no_grad()
    def generalized_advantage_estimation(self, rewards, values, terminated, last_obs):
        """
        广义优势估计算法 (Generalized Advantage Estimation, GAE)
        
        Args:
            rewards: 奖励, (batch_size, 1)
            values: 价值估计, (batch_size, 1)
            terminated: 终止标志, (batch_size, 1)
            last_obs: 最后一个观测, (1, obs_dim)
        
        Returns:
            advantages: 优势值, (batch_size, 1)
            returns: 回报值, (batch_size, 1)
        """
        batch_size = rewards.shape[0]
        
        last_adv = 0
        last_value = self.critic(last_obs).squeeze() # (1,)
        advantages = torch.zeros_like(rewards) # (batch_size, 1)
        
        for i in reversed(range(batch_size)):
            if i == batch_size - 1:
                next_val = last_value # (1,)
            else:
                next_val = values[i+1] # (1,)
            
            delta = rewards[i] + self.gamma * next_val * (1 - terminated[i]) - values[i] # (1,)
            last_adv = delta + self.gamma * self.gae_lambda * last_adv * (1 - terminated[i]) # (1,)
            advantages[i] = last_adv # (1,)

        returns = advantages + values # (batch_size, 1)
        
        return advantages, returns # (batch_size, 1)
    
    def update(self, last_obs: np.ndarray, total_steps: Optional[int] = None) -> dict:
        """
        执行一次PPO更新
        
        Args:
            last_obs: 一次rollout的最后一个观测, (obs_dim,)
            total_steps: update总调用次数, 用于学习率衰减

        Returns:
            metrics: 指标字典
        """
        metrics = {}
        if not self.buffer.should_update():
            return metrics
        
        self.global_step += 1
        
        ## 1.经验回放(并清空缓存)
        batch = self.buffer.replay(device=self.device)
        obs_batch = batch["obs"] # (batch_size, obs_dim)
        action_batch = batch["action"] # (batch_size, act_dim)
        reward_batch = batch["reward"] # (batch_size, 1)
        terminated_batch = batch["terminated"] # (batch_size, 1)
        old_log_prob_batch = batch["log_prob"] # (batch_size, 1)
        old_value_batch = batch["value"] # (batch_size, 1)
        
        ## 2.广义优势估计
        last_obs = torch.tensor(last_obs, dtype=torch.float32, device=self.device).view(1, -1) # (1, obs_dim)
        advantages_batch, returns_batch = self.generalized_advantage_estimation(reward_batch, old_value_batch, terminated_batch, last_obs) # (batch_size, 1)
        
        ## 3.使用同一批经验多次更新, 提高样本效率
        # 只有第一次是严格意义上的on-policy更新, 后续是off-policy, PPO的信赖域约束允许更新多次
        metrics_list = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "clip_fraction": [],
            "mean_log_prob": [],
            "entropy": [],
            "kl_div": [],
            "mean_value": [],
        }

        batch_size = reward_batch.shape[0]
        num_batches = batch_size // self.micro_batch_size
        should_break = False

        for epoch in range(self.ppo_epochs):
            # 打乱数据顺序
            indices = torch.randperm(batch_size)
            obs_shuffled = obs_batch[indices]
            action_shuffled = action_batch[indices]
            advantages_shuffled = advantages_batch[indices]
            returns_shuffled = returns_batch[indices]
            old_log_prob_shuffled = old_log_prob_batch[indices]
            old_value_shuffled = old_value_batch[indices]
            
            # 小批次更新
            for i in range(num_batches):
                # 提取micro batch
                start_idx = i * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                
                obs = obs_shuffled[start_idx:end_idx] # (micro_batch_size, obs_dim)
                action = action_shuffled[start_idx:end_idx] # (micro_batch_size, act_dim)
                advantages = advantages_shuffled[start_idx:end_idx] # (micro_batch_size, 1)
                returns = returns_shuffled[start_idx:end_idx] # (micro_batch_size, 1)
                old_log_prob = old_log_prob_shuffled[start_idx:end_idx] # (micro_batch_size, 1)
                old_value = old_value_shuffled[start_idx:end_idx] # (micro_batch_size, 1)

                if self.normalize_advantage and advantages.shape[0] > 1:
                    with torch.no_grad():
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 计算当前策略的价值
                value = self.critic(obs) # (micro_batch_size, 1)
                
                # 计算老action的新对数概率
                _, new_log_prob = self.actor(obs, action=action, compute_log_prob=True) # (micro_batch_size, 1)
                
                # 由于tanh分布变换, 需要用蒙特卡洛近似计算熵：H[π] = -E[log π(a|s)]
                entropy = -new_log_prob.mean() # ()
                
                # 计算Surrogate Objective (clip版)
                log_ratio = new_log_prob - old_log_prob # (micro_batch_size, 1)
                ratio = torch.exp(log_ratio) # (micro_batch_size, 1)
                surr1 = advantages * ratio # (micro_batch_size, 1)
                surr2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) # (micro_batch_size, 1)
                actor_loss = -torch.min(surr1, surr2).mean() # ()
                
                # 计算Value损失
                if self.clip_range_vf is None:
                    value_pred = value # (micro_batch_size, 1)
                else:
                    value_pred = old_value + torch.clamp(value - old_value, -self.clip_range_vf, self.clip_range_vf) # (micro_batch_size, 1)
                critic_loss = F.mse_loss(returns, value_pred) # ()

                # 计算Entropy损失
                entropy_loss = -entropy # ()
                
                # PPO总损失
                total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
                
                with torch.no_grad():
                    kl_div = torch.mean((ratio - 1) - log_ratio).item() # float
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_range).float()).item() # float
                    metrics_list["actor_loss"].append(actor_loss.item())
                    metrics_list["critic_loss"].append(critic_loss.item())
                    metrics_list["entropy_loss"].append(entropy_loss.item())
                    metrics_list["total_loss"].append(total_loss.item())
                    metrics_list["clip_fraction"].append(clip_fraction)
                    metrics_list["mean_log_prob"].append(-entropy.item())
                    metrics_list["entropy"].append(entropy.item())
                    metrics_list["kl_div"].append(kl_div)
                    metrics_list["mean_value"].append(value.mean().item())
                
                # KL达到阈值提前跳出, 不执行更新
                if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                    should_break = True
                    print(f"Early stopping at epoch {epoch} due to reaching max kl: {kl_div:.2f}")
                    break

                # 参数更新
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer_actor.step()
                self.optimizer_critic.step()
            # end mini batch

            if should_break:
                break
        # end epoch

        ## 4.更新学习率
        if self.decay_lr:
            assert total_steps is not None, "total_steps must be provided when decay_lr is True"
            metrics["lr_actor"] = lr_schedule(self.optimizer_actor, self.global_step, total_steps)
            metrics["lr_critic"] = lr_schedule(self.optimizer_critic, self.global_step, total_steps)

        ## 5.记录指标
        with torch.no_grad():
            for key in metrics_list:
                metrics[key] = np.mean(metrics_list[key])
            metrics["mean_return"] = returns_batch.mean().item()
            metrics["mean_advantage"] = advantages_batch.mean().item()
        
        self.writer.add_scalar("Loss/actor", metrics["actor_loss"], self.global_step)
        self.writer.add_scalar("Loss/critic", metrics["critic_loss"], self.global_step)
        self.writer.add_scalar("Loss/entropy", metrics["entropy_loss"], self.global_step)
        self.writer.add_scalar("Loss/total", metrics["total_loss"], self.global_step)
        self.writer.add_scalar("Loss/clip_fraction", metrics["clip_fraction"], self.global_step) # 裁剪比例

        self.writer.add_scalar("State Value/mean_value", metrics["mean_value"], self.global_step)
        self.writer.add_scalar("State Value/mean_return", metrics["mean_return"], self.global_step)
        self.writer.add_scalar("State Value/mean_advantage", metrics["mean_advantage"], self.global_step)

        self.writer.add_scalar("Policy/mean_log_prob", metrics["mean_log_prob"], self.global_step)
        self.writer.add_scalar("Policy/entropy", metrics["entropy"], self.global_step)
        self.writer.add_scalar("Policy/kl_div", metrics["kl_div"], self.global_step)
        
        if self.decay_lr:
            self.writer.add_scalar("Learning Rate/actor", metrics["lr_actor"], self.global_step)
            self.writer.add_scalar("Learning Rate/critic", metrics["lr_critic"], self.global_step)
        
        print(
            f"Global Step: {self.global_step} "
            f"| actor_loss: {metrics['actor_loss']:.4f} "
            f"| critic_loss: {metrics['critic_loss']:.4f} "
            f"| total_loss: {metrics['total_loss']:.4f} "
            f"| clip_fraction: {metrics['clip_fraction']:.4f} "
            f"| mean_log_prob: {metrics['mean_log_prob']:.4f} "
            f"| entropy: {metrics['entropy']:.4f} "
            f"| kl_div: {metrics['kl_div']:.4f} "
            f"| mean_value: {metrics['mean_value']:.4f} "
            f"| mean_return: {metrics['mean_return']:.4f} "
            f"| mean_advantage: {metrics['mean_advantage']:.4f} "
        )
        return metrics
    
    def train(self, max_env_steps: int):
        """
        PPO训练
        
        Args:
            max_env_steps: 最大环境步数
        """
        total_update_steps = max_env_steps // self.rollout_length
        env_step = 0
        episode = 0
        
        obs, _ = self.env.reset()
        episode_return, episode_length = 0, 0
        
        while env_step < max_env_steps:
            # 1.执行一次rollout
            for _ in range(self.rollout_length):
                # 环境交互
                action, log_prob, value = self.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                episode_return += reward
                episode_length += 1
                env_step += 1
                
                # 保存transition
                self.buffer.store(obs, action, reward, terminated, value, log_prob)
                
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
                
                # 检查是否达到最大步数
                if env_step >= max_env_steps:
                    break
            #end rollout
            
            # 2.PPO更新
            self.update(next_obs, total_update_steps)
        
        # 关闭tensorboard
        self.writer.close()
    
    @torch.no_grad()
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
        info = \
f"""============= Proximal Policy Optimization (PPO) =============
Environment:
    id: {self.env.env_name}
    observation_space: {self.env.observation_space}
    action_space: {self.env.action_space}
Parameters:
    gamma: {self.gamma}
    gae_lambda: {self.gae_lambda}
    rollout_length: {self.rollout_length}
    micro_batch_size: {self.micro_batch_size}
    ppo_epochs: {self.ppo_epochs}
    target_kl: {self.target_kl}
    clip_range: {self.clip_range}
    clip_range_vf: {self.clip_range_vf}
    value_coef: {self.value_coef}
    entropy_coef: {self.entropy_coef}
    max_grad_norm: {self.max_grad_norm}
    normalize_advantage: {self.normalize_advantage}
===================== Actor Model =====================
{self.actor}
===================== Critic Model =====================
{self.critic}
===================== Optimizers =====================
Actor Optimizer: {self.optimizer_actor}
Critic Optimizer: {self.optimizer_critic}
====================================================="""
        return info