"""强化学习倒立摆控制"""
import time
import pathlib
import numpy as np
import gymnasium as gym
from controller.ai.rl import RLController
from controller.utils import matplotlib_context, setup_seed, tic, toc


def train_sac(max_env_steps, onnx_path, save_stochastic=True):
    from controller.ai.rl import SAC
    env = gym.make("Pendulum-v1")
    sac = SAC(
        env=env,
        gamma=0.99,
        alpha=0.2,
        batch_size=64,
        buffer_size=10000,
        lr_alpha=1e-3,
        lr_actor=1e-3,
        lr_critic=1e-3,
        decay_lr=False,
        tau=0.005,
        actor_mlp_sizes=[128, 128, 128],
        critic_mlp_sizes=[128, 128, 128],
    )
    print(sac)
    time.sleep(5)

    print("\n====================模型训练开始====================")
    sac.train(max_env_steps=max_env_steps, update_freq=10)
    sac.save_onnx(onnx_path, deterministic=not save_stochastic)
    print("====================模型训练完成====================\n")


def train_ppo(max_env_steps, onnx_path, save_stochastic=True):
    from controller.ai.rl import PPO
    env = gym.make("Pendulum-v1")
    ppo = PPO(
        env=env,
        gamma=0.99,
        gae_lambda=0.95,
        rollout_length=2048,
        micro_batch_size=64,
        ppo_epochs=10,
        target_kl=0.5,
        clip_range=0.2,
        clip_range_vf=None,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        normalize_advantage=True,
        lr_actor=3e-4,
        lr_critic=3e-4,
        decay_lr=False,
        actor_mlp_sizes=[128, 128, 128],
        critic_mlp_sizes=[128, 128, 128],
    )
    print(ppo)
    time.sleep(5)

    print("\n====================模型训练开始====================")
    ppo.train(max_env_steps=max_env_steps)
    ppo.save_onnx(onnx_path, deterministic=not save_stochastic)
    print("====================模型训练完成====================\n")


def rl_pendulum_control(model_path):
    # 仿真参数
    dt = 0.05  # 与gym环境的dt保持一致, 必须0.05
    total_time = 50.0
    time_steps = int(total_time / dt)
    
    # 实例化RL控制器
    rl_ctrl = RLController(model_path, dt)
    print("RL倒立摆控制器参数:")
    print(rl_ctrl)
    
    # 仿真    
    time.sleep(5)
    print("\n====================仿真开始====================")
    env = gym.make("Pendulum-v1", max_episode_steps=time_steps, render_mode="human")
    # 强化学习不要求初始状态位于稳定点附近, 任意初始状态均能收敛
    obs, info = env.reset()
    rl_ctrl.reset()
    
    tic()
    for _ in range(time_steps):
        tic()
        u = rl_ctrl(obs)
        toc("RL控制律求解")

        obs, reward, terminated, truncated, info = env.step(u)
        if terminated or truncated:
            break
    toc("仿真")

    # 获取最终状态
    θf, dθf = env.unwrapped.state
    env.close()
    
    # 输出
    rl_ctrl.show(name=f'{ALGO} Pendulum-v1')
    print('倒立摆控制最终状态:')
    print(f'  摆杆角度: {θf:.4f}')
    print(f'  摆杆角速度: {dθf:.4f}')

    # 计算控制精度
    angle_error = np.abs(θf)
    angular_velocity_error = np.abs(dθf)
    print('控制精度:')
    print(f'  角度误差: {angle_error:.4f}')
    print(f'  角速度误差: {angular_velocity_error:.4f}')



if __name__ == '__main__':
    setup_seed(114514)
    
    ALGO = "PPO" # PPO 或 SAC
    MAX_ENV_STEPS = 100000
    MODEL_PATH = pathlib.Path(f"models/{ALGO}_pendulum.onnx")

    # 训练模型
    if not MODEL_PATH.is_file():
        if ALGO == "SAC":
            train_sac(MAX_ENV_STEPS, MODEL_PATH, save_stochastic=True)
        else:
            train_ppo(MAX_ENV_STEPS, MODEL_PATH, save_stochastic=False)

    # 使用模型
    with matplotlib_context():
        rl_pendulum_control(MODEL_PATH)
