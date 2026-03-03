# -*- coding: utf-8 -*-
import numpy as np
import gymnasium as gym
from controller import LQR, LTISystem
from controller.utils import matplotlib_context, setup_seed

def lqr_pendulum_control():
    # 仿真参数
    dt = 0.05  # 与gym环境的dt保持一致, 必须0.05
    total_time = 50.0
    time_steps = int(total_time / dt)
    
    # 倒立摆系统参数（与gym环境一致）
    m = 1.0  # 摆杆质量
    l = 1.0  # 摆杆长度
    g = 10  # 重力加速度
    
    # 系统矩阵（线性化模型，平衡点在竖直向上位置）
    # 状态向量：[theta, theta_dot]^T，其中theta是摆杆与竖直方向的夹角
    # 控制量：u是施加在摆杆上的力矩
    # 线性化动力学方程：theta'' = (3*g/(2*l))*theta + (3/(m*l**2))*u
    A = [[0, 1],
         [3*g/(2*l), 0]]
    B = [[0],
         [3/(m*l**2)]]
    
    sys = LTISystem(A, B)
    
    # 权重矩阵
    Q = np.diag([1000, 100]) # 角度误差处罚, 角速度误差处罚
    R = 0.01 # 控制惩罚
    
    # 初始化LQR控制器
    lqr_controller = LQR(sys, Q, R, dt)
    print("LQR倒立摆控制器参数:")
    print(lqr_controller)
    
    # 仿真
    env = gym.make("Pendulum-v1", g=g, max_episode_steps=time_steps, render_mode="human")
    # 初始状态需要接近线性化平衡点的位置，否则LQR控制器无法收敛
    obs, info = env.reset(options={"x_init": 0.1, "y_init": 0.01}) # 平衡状态为 0
    for _ in range(time_steps):
        x = np.array(env.unwrapped.state) # 状态向量：[theta, theta_dot]
        u = lqr_controller(x)
        obs, reward, terminated, truncated, info = env.step(u)
        if terminated or truncated:
            break
    
    # 获取最终状态
    θf, dθf = env.unwrapped.state
    env.close()
    
    # 输出
    lqr_controller.show(name='Pendulum-v1')
    print('倒立摆控制最终状态:')
    print(f'  摆杆角度: {θf:.4f}')
    print(f'  摆杆角速度: {dθf:.4f}')
    print('LQR性能指标J:')
    print(f'  J = {lqr_controller.J:.4f}')
    
    # 计算控制精度
    angle_error = np.abs(θf)
    angular_velocity_error = np.abs(dθf)
    print('控制精度:')
    print(f'  角度误差: {angle_error:.4f}')
    print(f'  角速度误差: {angular_velocity_error:.4f}')


if __name__ == '__main__':
    setup_seed(114514)
    with matplotlib_context():
        lqr_pendulum_control()
