# -*- coding: utf-8 -*-
import numpy as np
from controller.mimo import LQR_OutputRegulator
from controller.utils import matplotlib_context

def continuous_lqr_output_regulator():
    # 仿真参数
    dt = 0.01
    total_time = 5.0
    time_steps = int(total_time / dt)
    
    # 系统矩阵
    # 双积分器系统：x'' = u
    # 状态方程：[x', v']^T = [0, 1; 0, 0] * [x, v]^T + [0; 1] * u
    A = [[0, 1],
        [0, 0]]
    B = [[0],
        [1]]
    
    # 输出矩阵：测量位置
    C = [[1, 0]]
    
    # 权重矩阵
    Qy = 10
    R = 0.1
    
    # 初始化LQR控制器
    lqr_controller = LQR_OutputRegulator(A, B, C, Qy, R, dt, discrete=False)
    print("\n连续系统LQR输出调节器参数:")
    print(lqr_controller)
    
    # 初始状态：位置为1，速度为0
    x = np.array([1, 0])
    real_y_list = []
    
    # 仿真
    for _ in range(time_steps):
        # 状态估计器 (实现过程略, 这里简单加点噪声表示估计结果)
        x_hat = x + np.random.normal(0, 0.01, size=x.shape)
        # 状态反馈控制
        u = lqr_controller(x_hat)
        # ode积分
        x_dot = A @ x + B @ u
        x = x + x_dot * dt
        y = C @ x
        real_y_list.append(y[0]) # 需要手动记录真正的y，控制器自动迭代的y基于x估计值, 绘图不好看
        
    # 输出
    lqr_controller.show(name='Continuous', real_y_list=real_y_list)
    print('连续系统最终状态:')
    print(f'  位置: {x[0]:.4f}')
    print(f'  速度: {x[1]:.4f}')
    print('连续系统最终输出:')
    print(f'  位置: {y[0]:.4f}')
    print('连续系统性能指标J:')
    print(f'  J = {lqr_controller.J:.4f}')


def discrete_lqr_output_regulator():
    # 仿真参数
    dt = 0.01
    total_time = 5.0
    time_steps = int(total_time / dt)
    
    # 系统矩阵
    # 双积分器系统：x'' = u
    # 连续状态方程：[x', v']^T = [0, 1; 0, 0] * [x, v]^T + [0; 1] * u
    # 离散状态方程：x(k+1) = A_d * x(k) + B_d * u(k)
    A_cont = [[0, 1],
              [0, 0]]
    B_cont = [[0],
              [1]]
    
    # 离散化系统矩阵
    A = np.eye(2) + np.array(A_cont) * dt
    B = np.array(B_cont) * dt
    
    # 输出矩阵：测量位置
    C = [[1, 0]]
    
    # 权重矩阵
    Qy = 10
    R = 0.1
    
    # 初始化LQR控制器（设置为离散系统）
    lqr_controller = LQR_OutputRegulator(A, B, C, Qy, R, dt, discrete=True)
    print("\n离散系统LQR输出调节器参数:")
    print(lqr_controller)
    
    # 初始状态：位置为1，速度为0
    x = np.array([1, 0])
    real_y_list = []
    
    # 仿真
    for _ in range(time_steps):
        # 状态估计器 (实现过程略, 这里简单加点噪声表示估计结果)
        x_hat = x + np.random.normal(0, 0.01, size=x.shape)
        # 状态反馈控制
        u = lqr_controller(x_hat)
        # 离散状态更新
        x = A @ x + B @ u
        y = C @ x
        real_y_list.append(y[0]) # 需要手动记录真正的y，控制器自动迭代的y基于x估计值, 绘图不好看
    
    # 输出
    lqr_controller.show(name='Discrete', real_y_list=real_y_list)
    print('离散系统最终状态:')
    print(f'  位置: {x[0]:.4f}')
    print(f'  速度: {x[1]:.4f}')
    print('离散系统最终输出:')
    print(f'  位置: {y[0]:.4f}')
    print('离散系统性能指标J:')
    print(f'  J = {lqr_controller.J:.4f}')


if __name__ == '__main__':
    with matplotlib_context():
        continuous_lqr_output_regulator()
    with matplotlib_context():
        discrete_lqr_output_regulator()