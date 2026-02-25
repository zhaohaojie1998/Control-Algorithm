# -*- coding: utf-8 -*-
import numpy as np
from controller.mimo import LQR_StateRegulator
from controller.utils import matplotlib_context

def test_lqr_state_regulator():
    # 系统参数
    dt = 0.01
    total_time = 5.0
    time_steps = int(total_time / dt)
    
    # 系统矩阵
    # 双积分器系统：x'' = u
    # 状态方程：[x', v']^T = [0, 1; 0, 0] * [x, v]^T + [0; 1] * u
    A = np.array([[0, 1],
                  [0, 0]])
    B = np.array([[0],
                  [1]])
    
    # 权重矩阵
    Q = np.diag([10, 1])
    R = 0.1

    # 初始化LQR控制器
    lqr_controller = LQR_StateRegulator(A, B, Q, R, dt, discrete=False)
    print(lqr_controller)
    
    # 初始状态：位置为1，速度为0
    x = np.array([1, 0])
    
    # 仿真
    for _ in range(time_steps):
        # 计算控制输入
        u = lqr_controller(x)
        # 更新状态
        x_dot = A @ x + B @ u
        x = x + x_dot * dt
    
    # 绘制结果
    lqr_controller.show()


if __name__ == '__main__':
    with matplotlib_context():
        test_lqr_state_regulator()
