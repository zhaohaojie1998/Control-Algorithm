"""LQR 示例"""
import numpy as np
from controller import LQR, LTISystem
from controller.utils import matplotlib_context

def run_lqr_simulation(discrete=False, output_regulator=False):
    """
    运行LQR仿真
    
    参数:
    - discrete: bool, 是否使用离散系统
    - output_regulator: bool, 是否使用输出调节器
    """
    regulator_type = "输出调节器" if output_regulator else "状态调节器"
    system_type = "离散" if discrete else "连续"
    print(f"\n============= {regulator_type}仿真, {system_type}时间系统 ===============")

    # 仿真参数
    dt = 0.01
    total_time = 5.0
    time_steps = int(total_time / dt)
    
    # 系统矩阵
    # 双积分器系统：x'' = u
    # 状态方程：[x', v']^T = [0, 1; 0, 0] * [x, v]^T + [0; 1] * u
    A_cont = [[0, 1],
              [0, 0]]
    B_cont = [[0],
              [1]]
    
    # 离散化系统矩阵
    A_dis = np.eye(2) + np.array(A_cont) * dt
    B_dis = np.array(B_cont) * dt

    # 构造LTI系统
    A = A_dis if discrete else A_cont
    B = B_dis if discrete else B_cont
    C = [[1, 0]] if output_regulator else None
    Ts = dt if discrete else None
    sys = LTISystem(A, B, C, Ts=Ts)
    print("能控性:", sys.is_controllable())
    print("能观性:", sys.is_observable())
    print("可镇定性:", sys.is_stabilizable())
    print("可检测性:", sys.is_detectable())
    print("开环稳定:", sys.is_stable())
    print("Laypunov开环稳定:", sys.is_lyapunov_stable(np.eye(2)))

    # LQR权重矩阵
    if output_regulator:
        Q = 10  # 输出权重
    else:
        Q = np.diag([10, 1])  # 状态权重
    R = 0.1
    
    # 初始化LQR控制器
    lqr_controller = LQR(sys, Q, R, dt)
    
    print(f"\n{system_type}系统LQR{regulator_type}参数:")
    print(lqr_controller)
    print("LQR闭环稳定:", lqr_controller.stable)
    print()
    
    # 初始状态：位置为1，速度为0
    x = np.array([1, 0])
    real_y_list = []
    
    # 仿真
    for _ in range(time_steps):
        if output_regulator:
            # 状态估计器 (实现过程略, 这里简单加点噪声表示估计结果)
            x_hat = x + np.random.normal(0, 0.01, size=x.shape)
            # 状态反馈控制
            u = lqr_controller(x_hat)
        else:
            # 直接使用真实状态
            u = lqr_controller(x)
        
        # 更新状态
        if discrete:
            # 离散状态更新
            x = A @ x + B @ u
        else:
            # 连续状态更新 (ode积分)
            x_dot = A @ x + B @ u
            x = x + x_dot * dt
        
        # 记录输出（仅输出调节器需要）
        if output_regulator:
            y = C @ x
            real_y_list.append(y)
    
    # 输出
    name = f'{"Discrete" if discrete else "Continuous"} {"Output" if output_regulator else "State"} Regulator'
    if output_regulator:
        lqr_controller.show(name=name, real_response=real_y_list)
        y = C @ x
        print(f'{system_type}系统最终输出:')
        print(f'  位置: {y[0]:.4f}')
    else:
        lqr_controller.show(name=name)
    
    print(f'{system_type}系统最终状态:')
    print(f'  位置: {x[0]:.4f}')
    print(f'  速度: {x[1]:.4f}')
    print(f'{system_type}系统性能指标J:')
    print(f'  J = {lqr_controller.J:.4f}')



if __name__ == '__main__':
    # 连续系统状态调节器
    with matplotlib_context():
        run_lqr_simulation(discrete=False, output_regulator=False)
    
    # 离散系统状态调节器
    with matplotlib_context():
        run_lqr_simulation(discrete=True, output_regulator=False)
    
    # 连续系统输出调节器
    with matplotlib_context():
        run_lqr_simulation(discrete=False, output_regulator=True)
    
    # 离散系统输出调节器
    with matplotlib_context():
        run_lqr_simulation(discrete=True, output_regulator=True)
