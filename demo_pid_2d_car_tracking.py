"""2D轨迹跟踪demo - 二轮小车"""
from controller import PID, PIDConfig
from math import inf, cos, sin, sqrt
import numpy as np

WITH_NOISE = False # 是否存在干扰
DT = 0.01  # 步长

# 位置控制
pos_pid_cfg = PIDConfig(
    name = "Position",
    dt = DT,  # 控制器步长
    dim = 2,  # 输入维度 (x, y)
    # PID控制器增益
    Kp = [1.0, 1.0],     # 比例增益 [x, y]
    Ki = [0.0, 0.0],     # 积分增益
    Kd = [0.5, 0.5],     # 微分增益
    # 抗积分饱和
    u_max = [1.0, 1.0],  # 控制律上限 [x, y]
    u_min = [-1.0, -1.0], # 控制律下限 [x, y]
    Kaw = [0.2, 0.2],    # 抗饱和参数
    ins_max_err = [inf, inf], # 积分器分离阈值
)

# 速度控制
speed_pid_cfg = PIDConfig(
    name = "Speed",
    dt = DT,  # 控制器步长
    dim = 1,  # 输入维度 (v)
    # PID控制器增益
    Kp = [1.0],         # 比例增益
    Ki = [0.0],         # 积分增益
    Kd = [0.5],         # 微分增益
    # 抗积分饱和
    u_max = [2.0],      # 控制律上限
    u_min = [0.0],      # 控制律下限
    Kaw = [0.2],        # 抗饱和参数
    ins_max_err = [inf], # 积分器分离阈值
)

# 初始化PID控制器
pos_controller = PID(pos_pid_cfg)  # 位置控制器
speed_controller = PID(speed_pid_cfg)  # 速度控制器


#----------------------------- ↓↓↓↓↓ 二轮小车动力学模型 ↓↓↓↓↓ ------------------------------#
MAX_SPEED = 2.0  # 最大线速度
MAX_ANGULAR_SPEED = 3.0  # 最大角速度

class TwoWheelCarModel:
    """二轮小车（差分驱动）动力学模型
    
    状态变量:
    - x: x轴位置
    - y: y轴位置
    - θ: 朝向角度
    - v: 线速度
    - ω: 角速度
    
    控制输入:
    - u: [线速度指令, 角速度指令]
    """

    def __init__(self, dt, with_noise=True):
        self.dt = dt
        self.t = 0
        self.with_noise = with_noise
        # 初始状态 [x, y, θ, v, ω]
        self.s = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.u = np.array([0.0, 0.0], dtype=np.float32)  # 初始控制
    
    def __call__(self, u):
        self.t += self.dt
        u = np.clip(u, [-MAX_SPEED, -MAX_ANGULAR_SPEED], [MAX_SPEED, MAX_ANGULAR_SPEED])
        self.u = u
        self.s = self.ode_model(self.s, u)
        return self.position

    @property
    def states(self):
        """小车状态"""
        return self.s
    
    @property
    def position(self):
        """小车位置"""
        return self.s[:2]
    
    @property
    def orientation(self):
        """小车朝向"""
        return self.s[2]
    
    @property
    def velocity(self):
        """小车速度"""
        return self.s[3:]
    
    @property
    def control(self):
        """小车控制输入"""
        return self.u

    def ode_model(self, s, u):
        """动力学模型
        
        状态更新方程:
        >>> dx/dt = v * cos(θ)
        >>> dy/dt = v * sin(θ)
        >>> dθ/dt = ω
        >>> dv/dt = (u[0] - v) / τ
        >>> dω/dt = (u[1] - ω) / τ
        
        其中τ是时间常数，这里取0.1
        """
        x, y, θ, v, ω = s
        v_cmd, ω_cmd = u
        
        # 时间常数
        τ = 0.1
        
        # 状态更新
        x_new = x + v * cos(θ) * self.dt
        y_new = y + v * sin(θ) * self.dt
        θ_new = θ + ω * self.dt
        v_new = v + (v_cmd - v) / τ * self.dt
        ω_new = ω + (ω_cmd - ω) / τ * self.dt

        θ_new = np.arctan2(sin(θ_new), cos(θ_new)) # 归一化角度到 [-π, π]

        s_new = np.array([x_new, y_new, θ_new, v_new, ω_new])
        
        # 添加噪声
        if self.with_noise:
            noise = np.array([0.001, 0.001, 0.0005, 0.01, 0.01]) * np.random.randn(5)
            s_new += noise
        
        return s_new


#----------------------------- ↓↓↓↓↓ 参考轨迹设置 ↓↓↓↓↓ ------------------------------#
t_list = np.arange(0.0, 25.0, DT)
v_list = np.zeros((2, len(t_list))) # 参考速度轨迹

radius = 5.0  # 圆半径
omega = 0.2   # 角速度

for i, t in enumerate(t_list):
    # 参考位置
    ref_x = radius * cos(omega * t)
    ref_y = radius * sin(omega * t)
    
    # 参考速度
    ref_vx = -radius * omega * sin(omega * t)
    ref_vy = radius * omega * cos(omega * t)
    
    # 计算参考线速度和角速度
    ref_v = np.sqrt(ref_vx**2 + ref_vy**2)
    ref_theta = np.arctan2(ref_vy, ref_vx)
    
    # 下一个参考角度
    next_t = t + DT
    next_ref_x = radius * cos(omega * next_t)
    next_ref_y = radius * sin(omega * next_t)
    next_ref_theta = np.arctan2(next_ref_y - ref_y, next_ref_x - ref_x)
    
    # 计算参考角速度
    ref_omega = (next_ref_theta - ref_theta) / DT
    
    # 存储参考速度 [线速度, 角速度]
    v_list[:, i] = [ref_v, ref_omega]


#----------------------------- ↓↓↓↓↓ 轨迹跟踪控制仿真 ↓↓↓↓↓ ------------------------------#
plant = TwoWheelCarModel(DT, WITH_NOISE)

for i in range(len(t_list)):
    t = t_list[i]
    # 获取当前状态
    current_pos = plant.position
    current_theta = plant.orientation
    current_speed = plant.velocity[0]
    
    # 计算参考位置
    ref_x = radius * cos(omega * t)
    ref_y = radius * sin(omega * t)
    ref_pos = np.array([ref_x, ref_y])
    
    # 计算参考速度向量
    ref_vx = -radius * omega * np.sin(omega * t)
    ref_vy = radius * omega * np.cos(omega * t)
    ref_speed = np.sqrt(ref_vx**2 + ref_vy**2)
    
    # 步骤1：使用位置PID控制器计算位置误差控制量
    pos_control = pos_controller(ref_pos, current_pos)
    
    # 步骤2：计算目标速度向量 (位置控制量与参考速度向量相加)
    target_vx = ref_vx + pos_control[0]
    target_vy = ref_vy + pos_control[1]
    
    # 步骤3：计算目标速度和目标方向
    target_speed = np.sqrt(target_vx**2 + target_vy**2)
    target_speed = np.clip(target_speed, 0, 2.0)
    target_dir = np.arctan2(target_vy, target_vx)
    
    # 步骤4：使用速度PID控制器计算速度控制量
    speed_control = speed_controller([target_speed], [current_speed])[0]
    speed_control = np.clip(speed_control, 0, 2.0)
    
    # 步骤6：计算角度误差, 归一化到 [-π, π]
    angle_error = target_dir - current_theta
    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
    
    # 步骤7：计算角速度控制量
    angular_speed = 2.0 * angle_error
    angular_speed = np.clip(angular_speed, -2.0, 2.0)
    
    # 最终控制输入
    final_control = np.array([speed_control, angular_speed])
    
    # 更新状态
    plant(final_control)
    
    # 打印调试信息
    if i % 100 == 0:
        distance_error = np.sqrt((ref_pos[0] - current_pos[0])**2 + (ref_pos[1] - current_pos[1])**2)
        print(f"Time: {t:.2f}, Position: ({current_pos[0]:.2f}, {current_pos[1]:.2f}), Error: {distance_error:.2f}, Speed: {current_speed:.2f}, Target Speed: {target_speed:.2f}")

# 显示控制器信息
pos_controller.show(save_img=True)
speed_controller.show(save_img=True)
