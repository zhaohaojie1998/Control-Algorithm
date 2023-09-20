# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:19:37 2022

@author: ZHAOHAOJIE
"""

''' 无人机轨迹跟踪控制 demo '''
from ctrl import utils
from ctrl import ADRC
from ctrl import PID, IncrementPID
from math import inf
import numpy as np



# 选择你的唱跳rap篮球
CTRL = 'ADRC'
#CTRL = 'PID'
#CTRL = 'IncrementPID'
WITH_NOISE = True # 飞行器是否受到扰动, 设置成False得重新调参


# ADRC调参
ADRCConfig = ADRC.getConfig()
adrc_cfg = ADRCConfig(
    dt = 0.001,
    dim = 3,
    # 跟踪微分器
    r = 60,                 # 快速跟踪因子
    # 扩张状态观测器
    b0 = 1,                 # 被控系统系数
    delta = 0.015,          # fal(e, alpha, delta)函数线性区间宽度
    eso_beta01 = 150,           # ESO反馈增益1
    eso_beta02 = 250,           # ESO反馈增益2
    eso_beta03 = [350,400,500], # ESO反馈增益3
    # 非线性状态反馈控制率
    nlsef_beta1 = 20,                   # 跟踪输入信号增益1
    nlsef_beta2 = [10,15,20],           # 跟踪输入信号增益2
    nlsef_alpha1 = 200/201,             # 0 < alpha1 < 1
    nlsef_alpha2 = [201/200,1.01,1.02], # alpha2 > 1 
)


# PID调参
PIDConfig = PID.getConfig()
pid_cfg = PIDConfig(
    dt = 0.001,
    dim = 3,
    # PID控制器增益
    Kp = [30,20,30],       # 比例增益
    Ki = [0.01,0.01,100],  # 积分增益
    Kd = [3000,2800,5000], # 微分增益
    # 抗积分饱和
    u_max = 200,           # 控制律上限
    u_min = -200,          # 控制律下限
    Kaw = 0.2,             # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和
    max_err = [inf,inf,1], # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器
)




if CTRL == 'ADRC':
    dt = adrc_cfg.dt
    ctrl = ADRC(adrc_cfg)
elif CTRL == 'PID':
    dt = pid_cfg.dt
    ctrl = PID(pid_cfg)
else:
    dt = pid_cfg.dt
    ctrl = IncrementPID(pid_cfg)




#----------------------------- ↓↓↓↓↓ 飞行动力学模型 ↓↓↓↓↓ ------------------------------#
ODE_TIMES = 2  # 一个dt区间积分几次
class LinearModel:
    """线性加速度控制模型\n
    东北天坐标系, Oz指天\n
    s = [x, y, z, Vx, Vy, Vz]\n
    u = [ax, ay, az]\n
    """

    def __init__(self):
        self.with_noise = WITH_NOISE
        self.dt = dt
        self.t = 0
        self.u = np.zeros(3, dtype=np.float32)
        self.s = np.array([5,-2,-5,3,3,-2]) #! 初始化状态
    
    def __call__(self, u):
        # 更新状态
        self.t += self.dt
        self.u = u
        self.s = self.ode_model(self.s, u)
        return self.position

    @property
    def states(self):
        """无人机状态"""
        return self.s
    
    @property
    def position(self):
        """无人机位置"""
        return self.s[:3]
    
    @property
    def control(self):
        """无人机加速度控制"""
        return self.u

    def ode_model(self, s, u):
        """
        >>> dx/dt = Vx
        >>> dy/dt = Vy
        >>> dz/dt = Vz
        >>> dVx/dt = ax
        >>> dVy/dt = ay
        >>> dVz/dt = az
        """
        s1 = np.zeros_like(s)
        s1[0] = s[0] + self.dt * s[3]
        s1[1] = s[1] + self.dt * s[4]
        s1[2] = s[2] + self.dt * s[5]
        s1[3] = s[3] + self.dt * u[0]
        s1[4] = s[4] + self.dt * u[1]
        s1[5] = s[5] + self.dt * u[2]
        if self.with_noise:
            return s1 + 0.01*np.random.randn(6)
        return s1




#----------------------------- ↓↓↓↓↓ 参考轨迹设置 ↓↓↓↓↓ ------------------------------#
t_list = np.arange(0.0, 10.0, dt)
vx_list = 10*np.cos(t_list)
vy_list = 10*np.sin(t_list)
vz_list = np.linspace(-10, 10, len(t_list), endpoint=True)
v_list = np.vstack((vx_list,vy_list,vz_list))






#----------------------------- ↓↓↓↓↓ 轨迹跟踪控制仿真 ↓↓↓↓↓ ------------------------------#
plant = LinearModel()
print(ctrl)
utils.tic()
for i in range(len(t_list)):
    t = t_list[i]
    v = v_list[:, i]
    # 更新控制
    u = ctrl(v, plant.position)
    # 更新状态
    plant(u)
utils.toc()
ctrl.show()

