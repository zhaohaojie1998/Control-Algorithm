# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 19:19:37 2022

@author: AI
"""

from ADRC import ADRC
from PID import PID, IncrementPID
import pylab as pl
from pylab import inf
import utils


class ADRCConfig:
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 3           # 控制器维度 (int)
        ''' dim > 1 同时超参数为 list 数据类型时相当于同时设计了 dim 个控制器        '''
        ''' 必须满足 len(超参) == dim 或 len(超参) == 1 或 超参为float类型           '''
        ''' dim > 1 时超参数也可为float，此时相当于设计了1个控制器，控制效果可能不好 '''
        
        # 跟踪微分器
        self.h = self.dt                        # 滤波因子，系统调用步长 (float)
        self.r = 60                             # 快速跟踪因子 (float or list)
        # 扩张状态观测器
        self.b0 = 1                             # 被控系统系数 (float or list)
        self.delta = 0.015                      # fal(e, alpha, delta)函数线性区间宽度 (float or list)
        self.beta01 = 150                       # ESO反馈增益1 (float or list)
        self.beta02 = 250                       # ESO反馈增益2 (float or list)
        self.beta03 = [350,400,500]             # ESO反馈增益3 (float or list)
        # 非线性状态反馈控制率
        self.alpha1 = 200/201                   # 0 < alpha1 < 1   (float or list)
        self.alpha2 = [201/200,1.01,1.02]       # alpha2 > 1       (float or list)
        self.beta1 = 20                         # 跟踪输入信号增益 (float or list)
        self.beta2 = [10,15,20]                 # 跟踪微分信号增益 (float or list)
        
class PIDConfig:
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 3           # 控制器维度 (int)
        # PID控制器增益
        self.Kp = [30,20,30]           # 比例增益 (float or list)
        self.Ki = [0.01,0.01,100]      # 积分增益 (float or list)
        self.Kd = [3000,2800,5000]     # 微分增益 (float or list)
        # 抗积分饱和
        self.u_max = 200               # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
        self.u_min = -200              # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)
        self.Kaw = 0.2                 # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和 (float or list)
        self.max_err = [inf,inf,1]     # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器 (float or list)

class IncrementPIDConfig:
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 3           # 控制器维度 (int)
        # PID控制器增益
        self.Kp = [30,20,30]           # 比例增益 (float or list)
        self.Ki = [0.01,0.01,1000]      # 积分增益 (float or list)
        self.Kd = [3000,2800,5000]     # 微分增益 (float or list)
        # 抗积分饱和
        self.u_max = 200               # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
        self.u_min = -200              # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)
        self.Kaw = 0.2                 # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和 (float or list)
        self.max_err = [inf,inf,1]     # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器 (float or list)
        
        
dt = ADRCConfig().dt
dt = PIDConfig().dt
ctrl = ADRC(cfg = ADRCConfig())
ctrl = PID(cfg = PIDConfig())

        
        
# 三维信号跟踪
def PlantModel(x0, u, t, dt):
    x1 = pl.zeros(6)
    x1[0] = x0[0] + dt * x0[3] + 0.01*pl.randn(1)
    x1[1] = x0[1] + dt * x0[4] + 0.01*pl.randn(1)
    x1[2] = x0[2] + dt * x0[5] + 0.01*pl.randn(1)
    x1[3] = x0[3] + dt * u[0] + 0.01*pl.randn(1)
    x1[4] = x0[4] + dt * u[1] + 0.01*pl.randn(1)
    x1[5] = x0[5] + dt * u[2] + 0.01*pl.randn(1)
    return x1

t_list = pl.arange(0.0, 10.0, dt)
vx_list = 10*pl.cos(t_list)
vy_list = 10*pl.sin(t_list)
vz_list = pl.linspace(-10, 10, len(t_list), endpoint=True)
v_list = pl.vstack((vx_list,vy_list,vz_list))

u = pl.zeros(3)
x = pl.array([5,-2,-5,3,3,-2])
#x = pl.array([10,0,-10,3,3,-2])

print(ctrl)
utils.tic()
for i in range(len(t_list)):
    t = t_list[i]
    v = v_list[:, i]

    # 更新状态
    x = PlantModel(x, u, t, dt)

    # 更新控制
    u = ctrl(v, x[0:3])

    
utils.toc()
ctrl.show()

