# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:43:28 2022

@author: HJ
"""

''' PID '''
# model free controller

import pylab as pl
from pylab import inf
from copy import deepcopy
from common import BaseController, demo0


class PIDConfig:
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 1           # 控制器维度 (int)
        # PID控制器增益
        self.Kp = 5           # 比例增益 (float or list)
        self.Ki = 0.001       # 积分增益 (float or list)
        self.Kd = 10          # 微分增益 (float or list)
        # 抗积分饱和
        self.u_max = inf      # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
        self.u_min = -inf     # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)
        self.Kaw = 0.2        # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和 (float or list)
        self.max_err = inf    # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器 (float or list)



''' 位置式PID控制算法 '''
class PID(BaseController):
    def __init__(self, cfg):
        super().__init__()
        self.name = 'PID'      # 算法名称
        self.dt = cfg.dt       # 仿真步长
        self.dim = cfg.dim     # 控制器维度n
        
        # PID超参（不需要遍历的数据设置为一维数组）
        self.Kp = pl.array(cfg.Kp).flatten() # Kp array(n,) or array(1,)
        self.Ki = pl.array(cfg.Ki).flatten() # Ki array(n,) or array(1,)
        self.Kd = pl.array(cfg.Kd).flatten() # Kd array(n,) or array(1,)
        self.Kaw = pl.array(cfg.Kaw).flatten() / self.Kd # Kaw取 0.1~0.3 Kd
        
        # 抗积分饱和PID（需要遍历的数据设置为一维数组，且维度保持和dim一致）
        self.u_max = pl.array(cfg.u_max).flatten() # array(1,) or array(n,)
        self.u_max = self.u_max.repeat(self.dim) if len(self.u_max) == 1 else self.u_max # array(n,)
        self.u_min = pl.array(cfg.u_min).flatten() # array(1,) or array(n,)
        self.u_min = self.u_min.repeat(self.dim) if len(self.u_min) == 1 else self.u_min # array(n,)
        self.max_err = pl.array(cfg.max_err).flatten() # array(1,) or array(n,)
        self.max_err = self.max_err.repeat(self.dim) if len(self.max_err) == 1 else self.u_min # array(n,)
        
        # 控制器初始化
        self.u = pl.zeros(self.dim)            # array(n,)
        self.error_last = pl.zeros(self.dim)   # array(n,)
        self.integration = pl.zeros(self.dim)  # array(n,)
        self.t = 0
        
        # 存储器
        self.list_e = []    # 误差
        self.list_d = []    # 误差微分
        self.list_i = []    # 误差积分
    
    # PID控制器（v为参考轨迹，y为实际轨迹或其观测值）
    def __call__(self, v, y):
        # 计算PID误差
        error = pl.array(v - y).flatten()              # P偏差 array(n,)
        differential = error - self.error_last         # D偏差 array(n,)
        
        # 抗积分饱和算法
        beta = self.anti_integral_windup(error, method=2) # 积分分离参数 array(n,)
        
        # 控制量
        self.u = self.Kp * error + beta * self.Ki * self.integration + self.Kd * differential
        self.u = pl.clip(self.u, self.u_min, self.u_max)
        self.error_last = deepcopy(error)
        
        # 存储绘图数据
        self.list_t.append(self.t)
        self.list_u.append(self.u)
        self.list_y.append(y)
        self.list_v.append(v)
        self.list_e.append(error)
        self.list_d.append(differential)
        self.list_i.append(self.integration)
        
        self.t += self.dt
        return self.u
    
    # 抗积分饱和算法 + 积分分离
    def anti_integral_windup(self, error, method = 2):
        beta = pl.zeros(self.dim) # 积分分离参数
        gamma = pl.zeros(self.dim) if method < 2 else None # 方法1的抗积分饱和参数
        for i in range(self.dim):
            # 积分分离，误差超限去掉积分控制
            beta[i] = 0 if abs(error[i]) > self.max_err[i] else 1 
            
            # 算法1
            if method < 2:
                # 控制超上限累加负偏差，误差超限不累加
                if self.u[i] > self.u_max[i]:
                    if error[i] < 0:
                        gamma[i] = 1 # 负偏差累加
                    else:
                        gamma[i] = 0 # 正偏差不累加
                # 控制超下限累加正偏差，误差超限不累加
                elif self.u[i] < self.u_max[i]:
                    if error[i] > 0:
                        gamma[i] = 1 # 正偏差累加
                    else:
                        gamma[i] = 0 # 负偏差不累加
                else:
                    gamma[i] = 1 # 控制不超限，正常累加偏差
                #end if
            #end if
        #end for
                
        # 抗饱和算法1
        self.integration += error if method > 1 else beta * gamma * error # 正常积分PID
        # self.integration += error/2 if method > 1 else beta * gamma * error/2 # 梯形积分PID
        
        # 反馈抑制抗饱和算法 back-calculation
        if method > 1:
            antiWindupError = pl.clip(self.u, self.u_min, self.u_max) - self.u
            self.integration += self.Kaw * antiWindupError # 累计误差加上个控制偏差的反馈量
        
        return beta
            
    
    def show(self, save = False):
        # 响应曲线 与 控制曲线
        self.basic_plot(save)
        
        # 误差曲线
        self._figure(fig_name='Error Curve', t=self.list_t,
                     y1=self.list_e, y1_label='error',
                     xlabel='time', ylabel='error signal', save=save)
        self._figure(fig_name='Differential of Error Curve', t=self.list_t,
                     y1=self.list_d, y1_label='differential of error',
                     xlabel='time', ylabel='error differential signal', save=save)
        self._figure(fig_name='Integration of Error Curve', t=self.list_t,
                     y1=self.list_i, y1_label='integration of error',
                     xlabel='time', ylabel='error integration signal', save=save)
        
        # 理想轨迹跟踪曲线
        self._figure3D('轨迹跟踪控制', save=save)
        
        # 显示图像
        pl.show()
        
        
        
        
''' 增量式PID控制算法 '''
class IncrementPID(PID):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.name = 'IncrementPID'             # 算法名称
        self.error_last2 = pl.zeros(self.dim)  # e(k-2)
        self.error_sum = pl.zeros(self.dim)    # 这里integration是积分增量,error_sum是积分
        
    def __call__(self, v, y):
        # 计算PID误差
        error = pl.array(v - y).flatten()              # P偏差 array(n,)
        differential = error - self.error_last         # D偏差 array(n,)
        
        # 抗积分饱和算法
        self.integration = pl.zeros(self.dim)             # 积分增量 integration = error - 反馈信号
        beta = self.anti_integral_windup(error, method=2) # 积分分离参数 array(n,)
        
        # 控制量
        u0 = self.Kp * (error - self.error_last) + beta * self.Ki * self.integration \
             + self.Kd * (error - 2*self.error_last + self.error_last2)
        self.u = u0 + self.u # 增量式PID对u进行clip后有超调
        
        self.error_last2 = deepcopy(self.error_last)
        self.error_last = deepcopy(error)
        self.error_sum += self.integration         # 积分绘图用
        
        # 存储绘图数据
        self.list_t.append(self.t)
        self.list_u.append(self.u)
        self.list_y.append(y)
        self.list_v.append(v)
        self.list_e.append(error)
        self.list_d.append(differential)
        self.list_i.append(self.error_sum)
        
        self.t += self.dt
        return self.u
        
    
    

'debug'
if __name__ == '__main__':
    cfg = PIDConfig()
    demo0(PID, cfg)
