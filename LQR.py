# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:19:50 2022

@author: HJ
"""

''' LQR '''
# model based controller

import pylab as pl
from copy import deepcopy
from common import BaseController, SignalLike, StepDemo


# LQR控制器初始参数设置
class LQRConfig:
    """LQR控制算法参数
    """
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 1           # 控制器维度 (int)
        
        self.Q = []
        self.Qf = []
        self.R = []
        
        # model init
        self.A = []
        self.B = []
        
        

# 线性二次型调节器(LQR)控制算法
class LQR(BaseController):
    """线性二次型调节器(LQR)控制算法"""

    def __init__(self, cfg):
        super().__init__()
        self.name = 'LQR'      # 算法名称
        self.dt = cfg.dt       # 仿真步长
        self.dim = cfg.dim     # 状态方程维度n
        

        # 控制器初始化
        self.u = pl.zeros(self.dim)            # array(n,)

        self.t = 0
        
        # 存储器
        self.logger.e = []    # 误差
        self.logger.d = []    # 误差微分
        self.logger.i = []    # 误差积分
    
    # LQR控制器（v为参考轨迹, y为实际轨迹或其观测值, AB为时变状态方程）
    def __call__(self, v:SignalLike, y:SignalLike, At=None, Bt=None):
        x = pl.array(v-y)
        
        # A = # 转换成二维array数组或矩阵
        # B = # 转换成二维array数组或矩阵
        
        self.A = At or self.A
        self.B = Bt or self.B
        P = self._solve_riccati(self.A, self.B)
        
        K = pl.inv(self.R) @ self.B.t @ P  # 矩阵运算
        self.u = -K @ x
        
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)

        
        self.t += self.dt
        return self.u
    
    # 求解黎卡提方程
    def _solve_riccati(self, A, B):
        # A'P + PA + Q - PBR^(-1)B'P = 0
        P = None
        return P
            
    
    def show(self, *, save = False):
        # 响应曲线 与 控制曲线
        super().show(save=save)
        
        # 误差曲线
        
        # 3D数据轨迹跟踪曲线
        self._figure3D(save=save)
        
        # 显示图像
        pl.show()
        
        
        
