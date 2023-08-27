# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:43:28 2022

@author: HJ
"""

''' Fuzzy PID '''
# model free controller

import pylab as pl
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from dataclasses import dataclass
from copy import deepcopy

if __name__ == '__main__':
    import sys, os
    ctrl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ctrl包所在的目录
    sys.path.append(ctrl_dir)
    
from ctrl.pid import PID, SignalLike
from ctrl.demo import *

__all__ = ['PIDConfig', 'FuzzyPID']


# FuzzyPID控制器参数
@dataclass
class PIDConfig:
    """PID控制算法参数
    :param dt: float, 控制器步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, PID输出u也为dim维
    :param Kp: SignalLike, PID比例增益系数
    :param Ki: SignalLike, PID积分增益系数
    :param Kd: SignalLike, PID微分增益系数
    :param u_max: SignalLike, 控制律上限, 范围: (u_min, inf], 取inf时不设限
    :param u_min: SignalLike, 控制律下限, 范围: [-inf, u_max), 取-inf时不设限
    :param Kaw: SignalLike, 抗积分饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和
    :param max_err: SignalLike, 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器
    :Type : SignalLike = float (标量) | list / ndarray (一维数组即向量)\n
    备注:\n
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的PID控制器, 必须满足dim==len(SignalLike)\n
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的PID控制器, 控制效果可能不好\n
    """
    dt: float = 0.01             # 控制器步长 (float)
    dim: int = 1                 # 输入维度 (int)
    # PID控制器增益
    Kp: SignalLike = 5           # 比例增益 (float or list)
    Ki: SignalLike = 0.1         # 积分增益 (float or list)
    Kd: SignalLike = 0.1         # 微分增益 (float or list)
    # 抗积分饱和
    u_max: SignalLike = pl.inf   # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
    u_min: SignalLike = -pl.inf  # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)
    Kaw: SignalLike = 0.2        # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和 (float or list)
    max_err: SignalLike = pl.inf # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器 (float or list)



# FuzzyPID控制算法
class FuzzyPID(PID):
    """模糊PID控制算法"""

    def __init__(self, cfg: PIDConfig):
        super().__init__(cfg)
        self.name = 'FuzzyPID' # 算法名称
        
        # Fuzzy控制律
        self.Kp_init = deepcopy(self.Kp)
        self.Ki_init = deepcopy(self.Ki)
        self.Kd_init = deepcopy(self.Kd)

        
        # 存储器
        self.logger.kp = []
        self.logger.ki = []
        self.logger.kd = []

    # 模糊PID控制
    def update_gain(self, v, y):
        self.Kp = 0
        self.Ki = 0.0
        self.Kd = 0.0


    # 模糊PID控制器
    def __call__(self, v, y) -> pl.ndarray:
        self.update_gain(v, y)
        self.logger.kp.append(self.Kp)
        self.logger.ki.append(self.Ki)
        self.logger.kd.append(self.Kd)
        return super().__call__(v, y)

            
    # 绘图输出
    def show(self, *, save = False):
        super().show(save=save)
        self._figure(fig_name='Proportional Gain', t=self.logger.t,
                     y1=self.logger.kp, y1_label='Kp',
                     xlabel='time', ylabel='gain', save=save)
        self._figure(fig_name='Differential Gain', t=self.logger.t,
                     y1=self.logger.kd, y1_label='Kd',
                     xlabel='time', ylabel='gain', save=save)
        self._figure(fig_name='Integral Gain', t=self.logger.t,
                     y1=self.logger.ki, y1_label='Ki',
                     xlabel='time', ylabel='gain', save=save)
        pl.show()
        
        
        
        





'debug'
if __name__ == '__main__':
    with_noise = True
    cfg = PIDConfig()
    StepDemo(FuzzyPID, cfg, with_noise)
    CosDemo(FuzzyPID, cfg, with_noise)
