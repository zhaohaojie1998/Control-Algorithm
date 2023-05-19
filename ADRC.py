# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:27:34 2022

@author: HJ
"""

''' ADRC '''
# model free controller

import pylab as pl
from pylab import sign, sqrt
from dataclasses import dataclass
from common import BaseController, SignalLike, ListLike, StepDemo


# ADRC控制器参数
@dataclass
class ADRCConfig:
    """ADRC自抗扰控制算法参数
    :param dt: float, 仿真步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, ADRC输出u也为dim维
    :param h: float, 跟踪微分器(TD)滤波因子, 系统调用步长, 默认None设置成dt
    :param r: SignalLike, 跟踪微分器(TD)快速跟踪因子
    :param b0: SignalLike, 扩张状态观测器(ESO)被控系统系数
    :param delta: SignalLike, ESO的fal(e, alpha, delta)函数线性区间宽度
    :param beta01: SignalLike, ESO的反馈增益1
    :param beta02: SignalLike, ESO的反馈增益2
    :param beta03: SignalLike, ESO的反馈增益3
    :param alpha1: SignalLike, 非线性反馈控制律(NLSEF)参数, 0 < alpha1 < 1
    :param alpha2: SignalLike, NLSEF参数, alpha2 > 1
    :param beta1: SignalLike, NLSEF参数, 跟踪输入信号的增益
    :param beta2: SignalLike, NLSEF参数, 跟踪微分信号的增益
    :Type : SignalLike = float (标量) 或 list / ndarray (一维数组即向量)\n
    备注:\n
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的控制器, 必须满足dim==len(SignalLike)\n
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的控制器, 控制效果可能不好\n
    """

    dt: float = 0.001              # 仿真步长 (float)
    dim: int = 1                   # 输入维度 (int)
    # 跟踪微分器
    h: float = None                # 滤波因子，系统调用步长，默认None设置成dt (float)
    r: SignalLike = 100.           # 快速跟踪因子 (float or list)
    # 扩张状态观测器
    b0: SignalLike = 133.          # 被控系统系数 (float or list)
    delta: SignalLike = 0.015      # fal(e, alpha, delta)函数线性区间宽度 (float or list)
    beta01: SignalLike = 150.      # ESO反馈增益1 (float or list)
    beta02: SignalLike = 250.      # ESO反馈增益2 (float or list)
    beta03: SignalLike = 550.      # ESO反馈增益3 (float or list)
    # 非线性状态反馈控制率
    alpha1: SignalLike = 200/201   # 0 < alpha1 < 1  (float or list)
    alpha2: SignalLike = 201/200   # alpha2 > 1      (float or list)
    beta1: SignalLike = 10.        # 跟踪输入信号增益 (float or list)
    beta2: SignalLike = 0.0009     # 跟踪微分信号增益 (float or list)

    def __post_init__(self):
        if self.h is None:
            self.h = self.dt



# ADRC自抗扰算法
class ADRC(BaseController):
    """ADRC自抗扰控制"""

    def __init__(self, cfg: ADRCConfig):
        super().__init__()
        self.name = 'ADRC'       # 算法名称
        self.dt = cfg.dt         # 仿真步长
        self.dim = cfg.dim       # 反馈信号y和跟踪信号v的维度
        # TD超参
        self.r = pl.array(cfg.r).flatten()           # 快速跟踪因子
        self.h = pl.array(cfg.h).flatten()           # 滤波因子，系统调用步长
        # ESO超参
        self.b0 = pl.array(cfg.b0).flatten()         # 系统系数
        self.delta = pl.array(cfg.delta).flatten()   # fal(e, alpha, delta)函数线性区间宽度        
        self.beta01 = pl.array(cfg.beta01).flatten() # ESO反馈增益1
        self.beta02 = pl.array(cfg.beta02).flatten() # ESO反馈增益2
        self.beta03 = pl.array(cfg.beta03).flatten() # ESO反馈增益3
        # NLSEF超参
        self.alpha1 = pl.array(cfg.alpha1).flatten() # 0 < alpha1 < 1 < alpha2
        self.alpha2 = pl.array(cfg.alpha2).flatten() # alpha2 > 1
        self.beta1 = pl.array(cfg.beta1).flatten()   # 跟踪输入信号增益
        self.beta2 = pl.array(cfg.beta2).flatten()   # 跟踪微分信号增益
        
        # 控制器初始化
        self.v1 = pl.zeros(self.dim) # array(dim,)
        self.v2 = pl.zeros(self.dim) # array(dim,)
        self.z1 = pl.zeros(self.dim) # array(dim,)
        self.z2 = pl.zeros(self.dim) # array(dim,)
        self.z3 = pl.zeros(self.dim) # array(dim,)
        self.u = pl.zeros(self.dim) # array(dim,)
        self.t = 0
        
        # 存储器
        self.logger.v1 = []    # 观测
        self.logger.e1 = []    # 误差1
        self.logger.e2 = []    # 误差2
        self.logger.z3 = []    # 干扰
    
    # ADRC控制器（v为参考轨迹，y为实际轨迹）
    def __call__(self, v, y):
        # TD
        self._TD(v)
        # ESO
        self._ESO(y)
        self.z1 = pl.nan_to_num(self.z1)
        self.z2 = pl.nan_to_num(self.z2)
        self.z3 = pl.nan_to_num(self.z3)
        # NLSEF
        e1 = self.v1 - self.z1
        e2 = self.v2 - self.z2
        u0 = self._NLSEF(e1, e2)
        # 控制量
        self.u = u0 - self.z3 / self.b0
        
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)    
        self.logger.v1.append(self.v1)
        self.logger.e1.append(self.v1 - self.z1)
        self.logger.e2.append(self.v2 - self.z2)
        self.logger.z3.append(self.z3)
        self.t += self.dt
        return self.u
    
    # 跟踪微分器
    def _TD(self, v):
        fh = self._fhan(self.v1 - v, self.v2, self.r, self.h)
        self.v1 = self.v1 + self.h * self.v2
        self.v2 = self.v2 + self.h * fh
    
    # 扩张状态观测器
    def _ESO(self, y):
        e = self.z1 - y
        fe = self._fal(e, 1/2, self.delta)
        fe1 = self._fal(e, 1/4, self.delta)
        self.z1 = self.z1 + self.h * (self.z2 - self.beta01 * e)
        self.z2 = self.z2 + self.h * (self.z3 - self.beta02 * fe + self.b0 * self.u)
        self.z3 = self.z3 + self.h * (- self.beta03 * fe1)
    
    # 非线性状态反馈控制率
    def _NLSEF(self, e1, e2):
        # u0 = self.beta1 * e1 + self.beta2 * e2
        u0 = self.beta1 * self._fal(e1, self.alpha1, self.delta) + self.beta2 * self._fal(e2, self.alpha2, self.delta)
        # u0 = -self.fhan(e1, e2, self.r, self.h)
        # c = 1.5
        # u0 = -self.fhan(e1, c*e2, self.r, self.h)
        return u0 # (dim, )
    
    def _fhan(self, x1, x2, r, h):
        def fsg(x, d):
            return (sign(x + d) - sign(x - d)) / 2
        d = r * h**2  # array(dim,)
        a0 = h * x2   # array(dim,)
        y = x1 + a0   # array(dim,)
        a1 = sqrt(d * (d + 8*abs(y)))  # array(dim,)
        a2 = a0 + sign(y) * (a1 - d) / 2  # array(dim,)
        a = (a0 + y) * fsg(y, d) + a2 * (1 - fsg(y, d))  # array(dim,)
        fh = -r * (a/d) * fsg(y, d) - r * sign(a) * (1 - fsg(a, d))  # array(dim,)
        return fh
    
    def _fal(self, e, alpha, delta):
        ##  alpha和delta维度可以为1，也可以为dim    ##
        ##  数据类型可以为 int float list array   ##
        alpha = pl.array(alpha).flatten() # array(m,) m = 1 or m = dim
        delta = pl.array(delta).flatten() # array(m,) m = 1 or m = dim
        alpha = alpha.repeat(self.dim) if len(alpha) == 1 else alpha # array(dim,)
        delta = delta.repeat(self.dim) if len(delta) == 1 else delta # array(dim,)
        
        fa = pl.zeros(self.dim) # array(dim,)
        for i in range(self.dim):
            if abs(e[i]) <= delta[i]:
                fa[i] = e[i] / delta[i]**(alpha[i]-1)
            else:
                fa[i] = abs(e[i])**alpha[i] * sign(e[i])
        return fa
    
    def show(self, *, save: bool = False, interference: ListLike = None):
        """控制器控制效果绘图输出
        :param save: bool, 是否存储绘图
        :param interference: ListLike, 实际干扰数据, 用于对比ADRC控制器估计的干扰是否准确
        """
        # 响应曲线 与 控制曲线
        super().show(save=save)
        
        # TD曲线
        self._figure(fig_name='Tracking Differentiator (TD)', t=self.logger.t,
                     y1=self.logger.v1, y1_label='td',
                     y2=self.logger.v, y2_label='input',
                     xlabel='time', ylabel='response signal', save=save)
        
        # 误差曲线
        self._figure(fig_name='Error Curve', t=self.logger.t,
                     y1=self.logger.e1, y1_label='error',
                     xlabel='time', ylabel='error signal', save=save)
        
        self._figure(fig_name='Differential of Error Curve', t=self.logger.t,
                     y1=self.logger.e2, y1_label='differential estimation of error',
                     xlabel='time', ylabel='error differential signal', save=save)
        
        # 干扰估计曲线
        if interference is not None:
            interference = interference if len(interference) == len(self.logger.t) else None
        self._figure(fig_name='Interference Estimation', t=self.logger.t,
                     y1=self.logger.z3, y1_label='interference estimation',
                     y2=interference, y2_label='real interference',
                     xlabel='time', ylabel='interference signal', save=save)
        
        # 显示图像
        pl.show()
            



__all__ = ['ADRCConfig', 'ADRC']




'debug'
if __name__ == '__main__':
    cfg = ADRCConfig()
    StepDemo(ADRC, cfg)
