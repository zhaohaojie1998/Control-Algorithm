# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:27:34 2022

@author: HJ
"""

''' ADRC '''
# model free controller
from typing import Union
from dataclasses import dataclass
import numpy as np

if __name__ == '__main__':
    import sys, os
    ctrl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ctrl包所在的目录
    sys.path.append(ctrl_dir)

from ctrl.common import BaseController, SignalLike, ListLike, NdArray
from ctrl.demo import *

__all__ = ['ADRCConfig', 'ADRC']


# ADRC控制器参数
@dataclass
class ADRCConfig:
    """ADRC自抗扰控制算法参数
    :param dt: float, 控制器步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, ADRC输出u也为dim维
    :param h: float, 跟踪微分器(TD)滤波因子, 系统调用步长, 默认None设置成dt
    :param r: SignalLike, 跟踪微分器(TD)快速跟踪因子
    :param b0: SignalLike, 扩张状态观测器(ESO)被控系统系数
    :param delta: SignalLike, ESO的fal(e, alpha, delta)函数线性区间宽度
    :param beta01: SignalLike, ESO的反馈增益1
    :param beta02: SignalLike, ESO的反馈增益2
    :param beta03: SignalLike, ESO的反馈增益3
    :param beta1: SignalLike, NLSEF参数, 跟踪输入信号的增益
    :param beta2: SignalLike, NLSEF参数, 跟踪微分信号的增益
    :param alpha1: SignalLike, 非线性反馈控制律(NLSEF)参数, 0 < alpha1 < 1
    :param alpha2: SignalLike, NLSEF参数, alpha2 > 1
    :Type : SignalLike = float (标量) 或 list / ndarray (一维数组即向量)\n
    备注:\n
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的ADRC控制器, 必须满足dim==len(SignalLike)\n
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的ADRC控制器, 控制效果可能不好\n
    """

    dt: float = 0.001              # 控制器步长 (float)
    dim: int = 1                   # 输入维度 (int)
    # 跟踪微分器
    h: float = None                # 滤波因子，系统调用步长，默认None设置成dt (float)
    r: SignalLike = 100.           # 快速跟踪因子 (float or list)
    # 扩张状态观测器
    b0: SignalLike = 133.          # 被控系统系数 (float or list)
    delta: SignalLike = 0.015      # fal(e, alpha, delta)函数线性区间宽度 (float or list)
    eso_beta01: SignalLike = 150.  # ESO反馈增益1 (float or list)
    eso_beta02: SignalLike = 250.  # ESO反馈增益2 (float or list)
    eso_beta03: SignalLike = 550.  # ESO反馈增益3 (float or list)
    # 非线性状态反馈控制率
    nlsef_beta1: SignalLike = 10.        # 跟踪输入信号增益 (float or list)
    nlsef_beta2: SignalLike = 0.0009     # 跟踪微分信号增益 (float or list)
    nlsef_alpha1: SignalLike = 200/201   # 0 < alpha1 < 1  (float or list)
    nlsef_alpha2: SignalLike = 201/200   # alpha2 > 1      (float or list)
    # 控制约束
    u_max: SignalLike = float('inf')   # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
    u_min: SignalLike = float('-inf')  # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)

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
        self.r = self._reshape_param(cfg.r, self.dim) # 快速跟踪因子
        self.h = self._reshape_param(cfg.h, self.dim) # 滤波因子，系统调用步长
        # ESO超参
        self.b0 = self._reshape_param(cfg.b0, self.dim)             # 系统系数
        self.delta = self._reshape_param(cfg.delta, self.dim)       # fal(e, alpha, delta)函数线性区间宽度        
        self.beta01 = self._reshape_param(cfg.eso_beta01, self.dim) # ESO反馈增益1
        self.beta02 = self._reshape_param(cfg.eso_beta02, self.dim) # ESO反馈增益2
        self.beta03 = self._reshape_param(cfg.eso_beta03, self.dim) # ESO反馈增益3
        # NLSEF超参
        self.beta1 = self._reshape_param(cfg.nlsef_beta1, self.dim)   # 跟踪输入信号增益
        self.beta2 = self._reshape_param(cfg.nlsef_beta2, self.dim)   # 跟踪微分信号增益
        self.alpha1 = self._reshape_param(cfg.nlsef_alpha1, self.dim) # 0 < alpha1 < 1 < alpha2
        self.alpha2 = self._reshape_param(cfg.nlsef_alpha2, self.dim) # alpha2 > 1
        # 控制约束
        self.u_max = self._reshape_param(cfg.u_max, self.dim)
        self.u_min = self._reshape_param(cfg.u_min, self.dim)
        # 控制器初始化
        self.v1 = np.zeros(self.dim)
        self.v2 = np.zeros(self.dim)
        self.z1 = np.zeros(self.dim)
        self.z2 = np.zeros(self.dim)
        self.z3 = np.zeros(self.dim)
        self.u = np.zeros(self.dim)
        self.t = 0
        
        # 存储器
        self.logger.v1 = []    # 观测
        self.logger.e1 = []    # 误差1
        self.logger.e2 = []    # 误差2
        self.logger.z3 = []    # 干扰
    

    # ADRC控制器（v为参考轨迹，y为实际轨迹）
    def __call__(self, v, y, *, ctrl_method=1):
        v = np.array(v)
        y = np.array(y)
        # TD
        self._TD(v)
        # ESO
        self._ESO(y)
        self.z1 = np.nan_to_num(self.z1)
        self.z2 = np.nan_to_num(self.z2)
        self.z3 = np.nan_to_num(self.z3)
        # NLSEF
        e1 = self.v1 - self.z1
        e2 = self.v2 - self.z2
        u0 = self._NLSEF(e1, e2, ctrl_method)
        # 控制量
        self.u = u0 - self.z3 / self.b0
        self.u = np.clip(self.u, self.u_min, self.u_max)
        self.t += self.dt
        
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)    
        self.logger.v1.append(self.v1)
        self.logger.e1.append(self.v1 - self.z1)
        self.logger.e2.append(self.v2 - self.z2)
        self.logger.z3.append(self.z3)
        return self.u
    

    # 跟踪微分器
    def _TD(self, v: NdArray):
        fh = self._fhan(self.v1 - v, self.v2, self.r, self.h)
        self.v1 = self.v1 + self.h * self.v2
        self.v2 = self.v2 + self.h * fh
    

    # 扩张状态观测器
    def _ESO(self, y: NdArray):
        e = self.z1 - y
        fe = self._fal(e, 1/2, self.delta)
        fe1 = self._fal(e, 1/4, self.delta)
        self.z1 = self.z1 + self.h * (self.z2 - self.beta01 * e)
        self.z2 = self.z2 + self.h * (self.z3 - self.beta02 * fe + self.b0 * self.u)
        self.z3 = self.z3 + self.h * (- self.beta03 * fe1)
    

    # 非线性状态误差反馈控制律
    def _NLSEF(self, e1: NdArray, e2: NdArray, ctrl_method=1):
        ctrl_method %= 4
        if ctrl_method == 0:
            u0 = self.beta1 * e1 + self.beta2 * e2
        elif ctrl_method == 1:
            u0 = self.beta1 * self._fal(e1, self.alpha1, self.delta) + self.beta2 * self._fal(e2, self.alpha2, self.delta)
        elif ctrl_method == 2:
            u0 = -self.fhan(e1, e2, self.r, self.h)
        else:
            c = 1.5
            u0 = -self.fhan(e1, c*e2, self.r, self.h)
        return u0 # (dim, )
    

    @staticmethod
    def _fhan(x1: NdArray, x2: NdArray, r: NdArray, h: NdArray):
        def fsg(x, d):
            return (np.sign(x + d) - np.sign(x - d)) / 2
        d = r * h**2
        a0 = h * x2
        y = x1 + a0
        a1 = np.sqrt(d * (d + 8*abs(y)) + 1e-8)
        a2 = a0 + np.sign(y) * (a1 - d) / 2
        a = (a0 + y) * fsg(y, d) + a2 * (1 - fsg(y, d))
        fh = -r * (a/d) * fsg(y, d) - r * np.sign(a) * (1 - fsg(a, d))
        return fh
    
    @staticmethod
    def _fal(err: NdArray, alpha: Union[NdArray, float], delta: NdArray):
        if not isinstance(alpha, NdArray):
            alpha = np.ones_like(err) * alpha
        fa = np.zeros_like(err)
        mask = np.abs(err) <= delta
        fa[mask] = err[mask] / delta[mask]**(alpha[mask] - 1)
        fa[~mask] = np.abs(err[~mask])**alpha[~mask] * np.sign(err[~mask])
        return fa
    

    # 输出
    def show(self, interference: ListLike = None, *, save=False, show_img=True):
        """控制器控制效果绘图输出
        :param interference: ListLike, 实际干扰数据, 用于对比ADRC控制器估计的干扰是否准确
        :param save: bool, 是否存储绘图
        :param show_img: bool, 是否CMD输出图像
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
        if show_img:
            self._show_img()






'debug'
if __name__ == '__main__':
    with_noise = True
    cfg = ADRCConfig()
    StepDemo(ADRC, cfg, with_noise)
    CosDemo(ADRC, cfg, with_noise)
