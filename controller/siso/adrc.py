# -*- coding: utf-8 -*-
"""
ADRC自抗扰控制器
Created on Sat Jun 18 15:27:34 2022

@author: https://github.com/zhaohaojie1998
"""

''' ADRC '''
# model free controller
from typing import Union, Optional
from dataclasses import dataclass, field
import numpy as np

from ..base import BaseController
from ..types import SignalLike, ListLike, NdArray

__all__ = ['ADRCConfig', 'ADRC']


# ADRC控制器参数
@dataclass
class ADRCConfig:
    """ADRC自抗扰控制算法参数
    
    参数列表：
    - dt: float, 控制器步长, 默认值 0.001
    - dim: int, 输入信号维度, 即控制器输入v、y的维度, ADRC输出u也为dim维, 默认值 1
    - h: float, 跟踪微分器(TD)滤波因子, 系统调用步长, 默认None设置成dt, 默认值 None
    - r: SignalLike, 跟踪微分器(TD)快速跟踪因子, 默认值 100.
    - b0: SignalLike, 扩张状态观测器(ESO)被控系统系数, 默认值 133.
    - delta: SignalLike, fal(e, alpha, delta)函数线性区间宽度, 默认值 0.015
    - eso_beta01: SignalLike, ESO的反馈增益1, 默认值 150.
    - eso_beta02: SignalLike, ESO的反馈增益2, 默认值 250.
    - eso_beta03: SignalLike, ESO的反馈增益3, 默认值 550.
    - nlsef_beta1: SignalLike, NLSEF参数, 跟踪输入信号的增益, 默认值 10.
    - nlsef_beta2: SignalLike, NLSEF参数, 跟踪微分信号的增益, 默认值 0.0009
    - nlsef_alpha1: SignalLike, 非线性反馈控制律(NLSEF)参数, 0 < alpha1 < 1, 默认值 200/201
    - nlsef_alpha2: SignalLike, NLSEF参数, alpha2 > 1, 默认值 201/200
    - u_max: SignalLike, 控制律上限, 范围: (u_min, inf], 取inf时不设限, 默认值 inf
    - u_min: SignalLike, 控制律下限, 范围: [-inf, u_max), 取-inf时不设限, 默认值 -inf
    
    类型说明：
    SignalLike = float (标量) | list / ndarray (一维数组即向量)
    
    备注：
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的ADRC控制器, 必须满足dim==len(SignalLike)
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的ADRC控制器, 控制效果可能不好
    """
    dt: float = 0.001              # 控制器步长 (float)
    dim: int = 1                   # 输入维度 (int)
    # 跟踪微分器
    h: Optional[float] = None      # 滤波因子，系统调用步长，默认None设置成dt (float)
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
    
    def build(self):
        """构建ADRC控制器"""
        return ADRC(self)



# ADRC自抗扰算法
class ADRC(BaseController):
    """ADRC自抗扰控制"""

    def __init__(self, cfg: ADRCConfig):
        super().__init__()
        self.dt = cfg.dt         # 仿真步长
        self.dim = cfg.dim       # 反馈信号y和跟踪信号v的维度
        # TD超参
        self.r = self._reshape_param(cfg.r, self.dim) # 快速跟踪因子
        self.h = self._reshape_param(cfg.h, self.dim) # 滤波因子，系统调用步长
        # ESO超参
        self.b0 = self._reshape_param(cfg.b0, self.dim)             # 系统系数
        self.delta = self._reshape_param(cfg.delta, self.dim)       # fal(e, alpha, delta)函数线性区间宽度        
        self.eso_beta01 = self._reshape_param(cfg.eso_beta01, self.dim) # ESO反馈增益1
        self.eso_beta02 = self._reshape_param(cfg.eso_beta02, self.dim) # ESO反馈增益2
        self.eso_beta03 = self._reshape_param(cfg.eso_beta03, self.dim) # ESO反馈增益3
        # NLSEF超参
        self.nlsef_beta1 = self._reshape_param(cfg.nlsef_beta1, self.dim)   # 跟踪输入信号增益
        self.nlsef_beta2 = self._reshape_param(cfg.nlsef_beta2, self.dim)   # 跟踪微分信号增益
        self.nlsef_alpha1 = self._reshape_param(cfg.nlsef_alpha1, self.dim) # 0 < alpha1 < 1 < alpha2
        self.nlsef_alpha2 = self._reshape_param(cfg.nlsef_alpha2, self.dim) # alpha2 > 1
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
    def __call__(self, y, v=None, *, ctrl_method=1) -> NdArray:
        y = np.array(y)
        v = np.array(v) if v is not None else np.zeros_like(y)
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
        self.z1 = self.z1 + self.h * (self.z2 - self.eso_beta01 * e)
        self.z2 = self.z2 + self.h * (self.z3 - self.eso_beta02 * fe + self.b0 * self.u)
        self.z3 = self.z3 + self.h * (- self.eso_beta03 * fe1)
    
    # 非线性状态误差反馈控制律
    def _NLSEF(self, e1: NdArray, e2: NdArray, ctrl_method=1) -> NdArray:
        ctrl_method %= 4
        if ctrl_method == 0:
            u0 = self.nlsef_beta1 * e1 + self.nlsef_beta2 * e2
        elif ctrl_method == 1:
            u0 = self.nlsef_beta1 * self._fal(e1, self.nlsef_alpha1, self.delta) + self.nlsef_beta2 * self._fal(e2, self.nlsef_alpha2, self.delta)
        elif ctrl_method == 2:
            u0 = -self.fhan(e1, e2, self.r, self.h)
        else:
            c = 1.5
            u0 = -self.fhan(e1, c*e2, self.r, self.h)
        return u0 # (dim, )
    
    @staticmethod
    def _fhan(x1: NdArray, x2: NdArray, r: NdArray, h: NdArray) -> NdArray:
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
    def _fal(err: NdArray, alpha: Union[NdArray, float], delta: NdArray) -> NdArray:
        if not isinstance(alpha, NdArray):
            alpha = np.ones_like(err) * alpha
        fa = np.zeros_like(err)
        mask = np.abs(err) <= delta
        fa[mask] = err[mask] / delta[mask]**(alpha[mask] - 1)
        fa[~mask] = np.abs(err[~mask])**alpha[~mask] * np.sign(err[~mask])
        return fa
    
    # 输出
    def show(self, name='', save_img=False, interference: ListLike = None):
        """控制器控制效果绘图输出
        :param name: str, 图像名称
        :param save_img: bool, 是否存储绘图
        :param interference: ListLike, 实际干扰数据, 用于对比ADRC控制器估计的干扰是否准确
        """
        # 响应曲线 与 控制曲线
        super().show(name=name, save_img=save_img)
        # TD曲线
        self._add_figure(name=name, title='Tracking Differentiator (TD)', t=self.logger.t,
                     y1=self.logger.v1, y1_label='td',
                     y2=self.logger.v, y2_label='input',
                     xlabel='time', ylabel='response signal', save_img=save_img)
        # 误差曲线
        self._add_figure(name=name, title='Error Curve', t=self.logger.t,
                     y1=self.logger.e1, y1_label='error',
                     xlabel='time', ylabel='error signal', save_img=save_img)
        self._add_figure(name=name, title='Differential of Error Curve', t=self.logger.t,
                     y1=self.logger.e2, y1_label='differential estimation of error',
                     xlabel='time', ylabel='error differential signal', save_img=save_img)
        # 干扰估计曲线
        if interference is not None:
            interference = interference if len(interference) == len(self.logger.t) else None
        self._add_figure(name=name, title='Interference Estimation', t=self.logger.t,
                     y1=self.logger.z3, y1_label='interference estimation',
                     y2=interference, y2_label='real interference',
                     xlabel='time', ylabel='interference signal', save_img=save_img)
        
    def __repr__(self):
        info = \
f"""{self.__class__.__name__} Controller (dt={self.dt}):
    h={self.h}, r={self.r}, b0={self.b0}, delta={self.delta}
    eso_beta01={self.eso_beta01}, eso_beta02={self.eso_beta02}, eso_beta03={self.eso_beta03}
    nlsef_beta1={self.nlsef_beta1}, nlsef_beta2={self.nlsef_beta2}, nlsef_alpha1={self.nlsef_alpha1}, nlsef_alpha2={self.nlsef_alpha2}
    u_min={self.u_min}, u_max={self.u_max}"""
        return info