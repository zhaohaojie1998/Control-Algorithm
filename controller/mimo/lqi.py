# -*- coding: utf-8 -*-
"""
LQI控制器 (带积分器的LQR控制器, 用于输出跟踪问题)
模型:
    dx/dt = A x + B u
    y = C x + D u

@author: https://github.com/zhaohaojie1998
"""

''' LQI '''
# model based controller
from typing import Union, Optional

import numpy as np

from ..core import LTISystem
from ..base import BaseController
from ..types import MatLike, SignalLike, ScalarLike, ListLike, NdArray

__all__ = [
    "LQI"
]


class LQI(BaseController):
    """线性二次型积分型控制器"""

    def __init__(
        self,
        system: LTISystem,
        Qy: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        dt: float,
        handle_cross_terms: bool = False,
    ):
        """
        Args:
            system (LTISystem): 系统模型
            Qy (Union[MatLike, ScalarLike]): 输出权重矩阵, 标量时扩展为对角矩阵
            R (Union[MatLike, ScalarLike]): 输入权重矩阵, 标量时扩展为对角矩阵
            dt (float): 控制器步长
            handle_cross_terms (bool, optional): 是否考虑由D引起的输出交叉项 2 x^T C^T Qy D u, 默认为False
        """
        
        super().__init__()
        self.t = 0.0
        self.dt = dt

        # 系统模型
        self.system = system

        # 求解LQI问题
        self.controler, lqi_info = self.system.design_lqi(Qy, R, handle_cross_terms)
        self.Kx = lqi_info["Kx"]
        self.Ki = lqi_info["Ki"]
        self.λ = lqi_info["λ"]
        self.stable = lqi_info["stable"]
        
        # 绘图数据
        self.Qy = self._reshape_scalar(Qy, self.system.dim_y, mode='diag') # 用于计算指标
        self.R = self._reshape_scalar(R, self.system.dim_u, mode='diag') # 用于计算指标
        self.J = 0.0 # 性能指标
        self.logger.J = [] # 指标曲线
        self.logger.e = [] # 误差曲线
        self.logger.i = [] # 误差积分曲线

    @property
    def discrete(self) -> bool:
        """是否为离散时间系统"""
        return self.system.discrete
    
    # 重置控制器状态
    def reset(self, integral: Optional[SignalLike] = None):
        """
        Args:
            integral (Optional[SignalLike]): 初始积分值, 维度(dim_y, ), 默认None为全零向量
        """
        if integral is not None:
            integral = np.asarray(integral).ravel()
            assert integral.size == self.system.dim_y, "初始积分值维度必须为dim_y"
        super().reset()
        self.controler.reset(integral) # 重置积分项
        self.t = 0.0
        self.J = 0.0

    # LQI控制器（x为状态向量的观测值，y为系统输出向量，v为参考输出向量）
    def __call__(self, x: SignalLike, y: SignalLike, v: SignalLike) -> NdArray:
        """
        Args:
            x (SignalLike): 状态向量的观测值
            y (SignalLike): 系统输出向量
            v (SignalLike): 要跟踪的参考输出
        
        Returns:
            u (NdArray): LQI控制量
        """
        x = np.array(x).ravel()
        assert x.size == self.system.dim_x, "x必须为状态向量, 维度必须为dim_x"
        y = np.array(y).ravel()
        assert y.size == self.system.dim_y, "y必须为输出向量, 维度必须为dim_y"
        v = np.array(v).ravel()
        assert v.size == self.system.dim_y, "v必须为参考向量, 维度必须为dim_y"

        # 控制律求解
        u = self.controler(x, y, v, dt=self.dt)
        self.t += self.dt

        # 绘图数据求解
        error = v - y
        if self.discrete: # scipy求解器使用不带1/2的性能指标
            self.J += (error.T @ self.Qy @ error + u.T @ self.R @ u)
        else:
            self.J += (error.T @ self.Qy @ error + u.T @ self.R @ u) * self.dt

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.y.append(y)
        self.logger.u.append(u)
        self.logger.v.append(v)
        self.logger.J.append(self.J)
        self.logger.e.append(error)
        self.logger.i.append(self.controler.integral)
        return u

    # 绘图输出
    def show(self, name='', save_img=False, real_response: Optional[ListLike] = None):
        """
        Args:
            name (str): 控制器名称
            save_img (bool): 是否存储绘图
            real_response (Optional[ListLike]): 实际响应, 非None时覆盖由观测器计算得到的假响应, 从而使绘制的响应曲线更真实
        """
        if real_response is not None and len(real_response) == len(self.logger.t):
            self.logger.y = real_response
        super().show(name=name, save_img=save_img)
        # 性能指标曲线
        self._add_figure(name=name, title='Performance Index', t=self.logger.t,
                            y1=self.logger.J, y1_label='J',
                            xlabel='time', ylabel='total', save_img=save_img)
        # 误差曲线
        self._add_figure(name=name, title='Error Curve', t=self.logger.t,
                            y1=self.logger.e, y1_label='error',
                            xlabel='time', ylabel='error signal', save_img=save_img)
        # 误差积分曲线
        self._add_figure(name=name, title='Integration of Error Curve', t=self.logger.t,
                            y1=self.logger.i, y1_label='integration of error',
                            xlabel='time', ylabel='error integration signal', save_img=save_img)
    
    def __repr__(self):
        info = \
f"""{self.__class__.__name__} Controller (dt={self.dt}, discrete={self.discrete}):
    Kx = {self.Kx},
    Ki = {self.Ki},
    λ = {self.λ},
    stable = {self.stable}"""
        return info