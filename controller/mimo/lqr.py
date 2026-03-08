# -*- coding: utf-8 -*-
"""
LQR调节器 (无限时域+线性时不变系统)
模型:
    dx/dt = A x + B u
    y = C x

@author: https://github.com/zhaohaojie1998
"""

''' LQR '''
# model based controller
from typing import Union, Optional

import numpy as np

from ..core import LTISystem
from ..base import BaseController
from ..types import MatLike, SignalLike, ScalarLike, ListLike, NdArray

__all__ = [
    "LQR"
]


class LQR(BaseController):
    """线性二次型调节器"""

    def __init__(
        self,
        system: LTISystem,
        Q: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        dt: float,
    ):
        """
        Args:
            system (LTISystem): 系统模型, C=None时为状态调节器, 否则为输出调节器
            Q (Union[MatLike, ScalarLike]): 状态/输出权重矩阵, 标量时扩展为对角矩阵
            R (Union[MatLike, ScalarLike]): 输入权重矩阵, 标量时扩展为对角矩阵
            dt (float): 控制器步长
        """
        super().__init__()
        self.t = 0.0
        self.dt = dt

        # 系统模型
        self.system = system
        self.is_y_regulator = self.system.C is not None

        # 求解LQR问题
        if self.is_y_regulator:
            self.control_law, lqr_info = self.system.design_lqry(Q, R)
        else:
            self.control_law, lqr_info = self.system.design_lqr(Q, R)
        self.K = lqr_info["K"]
        self.λ = lqr_info["λ"]
        self.stable = lqr_info["stable"]
        
        # 绘图数据
        dim_q = self.system.dim_y if self.is_y_regulator else self.system.dim_x
        self.Q = self._reshape_scalar(Q, dim_q, mode='diag') # 用于计算指标
        self.R = self._reshape_scalar(R, self.system.dim_u, mode='diag') # 用于计算指标
        self.J = 0.0 # 性能指标
        self.logger.J = [] # 指标曲线

    @property
    def discrete(self) -> bool:
        """是否为离散时间系统"""
        return self.system.discrete
    
    # 重置控制器状态
    def reset(self):
        super().reset()
        self.t = 0.0
        self.J = 0.0

    # LQR控制器（x为状态向量的观测值）
    def __call__(self, x: SignalLike) -> NdArray:
        """
        Args:
            x (SignalLike): 状态向量的观测值

        Returns:
            NdArray: LQR控制量
        """
        x = np.array(x).ravel()
        assert x.size == self.system.dim_x, "输入必须为状态向量, 维度必须为dim_x"

        # 控制律求解
        u = self.control_law(x)
        self.t += self.dt

        # 绘图数据求解
        y = self.system.C @ x if self.is_y_regulator else x
        if self.discrete: # scipy求解器使用不带1/2的性能指标
            self.J += (y.T @ self.Q @ y + u.T @ self.R @ u)
        else:
            self.J += (y.T @ self.Q @ y + u.T @ self.R @ u) * self.dt

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.y.append(y)
        self.logger.u.append(u)
        self.logger.v.append([0])
        self.logger.J.append(self.J)
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
    
    def __repr__(self):
        info = \
f"""{self.__class__.__name__} Controller (dt={self.dt}, discrete={self.discrete}):
    K = {self.K},
    λ = {self.λ},
    stable = {self.stable}"""
        return info