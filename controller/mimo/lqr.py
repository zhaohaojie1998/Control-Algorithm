# -*- coding: utf-8 -*-
"""
LQR调节器 (无限时域+线性时不变系统)
模型:
    dx = A x + B u
    y = C x

@author: https://github.com/zhaohaojie1998
"""

''' LQR '''
# model based controller
from typing import Union, Optional

import numpy as np

from ..core import LTISystem
from ..base import BaseController
from ..types import MatLike, SignalLike, ScalarLike, ListLike


class LQR(BaseController):
    mode: str

    def __init__(
        self,
        system: LTISystem,
        Q: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        dt: float,
    ):
        super().__init__()
        self.t = 0.0
        self.dt = dt

        # 系统模型
        self.system = system
        self.is_y_egulator = self.system.C is not None

        # 求解LQR问题
        if self.is_y_egulator:
            self.control_law, lqr_info = self.system.design_lqry(Q, R)
        else:
            self.control_law, lqr_info = self.system.design_lqr(Q, R)
        self.K = lqr_info["K"]
        self.λ = lqr_info["λ"]
        self.stable = lqr_info["stable"]
        
        # 绘图数据
        dim_q = self.system.dim_y if self.is_y_egulator else self.system.dim_x
        self.Q = self._reshape_scalar(Q, dim_q, mode='eye') # 用于计算指标
        self.R = self._reshape_scalar(R, self.system.dim_u, mode='eye') # 用于计算指标
        self.J = 0.0 # 性能指标
        self.logger.J = [] # 指标曲线

    def __call__(self, x: SignalLike) -> SignalLike:
        x = np.array(x).flatten()
        assert x.size == self.system.dim_x, "输入必须为状态向量, 维度必须为dim_x"

        # 控制律求解
        u = self.control_law(x)
        self.t += self.dt

        # 绘图数据求解
        y = self.system.C @ x if self.is_y_egulator else x
        if self.system.discrete:
            self.J += 0.5 * (y.T @ self.Q @ y + u.T @ self.R @ u)
        else:
            self.J += 0.5 * (y.T @ self.Q @ y + u.T @ self.R @ u) * self.dt

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.y.append(y)
        self.logger.u.append(u)
        self.logger.v.append([0])
        self.logger.J.append(self.J)
        return u

    # 绘图输出
    def show(self, name='', save_img=False, real_response: Optional[ListLike] = None):
        """控制器控制效果绘图输出
        :param name: str, 控制器名称
        :param save_img: bool, 是否存储绘图
        :param real_response: Optional[ListLike], 实际响应, 非None时覆盖由观测器计算得到的假响应, 从而使绘制的响应曲线更真实
        """
        if real_response is not None:
            real_response = np.asarray(real_response).reshape(-1, self.system.dim_y)
            assert real_response.shape[0] == len(self.logger.t), "实际response长度与时间长度不一致"
            self.logger.y = real_response
        super().show(name=name, save_img=save_img)
        # 性能指标曲线
        if self.logger.J:
            self._add_figure(name=name, title='Performance Index', t=self.logger.t,
                             y1=self.logger.J, y1_label='J',
                             xlabel='time', ylabel='total', save_img=save_img)
    
    def __repr__(self):
        info = \
f"""{self.__class__.__name__} Controller (dt={self.dt}, discrete={self.system.discrete}):
    K = {self.K},
    λ = {self.λ},
    stable = {self.stable}"""
        return info