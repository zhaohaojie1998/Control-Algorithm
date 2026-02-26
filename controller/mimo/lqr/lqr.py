# -*- coding: utf-8 -*-
"""
LQR控制器 (无限时域+线性时不变系统)

@author: https://github.com/zhaohaojie1998
"""

''' LQR '''
from typing import Union, Optional

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ...base import BaseController
from ...types import MatLike, SignalLike, ScalarLike, ListLike


def solve_algebraic_riccati(A, B, Q, R, discrete=False):
    """求解代数黎卡提方程
    连续: A'P + PA - (PB) R^(-1) (B'P) + Q = 0
    离散: A'PA - P - (A'PB) (R + B'PB)^(-1) (B'PA) + Q = 0
    '表示转置
    """
    if discrete:
        P = solve_discrete_are(A, B, Q, R)
    else:
        P = solve_continuous_are(A, B, Q, R)
    return P


class LQR_StateRegulator(BaseController):
    """LQR状态调节器"""
    def __init__(
        self,
        A: MatLike,
        B: MatLike,
        Q: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        dt: float,
        discrete: bool = False,
        compute_J: bool = True,
    ):
        """LQR状态调节器
        Parameters
        ----------
        A : MatLike
            线性定常系统 A 矩阵, dim_x*dim_x维
        B : MatLike
            线性定常系统 B 矩阵, dim_x*dim_u维
        Q : MatLike | ScalarLike
            性能指标中的 Q 矩阵, dim_x*dim_x维, 取float时设置成 float*单位阵
        R : MatLike | ScalarLike
            性能指标中的 R 矩阵, dim_u*dim_u维, 取float时设置成 float*单位阵
        dt : float 
            控制器步长
        discrete : bool, optional
            是否为离散系统, 默认 False
        compute_J : bool, optional
            是否计算性能指标, 会造成额外计算开销, 默认 True
        """
        super().__init__()
        self.t = 0.0
        self.dt = dt
        self.discrete = discrete

        self.A = np.asarray(A)
        self.B = np.asarray(B)

        self.dim_x = self.A.shape[0]
        self.dim_u = self.B.shape[1]

        self.Q = self._reshape_scalar(Q, self.dim_x, mode='eye')
        self.R = self._reshape_scalar(R, self.dim_u, mode='eye')

        # 检查矩阵维度
        assert self.A.shape[0] == self.A.shape[1], "A必须为方阵"
        assert self.B.shape[0] == self.A.shape[0], "B矩阵维度必须为(dim_x, dim_u)"
        if hasattr(self, "C"):
            assert self.C.shape[1] == self.A.shape[0], "C矩阵维度必须为(dim_y, dim_x)"

        # 求解代数黎卡提方程和反馈增益
        self.P = solve_algebraic_riccati(self.A, self.B, self.Q, self.R, self.discrete)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P # K = R^(-1) * B^T * P

        # 统计指标
        self.compute_J = compute_J
        self.J = 0.0 # 性能指标
        self.logger.J = []

    def __call__(self, x: SignalLike, v=None) -> SignalLike:
        assert v is None, "状态调节器不需要参考信号"
        x = np.array(x).flatten()
        assert x.size == self.dim_x, "输入必须为状态向量, 维度必须为dim_x"

        # 状态负反馈控制
        u = - self.K @ x # u = -Kx
        self.t += self.dt

        # 计算性能指标
        if self.compute_J:
            if self.discrete:
                self.J += 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u)
            else:
                self.J += 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u) * self.dt
            self.logger.J.append(self.J)

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.y.append(x)
        self.logger.u.append(u)
        self.logger.v.append([0])
        return u
    
    # 绘图输出
    def show(self, name='', save_img=False):
        super().show(name=name, save_img=save_img)
        # 性能指标曲线
        if self.compute_J:
            self._add_figure(name=name, title='Performance Index', t=self.logger.t,
                             y1=self.logger.J, y1_label='J',
                             xlabel='time', ylabel='total', save_img=save_img)   

    def __repr__(self):
        return f"{self.__class__.__name__} (dt={self.dt}, discrete={self.discrete}, K={self.K})"



class LQR_OutputRegulator(LQR_StateRegulator):
    """LQR输出调节器"""
    def __init__(
        self,
        A: MatLike,
        B: MatLike,
        C: MatLike,
        Qy: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        dt: float,
        discrete: bool = False,
        compute_J: bool = True,
    ):
        """LQR输出调节器
        Parameters
        ----------
        A : MatLike
            线性定常系统 A 矩阵, dim_x*dim_x维
        B : MatLike
            线性定常系统 B 矩阵, dim_x*dim_u维
        C : MatLike
            线性定常系统 C 矩阵, dim_y*dim_x维
        Qy : MatLike | ScalarLike
            性能指标中的 Qy 矩阵, dim_y*dim_y维, 取float时设置成 float*单位阵
        R : MatLike | ScalarLike
            性能指标中的 R 矩阵, dim_u*dim_u维, 取float时设置成 float*单位阵
        dt : float 
            控制器步长
        discrete : bool, optional
            是否为离散系统, 默认 False
        compute_J : bool, optional
            是否计算性能指标, 会造成额外计算开销, 默认 True
        """
        self.C = np.asarray(C)
        self.dim_y = self.C.shape[0]
        self.Qy = self._reshape_scalar(Qy, self.dim_y, mode='eye')
        # 等效状态调节器
        Q = self.C.T @ self.Qy @ self.C # Q = C^T * Qy * C
        super().__init__(A, B, Q, R, dt, discrete, compute_J)

    def __call__(self, x: SignalLike, v=None) -> SignalLike:
        x = np.asarray(x).flatten()
        u = super().__call__(x, v)
        self.logger.y[-1] = self.C @ x
        return u

    # 绘图输出
    def show(self, name='', save_img=False, real_output: Optional[ListLike] = None):
        """控制器控制效果绘图输出
        :param name: str, 控制器名称
        :param save_img: bool, 是否存储绘图
        :param real_output: Optional[ListLike], 实际输出, 非None时覆盖由y=Cx计算得到的假输出, 从而使绘制的响应曲线更真实
        """
        if real_output is not None:
            real_output = np.asarray(real_output).reshape(-1, self.dim_y)
            assert real_output.shape[0] == len(self.logger.t), "实际输出长度与时间长度不一致"
            self.logger.y = real_output
        super().show(name=name, save_img=save_img)



class LQR_OutputTracker(BaseController):
    """LQR输出跟踪器"""
    pass
