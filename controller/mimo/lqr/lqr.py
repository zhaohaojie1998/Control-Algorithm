# -*- coding: utf-8 -*-
"""
LQR控制器 (无限时域+线性时不变系统)

@author: https://github.com/zhaohaojie1998
"""

''' LQR '''
from typing import Union

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ...base import BaseController
from ...types import MatLike, SignalLike, ScalarLike


def solve_riccati(A, B, Q, R, discrete=False):
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
        assert self.A.shape[0] == self.A.shape[1], "A必须为方阵"
        self.B = np.asarray(B)
        assert self.B.shape[0] == self.A.shape[0], "B矩阵维度必须为(dim_x, dim_u)"

        self.dim_x = self.A.shape[0]
        self.dim_u = self.B.shape[1]

        self.Q = self._reshape_scalar(Q, self.dim_x, mode='eye')
        self.R = self._reshape_scalar(R, self.dim_u, mode='eye')

        # 求解反馈增益
        self.P = solve_riccati(self.A, self.B, self.Q, self.R, self.discrete)
        self.K = -np.linalg.inv(self.R) @ self.B.T @ self.P # K = -R^(-1) * B^T * P

        # 统计指标
        self.compute_J = compute_J
        self.J = 0.0 # 性能指标
        self.logger.J = []

    def __call__(self, x: SignalLike, v=None) -> SignalLike:
        assert v is None, "状态调节器不需要参考信号"
        x = np.array(x).flatten()

        # 状态反馈控制
        u = self.K @ x
        self.t += self.dt

        # 计算性能指标
        if self.compute_J:
            if self.discrete:
                self.J += 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u)
            else:
                self.J += 0.5 * (x.T @ self.Q @ x + u.T @ self.R @ u) * self.dt
            self.logger.J.append(self.J)

        # loggging
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




class LQR_OutputRegulator(BaseController):
    """LQR输出调节器（直接输出反馈）"""
    def __init__(
        self,
        A: MatLike,
        B: MatLike,
        C: MatLike,
        D: Union[MatLike, None],
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
        D : MatLike | None
            线性定常系统 D 矩阵, dim_y*dim_u维, 取None时设为0
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
        super().__init__()
        self.t = 0.0
        self.dt = dt
        self.discrete = discrete

        self.A = np.asarray(A)
        assert self.A.shape[0] == self.A.shape[1], "A必须为方阵"
        self.B = np.asarray(B)
        assert self.B.shape[0] == self.A.shape[0], "B矩阵维度必须为(dim_x, dim_u)"
        self.C = np.asarray(C)
        assert self.C.shape[1] == self.A.shape[0], "C矩阵维度必须为(dim_y, dim_x)"

        self.dim_x = self.A.shape[0]
        self.dim_y = self.C.shape[0]
        self.dim_u = self.B.shape[1]
        
        self.D = np.zeros((self.dim_y, self.dim_u)) if D is None else np.asarray(D)
        assert self.D.shape == (self.dim_y, self.dim_u), "D矩阵维度必须为(dim_y, dim_u)"
        
        self.Qy = self._reshape_scalar(Qy, self.dim_y, mode='eye')
        self.R = self._reshape_scalar(R, self.dim_u, mode='eye')

        raise NotImplementedError("输出反馈增益计算暂未实现")
        self.F = ... # 输出反馈增益

        # 统计指标
        self.compute_J = compute_J
        self.J = 0.0 # 性能指标
        self.logger.J = []

    def __call__(self, y: SignalLike, v=None) -> SignalLike:
        assert v is None, "输出调节器不需要参考信号"
        y = np.array(y).flatten()
        
        # 输出反馈控制
        u = self.F @ y
        self.t += self.dt

        # 计算性能指标
        if self.compute_J:
            if self.discrete:
                self.J += 0.5 * (y.T @ self.Qy @ y + u.T @ self.R @ u)
            else:
                self.J += 0.5 * (y.T @ self.Qy @ y + u.T @ self.R @ u) * self.dt
            self.logger.J.append(self.J)

        # logging
        self.logger.t.append(self.t)
        self.logger.y.append(y)
        self.logger.u.append(u)
        self.logger.v.append([0])
        return u

    def __repr__(self):
        return f"{self.__class__.__name__} (dt={self.dt}, discrete={self.discrete}, F={self.F})"



class LQR_OutputTracker(BaseController):
    """LQR输出跟踪器"""
    pass
