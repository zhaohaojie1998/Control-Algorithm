"""
线性定常系统

@author: https://github.com/zhaohaojie1998
"""
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

from typing import Optional, Literal, Union
from ..base import BaseController
from ..types import MatLike, ScalarLike

from .ltialg import (
    get_controllable_matrix,
    is_controllable,
    get_observable_matrix,
    is_observable,

    is_stable,
    is_lyapunov_stable,

    get_uncontrollable_modes,
    is_uncontrollable_stable,
    get_unobservable_modes,
    is_unobservable_stable,

    solve_lqr,
    solve_lqry,
    solve_lqi,
)
reshape_scalar = BaseController._reshape_scalar

__all__ = [
    "LTISystem",
]


# -------------------------- 线性定常系统 --------------------------

# LTI系统
class LTISystem:
    """线性定常系统
    dx = Ax + Bu
    y = Cx + Du
    """
    def __init__(
        self,
        A: MatLike,
        B: MatLike,
        C: Optional[MatLike] = None,
        D: Optional[MatLike] = None,
        Ts: Optional[float] = None,
    ):
        """线性定常系统
        
        Args:
            A (MatLike): 状态矩阵
            B (MatLike): 输入矩阵
            C (Optional[MatLike]): 输出矩阵, 为None时没有观测方程
            D (Optional[MatLike]): 输出矩阵, 为None时没有观测方程
            Ts (Optional[float]): 采样时间, 为None时为连续时间系统
        """
        if C is None and D is not None:
            raise ValueError("C为None时D必须为None")

        self.A = np.asarray(A)
        self.B = np.asarray(B)
        self.C = np.asarray(C) if C is not None else None
        self.D = np.asarray(D) if D is not None else None
        self.Ts = Ts
        
        self.dim_x = self.A.shape[0]
        self.dim_u = self.B.shape[1]
        self.dim_y = self.C.shape[0] if self.C is not None else None

        # 检查矩阵维度
        assert self.A.shape[0] == self.A.shape[1], "A必须为方阵"
        assert self.B.shape[0] == self.A.shape[0], "B矩阵维度必须为(dim_x, dim_u)"
        if self.C is not None:
            assert self.C.shape[1] == self.A.shape[0], "C矩阵维度必须为(dim_y, dim_x)"
        if self.D is not None:
            assert self.D.shape[0] == self.C.shape[0], "D矩阵维度必须为(dim_y, dim_u)"

        # 能控、能观、稳定
        self.Sc = get_controllable_matrix(self.A, self.B)
        """可控制性矩阵"""
        self.So = get_observable_matrix(self.A, self.C) if self.C is not None else None
        """可观性矩阵"""
        self._is_controllable = nla.matrix_rank(self.Sc) == self.dim_x
        self._is_observable = nla.matrix_rank(self.So) == self.dim_x if self.So is not None else None
        self._is_stable = is_stable(self.A, self.discrete)

        # 是否可镇定
        self._uncontrollable_modes = get_uncontrollable_modes(self.A, self.B)
        self._is_uncontrollable_stable = is_uncontrollable_stable(self.A, self.B, self.discrete)

        # 是否可检测
        self._unobservable_modes = get_unobservable_modes(self.A, self.C) if self.C is not None else None
        self._is_unobservable_stable = is_unobservable_stable(self.A, self.C, self.discrete) if self.C is not None else None

    @property
    def discrete(self):
        """是否离散时间系统"""
        return self.Ts is not None or self.Ts == 0
    
    def step(self, x: MatLike, u: MatLike, dt: Optional[float] = None):
        """执行一步仿真，更新系统状态和输出

        Args:
            x (MatLike): 系统状态
            u (MatLike): 控制量
            dt (Optional[float]): 积分(欧拉法)时间步长. 连续系统必须指定.

        Returns:
            x (MatLike): 系统状态
            y (MatLike, optional): 输出. 无观测方程时为None
        """
        x = np.asarray(x).ravel()
        u = np.asarray(u).ravel()
        if self.discrete:
            next_x = self.A @ x + self.B @ u
        else:
            assert dt is not None, "连续时间系统必须指定积分时间步长"
            x_dot = self.A @ x + self.B @ u
            next_x = x + x_dot * dt
        
        next_y = None
        if self.C is not None:
            next_y = self.C @ next_x
            if self.D is not None:
                next_y += self.D @ u
        return next_x, next_y

    # LTI系统性质
    def is_controllable(self, with_matrix=False):
        """是否完全能控"""
        if with_matrix:
            return self._is_controllable, self.Sc
        return self._is_controllable
    
    def is_observable(self, with_matrix=False):
        """是否完全能观"""
        if with_matrix:
            return self._is_observable, self.So
        return self._is_observable
    
    def is_stable(self):
        """是否稳定"""
        return self._is_stable
    
    def is_lyapunov_stable(self, Q: MatLike):
        """是否Lyapunov稳定"""
        Q = reshape_scalar(Q, self.dim_x, mode="diag")
        return is_lyapunov_stable(self.A, Q, self.discrete)
    
    def get_uncontrollable_modes(self):
        """获取系统的不可控模态（特征值）"""
        return self._uncontrollable_modes
    
    def is_stabilizable(self):
        """判断不可控模态是否稳定 (是否可镇定)"""
        return self._is_uncontrollable_stable
    
    def get_unobservable_modes(self):
        """获取系统的不可观模态（特征值）"""
        return self._unobservable_modes
    
    def is_detectable(self):
        """判断不可观模态是否稳定 (是否可检测)"""
        return self._is_unobservable_stable

    def __repr__(self):
        return (
            f"LTISystem(A={self.A}, B={self.B}, C={self.C}, D={self.D}, "
            f"discrete={self.discrete})"
            f"\n  is_controllable={self._is_controllable}, "
            f"\n  is_observable={self._is_observable}, "
            f"\n  is_stable={self._is_stable}, "
            f"\n  is_stabilizable={self._is_uncontrollable_stable}, "
            f"\n  is_detectable={self._is_unobservable_stable}"
        )
    
    # LQR设计
    def design_lqr(
        self,
        Q: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
    ):
        """
        设计LQR状态调节器
        
        Args:
            Q (MatLike | ScalarLike): 状态权重矩阵, 如果为标量, 则视为对角矩阵
            R (MatLike | ScalarLike): 控制输入权重矩阵, 如果为标量, 则视为对角矩阵

        Returns:
            controller (callable): 状态调节器 u = -Kx
            info (dict): 包含LQR设计信息, 包括状态增益矩阵K, Riccati矩阵P, 特征值λ, 是否稳定stable
        """
        # 能控能观检查
        if not self.is_stabilizable():
            raise RuntimeError("系统不具备可镇定性, 无法设计LQR状态调节器")

        # 调节器问题 u = -Kx, 状态跟踪问题 u = -Ke + uf
        R = reshape_scalar(R, self.dim_u, mode="diag")
        Q = reshape_scalar(Q, self.dim_x, mode="diag")
        K, P, λ, stable = solve_lqr(self.A, self.B, Q, R, self.discrete)
        if not stable:
            print(f"[warning] LQR闭环不稳定, 闭环特征值为{λ}")
        
        controller = lambda x: -K @ x  # 状态调节器问题
        info = {
            "K": K,
            "P": P,
            "λ": λ,
            "stable": stable,
        }

        return controller, info
    
    def design_lqry(
        self,
        Qy: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
    ):
        """
        设计LQR输出调节器, 不支持有D矩阵
        
        Args:
            Qy (MatLike | ScalarLike): 输出权重矩阵, 如果为标量, 则视为对角矩阵
            R (MatLike | ScalarLike): 控制输入权重矩阵, 如果为标量, 则视为对角矩阵

        Returns:
            controller (callable): 输出调节器 u = -Kx
            info (dict): 包含LQR设计信息, 包括状态增益矩阵K, Riccati矩阵P, 特征值λ, 是否稳定stable
        """
        assert self.C is not None, "未设置输出矩阵C, 无法设计LQR输出调节器"
        assert self.D is None, "有D矩阵时LQR输出调节器公式过于复杂, 这里不支持, 请使用LQI输出跟踪器跟踪0信号"
        # 能控能观检查
        if not self.is_stabilizable():
            print("[warning] 系统不具备可镇定性, 无法设计LQR输出调节器")
        if not self.is_detectable():
            print("[warning] 系统不具备可检测定性, 无法设计LQR输出调节器")
        
        # 输出调节器问题: u = -Kx
        R = reshape_scalar(R, self.dim_u, mode="diag")
        Qy = reshape_scalar(Qy, self.dim_y, mode="diag")
        K, P, λ, stable = solve_lqry(self.A, self.B, self.C, Qy, R, self.discrete)
        if not stable:
            print(f"[warning] LQR闭环不稳定, 闭环特征值为{λ}")
        
        controller = lambda x: -K @ x  # 输出调节器问题
        info = {
            "K": K,
            "P": P,
            "λ": λ,
            "stable": stable,
        }

        return controller, info
    
    def design_lqi(
        self,
        Qy: Union[MatLike, ScalarLike],
        R: Union[MatLike, ScalarLike],
        handle_cross_terms: bool = False,
    ):
        """
        设计LQR输出跟踪器 (LQI是输出跟踪器的一种实现)
        
        Args:
            Qy (MatLike | ScalarLike): 输出权重矩阵, 如果为标量, 则视为对角矩阵
            R (MatLike | ScalarLike): 控制输入权重矩阵, 如果为标量, 则视为对角矩阵
            handle_cross_terms (bool): 是否考虑由D引起的输出交叉项 2 x^T C^T Qy D u, 默认False

        Returns:
            controller (callable): 输入跟踪器 u = -Kx @ x - Ki ∫(r - y)dt
            info (dict): 包含LQI设计信息, 包括状态增益矩阵K, 积分增益Ki, Riccati矩阵P, 特征值λ, 是否稳定stable
        """
        assert self.C is not None, "未设置输出矩阵C, 无法设计LQI输出跟踪器"
        if self.D is None:
            handle_cross_terms = False
            D = np.zeros((self.dim_y, self.dim_u))
        else:
            D = self.D
        
        # 能控能观检查 (输出跟踪期望值不为0, 要求相比调节器更严格, 需要完全能控能观)
        if not self.is_controllable():
            print("[warning] 系统不具备能控性, 无法设计LQI输出跟踪器")
        if not self.is_observable():
            print("[warning] 系统不具备能观性, 无法设计LQI输出跟踪器")

        # 输出跟踪器问题: u = -Kx @ x - Ki ∫(r - y)dt
        R = reshape_scalar(R, self.dim_u, mode="diag")
        Qy = reshape_scalar(Qy, self.dim_y, mode="diag")
        Kx, Ki, P, λ, stable = solve_lqi(self.A, self.B, self.C, D, Qy, R, self.discrete, handle_cross_terms)
        if not stable:
            print(f"[warning] LQI闭环不稳定, 闭环特征值为{λ}")
        
        class LQI:
            def __init__(this, Kx: np.ndarray, Ki: np.ndarray, discrete: bool):
                this.Kx = Kx
                this.Ki = Ki
                this.dim_y = Ki.shape[1]
                this.discrete = discrete
                this.integral = np.zeros(this.dim_y)

            def reset(this, integral: np.ndarray = None):
                if integral is not None:
                    this.integral = integral.ravel()
                else:
                    this.integral = np.zeros(this.dim_y)

            def __call__(this, x: np.ndarray, y: np.ndarray, r: np.ndarray, dt: float = None):
                if this.discrete:
                    this.integral += r.ravel() - y.ravel()
                else:
                    assert dt is not None, "连续系统需要提供时间步长dt"
                    this.integral += (r.ravel() - y.ravel()) * dt
                return -this.Kx @ x.ravel() - this.Ki @ this.integral

        controller = LQI(Kx, Ki, self.discrete)
        info = {
            "Kx": Kx,
            "Ki": Ki,
            "P": P,
            "λ": λ,
            "stable": stable,
        }

        return controller, info