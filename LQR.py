# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 15:19:50 2022

@author: HJ
"""

''' LQR '''
# model based controller
from typing import Union
import pylab as pl
from copy import deepcopy
from dataclasses import dataclass
from common import BaseController, SignalLike, StepDemo

EyeLike = Union[int, float]
"""类似单位矩阵的矩阵简单表示\n
当数据为标量时, 可认为是 矩阵=标量*单位阵
"""

MatLike = Union[list, pl.ndarray]
"""矩阵数据类型"""





# LQR控制器初始参数设置
@dataclass
class LQRConfig:
    """LQR控制算法参数\n

    状态方程
    -------
    dx = Ax + Bu; 
    y = Cx     

    性能指标
    -------  
    J = 1/2 [vf-yf]'Qf[vf-yf] + 1/2 ∫ [v-y]'Q[v-y] + u'Ru dt

    参数设置
    -------
    A : MatLike
        当前时刻 A 矩阵, n*n维
    B : MatLike
        当前时刻 B 矩阵, n*dim_u维
    C : MatLike | None
        当前时刻 C 矩阵, dim*n维, 取None时 y=x, 即x信号跟踪v信号
    x : SignalLike | None
        当前时刻状态向量 x, 向量长度为n, 取None时 x=0
    dt : float 
        控制器步长
    tf : float
        终端时刻 tf, 默认 tf = inf
    Q : MatLike | EyeLike
        当前时刻 Q 矩阵, dim*dim维, 取float时设置成 float*单位阵
    Qf : MatLike | EyeLike
        终端时刻 Qf 矩阵, dim*dim维, 取float时设置成 float*单位阵
    R : MatLike | EyeLike
        当前时刻 R 矩阵, dim_u*dim_u维, 取float时设置成 float*单位阵
    max_iter : int
        求解黎卡提方程最大迭代次数
    err_stop : float
        当黎卡提矩阵误差小于多少时停止迭代
    """

    # 线性系统初始时刻状态方程
    A: MatLike            # 初始 A 矩阵 (n, n    )        
    B: MatLike            # 初始 B 矩阵 (n, dim_u)
    C: MatLike = None     # 初始 C 矩阵 (dim, n  ) 取None表示y=x 
    x: SignalLike = None  # 初始 x 向量 (n,      ) 取None表示x0=0 

    # LQR控制器初始参数
    dt = 0.001                         # 仿真步长 (float)
    tf = pl.inf                        # 终止时刻 (float)
    Q: Union[MatLike, EyeLike] = 1e-4  # 当前时刻 Q 矩阵 (dim, dim), float设置成 float*单位阵 (list or float)
    Qf: Union[MatLike, EyeLike] = 1e-4 # 终止时刻 Q 矩阵 (dim, dim), float设置成 float*单位阵 (list or float)
    R: Union[MatLike, EyeLike] = 1e4   # 当前时刻 R 矩阵 (dim_u, dim_u), float设置成 float*单位阵 (list or float)

    # LQR控制器终端参数
    Cf = ... # Pf = Cf' Qf Cf
    vf = ... # gf = Cf' Qf vf

    # 黎卡提求解参数
    max_iter: int = 1000    # 最大迭代次数 (int)
    err_stop: float = 0.001 # 停止迭代误差 (float)            
    
    def __post_init__(self):
        # list -> ndarray
        self.A = pl.array(self.A)
        self.B = pl.array(self.B)
        if self.C is None:
            self.C = pl.eye(self.A.shape[0])    # C = 1, y = x
        self.C = pl.array(self.C)
        if self.x is None:
            self.x = pl.zeros(self.A.shape[0]) # x0 = 0
        self.x = pl.array(self.x).flatten()   # x: (n, )
        self.Q = pl.array(self.Q)
        self.Qf = pl.array(self.Qf)
        self.R = pl.array(self.R)
        # input dim and output dim
        self.dim = self.C.shape[0]              # C: (dim, n)
        self.dim_u = self.B.shape[-1]           # B: (n, dim_u)
        # reshape Q/R
        if self.Q.ndim == 0:
            self.Q = self.Q * pl.eye(self.dim)  # Q: (dim, dim)
        if self.Qf.ndim == 0:
            self.Qf = self.Qf * pl.eye(self.dim) #NOTE 用 *= 会报错
        if self.R.ndim == 0:
            self.R = self.R * pl.eye(self.dim_u)# R: (dim_u, dim_u)
        


'''
NOTE 设计思路
先判断 tf 有没有界, 再看是否有 C, 最后看是否已知 vt 轨迹


---------- 输出跟踪器 ----------
vt != 0

情况0 tf != inf
Cf = None, 要求C不可变, 不然没法解
vf = None, 没法解
vt 轨迹不知道, 没法解

情况1 tf = 999999999999
vt = const, 时不变系统, 有近似解

---------- 输出调节器 ----------
vt 始终为 0

看是否能控能观, 根据结果求解


---------- 状态调节器 ----------
C = None, vt = 0


---------- 状态跟踪器 ----------
C = None, vt != 0



'''







# 线性二次型调节器(LQR)控制算法
class LQR(BaseController):
    """线性二次型调节器(LQR)控制算法
    >>> dx = Ax + Bu # 状态方程
    >>> y = Cx       # 输出方程
    >>> J = 1/2 [vf-yf]'Qf[vf-yf] + 1/2 ∫ [v-y]'Q[v-y] + u'Ru dt
    """

    def __init__(self, cfg: LQRConfig):
        super().__init__()
        self.name = 'LQR'      # 算法名称
        self.dt = cfg.dt       # 仿真步长
        self.tf = cfg.tf       # 终止时刻
        self.dim = cfg.dim     # 输入维度 dim
        self.dim_u = cfg.dim_u # 控制维度 dim_u
        
        # 模型初始化
        self.A = cfg.A    # t时刻 A 矩阵
        self.B = cfg.B    # t时刻 B 矩阵
        self.C = cfg.C    # t时刻 C 矩阵
        self.x = cfg.x    # t时刻 x 向量

        # LQR参数初始化
        self.Q = cfg.Q    # t 时刻 Q 矩阵
        self.Qf = cfg.Qf  # tf时刻 Q 矩阵 -> 常矩阵
        self.R = cfg.R    # t 时刻 R 矩阵

        # 黎卡提求解参数
        self.max_iter = cfg.max_iter # 黎卡提求解最大迭代次数
        self.err_stop = cfg.err_stop # 黎卡提求解迭代终止条件
        
        # 控制器初始化
        self.u = pl.zeros(self.dim_u) # (dim_u, )
        self.t = 0

        # 求解问题判定, 没法解的问题报错
        if 0:
            raise NotImplementedError("无法求解")
        
        # 存储器
        self.logger.J = []    # 性能指标
        self.logger.e = []    # 跟踪误差
        self.logger.pe = []   # 黎卡提求解误差
    
    # LQR控制器（v为参考轨迹, y为实际轨迹或其观测值）
    def __call__(
            self, 
            v: SignalLike, 
            y: SignalLike, 
            At: MatLike = None, 
            Bt: MatLike = None, 
            Ct: MatLike = None, 
            Qt: Union[MatLike, EyeLike] = None, 
            Rt: Union[MatLike, EyeLike] = None,
        ) -> pl.ndarray:
        """
        控制器输入输出接口

        Ctrller
        ------
        控制y信号跟踪v信号, 输出控制量u\n
        dx = Ax + bu \n
        y = Cx \n
        y -> v

        Params
        ------
        v : SignalLike (标量或向量)
            控制器输入信号, 即理想信号
        y : SignalLike (标量或向量)
            控制器反馈信号, 即实际信号

        Time Varying System Params
        --------------------------
        At : MatLike (矩阵)
            当前时刻A矩阵
        Bt : MatLike (矩阵)
            当前时刻B矩阵
        Ct : MatLike (矩阵)
            当前时刻C矩阵
        Qt : MatLike (矩阵或标量)
            当前时刻Q矩阵, 取标量时 Qt <- Qt*E
        Rt : MatLike (矩阵或标量)
            当前时刻R矩阵, 取标量时 Rt <- Rt*E

        Return
        ------
        u : ndarray (向量)
            输出控制量u, 输入为标量时输出也为向量
        """

        # 线性时变系统
        if At is not None:
            self.A = pl.array(At) # n*n
        if Bt is not None:
            self.B = pl.array(Bt) # n*dim_u
        if Ct is not None:
            self.C = pl.array(Ct) # dim*n
        if Qt is not None:
            self.Q = pl.array(Qt) # dim*dim
            if self.Q.ndim == 0:  # 传入标量情况
                self.Q = self.Q * pl.eye(self.dim)
        if Rt is not None:
            self.R = pl.array(Rt) # dim_u*dim_u
            if self.R.ndim == 0:  # 传入标量情况
                self.R = self.R * pl.eye(self.dim_u)

        # LQR问题求解
        x = pl.array(v-y).flatten()
        

        P = self._solve_riccati(self.A, self.B)
        
        K = pl.inv(self.R) @ self.B.t @ P  # 矩阵运算

        self.u = -K @ x
        
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)

        
        self.t += self.dt
        return self.u
    
    # 求解黎卡提方程
    def _solve_riccati(self, A, B):
        # A'P + PA + Q - PBR^(-1)B'P = 0
        P = None
        return P
            
    
    def show(self, *, save = False):
        # 响应曲线 与 控制曲线
        super().show(save=save)
        
        # 性能指标曲线
        self._figure(fig_name='Performance Index', t=self.logger.t,
                     y1=self.logger.J, y1_label='J',
                     xlabel='time', ylabel='J(x, u)', save=save)
        
        # 显示图像
        pl.show()
        
        
        
