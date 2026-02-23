# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:43:28 2022

@author: https://github.com/zhaohaojie1998
"""

''' PID '''
# model free controller
from typing import Literal
from dataclasses import dataclass
import numpy as np

from ..common import BaseController
from ..types import SignalLike, NdArray

__all__ = ['PIDConfig', 'PID', 'IncrementPID']


# PID控制器参数
@dataclass
class PIDConfig:
    """PID控制算法参数
    :param name: str, 控制器名称, 例如position, speed等
    :param dt: float, 控制器步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, PID输出u也为dim维
    :param Kp: SignalLike, PID比例增益系数
    :param Ki: SignalLike, PID积分增益系数
    :param Kd: SignalLike, PID微分增益系数
    :param u_max: SignalLike, 控制律上限, 范围: (u_min, inf], 取inf时不设限
    :param u_min: SignalLike, 控制律下限, 范围: [-inf, u_max), 取-inf时不设限
    :param Kaw: SignalLike, 抗积分饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和
    :param ins_max_err: SignalLike, 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器
    :param Kd: SignalLike, 前馈控制增益系数, 默认0
    :Type : SignalLike = float (标量) | list / ndarray (一维数组即向量)\n
    备注:\n
    dim>1时SignalLike为向量时, 相当于同时设计了dim个不同的PID控制器, 必须满足dim==len(SignalLike)\n
    dim>1时SignalLike为标量时, 相当于设计了dim个参数相同的PID控制器, 控制效果可能不好\n
    """
    name: str = ''               # 控制器名称 (str)
    dt: float = 0.01             # 控制器步长 (float)
    dim: int = 1                 # 输入维度 (int)
    # PID控制器增益
    Kp: SignalLike = 5           # 比例增益 (float or list)
    Ki: SignalLike = 0.0         # 积分增益 (float or list)
    Kd: SignalLike = 0.1         # 微分增益 (float or list)
    # 抗积分饱和
    u_max: SignalLike = float('inf')   # 控制律上限, 范围: (u_min, inf], 取inf时不设限 (float or list)
    u_min: SignalLike = float('-inf')  # 控制律下限, 范围: [-inf, u_max), 取-inf时不设限 (float or list)
    Kaw: SignalLike = 0.2              # 抗饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和 (float or list)
    ins_max_err: SignalLike = float('inf') # 积分器分离阈值, 范围: (0, inf], 取inf时不分离积分器 (float or list)
    # 前馈控制
    Kf: SignalLike = 0.0         # 前馈控制增益 (float or list)

    def build(self, pid_type: Literal["PID", "IncrementPID"] = "PID"):
        """构建PID控制器"""
        if pid_type == 'PID':
            return PID(self)
        else:
            return IncrementPID(self)



# 位置式 PID
class PID(BaseController):
    """位置式PID控制算法"""

    def __init__(self, cfg: PIDConfig):
        super().__init__()
        self.name = cfg.name   # 控制器名称
        self.dt = cfg.dt       # 控制器步长
        self.dim = cfg.dim     # 反馈信号y和跟踪信号v的维度
        # PID超参
        self.Kp = self._reshape_param(cfg.Kp, self.dim) # Kp
        self.Ki = self._reshape_param(cfg.Ki, self.dim) # Ki
        self.Kd = self._reshape_param(cfg.Kd, self.dim) # Kd
        self.Kaw = self._reshape_param(cfg.Kaw, self.dim) / (self.Kd + 1e-8) # Kaw 取 0.1~0.3 Kd
        self.Kf = self._reshape_param(cfg.Kf, self.dim) # Kf
        # 抗积分饱和PID
        self.u_max = self._reshape_param(cfg.u_max, self.dim)
        self.u_min = self._reshape_param(cfg.u_min, self.dim)
        self.ins_max_err = self._reshape_param(cfg.ins_max_err, self.dim) # 分离阈值
        # 控制器初始化
        self.u = np.zeros(self.dim)
        self.error = np.zeros(self.dim)      # 误差
        self.last_error = np.zeros(self.dim) # 上一时刻误差
        self.error_diff = np.zeros(self.dim) # 误差微分
        self.error_sum = np.zeros(self.dim)  # real误差积分
        self.anti_error = np.zeros(self.dim) # anti误差积分
        self.t = 0

        # 存储器
        self.logger.e = []    # 误差
        self.logger.d = []    # 误差微分
        self.logger.i = []    # 误差积分
    
    # 计算 PID 误差
    def _update_pid_error(self, v, y):
        self.error = (np.array(v) - y).flatten()                   # P偏差
        self.error_diff = (self.error - self.last_error) / self.dt # D偏差
        self.error_sum += self.error * self.dt                     # I偏差

    # PID控制器（v为参考轨迹，y为实际轨迹或其观测值）
    def __call__(self, v, y, y_expected = None, *, anti_windup_method=1) -> NdArray:
        # PID误差
        self._update_pid_error(v, y)
        # 抗积分饱和算法
        integration = self._Anti_Windup(anti_windup_method) # 是否积分分离
        # PID+前馈控制
        self.u = self.Kp * self.error + integration + self.Kd * self.error_diff
        if y_expected is not None:
            self.u += self.Kf * (np.array(y_expected) - y)
        self.u = np.clip(self.u, self.u_min, self.u_max)
        self.t += self.dt
        self.last_error[:] = self.error

        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)
        self.logger.e.append(self.error)
        self.logger.d.append(self.error_diff)
        self.logger.i.append(self.error_sum)
        return self.u
    
    # 抗积分饱和算法 + 积分分离
    def _Anti_Windup(self, method = 1):
        """输出积分项"""
        method = method % 2
        # 是否积分分离 (remove=0时分离)
        remove = np.ones(self.dim) 
        mask = np.abs(self.error) > self.ins_max_err # 误差超限时分离积分器
        remove[mask] = 0 
        # 算法0 - 反向累加偏差
        if method == 0:
            gamma = np.ones(self.dim)
            mask1 = self.u > self.u_max and self.error >= 0 # 控制超上限累加负偏差(gamma=1), 正偏差不累加(gamma=0)
            mask2 = self.u < self.u_min and self.error <= 0 # 控制超下限累加正偏差(gamma=1), 负偏差不累加(gamma=0)
            gamma[mask1] = 0
            gamma[mask2] = 0
            self.anti_error += remove * gamma * self.error * self.dt
        # 算法1 - 反馈抑制抗饱和算法
        else:
            antiWindupError = np.clip(self.u, self.u_min, self.u_max) - self.u
            self.anti_error += self.error * self.dt + self.Kaw * antiWindupError # 累计误差加上个控制偏差的反馈量
        # 计算积分项
        integration = remove * self.Ki * self.anti_error
        return integration
    
    # 输出
    def show(self, *, save_img=False, show_img=True):
        # 响应曲线 与 控制曲线
        super().show(save_img=save_img)
        # 误差曲线
        self._figure(fig_name='Error Curve', t=self.logger.t,
                     y1=self.logger.e, y1_label='error',
                     xlabel='time', ylabel='error signal', save_img=save_img)
        self._figure(fig_name='Differential of Error Curve', t=self.logger.t,
                     y1=self.logger.d, y1_label='differential of error',
                     xlabel='time', ylabel='error differential signal', save_img=save_img)
        self._figure(fig_name='Integration of Error Curve', t=self.logger.t,
                     y1=self.logger.i, y1_label='integration of error',
                     xlabel='time', ylabel='error integration signal', save_img=save_img)
        # 显示图像
        if show_img:
            self._show_img()



# 增量式 PID
class IncrementPID(PID):
    """增量式PID控制算法"""

    def __init__(self, cfg: PIDConfig):
        super().__init__(cfg)
        self.name = 'IncrementPID'                # 算法名称
        self.last_last_error = np.zeros(self.dim) # e(k-2)
        
    def __call__(self, v, y, *, anti_windup_method=0):
        # PID误差
        self._update_pid_error(v, y)
        # 抗积分饱和算法
        self.anti_error = np.zeros(self.dim)
        integration = self._Anti_Windup(anti_windup_method)
        # 控制量
        u0 = self.Kp * (self.error - self.last_error) + integration + self.Kd * (self.error - 2*self.last_error + self.last_last_error) / self.dt
        self.u += u0      # note 增量式PID对u进行clip后有超调
        self.t += self.dt
        self.last_last_error[:] = self.last_error
        self.last_error[:] = self.error
        
        # 存储绘图数据
        self.logger.t.append(self.t)
        self.logger.u.append(self.u)
        self.logger.y.append(y)
        self.logger.v.append(v)
        self.logger.e.append(self.error)
        self.logger.d.append(self.error_diff)
        self.logger.i.append(self.error_sum)
        return self.u