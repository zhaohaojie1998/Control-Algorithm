# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:43:28 2022

@author: HJ
"""

''' Fuzzy PID '''
# model free controller

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as f_ctrl
from dataclasses import dataclass
from copy import deepcopy

if __name__ == '__main__':
    import sys, os
    ctrl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ctrl包所在的目录
    sys.path.append(ctrl_dir)
    
from ctrl.pid import PID, SignalLike
from ctrl.demo import *

__all__ = ['PIDConfig', 'FuzzyPID']


# FuzzyPID控制器参数
@dataclass
class FuzzyPIDConfig:
    """PID控制算法参数
    :param dt: float, 控制器步长
    :param dim: int, 输入信号维度, 即控制器输入v、y的维度, PID输出u也为dim维
    :param Kp: SignalLike, PID比例增益系数
    :param Ki: SignalLike, PID积分增益系数
    :param Kd: SignalLike, PID微分增益系数
    :param u_max: SignalLike, 控制律上限, 范围: (u_min, inf], 取inf时不设限
    :param u_min: SignalLike, 控制律下限, 范围: [-inf, u_max), 取-inf时不设限
    :param Kaw: SignalLike, 抗积分饱和参数, 最好取: 0.1~0.3, 取0时不抗饱和
    :param max_Kp_add: SignalLike, Kp浮动范围, >0
    :param max_Ki_add: SignalLike, Ki浮动范围, >0
    :param max_Kd_add: SignalLike, Kd浮动范围, >0
    :param max_err: SignalLike, 积分器分离阈值, 模糊系统error输入范围, >0, 取inf时不分离积分器
    :param max_err_sum: SignalLike, 模糊系统error_sum输入范围, >0
    :param max_err_diff: SignalLike, 模糊系统error_diff输入范围, >0
    :param Kf: SignalLike, 前馈控制增益系数, 默认0
    :Type : SignalLike = float (标量) | list / ndarray (一维数组即向量)\n
    """
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
    # Fuzzy控制
    max_Kp_add: SignalLike = 0.5  # 模糊Kp调节阈值, (float or list)
    max_Ki_add: SignalLike = 0.5  # 模糊Kp调节阈值, (float or list)
    max_Kd_add: SignalLike = 0.5  # 模糊Kp调节阈值, (float or list)
    max_err: SignalLike = 10      # 模糊误差输入阈值, 积分器分离阈值, (float or list)
    max_err_sum: SignalLike = 10  # 模糊误差积分输入阈值, (float or list)
    max_err_diff: SignalLike = 10 # 模糊误差微分输入阈值, (float or list)
    # 前馈控制
    Kf: SignalLike = 0.0          # 前馈控制增益 (float or list)
    




# FuzzyPID控制算法
class FuzzyPID(PID):
    """模糊PID控制算法"""

    def __init__(self, cfg: FuzzyPIDConfig):
        super().__init__(cfg)
        self.name = 'FuzzyPID'
        # 模糊输入
        max_err = self._reshape_param(cfg.max_err, self.dim)
        max_err_sum = self._reshape_param(cfg.max_err_sum, self.dim)
        max_err_diff = self._reshape_param(cfg.max_err_diff, self.dim)
        # 模糊输出
        max_Kp_add = self._reshape_param(cfg.max_Kp_add, self.dim)
        max_Ki_add = self._reshape_param(cfg.max_Ki_add, self.dim)
        max_Kd_add = self._reshape_param(cfg.max_Kd_add, self.dim)
        # 超参范围
        self.Kp_max, self.Kp_min = self.Kp + max_Kp_add, self.Kp - max_Kp_add
        self.Ki_max, self.Ki_min = self.Ki + max_Ki_add, self.Ki - max_Ki_add
        self.Kd_max, self.Kd_min = self.Kd + max_Kd_add, self.Kd - max_Kd_add
        # Fuzzy仿真系统
        self.fuzzy_sim = [self._fuzzy_init(*params) for params in zip(max_Kp_add, max_Ki_add, max_Kd_add, max_err, max_err_sum, max_err_diff)]
        
        # 存储器
        self.logger.kp = []
        self.logger.ki = []
        self.logger.kd = []


    # 设置模糊规则
    def _fuzzy_init(self, max_Kp_add=0.5, max_Ki_add=0.5, max_Kd_add=0.5, max_err=10.0, max_err_sum=10.0, max_err_diff=10.0, num=10):
        """ 设置模糊规则(1个dim) """
        # fuzzy input
        f_error = f_ctrl.Antecedent(np.linspace(-max_err, max_err, num), 'error')
        f_error_sum = f_ctrl.Antecedent(np.linspace(-max_err_sum, max_err_sum, num), 'error_sum')
        f_error_diff = f_ctrl.Antecedent(np.linspace(-max_err_diff, max_err_diff, num), 'error_diff')
        # fuzzy output
        f_Kp_add = f_ctrl.Consequent(np.linspace(-max_Kp_add, max_Kp_add, num), 'Kp_add')
        f_Ki_add = f_ctrl.Consequent(np.linspace(-max_Ki_add, max_Ki_add, num), 'Ki_add')
        f_Kd_add = f_ctrl.Consequent(np.linspace(-max_Kd_add, max_Kd_add, num), 'Kd_add')
        # 设置隶属度函数
        f_error.automf(3)
        f_error_sum.automf(3)
        f_error_diff.automf(3)
        f_Kp_add.automf(3)
        f_Ki_add.automf(3)
        f_Kd_add.automf(3)
        # 设置模糊规则 # bug: 暂时有问题
        rules = [
            f_ctrl.Rule(f_error['poor'], f_Kp_add['poor']),
            f_ctrl.Rule(f_error['average'], f_Kp_add['average']),
            f_ctrl.Rule(f_error['good'], f_Kp_add['good']),
            f_ctrl.Rule(f_error_sum['poor'], f_Ki_add['poor']),
            f_ctrl.Rule(f_error_sum['average'], f_Ki_add['average']),
            f_ctrl.Rule(f_error_sum['good'], f_Ki_add['good']),
            f_ctrl.Rule(f_error_diff['poor'], f_Kd_add['poor']),
            f_ctrl.Rule(f_error_diff['average'], f_Kd_add['average']),
            f_ctrl.Rule(f_error_diff['good'], f_Kd_add['good'])
        ]
        # 设置模糊推理系统
        fuzzy_sys = f_ctrl.ControlSystem(rules)
        fuzzy_sim = f_ctrl.ControlSystemSimulation(fuzzy_sys)
        return fuzzy_sim


    # 计算 PID 误差
    def _update_pid_error(self, v, y):
        """在update_gain中计算误差"""
        return


    # 模糊 PID 控制
    def _update_gain(self, v, y):
        """更新PID增益"""
        self.error = (np.array(v) - y).flatten()                   # P偏差
        self.error_diff = (self.error - self.last_error) / self.dt # D偏差
        self.error_sum += self.error * self.dt                     # I偏差

        for i in range(self.dim):
            self.fuzzy_sim[i].input['error'] = self.error[i]
            self.fuzzy_sim[i].input['error_sum'] = self.error_sum[i]
            self.fuzzy_sim[i].input['error_diff'] = self.error_diff[i]
            self.fuzzy_sim[i].compute()
            self.Kp[i] += self.fuzzy_sim[i].output['Kp_add']
            self.Ki[i] += self.fuzzy_sim[i].output['Ki_add']
            self.Kd[i] += self.fuzzy_sim[i].output['Kd_add']

        self.Kp = np.clip(self.Kp, self.Kp_min, self.Kp_max)
        self.Ki = np.clip(self.Ki, self.Ki_min, self.Ki_max)
        self.Kd = np.clip(self.Kd, self.Kd_min, self.Kd_max)


    # 模糊PID控制器
    def __call__(self, v, y, y_expected = None, *, anti_windup_method=1):
        self._update_gain(v, y)
        self.logger.kp.append(self.Kp)
        self.logger.ki.append(self.Ki)
        self.logger.kd.append(self.Kd)
        return super().__call__(v, y, y_expected, anti_windup_method=anti_windup_method)
    

    # 绘图输出
    def show(self, *, save=False, show_img=True):
        super().show(save=save, show_img=False)
        self._figure(fig_name='Proportional Gain', t=self.logger.t,
                     y1=self.logger.kp, y1_label='Kp',
                     xlabel='time', ylabel='gain', save=save)
        self._figure(fig_name='Differential Gain', t=self.logger.t,
                     y1=self.logger.kd, y1_label='Kd',
                     xlabel='time', ylabel='gain', save=save)
        self._figure(fig_name='Integral Gain', t=self.logger.t,
                     y1=self.logger.ki, y1_label='Ki',
                     xlabel='time', ylabel='gain', save=save)
        if show_img:
            self._show_img()
        
        
        
        





'debug'
if __name__ == '__main__':
    with_noise = True
    cfg = FuzzyPIDConfig()
    StepDemo(FuzzyPID, cfg, with_noise)
    #CosDemo(FuzzyPID, cfg, with_noise)
