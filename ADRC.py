# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 15:27:34 2022

@author: HJ
"""

''' ADRC '''
# model free controller

import pylab as pl
from pylab import sign, sqrt
from common import BaseController, demo0


class ADRCConfig:
    def __init__(self):
        self.dt = 0.001        # 仿真步长 (float)
        self.dim = 1           # 控制器维度 (int)
        ''' dim > 1 同时超参数为 list 数据类型时相当于同时设计了 dim 个控制器        '''
        ''' 必须满足 len(超参) == dim 或 len(超参) == 1 或 超参为float类型           '''
        ''' dim > 1 时超参数也可为float，此时相当于设计了1个控制器，控制效果可能不好 '''
        
        # 跟踪微分器
        self.h = self.dt       # 滤波因子，系统调用步长 (float)
        self.r = 100           # 快速跟踪因子 (float or list)
        # 扩张状态观测器
        self.b0 = 133          # 被控系统系数 (float or list)
        self.delta = 0.015     # fal(e, alpha, delta)函数线性区间宽度 (float or list)
        self.beta01 = 150      # ESO反馈增益1 (float or list)
        self.beta02 = 250      # ESO反馈增益2 (float or list)
        self.beta03 = 550      # ESO反馈增益3 (float or list)
        # 非线性状态反馈控制率
        self.alpha1 = 200/201  # 0 < alpha1 < 1   (float or list)
        self.alpha2 = 201/200  # alpha2 > 1       (float or list)
        self.beta1 = 10        # 跟踪输入信号增益 (float or list)
        self.beta2 = 0.0009    # 跟踪微分信号增益 (float or list)



''' ADRC自抗扰算法 '''
class ADRC(BaseController):
    def __init__(self, cfg):
        super().__init__()
        self.name = 'ADRC'       # 算法名称
        self.dt = cfg.dt         # 仿真步长
        self.dim = cfg.dim       # 控制器维度n
        # TD超参
        self.r = pl.array(cfg.r).flatten()           # 快速跟踪因子
        self.h = pl.array(cfg.h).flatten()           # 滤波因子，系统调用步长
        # ESO超参
        self.b0 = pl.array(cfg.b0).flatten()         # 系统系数
        self.delta = pl.array(cfg.delta).flatten()   # fal(e, alpha, delta)函数线性区间宽度        
        self.beta01 = pl.array(cfg.beta01).flatten() # ESO反馈增益1
        self.beta02 = pl.array(cfg.beta02).flatten() # ESO反馈增益2
        self.beta03 = pl.array(cfg.beta03).flatten() # ESO反馈增益3
        # NLSEF超参
        self.alpha1 = pl.array(cfg.alpha1).flatten() # 0 < alpha1 < 1 <alpha2
        self.alpha2 = pl.array(cfg.alpha2).flatten()
        self.beta1 = pl.array(cfg.beta1).flatten()   # 跟踪输入信号增益
        self.beta2 = pl.array(cfg.beta2).flatten()   # 跟踪微分信号增益
        
        # 控制器初始化
        self.v1 = pl.zeros(self.dim) # array(n,)
        self.v2 = pl.zeros(self.dim) # array(n,)
        self.z1 = pl.zeros(self.dim) # array(n,)
        self.z2 = pl.zeros(self.dim) # array(n,)
        self.z3 = pl.zeros(self.dim) # array(n,)
        self.u = pl.zeros(self.dim) # array(n,)
        self.t = 0
        
        # 存储器
        self.list_v1 = []    # 观测
        self.list_e1 = []    # 误差1
        self.list_e2 = []    # 误差2
        self.list_z3 = []    # 干扰
    
    # ADRC控制器（v为参考轨迹，y为实际轨迹）
    def __call__(self, v, y):
        # TD
        self.TD(v)
        # ESO
        self.ESO(y)
        self.z1 = pl.nan_to_num(self.z1)
        self.z2 = pl.nan_to_num(self.z2)
        self.z3 = pl.nan_to_num(self.z3)
        # NLSEF
        e1 = self.v1 - self.z1
        e2 = self.v2 - self.z2
        u0 = self.NLSEF(e1, e2)
        # 控制量
        self.u = u0 - self.z3 / self.b0
        
        # 存储绘图数据
        self.list_t.append(self.t)
        self.list_u.append(self.u)
        self.list_y.append(y)
        self.list_v.append(v)    
        self.list_v1.append(self.v1)
        self.list_e1.append(self.v1 - self.z1)
        self.list_e2.append(self.v2 - self.z2)
        self.list_z3.append(self.z3)
        self.t += self.dt
        return self.u
    
    # 跟踪微分器
    def TD(self, v):
        fh = self.fhan(self.v1 - v, self.v2, self.r, self.h)
        self.v1 = self.v1 + self.h * self.v2
        self.v2 = self.v2 + self.h * fh
    
    # 扩张状态观测器
    def ESO(self, y):
        e = self.z1 - y
        fe = self.fal(e, 1/2, self.delta)
        fe1 = self.fal(e, 1/4, self.delta)
        self.z1 = self.z1 + self.h * (self.z2 - self.beta01 * e)
        self.z2 = self.z2 + self.h * (self.z3 - self.beta02 * fe + self.b0 * self.u)
        self.z3 = self.z3 + self.h * (- self.beta03 * fe1)
    
    # 非线性状态反馈控制率
    def NLSEF(self, e1, e2):
        # u0 = self.beta1 * e1 + self.beta2 * e2
        u0 = self.beta1 * self.fal(e1, self.alpha1, self.delta) + self.beta2 * self.fal(e2, self.alpha2, self.delta)
        # u0 = -self.fhan(e1, e2, self.r, self.h)
        # c = 1.5
        # u0 = -self.fhan(e1, c*e2, self.r, self.h)
        return u0
    
    def fhan(self, x1, x2, r, h):
        def fsg(x, d):
            return (sign(x + d) - sign(x - d)) / 2
        d = r * h**2  # array(n,)
        a0 = h * x2   # array(n,)
        y = x1 + a0   # array(n,)
        a1 = sqrt(d * (d + 8*abs(y)))  # array(n,)
        a2 = a0 + sign(y) * (a1 - d) / 2  # array(n,)
        a = (a0 + y) * fsg(y, d) + a2 * (1 - fsg(y, d))  # array(n,)
        fh = -r * (a/d) * fsg(y, d) - r * sign(a) * (1 - fsg(a, d))  # array(n,)
        return fh
    
    def fal(self, e, alpha, delta):
        ##  alpha和delta维度可以为1，也可以为n    ##
        ##  数据类型可以为 int float list array   ##
        alpha = pl.array(alpha).flatten() # array(m,) m = 1 or m = n
        delta = pl.array(delta).flatten() # array(m,) m = 1 or m = n
        alpha = alpha.repeat(self.dim) if len(alpha) == 1 else alpha # array(n,)
        delta = delta.repeat(self.dim) if len(delta) == 1 else delta # array(n,)
        
        fa = pl.zeros(self.dim) # array(n,)
        for i in range(self.dim):
            if abs(e[i]) <= delta[i]:
                fa[i] = e[i] / delta[i]**(alpha[i]-1)
            else:
                fa[i] = abs(e[i])**alpha[i] * sign(e[i])
        return fa
    
    def show(self, save = False, interference = None):
        # 响应曲线 与 控制曲线
        self.basic_plot(save)
        
        # TD曲线
        self._figure(fig_name='Tracking Differentiator (TD)', t=self.list_t,
                     y1=self.list_v1, y1_label='td',
                     y2=self.list_v, y2_label='input',
                     xlabel='time', ylabel='response signal', save=save)
        
        # 误差曲线
        self._figure(fig_name='Error Curve', t=self.list_t,
                     y1=self.list_e1, y1_label='error',
                     xlabel='time', ylabel='error signal', save=save)
        
        self._figure(fig_name='Differential of Error Curve', t=self.list_t,
                     y1=self.list_e2, y1_label='differential estimation of error',
                     xlabel='time', ylabel='error differential signal', save=save)
        
        # 干扰估计曲线
        if interference is not None:
            interference = interference if len(interference) == len(self.t) else None
        self._figure(fig_name='Interference Estimation', t=self.list_t,
                     y1=self.list_z3, y1_label='interference estimation',
                     y2=interference, y2_label='real interference',
                     xlabel='time', ylabel='interference signal', save=save)
        
        # 理想轨迹跟踪
        self._figure3D('轨迹跟踪控制', save=save)

        # 显示图像
        pl.show()
            

    

'debug'
if __name__ == '__main__':
    cfg = ADRCConfig()
    demo0(ADRC, cfg)
