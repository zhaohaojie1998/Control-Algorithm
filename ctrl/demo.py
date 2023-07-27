# -*- coding: utf-8 -*-
"""
 Created on Wed Jul 26 2023 18:53:19
 Modified on 2023-7-26 18:53:19
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#

''' 跟踪控制测试用例 '''
import numpy as np
from ctrl.common import BaseDRLController, BaseSearchController
from ctrl.utils import TicToc



__all__ = [
    'StepDemo',
    'CosDemo',
]


# 一维被控模型 (u 为 1 维, 跟踪第 1 个 y)
class PlantModel:
    def __init__(self, dt: float, with_noise=True):
        self.dt = dt
        self.t = 0                             # 初始时刻
        self.x = np.zeros(3, dtype=np.float32) # 初始状态
        self.u = 0                             # 初始控制
        self.with_noise = with_noise           # 是否存在干扰

    def __call__(self, u):
        """更新状态和观测"""
        x_new = np.zeros_like(self.x)
        if self.with_noise:
            f = -25 * self.x[1] + 33 * np.sin(np.pi*self.t) + 0.01*np.random.randn(1)
            x_new[0] = self.x[0] + self.x[1] * self.dt + 0.001*np.random.randn(1)
            x_new[1] = self.x[1] + self.x[2] * self.dt + 0.001*np.random.randn(1)
            x_new[2] = f + 133 * u
        else:
            f = -25 * self.x[1] + 33 * np.sin(np.pi*self.t)
            x_new[0] = self.x[0] + self.x[1] * self.dt
            x_new[1] = self.x[1] + self.x[2] * self.dt
            x_new[2] = f + 133 * u
        
        self.x = x_new
        self.u = u
        self.t += self.dt
        return self.y
    
    @property
    def y(self):
        """观测方程"""
        return self.x[0]





# 一维阶跃信号跟踪Demo      
def StepDemo(algo, cfg, with_noise=True):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = algo(cfg)
    print(ctrl)
    # 生成参考轨迹
    t_list = np.arange(0.0, 10.0, dt)
    v_list = np.sign(np.sin(t_list))
    # 初始化被控对象
    plant = PlantModel(dt, with_noise)
    y = plant.y
    # 训练 DRL 控制器
    if isinstance(ctrl, BaseDRLController):
        ctrl.train(plant, len(t_list), 1000)
    # 仿真
    with TicToc(CN=False):
        for i in range(len(t_list)):
            # 获取参考轨迹
            v = v_list[i]
            # 控制信号产生
            if not isinstance(ctrl, BaseSearchController):
                u = ctrl(v, y)
            else:
                u = ctrl(v, plant)
            # 更新观测
            y = plant(u)
        #end
    #end
    ctrl.show(save = False)




# 一维余弦信号跟踪Demo
def CosDemo(algo, cfg, with_noise=True):
    # 实例化控制算法
    dt = cfg.dt
    ctrl = algo(cfg)
    print(ctrl)
    # 生成参考轨迹
    t_list = np.arange(0.0, 10.0, dt)
    v_list = np.cos(t_list)
    # 初始化被控对象
    plant = PlantModel(dt, with_noise)
    y = plant.y
    # 训练 DRL 控制器
    if isinstance(ctrl, BaseDRLController):
        ctrl.train(plant, len(t_list), 1000)
    # 仿真
    with TicToc(CN=False):
        for i in range(len(t_list)):
            # 获取参考轨迹
            v = v_list[i]
            # 控制信号产生
            if not isinstance(ctrl, BaseSearchController):
                u = ctrl(v, y)
            else:
                u = ctrl(v, plant)
            # 更新观测
            y = plant(u)
        #end
    #end
    ctrl.show(save = False)
