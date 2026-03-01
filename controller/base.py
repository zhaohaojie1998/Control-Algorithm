# -*- coding: utf-8 -*-
"""
控制器基类
Created on Sat Jul 23 23:59:33 2022

@author: https://github.com/zhaohaojie1998
"""

''' 控制器 '''
from typing import Optional, Union, Literal
from abc import ABC, abstractmethod

import pathlib
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from .types import ListLike, SignalLike, NdArray, ScalarLike


class Logger:
    pass


# 控制器
class BaseController(ABC):
    def __init__(self):
        # common参数
        self.dt = 0.001  # 控制器步长
        
        # 绘图数据存储器
        self.logger = Logger()
        self.logger.t = [] # 时间
        self.logger.u = [] # 控制量
        self.logger.y = [] # 状态或观测
        self.logger.v = [] # 参考信号
    
    def __repr__(self):
        return f"{self.__class__.__name__} (dt={self.dt})"

    @staticmethod
    def _reshape_param(param: Union[float, list[float], NdArray], dim: int) -> NdArray:
        """convert param to ndarray, shape=(dim, )"""
        param = np.asarray(param).flatten() # (dim, ) or (1, )
        if param.size != dim:
            assert param.size == 1, "param为float或dim维的ArrayLike"
            return np.repeat(param, dim) # (dim, )
        return param
    
    @staticmethod
    def _reshape_scalar(value: Union[ScalarLike, list[ScalarLike], NdArray], dim: int, mode: Literal['vector', 'eye']) -> NdArray:
        """convert number to eye(dim, dim) or vector(dim, )"""
        value = np.asarray(value)
        # matrix case
        if value.ndim == 2:
            assert value.shape == (dim, dim) and mode == 'eye', "矩阵参数必须为dim*dim方阵"
            return value
        
        # vector or scalar case
        value = value.flatten()
        if value.size != 1 and value.size != dim:
            raise ValueError(f"number={value} 不能转换为 dim={dim} 维的向量或方阵")
        
        if mode == 'vector':
            return np.full((dim, ), value) # (dim, )
        elif mode == 'eye':
            return np.eye(dim) * value # (dim, dim)
        else:
            raise ValueError(f"mode={mode}")
    
    @abstractmethod
    def __call__(self, x_or_y: SignalLike, v: Optional[SignalLike] = None) -> np.ndarray:
        """控制器接口

        Controller
        ------
        控制y信号跟踪v信号, 输出控制量u\n
                         ————————             ——————————         \n
         ref_signal ---> | ctrl | ---- u ---> | system | ---> y  \n
                         ————————             ——————————         \n
                             ↑                    |              \n
                             ---- real_signal -----              \n
        Params
        ------
        x_or_y : SignalLike (标量或向量)
            控制器反馈信号, 即实际信号
        v : Optional[SignalLike] (标量或向量)
            控制器跟踪的参考信号, 即理想信号; 对于状态/输出调节器, 设置为None

        Return
        ------
        u : ndarray (向量)
            输出控制量u, 输入为标量时输出也为向量
        """
        raise NotImplementedError
    
    # 绘图相关
    def _get_save_dir(self, name='') -> pathlib.Path:
        """获取图像保存目录
        :param name: str, 控制器名称
        """
        if not name:
            save_dir = pathlib.Path('figure', self.__class__.__name__)
        else:
            save_dir = pathlib.Path('figure', f"{name} {self.__class__.__name__}")
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir
    
    def show(self, name='', save_img=False):
        """控制器控制效果绘图输出
        :param name: str, 控制器名称
        :param save_img: bool, 是否存储绘图
        """
        # 响应曲线
        self._add_figure(name=name, title='Response Curve', t=self.logger.t,
                     y1=self.logger.y, y1_label='response',
                     y2=self.logger.v, y2_label='reference',
                     xlabel='time', ylabel='response signal', save_img=save_img)
        # 控制曲线
        self._add_figure(name=name, title='Control Law', t=self.logger.t,
                     y1=self.logger.u, y1_label='control',
                     xlabel='time', ylabel='control signal', save_img=save_img)
    
    def _add_figure(
            self,
            name: str,
            title: str,
            t: ListLike,
            y1: ListLike,
            y1_label: str,
            y2: ListLike = None,
            y2_label: str = None,
            xlabel: str = 'time',
            ylabel: str = 'signal',
            save_img: bool = False
        ):
        """新增 <时间-信号> 曲线"""
        have_y2 = bool(y2.size) if isinstance(y2, np.ndarray) else bool(y2)
        # 图例添加
        def get_label(label, data: np.ndarray):
            dim = data.shape[1] if len(data.shape) > 1 else 1
            if dim != 1:
                lb = [label+' {}'.format(i+1) for i in range(dim)] # NOTE: format格式化字符串的上古语法, 留着以免失传
            else:
                lb = label
            return lb
        lb1 = get_label(y1_label, np.asarray(y1))
        lb2 = get_label(y2_label, np.asarray(y2)) if have_y2 else y2_label

        # 绘图
        fig_name = f"{name} {self.__class__.__name__} {title}"
        plt.figure(fig_name, (10,5)) # 创建绘图窗口
        plt.clf() # 清除原图像
        plt.plot(t, y1, label=lb1) 
        if have_y2:
            plt.plot(t, y2, label=lb2, linestyle='-.') 
        plt.xlabel(xlabel, fontsize=20) # x轴标签
        plt.ylabel(ylabel, fontsize=20) # y轴标签
        plt.xticks(fontsize=20) # x轴刻度设置
        plt.yticks(fontsize=20) # y轴刻度设置
        plt.grid() # 生成网格
        plt.legend(loc='best', fontsize=20).set_draggable(True) # 显示图例
        plt.title(fig_name, fontsize=20) # 标题
        if save_img:
            path = self._get_save_dir(name) / f"{fig_name} {datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            plt.savefig(path)

    def show_trajectory(self, name='', save_img=False):
        """控制器轨迹跟踪效果绘图输出
        :param name: str, 控制器名称
        :param save_img: bool, 是否存储绘图
        默认不弹出窗口, 需要手动plt.show()或者使用上下文管理器
        """
        v = np.asarray(self.logger.v) # 参考轨迹
        y = np.asarray(self.logger.y) # 实际轨迹
        
        if v.size == 0 or y.size == 0: # shape = (n, )
            return
        dim = v.shape[1] if len(v.shape) > 1 else 0 # shape = (n, dim)
        if dim not in {2, 3}:
            return
        
        x_ref = v[:,0]; y_ref = v[:,1]
        x_real = y[:,0]; y_real = y[:,1]
        if dim == 3:
            z_ref = v[:,2]; z_real = y[:,2]
        
        # 绘制轨迹图
        fig_name = f"{name} {self.__class__.__name__} {dim}D Trajectory Tracking"

        fig = plt.figure(fig_name, (8,8))
        plt.clf()
        if dim == 3:
            ax = fig.add_subplot(projection='3d')
        else:
            ax = fig.add_subplot()
        
        ax.grid()
        ax.axis('auto')
        ax.set_aspect('equal')
        
        ref_data = (x_ref, y_ref) + ((z_ref,) if dim == 3 else ())
        real_data = (x_real, y_real) + ((z_real,) if dim == 3 else ())
        plot_data = [
            (*ref_data, 'r--', '参考轨迹', 2),
            (*real_data, 'b-', '实际轨迹', 2),
            (*(d[0] for d in ref_data), 'ro', '参考起点', 10),
            (*(d[0] for d in real_data), 'bo', '实际起点', 10),
            (*(d[-1] for d in ref_data), 'go', '参考终点', 10),
            (*(d[-1] for d in real_data), 'mo', '实际终点', 10),
        ]
        
        for *coords, fmt, label, size in plot_data:
            if size > 2:  # 标记点
                ax.plot(*coords, fmt, markersize=size, label=label)
            else:  # 轨迹线
                ax.plot(*coords, fmt, linewidth=size, label=label)
        
        if dim == 3:
            ax.set_zlabel('Z', fontsize=14)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.legend(loc='best', fontsize=12).set_draggable(True)
        plt.title(fig_name, fontsize=16)

        if save_img:
            path = self._get_save_dir(name) / f"{fig_name} {datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
            plt.savefig(path)