# -*- coding: utf-8 -*-
"""
控制器基类
Created on Sat Jul 23 23:59:33 2022

@author: https://github.com/zhaohaojie1998
"""

''' 控制器 '''
from typing import Callable, Union
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_str_time
from .types import ListLike, SignalLike, NdArray


class Logger:
    pass


# 控制器
class BaseController(ABC):
    def __init__(self):
        # common参数
        self.name = ''
        self.dt = 0.001  # 控制器步长
        self.dim = 1     # 反馈信号y和跟踪信号v的维度
        
        # 绘图数据存储器
        self.logger = Logger()
        self.logger.t = [] # 时间
        self.logger.u = [] # 控制
        self.logger.y = [] # 实际信号
        self.logger.v = [] # 输入信号
    
    @property
    def save_dir(self) -> Path:
        return Path('figure', str(self))

    def __str__(self):
        if not self.name:
            return self.__class__.__name__
        return f"{self.name} {self.__class__.__name__}"

    @staticmethod
    def _reshape_param(param: Union[float, list[float], NdArray], dim: int) -> NdArray:
        """float | array_like -> ndarray (dim, )"""
        param = np.array(param).flatten() # (dim0, ) or (1, )
        if len(param) != dim:
            assert len(param) == 1, "param为float或dim维的ArrayLike"
            return np.repeat(param, dim) # (dim, )
        return param

    @abstractmethod
    def __call__(self, v: SignalLike, y: SignalLike) -> np.ndarray:
        """控制器输入输出接口

        Ctrller
        ------
        控制y信号跟踪v信号, 输出控制量u\n
                ————————          ——————————         \n
         v ---> | ctrl | -- u --> | system | ---> y  \n
                ————————          ——————————         \n
                   ↑                  |              \n
                   -------- y ---------              \n
        Params
        ------
        v : SignalLike (标量或向量)
            控制器输入信号, 即理想信号
        y : SignalLike (标量或向量)
            控制器反馈信号, 即实际信号

        Return
        ------
        u : ndarray (向量)
            输出控制量u, 输入为标量时输出也为向量
        """
        raise NotImplementedError
    
    def show(self, *, save_img=False, show_img=True):
        """控制器控制效果绘图输出
        :param save_img: bool, 是否存储绘图
        :param show_img: bool, 是否CMD输出图像
        """
        # 绘图配置
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        plt.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
        plt.rcParams['figure.figsize'] =  [10, 8]             # 视图窗口大小，英寸表示(默认[6.4,4.8])
        plt.rcParams['font.size'] = 20                        # 所有字体大小(默认12)
        plt.rcParams['xtick.labelsize'] = 20                  # X轴标签字体大小
        plt.rcParams['ytick.labelsize'] = 20                  # Y轴标签字体大小
        plt.rcParams['axes.labelsize'] = 20                   # X、Y轴刻度标签字体大小
        plt.rcParams['axes.titlesize'] = 20                   # 坐标轴字体大小
        plt.rcParams['lines.linewidth'] = 1.5                 # 线宽(默认1.5)
        plt.rcParams['lines.markeredgewidth'] = 1             # 标记点附近线宽(默认1)
        plt.rcParams['lines.markersize'] = 6                  # 标记点大小(默认6)
        # plt.rcParams['agg.path.chunksize'] = 10000          # 解决绘图数据溢出报错
        plt.close('all')                                          # 关闭所有窗口
        # 响应曲线
        self._figure(fig_name='Response Curve', t=self.logger.t,
                     y1=self.logger.y, y1_label='Real Signal',
                     y2=self.logger.v, y2_label='Input Signal',
                     xlabel='time', ylabel='response signal', save_img=save_img)
        # 控制曲线
        self._figure(fig_name='Control Law', t=self.logger.t,
                     y1=self.logger.u, y1_label='Control Signal',
                     xlabel='time', ylabel='control signal', save_img=save_img)
        # 3D/2D轨迹曲线
        self._figure3D(save_img=save_img)
        self._figure2D(save_img=save_img)

        if show_img:
            plt.show()

    @staticmethod
    def _show_img():
        plt.show()
    
    def _figure(
            self, 
            fig_name: str, 
            t: ListLike, 
            y1: ListLike, 
            y1_label: str, 
            y2: ListLike = None, 
            y2_label: str = None, 
            xlabel: str = 'time', 
            ylabel: str = 'signal', 
            save_img: bool = False
        ):
        """绘制时间-信号曲线"""
        # 图例添加
        def get_label(label):
            if self.dim != 1:
                lb = [label+' {}'.format(i+1) for i in range(self.dim)] # NOTE: format格式化字符串的上古语法, 留着以免失传
            else:
                lb = label
            return lb
        lb1 = get_label(y1_label)
        lb2 = get_label(y2_label) if y2_label is not None else y2_label

        # 绘图
        fig_name = f"{str(self)} {fig_name}"
        plt.figure(fig_name, (10,8)) # 创建绘图窗口
        plt.clf() # 清除原图像
        plt.plot(t, y1, label=lb1) 
        if y2 is not None:
            plt.plot(t, y2, label=lb2, linestyle='-.') 
        plt.xlabel(xlabel, fontsize=20) # x轴标签
        plt.ylabel(ylabel, fontsize=20) # y轴标签
        plt.xticks(fontsize=20) # x轴刻度设置
        plt.yticks(fontsize=20) # y轴刻度设置
        plt.grid() # 生成网格
        plt.legend(loc='best', fontsize=20).set_draggable(True) # 显示图例
        plt.title(fig_name, fontsize=20) # 标题
        if save_img:
            path = self.save_dir / f"{fig_name} {get_str_time()}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
    
    def _figure3D(self, save_img=False):
        """绘制3D轨迹跟踪控制曲线, dim!=3不绘制"""
        if self.dim != 3:
            return
        v = np.array(self.logger.v) # 参考轨迹
        y = np.array(self.logger.y) # 实际轨迹
        x_ref = v[:,0]; y_ref = v[:,1]; z_ref = v[:,2]
        x_real = y[:,0]; y_real = y[:,1]; z_real = y[:,2]

        fig_name = f"{str(self)} 3D Trajectory Tracking"
        fig = plt.figure(fig_name, (10,8))
        plt.clf()     
        ax = fig.add_subplot(projection='3d')
        ax.grid()
        ax.axis('auto')
        ax.set_aspect('equal')
        ax.plot(x_ref, y_ref, z_ref, 'r--', label='参考轨迹', linewidth=2)
        ax.plot(x_real, y_real, z_real, 'b-', label='实际轨迹', linewidth=2)
        ax.plot(x_ref[0], y_ref[0], z_ref[0], 'ro', markersize=10, label='参考起点')
        ax.plot(x_real[0], y_real[0], z_real[0], 'bo', markersize=10, label='实际起点')
        ax.plot(x_ref[-1], y_ref[-1], z_ref[-1], 'go', markersize=10, label='参考终点')
        ax.plot(x_real[-1], y_real[-1], z_real[-1], 'mo', markersize=10, label='实际终点')
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Z', fontsize=14)
        ax.legend(loc='best', fontsize=12).set_draggable(True)
        plt.title(fig_name, fontsize=16)
        if save_img:
            path = self.save_dir / f"{fig_name} {get_str_time()}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)
    
    def _figure2D(self, save_img=False):
        """绘制2D轨迹跟踪控制曲线, dim!=2不绘制"""
        if self.dim != 2:
            return
        v = np.array(self.logger.v) # 参考轨迹
        y = np.array(self.logger.y) # 实际轨迹
        x_ref = v[:,0]; y_ref = v[:,1]
        x_real = y[:,0]; y_real = y[:,1]
        
        fig_name = f"{str(self)} 2D Trajectory Tracking"
        plt.figure(fig_name, (10,8))
        plt.clf()
        plt.plot(x_ref, y_ref, 'r--', label='参考轨迹', linewidth=2)
        plt.plot(x_real, y_real, 'b-', label='实际轨迹', linewidth=2)
        plt.plot(x_ref[0], y_ref[0], 'ro', markersize=10, label='参考起点')
        plt.plot(x_real[0], y_real[0], 'bo', markersize=10, label='实际起点')
        plt.plot(x_ref[-1], y_ref[-1], 'go', markersize=10, label='参考终点')
        plt.plot(x_real[-1], y_real[-1], 'mo', markersize=10, label='实际终点')
        plt.xlabel('X', fontsize=14)
        plt.ylabel('Y', fontsize=14)
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='best', fontsize=12).set_draggable(True)
        plt.title(fig_name, fontsize=16)
        if save_img:
            path = self.save_dir / f"{fig_name} {get_str_time()}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path)