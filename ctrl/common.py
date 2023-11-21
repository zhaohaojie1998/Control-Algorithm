# -*- coding: utf-8 -*-
"""
控制器基类
Created on Sat Jul 23 23:59:33 2022

@author: HJ https://github.com/zhaohaojie1998
"""

''' 控制器 '''
from typing import Union, Callable
from abc import ABC, abstractmethod
from pathlib import Path
import pylab as pl
from ctrl.utils import get_str_time


ListLike = Union[list, pl.ndarray]
"""绘图数据"""

SignalLike = Union[float, list, pl.ndarray]
"""输入信号/超参, 标量或向量"""

NdArray = pl.ndarray
"""控制信号, 向量"""




class Logger:
    pass


# 控制器
class BaseController(ABC):
    def __init__(self):
        # common参数
        self.name = 'Controller'
        self.dt = 0.001  # 控制器步长
        self.dim = 1     # 反馈信号y和跟踪信号v的维度
        
        # 绘图数据存储器
        self.logger = Logger()
        self.logger.t = [] # 时间
        self.logger.u = [] # 控制
        self.logger.y = [] # 实际信号
        self.logger.v = [] # 输入信号

        # 绘图存储地址
        self.save_dir = Path('figure', self.name)


    def __str__(self):
        return self.name +' Controller'
    
    
    @staticmethod
    def getConfig():
        """获取控制器的Config配置数据类"""
        pass


    @staticmethod
    def _reshape_param(param: Union[float, list[float], NdArray], dim: int) -> NdArray:
        """float | array_like -> ndarray (dim, )"""
        param = pl.array(param).flatten() # (dim0, ) or (1, )
        if len(param) != dim:
            assert len(param) == 1, "param为float或dim维的ArrayLike"
            return param.repeat(dim) # (dim, )
        return param


    @abstractmethod
    def __call__(self, v: SignalLike, y: SignalLike) -> pl.ndarray:
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
    
    
    def show(self, *, save=False, show_img=False):
        """控制器控制效果绘图输出
        :param save: bool, 是否存储绘图
        :param show_img: bool, 是否CMD输出图像
        """
        # 绘图配置
        pl.mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 修复字体bug
        pl.mpl.rcParams['axes.unicode_minus'] = False            # 用来正常显示负号
        pl.mpl.rcParams['figure.figsize'] =  [10, 8]             # 视图窗口大小，英寸表示(默认[6.4,4.8])
        pl.mpl.rcParams['font.size'] = 20                        # 所有字体大小(默认12)
        pl.mpl.rcParams['xtick.labelsize'] = 20                  # X轴标签字体大小
        pl.mpl.rcParams['ytick.labelsize'] = 20                  # Y轴标签字体大小
        pl.mpl.rcParams['axes.labelsize'] = 20                   # X、Y轴刻度标签字体大小
        pl.mpl.rcParams['axes.titlesize'] = 20                   # 坐标轴字体大小
        pl.mpl.rcParams['lines.linewidth'] = 1.5                 # 线宽(默认1.5)
        pl.mpl.rcParams['lines.markeredgewidth'] = 1             # 标记点附近线宽(默认1)
        pl.mpl.rcParams['lines.markersize'] = 6                  # 标记点大小(默认6)
        # pl.mpl.rcParams['agg.path.chunksize'] = 10000          # 解决绘图数据溢出报错
        pl.close('all')                                          # 关闭所有窗口
        # 响应曲线
        self._figure(fig_name='Response Curve', t=self.logger.t,
                     y1=self.logger.y, y1_label='Real Signal',
                     y2=self.logger.v, y2_label='Input Signal',
                     xlabel='time', ylabel='response signal', save=save)
        # 控制曲线
        self._figure(fig_name='Control Law', t=self.logger.t,
                     y1=self.logger.u, y1_label='Control Signal',
                     xlabel='time', ylabel='control signal', save=save)
        # 3D数据轨迹跟踪
        self._figure3D(save=save)

        if show_img:
            pl.show()


    @staticmethod
    def _show_img():
        pl.show()
        

    # 绘制时间-信号曲线
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
            save: bool = False
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
        pl.figure(f"{self.name} {fig_name}", (10,8)) # 创建绘图窗口
        pl.clf() # 清除原图像
        pl.plot(t, y1, label=lb1) 
        if y2 is not None:
            pl.plot(t, y2, label=lb2, linestyle='-.') 
        pl.xlabel(xlabel, fontsize = 20) # x轴标签
        pl.ylabel(ylabel, fontsize = 20) # y轴标签
        pl.xticks(fontsize = 20) # x轴刻度设置
        pl.yticks(fontsize = 20) # y轴刻度设置
        pl.grid() # 生成网格
        pl.legend(loc='best', fontsize = 20).set_draggable(True) # 显示图例
        pl.title(f"{self.name} {fig_name}", fontsize = 20)      # 标题
        if save:
            path = self.save_dir / f"{self.name}  {fig_name} {get_str_time()}.png"
            pl.savefig(path)
            
            
    # 无人机3D轨迹跟踪控制
    def _figure3D(self, fig_name='轨迹跟踪控制', save=False):
        """绘制无人机3D轨迹跟踪控制曲线, dim!=3不绘制"""
        if self.dim != 3:
            return
        v = pl.array(self.logger.v) # 参考轨迹
        y = pl.array(self.logger.y) # 实际轨迹
        x1 = v[:,0]; y1 = v[:,1]; z1 = v[:,2]
        x2 = y[:,0]; y2 = y[:,1]; z2 = y[:,2]
        
        fig = pl.figure(f"{self.name} {fig_name}", (10,8)) # 创建绘图窗口
        pl.clf()                              # 清除原图像       
        ax = fig.add_subplot(projection='3d') # 创建3D绘图
        ax.grid()                             # 生成网格
        ax.axis('auto')
        ax.set_aspect('equal')
        ax.plot(x1,y1,z1,'r',label='Ideal Trajectory')
        ax.plot(x2,y2,z2,'b',label='Real Trajectory')
        ax.set_xlabel('x')                    # x轴标签
        ax.set_ylabel('y')                    # y轴标签
        ax.set_zlabel('z')                    # z轴标签
        ax.legend(loc='best').set_draggable(True) # 显示图例
        pl.title(f"{self.name} {fig_name}")      # 标题
        if save:
            path = path = self.save_dir / f"{self.name}  {fig_name} {get_str_time()}.png"
            pl.savefig(path)






# Search控制器
class BaseSearchController(BaseController):
    """启发搜索控制器"""

    def __init__(self):
        super().__init__()
        self.name = 'DRL Controller'

    @abstractmethod
    def __call__(self, v: SignalLike, plant: Callable[[pl.ndarray], SignalLike]) -> pl.ndarray:
        """控制器输入输出接口

        Ctrller
        ------
        控制y信号跟踪v信号, 输出控制量u\n

                ——————————          ——————————         \n
         v ---> | search | -- u --> |  plant | ---> y  \n
                ——————————          ——————————         \n
                   ↑                    |              \n
                   --------- y ----------              \n

        Params
        ------
        v : SignalLike (标量或向量)
            控制器输入信号, 即理想信号
        plant : u -> y (输入u, 输出y)
            被控对象, 输入u, 输出观测y

        Return
        ------
        u : ndarray (向量)
            输出控制量u, 输入为标量时输出也为向量
        """
        raise NotImplementedError










# DRL控制器
class BaseDRLController(BaseController):
    """强化学习控制器"""

    def __init__(self):
        super().__init__()
        self.name = 'DRL Controller'
        self.model_dir = Path('networks', self.name)

    @abstractmethod
    def train(self, plant: Callable[[SignalLike], SignalLike], max_time_step: int, max_epoch: int):
        """强化学习训练接口"""

        # NOTE plant deepcopy 一下, 防止改变时序
        raise NotImplementedError

    def _env_reset(self):
        raise NotImplementedError
    
    def _env_step(self):
        raise NotImplementedError
